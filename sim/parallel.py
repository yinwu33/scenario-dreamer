"""Sharded rollout: the same step loop, split across worker processes.

The rollout is the DDPO iteration's long pole (~66% of wall clock, measured with
``scripts/profile_ddpo.py``), and almost all of it is pure-numpy per-scene work
in ``sim.world`` -- observation building, metric bookkeeping, rule-based
planning, dynamics -- on a machine with far more cores than the single-process
loop can use. This module hands each worker a contiguous slice of the batch and
lets it run the ordinary ``RolloutRunner`` loop over its own scenes.

WHY THIS IS BIT-EXACT
---------------------
Scenes are independent: every hook writes ``ctx.metrics[<name>][scene_idx]``,
the reward assembles per scene, and ``SimScene.compute_obs`` documents that
splitting one call into subset calls yields identical observations. So sharding
the per-scene work changes nothing -- with ONE exception.

A neural planner's ``plan`` fuses every scene's observations into a single GEMM,
and that result depends on the batch composition: measured on this checkpoint,
logits move by ~1.4e-5 between a 65-row batch and 8-row shards. That is small,
but an argmax over 91 discrete actions can flip on a near-tie, and one flipped
action diverges the whole trajectory. Such a planner therefore does NOT run in
the workers. ``PPOPlanner`` is split into a CPU half (``gather`` / ``scatter``)
and a GPU half (``forward``); the workers run the CPU half, and the parent runs
``forward`` ONCE per step on the concatenation of every worker's gather, in
ascending scene order -- byte-identical to the batch the single-process runner
would have assembled. Planners that are pure per-agent numpy (``idm``) carry no
such coupling and run entirely inside the workers.

Everything else -- the step loop, the hooks, the dynamics -- is literally the
same code object as the single-process path: ``_WorkerRunner`` subclasses
``RolloutRunner`` and overrides exactly two decisions (see ``sim.runner``).

SYNCHRONISATION
---------------
Workers advance in lockstep, one barrier round per step plus one per centrally
batched role::

    worker                                parent
    ------                                ------
    publish live-scene count
                          --- barrier ---
    read total                            read total, stop if 0
                          --- barrier ---
    gather() -> shm                       (per centrally batched role)
                          --- barrier ---
                                          compact shards into scene order
                                          planner.forward(...)  <- the ONE GEMM
                                          write actions/carry to shm
                          --- barrier ---
    scatter() from shm

A worker whose own scenes have all finished keeps participating in the barriers
(contributing zero rows) until every shard is done, so the rounds never desync.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np
import torch
from omegaconf import OmegaConf

from .planners import PlanItem, Planner
from .runner import ROLES, RolloutResult, RolloutRunner, SimulatorConfig
from .scenes import GeneratedScenes, slice_scenes

# A step is milliseconds; this only has to cover the first round, which also
# pays the workers' import and checkpoint load. Hitting it means a deadlock.
BARRIER_TIMEOUT_S = 600.0


# --------------------------------------------------------------------- layout
@dataclass(frozen=True)
class _BufferSpec:
    """Names and shapes of one centrally batched role's shared buffers.

    ``rows_per_worker`` is the per-worker capacity: a worker writes its gathered
    rows at ``rank * rows_per_worker``, and the parent compacts the regions into
    ascending scene order -- exactly the single-process concatenation.
    """

    role: str
    obs: tuple[str, tuple[int, ...]]
    lstm_h: tuple[str, tuple[int, ...]] | None
    lstm_c: tuple[str, tuple[int, ...]] | None
    out_actions: tuple[str, tuple[int, ...]]
    out_h: tuple[str, tuple[int, ...]] | None
    out_c: tuple[str, tuple[int, ...]] | None
    counts: tuple[str, tuple[int, ...]]
    offsets: tuple[str, tuple[int, ...]]
    rows_per_worker: int


class _Shm:
    """Owns shared-memory blocks and hands out numpy views on them.

    Views are cached per (name, shape, dtype) so asking for the same buffer every
    step does not re-wrap the block.
    """

    def __init__(self):
        self._blocks: dict[str, shared_memory.SharedMemory] = {}
        self._views: dict[tuple, np.ndarray] = {}

    def create(self, shape: tuple[int, ...], dtype) -> tuple[str, np.ndarray]:
        nbytes = max(int(np.prod(shape)) * np.dtype(dtype).itemsize, 1)
        blk = shared_memory.SharedMemory(create=True, size=nbytes)
        self._blocks[blk.name] = blk
        view = np.ndarray(shape, dtype=dtype, buffer=blk.buf)
        view[...] = 0
        self._views[(blk.name, tuple(shape), np.dtype(dtype).str)] = view
        return blk.name, view

    def attach(self, name: str, shape: tuple[int, ...], dtype) -> np.ndarray:
        key = (name, tuple(shape), np.dtype(dtype).str)
        if key in self._views:
            return self._views[key]
        if name not in self._blocks:
            self._blocks[name] = shared_memory.SharedMemory(name=name)
        view = np.ndarray(shape, dtype=dtype, buffer=self._blocks[name].buf)
        self._views[key] = view
        return view

    def close(self, unlink: bool) -> None:
        self._views.clear()
        for blk in self._blocks.values():
            blk.close()
            if unlink:
                blk.unlink()
        self._blocks.clear()


# ------------------------------------------------------------------- channels
class _WorkerChannel:
    """The worker's half of the lockstep protocol."""

    def __init__(self, rank: int, num_workers: int, barrier, active_name: str):
        self.rank = rank
        self.num_workers = num_workers
        self.barrier = barrier
        self.shm = _Shm()
        self.active = self.shm.attach(active_name, (num_workers,), np.int64)
        self.specs: dict[str, _BufferSpec] = {}
        self._bufs: dict[str, dict[str, np.ndarray | None]] = {}

    def bind(self, specs: dict[str, _BufferSpec]) -> None:
        """Attach this rollout's buffers (re-sent whenever the parent grows them)."""
        self.specs = specs
        self._bufs = {}
        for role, spec in specs.items():
            buf: dict[str, np.ndarray | None] = {
                "counts": self.shm.attach(*spec.counts, np.int64),
                "offsets": self.shm.attach(*spec.offsets, np.int64),
                "obs": self.shm.attach(*spec.obs, np.float32),
                "out_actions": self.shm.attach(*spec.out_actions, np.int64),
            }
            for key in ("lstm_h", "lstm_c", "out_h", "out_c"):
                ref = getattr(spec, key)
                buf[key] = None if ref is None else self.shm.attach(*ref, np.float32)
            self._bufs[role] = buf

    def sync_active(self, n_active: int) -> bool:
        """Publish this shard's live-scene count; True once EVERY shard is dry."""
        self.active[self.rank] = n_active
        self.barrier.wait(BARRIER_TIMEOUT_S)
        total = int(self.active.sum())
        self.barrier.wait(BARRIER_TIMEOUT_S)
        return total == 0

    def remote_forward(self, gathered: dict[str, dict]) -> dict[str, tuple]:
        """Settle EVERY centrally batched role in one barrier round.

        All roles plan from the same pre-step state, so their gathers are
        independent and can ride one rendezvous. Each extra round would cost a
        full barrier plus the slowest shard's tail, once per simulation step.
        """
        for role, g in gathered.items():
            spec = self.specs[role]
            buf = self._bufs[role]
            rows = int(g["obs"].shape[0])
            if rows > spec.rows_per_worker:
                raise RuntimeError(
                    f"role {role!r}: shard produced {rows} rows but the shared buffer "
                    f"holds {spec.rows_per_worker} per worker"
                )
            base = self.rank * spec.rows_per_worker
            buf["counts"][self.rank] = rows
            if rows:
                buf["obs"][base : base + rows] = g["obs"]
                if buf["lstm_h"] is not None:
                    buf["lstm_h"][base : base + rows] = g["lstm_h"]
                    buf["lstm_c"][base : base + rows] = g["lstm_c"]

        self.barrier.wait(BARRIER_TIMEOUT_S)   # shards written, parent may read
        self.barrier.wait(BARRIER_TIMEOUT_S)   # parent has published results

        out = {}
        for role, g in gathered.items():
            buf = self._bufs[role]
            rows = int(g["obs"].shape[0])
            if rows == 0:
                out[role] = (np.empty(0, dtype=np.int64), None, None)
                continue
            off = int(buf["offsets"][self.rank])
            out[role] = (
                buf["out_actions"][off : off + rows].copy(),
                None if buf["out_h"] is None else buf["out_h"][off : off + rows].copy(),
                None if buf["out_c"] is None else buf["out_c"][off : off + rows].copy(),
            )
        return out


class _WorkerRunner(RolloutRunner):
    """``RolloutRunner`` over one shard, kept in lockstep with its siblings.

    Only the two hook points documented in ``sim.runner`` differ; every other
    line of stepping and hook firing is inherited, which is what makes the
    sharded result bit-exact.
    """

    def __init__(self, planner_cfg, params: SimulatorConfig, *, chan: _WorkerChannel):
        super().__init__(planner_cfg, params, device="cpu")
        self.chan = chan
        self.remote_roles = [r for r in ROLES if self.planners[r].batched_across_scenes]

    def _should_stop(self, active: list[int]) -> bool:
        if not self.remote_roles:
            # Nothing is centrally batched, so no worker ever waits on another
            # and a dry shard can just stop -- exactly as the single-process loop
            # stops once its own scenes are done. This is the whole reason an
            # all-rule-based trio (idm-idm) shards with ZERO synchronisation.
            return not active
        return self.chan.sync_active(len(active))

    def _stage_plans(self, active, sims, role_ids) -> list:
        work = [(role, planner, self._role_items(role, active, sims, role_ids))
                for role, planner in self.planners.items()]
        # Gather every centrally batched role BEFORE the rendezvous, then settle
        # them all in one round. A worker joins even with zero rows: the round is
        # global.
        gathered = {role: planner.gather(items)
                    for role, planner, items in work if planner.batched_across_scenes}
        results = self.chan.remote_forward(gathered) if gathered else {}

        staged = []
        for role, planner, items in work:
            if planner.batched_across_scenes:
                actions, new_h, new_c = results[role]
                staged.append((planner, items,
                               planner.scatter(items, gathered[role], actions, new_h, new_c)))
            elif items:
                staged.append((planner, items, planner.plan(items)))
        return staged


# --------------------------------------------------------------------- worker
def _worker_main(rank, num_workers, planner_cfg, params, barrier, active_name, conn):
    """Persistent worker: build once, then serve rollouts until told to stop."""
    # No worker ever calls forward(), so none of them needs a CUDA context; and
    # the parallelism is across scenes, so a per-worker BLAS pool would only
    # oversubscribe the box.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    try:
        cfg = OmegaConf.create(planner_cfg)
        for role in ROLES:
            # Only the neural planner family declares a device; pin it to the
            # host so importing the checkpoint cannot touch the GPU.
            if "device" in cfg[role]:
                cfg[role].device = "cpu"
        runner = _WorkerRunner(cfg, params, chan=_WorkerChannel(
            rank, num_workers, barrier, active_name))
        conn.send(("ready", rank))

        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                return
            _, specs, lo, hi, scenes, hooks, record = msg
            runner.chan.bind(specs)
            result = runner.rollout(scenes, hooks=hooks, record_trajectories=record)
            conn.send(("result", lo, hi, result.metrics, result.trajectories))
    except BaseException:
        # Surface the failure instead of letting the parent block on a barrier
        # for BARRIER_TIMEOUT_S with no explanation: ship the traceback, then
        # break the barrier so the parent raises immediately.
        conn.send(("error", rank, traceback.format_exc()))
        barrier.abort()
        raise


# --------------------------------------------------------------------- parent
class ParallelRolloutRunner(RolloutRunner):
    """Drop-in ``RolloutRunner`` that shards the batch across worker processes.

    The parent keeps the real (GPU) planners and runs each centrally batched
    role's single forward; the workers do everything else. ``rollout`` has the
    same signature and returns the same ``RolloutResult``.
    """

    def __init__(self, planner_cfg, params: SimulatorConfig, *, num_workers: int,
                 train_batch_size: int, device: str | None = None, scene_block: int = 8):
        super().__init__(planner_cfg, params, device=device)
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        self.num_workers = int(num_workers)
        # Shard on multiples of this so a GRPO context's replicas -- which share
        # one map -- stay in a single worker and keep its lane-grid cache hot.
        self.scene_block = int(scene_block)
        if self.scene_block < 1:
            raise ValueError(f"scene_block must be >= 1, got {scene_block}")
        # Resolve against the LIVE node: the planner yamls interpolate
        # ${project_root}, which only exists while the subtree is attached to
        # its root config. Workers receive the resolved plain dict.
        # A worker pool the TRAINING batch cannot fill is a misconfiguration, and it
        # must fail here rather than after the first checkpoint: every rollout would
        # quietly run single-process and the run would just be slow for days.
        if not self.shardable(int(train_batch_size)):
            raise ValueError(
                f"rollout_workers={self.num_workers} cannot be fed by batch_size="
                f"{train_batch_size} at scene_block={self.scene_block}: that batch "
                f"is only {self._n_blocks(int(train_batch_size))} blocks. Lower "
                f"ddpo.rollout_workers, or lower scene_block (at the cost of "
                f"splitting GRPO context groups across workers)."
            )
        self._planner_cfg = OmegaConf.to_container(planner_cfg, resolve=True)
        # RolloutRunner._apply_conditioning draws from ONE rng, scene by scene,
        # whenever a role's conditioning field is a [lo, hi] range. Each worker
        # owns a separate rng, so a sharded run would consume that stream in a
        # different order and give the agents different driving styles. No
        # current planner yaml uses a range; fail loudly the day one does rather
        # than silently diverging from the single-process result.
        ranged = {
            role: sorted(k for k, v in cond.items() if isinstance(v, tuple))
            for role, cond in self.conditioning.items() if cond is not None
        }
        ranged = {role: keys for role, keys in ranged.items() if keys}
        if ranged:
            raise NotImplementedError(
                f"sim.parallel cannot shard a rollout with per-agent RANDOM conditioning "
                f"{ranged}: the draws come off a single rng in scene order, which sharding "
                f"reorders. Use scalar conditioning values, or run with num_workers=0."
            )
        self._remote_roles = [r for r in ROLES if self.planners[r].batched_across_scenes]
        self._procs: list = []
        self._conns: list = []
        self._barrier = None
        self._ctl = _Shm()      # the `active` counter, created once
        self._shm = _Shm()      # per-role buffers, rebuilt when the batch grows
        self._active = None
        self._active_name = ""
        self._specs: dict[str, _BufferSpec] = {}
        self._capacity = (0, 0)   # (scenes, rows_per_worker)

    # ------------------------------------------------------------ lifecycle
    def _n_blocks(self, num_scenes: int) -> int:
        """How many whole shard units this batch splits into."""
        return -(-num_scenes // self.scene_block)

    def shardable(self, num_scenes: int) -> bool:
        """Can this batch give every worker at least one block?"""
        return self._n_blocks(num_scenes) >= self.num_workers

    def _bounds(self, num_scenes: int) -> list[tuple[int, int]]:
        """Contiguous, block-aligned scene ranges, one per worker."""
        n_blocks = self._n_blocks(num_scenes)
        out = []
        for rank in range(self.num_workers):
            first = (n_blocks * rank) // self.num_workers
            last = (n_blocks * (rank + 1)) // self.num_workers
            out.append((first * self.scene_block,
                        min(last * self.scene_block, num_scenes)))
        return out

    def _start(self) -> None:
        if self._procs:
            return
        ctx = mp.get_context("spawn")
        self._barrier = ctx.Barrier(self.num_workers + 1)
        self._active_name, self._active = self._ctl.create((self.num_workers,), np.int64)
        for rank in range(self.num_workers):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_main,
                args=(rank, self.num_workers, self._planner_cfg, self.params,
                      self._barrier, self._active_name, child_conn),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._procs.append(proc)
            self._conns.append(parent_conn)
        for rank, conn in enumerate(self._conns):
            msg = conn.recv()
            if msg[0] != "ready":
                raise RuntimeError(f"rollout worker {rank} failed to start:\n{msg[-1]}")

    def close(self) -> None:
        for conn in self._conns:
            conn.send(("stop",))
        for proc in self._procs:
            proc.join(timeout=30)
        self._procs, self._conns = [], []
        self._shm.close(unlink=True)
        self._ctl.close(unlink=True)
        self._active, self._active_name = None, ""
        self._specs, self._capacity = {}, (0, 0)

    # -------------------------------------------------------------- buffers
    def _ensure_buffers(self, num_scenes: int, max_agents: int) -> None:
        """(Re)allocate the per-role shared buffers when the batch outgrows them."""
        if not self._remote_roles:
            return
        per_worker_scenes = max(hi - lo for lo, hi in self._bounds(num_scenes))
        rows_per_worker = max(per_worker_scenes * max_agents, self._capacity[1])
        if self._specs and self._capacity[0] >= num_scenes \
                and self._capacity[1] >= rows_per_worker:
            return

        fresh = _Shm()
        total_rows = rows_per_worker * self.num_workers
        specs: dict[str, _BufferSpec] = {}
        for role in self._remote_roles:
            planner = self.planners[role]
            recurrent = bool(planner.recurrent)
            hidden = int(planner.net.hidden_size) if recurrent else 0

            def mk(shape, dtype=np.float32):
                name, _ = fresh.create(shape, dtype)
                return name, tuple(shape)

            specs[role] = _BufferSpec(
                role=role,
                obs=mk((total_rows, int(planner.obs_dim))),
                lstm_h=mk((total_rows, hidden)) if recurrent else None,
                lstm_c=mk((total_rows, hidden)) if recurrent else None,
                out_actions=mk((total_rows,), np.int64),
                out_h=mk((total_rows, hidden)) if recurrent else None,
                out_c=mk((total_rows, hidden)) if recurrent else None,
                counts=mk((self.num_workers,), np.int64),
                offsets=mk((self.num_workers,), np.int64),
                rows_per_worker=rows_per_worker,
            )
        self._shm.close(unlink=True)
        self._shm = fresh
        self._specs = specs
        self._capacity = (max(num_scenes, self._capacity[0]), rows_per_worker)

    def _view(self, ref, dtype=np.float32):
        """Numpy view on a buffer named by a spec field; None for absent optionals."""
        return None if ref is None else self._shm.attach(ref[0], ref[1], dtype)

    # -------------------------------------------------------------- rollout
    @torch.no_grad()
    def rollout(self, scenes: GeneratedScenes, *, hooks: list,
                record_trajectories: bool = False) -> RolloutResult:
        num_scenes = int(scenes.num_scenes)
        if not self.shardable(num_scenes):
            # NOT a fallback that hides a problem: the single-process runner is the
            # reference this class reproduces bit-exactly, so routing to it changes
            # nothing about the result. Training scores small off-cycle batches
            # besides the main one -- the train-group diversity viz scores 4 scenes
            # -- and those cannot fill the pool. A pool the TRAINING batch cannot
            # fill is rejected in __init__ instead.
            return super().rollout(scenes, hooks=hooks,
                                   record_trajectories=record_trajectories)
        bounds = self._bounds(num_scenes)
        self._start()

        a_idx = scenes.agent_scene_idx.detach().cpu().numpy()
        max_agents = int(np.bincount(a_idx, minlength=num_scenes).max())
        self._ensure_buffers(num_scenes, max_agents)

        for conn, (lo, hi) in zip(self._conns, bounds):
            conn.send(("rollout", self._specs, lo, hi,
                       slice_scenes(scenes, lo, hi), hooks, record_trajectories))
        self._drive()

        shards = []
        for conn in self._conns:
            msg = conn.recv()
            if msg[0] == "error":
                raise RuntimeError(f"rollout worker {msg[1]} failed:\n{msg[2]}")
            shards.append(msg)
        return self._merge(shards, num_scenes, record_trajectories)

    def _drive(self) -> None:
        """Parent side of the lockstep loop: service barriers, run the forwards.

        Nothing to drive when no planner is centrally batched: the workers then
        never rendezvous, they just run their own scenes to completion.
        """
        if not self._remote_roles:
            return
        for _ in range(int(self.params.sim_steps)):
            self._barrier.wait(BARRIER_TIMEOUT_S)
            total = int(self._active.sum())
            self._barrier.wait(BARRIER_TIMEOUT_S)
            if total == 0:
                return
            self._barrier.wait(BARRIER_TIMEOUT_S)   # every role's shards written
            for role in self._remote_roles:
                self._central_forward(role)
            self._barrier.wait(BARRIER_TIMEOUT_S)   # results published

    def _central_forward(self, role: str) -> None:
        """The one batched forward for ``role``, on the whole batch in scene order."""
        spec = self._specs[role]
        counts = self._view(spec.counts, np.int64)
        # Compact the per-worker regions into ascending scene order -- byte-identical
        # to the single-process np.concatenate(obs_list).
        rows = np.concatenate([
            np.arange(rank * spec.rows_per_worker,
                      rank * spec.rows_per_worker + int(counts[rank]))
            for rank in range(self.num_workers)
        ]).astype(np.int64)
        self._view(spec.offsets, np.int64)[:] = np.concatenate([[0], np.cumsum(counts)[:-1]])

        lstm_h, lstm_c = self._view(spec.lstm_h), self._view(spec.lstm_c)
        actions, new_h, new_c = self.planners[role].forward(
            self._view(spec.obs)[rows],
            None if lstm_h is None else lstm_h[rows],
            None if lstm_c is None else lstm_c[rows],
        )
        n = rows.shape[0]
        if n:
            self._view(spec.out_actions, np.int64)[:n] = actions
            if new_h is not None:
                self._view(spec.out_h)[:n] = new_h
                self._view(spec.out_c)[:n] = new_c

    def _merge(self, shards, num_scenes: int, record_trajectories: bool) -> RolloutResult:
        """Stitch the shards' per-scene arrays back into one batch-wide result."""
        shards = sorted(shards, key=lambda m: m[1])
        metrics = {}
        for key in shards[0][3]:
            merged = np.concatenate([m[3][key] for m in shards])
            if merged.shape[0] != num_scenes:
                raise RuntimeError(
                    f"metric {key!r} merged to {merged.shape[0]} rows, expected {num_scenes}"
                )
            metrics[key] = merged
        trajectories = None
        if record_trajectories:
            trajectories = [traj for m in shards for traj in m[4]]
        return RolloutResult(metrics=metrics, trajectories=trajectories)
