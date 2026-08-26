"""ppo planner: frozen recurrent PufferDrive PPO net + numpy classic dynamics.

``plan`` gathers the driven agents' observations across scenes, batches them
through the frozen PPO network (the Drive policy port in
``nets/selfplay_drive/net.py``) to discrete actions; ``apply`` integrates them
with the bicycle ``step_dynamics`` model. Checkpoint / net arch / determinism
come from the role's ``cfgs/planner/ppo_*.yaml``.

This one class backs the whole **ppo family**: ``ppo_aggressive`` /
``ppo_normal`` / ``ppo_caution`` are the same checkpoint driven at different
``conditioning.collision_factor`` values (0 = reckless, 2 = maximally
defensive), so the PPO columns of the SUT x traffic table are behaviourally
distinct planners that share one set of weights. Each variant is a separate
``PLANNER_REGISTRY`` entry pointing here.

One instance drives one role's agents (see ``sim.runner.ROLES``); the LSTM
carry is kept per (scene, role), so several ppo instances -- or the same config
reused for every role -- never share or clobber recurrent state. The loaded net
itself is cached per resolved config, so identical role configs share one
frozen network (variants differing only in conditioning therefore cost nothing
extra: the conditioning lives in the agents' obs, not in the weights).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..world import SimScene
from .base import Planner, PlanItem


class PPOPlanner(Planner):
    # One GEMM over every item's observations -> the result depends on which
    # scenes shared the batch; sim.parallel keeps this forward in the parent.
    batched_across_scenes = True

    def __init__(self, planner_cfg, *, role: str, device: str | None = None):
        super().__init__(planner_cfg, role=role, device=device)
        from nets.selfplay_drive.net import OBS_DIM, load_net

        self.net = load_net(planner_cfg)
        self.obs_dim = OBS_DIM
        # `device` (the caller's rollout device) wins when given; otherwise the
        # net's own resolved device is authoritative.
        self.device = device or str(next(self.net.parameters()).device)
        self.deterministic = bool(self._require("deterministic"))
        self.recurrent = self.net.recurrent

    def _lstm_state_for(self, sim: SimScene) -> dict[str, np.ndarray]:
        """This role's recurrent carry for ``sim``, as CPU numpy.

        Kept on the host (not the net's device) so the carry can be gathered and
        scattered by a ``sim.parallel`` worker that owns no GPU. Values are the
        same zeros / same float32 the device tensors held; the single batched
        forward still happens on the net's device, so nothing about the numerics
        moves.
        """
        states = getattr(sim, "_planner_lstm_states", None)
        if states is None:
            states = sim._planner_lstm_states = {}
        state = states.get(self.role)
        if state is None or state["lstm_h"].shape[0] != sim.n:
            state = {
                "lstm_h": np.zeros((sim.n, self.net.hidden_size), dtype=np.float32),
                "lstm_c": np.zeros((sim.n, self.net.hidden_size), dtype=np.float32),
            }
            states[self.role] = state
        return state

    # ------------------------------------------------------------------------
    # plan() is split into a CPU half (gather / scatter) and a GPU half
    # (forward). Single-process, the three run back to back and this is exactly
    # the old plan(). Sharded, sim.parallel runs gather/scatter inside each
    # worker and calls forward ONCE in the parent on the concatenation of every
    # worker's gather -- so the network still sees one batch, in the same scene
    # order, with the same rows. That is what keeps a parallel rollout
    # bit-exact: re-batching this net perturbs its logits by ~1e-5 (enough to
    # flip an argmax at a near-tie and diverge the whole trajectory), so the
    # batch must never be split.
    # ------------------------------------------------------------------------

    def gather(self, items: Sequence[PlanItem]) -> dict:
        """CPU half: observations (+ recurrent carry) for every item, in item order."""
        obs_list = [sim.compute_obs(ids) for sim, ids in items]
        counts = np.array([ob.shape[0] for ob in obs_list], dtype=np.int64)
        total = int(counts.sum())
        out = {
            "obs": np.concatenate(obs_list) if total else np.zeros((0, self.obs_dim), np.float32),
            "counts": counts,
        }
        if self.recurrent and total:
            h, c = [], []
            for sim, ids in items:
                state = self._lstm_state_for(sim)
                h.append(state["lstm_h"][ids])
                c.append(state["lstm_c"][ids])
            out["lstm_h"] = np.concatenate(h)
            out["lstm_c"] = np.concatenate(c)
        return out

    def forward(self, obs: np.ndarray, lstm_h=None, lstm_c=None):
        """GPU half: the ONE batched net forward. Returns (actions, h, c)."""
        if obs.shape[0] == 0:
            return np.empty(0, dtype=np.int64), lstm_h, lstm_c
        obs_t = torch.as_tensor(obs, device=self.device)
        if self.recurrent:
            batch_state = {
                "lstm_h": torch.as_tensor(lstm_h, device=self.device),
                "lstm_c": torch.as_tensor(lstm_c, device=self.device),
            }
            actions = self.net.act(
                obs_t, state=batch_state, deterministic=self.deterministic
            ).cpu().numpy()
            return (
                actions,
                batch_state["lstm_h"].cpu().numpy(),
                batch_state["lstm_c"].cpu().numpy(),
            )
        actions = self.net.act(obs_t, deterministic=self.deterministic).cpu().numpy()
        return actions, None, None

    def scatter(self, items, gathered: dict, actions, lstm_h=None, lstm_c=None) -> list:
        """CPU half: write the carry back and split the actions per item."""
        counts = gathered["counts"]
        if self.recurrent and lstm_h is not None and int(counts.sum()):
            off = 0
            for (sim, ids), n in zip(items, counts):
                n = int(n)
                state = self._lstm_state_for(sim)
                state["lstm_h"][ids] = lstm_h[off : off + n]
                state["lstm_c"][ids] = lstm_c[off : off + n]
                off += n
        plans, off = [], 0
        for n in counts:
            n = int(n)
            plans.append(actions[off : off + n])
            off += n
        return plans

    def plan(self, items: Sequence[PlanItem]) -> list:
        gathered = self.gather(items)
        actions, h, c = self.forward(
            gathered["obs"], gathered.get("lstm_h"), gathered.get("lstm_c")
        )
        return self.scatter(items, gathered, actions, h, c)

    def apply(self, items: Sequence[PlanItem], plans: list) -> None:
        for (sim, ids), actions in zip(items, plans):
            sim.step_dynamics(actions, ids)
