"""smart planner: a SMART-style learned traffic model as a rollout role.

Drives the ``env`` (background traffic) role -- and optionally ``adv`` -- with a
transformer that predicts the next action from the agent's own recent motion,
its neighbours and the nearby centrelines. It has NO goal input, which is the
point: it imitates how logged traffic drives rather than being told where to go.
It is deliberately never the SUT, because ``Succ.`` is a goal check
(``SimScene.goal_step``) and a goal-free ego would produce a number that is not
comparable to the IDM and PPO columns.

Two properties keep it inside the existing machinery, both of which
``sim.planners.ctrl_sim`` does not have:

  * **Shardable.** ``gather`` returns one flat ``[rows, obs_dim]`` matrix, which
    is exactly what ``sim.parallel`` shuttles through shared memory, so workers
    run the CPU halves and the parent runs ONE batched forward -- the same
    split as ``sim.planners.ppo`` and the same reason it stays bit-exact.
  * **No integrator exception.** It emits an index into the shared 7x13
    accel/steer table, so ``SimScene.step_dynamics`` integrates it like every
    other planner.

The rolling history is planner-internal state kept per (scene, role), advanced
from the motion each agent actually made. It starts EMPTY and every history slot
carries a validity flag, so a freshly generated scene with no past is an ordinary
input rather than a cold start -- the model is trained with the visible history
length randomly masked for exactly this reason.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from sim.planners.base import Planner, PlanItem
from sim.world import SimScene

from . import observation
from .net import OBS_DIM, load_net
from .observation import POSE_SLOTS


class SMARTPlanner(Planner):
    # One batched forward over every item's rows -> the result depends on which
    # scenes shared the batch; sim.parallel keeps this forward in the parent.
    batched_across_scenes = True

    def __init__(self, planner_cfg, *, role: str, device: str | None = None):
        super().__init__(planner_cfg, role=role, device=device)
        self.net = load_net(planner_cfg)
        self.obs_dim = OBS_DIM
        self.device = device or str(next(self.net.parameters()).device)
        self.deterministic = bool(self._require("deterministic"))
        self.temperature = float(self._require("temperature"))
        # No recurrent carry: the history is rebuilt from observed poses inside
        # each worker, so nothing but the obs matrix has to cross shared memory.
        self.recurrent = False
        # Owned generator, advanced once per forward. forward() runs exactly once
        # per step in BOTH the single-process and the sharded path, so sampling
        # stays reproducible and shard-exact.
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(self._require("seed")))

    # ------------------------------------------------------------------ state
    def _poses_for(self, sim: SimScene) -> dict:
        """This role's rolling pose history for ``sim`` (planner-internal).

        Kept on the host so a ``sim.parallel`` worker that owns no GPU can
        advance it; a scene's history never leaves the worker that owns it.
        """
        store = getattr(sim, "_smart_poses", None)
        if store is None:
            store = sim._smart_poses = {}
        state = store.get(self.role)
        if state is None or state["poses"].shape[0] != sim.n:
            state = {
                "poses": np.zeros((sim.n, POSE_SLOTS, 3), dtype=np.float32),
                "filled": 0,
            }
            store[self.role] = state
        return state

    def _advance(self, sim: SimScene, state: dict) -> None:
        """Append the current pose of every agent, sliding the window."""
        poses = state["poses"]
        poses[:, :-1] = poses[:, 1:]
        poses[:, -1, 0] = sim.x
        poses[:, -1, 1] = sim.y
        poses[:, -1, 2] = sim.heading
        state["filled"] = min(state["filled"] + 1, POSE_SLOTS)

    def prime_history(self, sim: SimScene, poses: np.ndarray) -> None:
        """Seed the rolling history with real past poses, oldest first.

        A generated scene has no past and starts cold, which is the case the
        random history masking in training exists for. A LOGGED scene does have a
        past, so an evaluation can hand it over and measure how much the model
        gains from it -- the gap between the two is the direct test of whether the
        masking worked. ``poses`` is [A, k, 3] with k <= POSE_SLOTS.
        """
        state = self._poses_for(sim)
        k = poses.shape[1]
        if k > POSE_SLOTS:
            raise ValueError(f"at most {POSE_SLOTS} poses fit the history, got {k}")
        state["poses"][:, POSE_SLOTS - k :] = poses
        state["filled"] = k

    # ------------------------------------------------------------------- obs
    def _observe(self, sim: SimScene, ids: np.ndarray, state: dict) -> np.ndarray:
        """[len(ids), OBS_DIM] -- the SAME builder ``smart.dataset`` trains on."""
        return observation.build(
            observation.frame_from_sim(sim),
            observation.grid_from_sim(sim),
            ids,
            state["poses"],
            POSE_SLOTS - state["filled"],
        )

    # ------------------------------------------------------------------ plan
    def gather(self, items: Sequence[PlanItem]) -> dict:
        """CPU half: advance each scene's history, then build the obs matrix."""
        obs_list = []
        for sim, ids in items:
            state = self._poses_for(sim)
            self._advance(sim, state)
            obs_list.append(self._observe(sim, ids, state))
        counts = np.array([ob.shape[0] for ob in obs_list], dtype=np.int64)
        total = int(counts.sum())
        return {
            "obs": np.concatenate(obs_list) if total else np.zeros((0, OBS_DIM), np.float32),
            "counts": counts,
        }

    def forward(self, obs: np.ndarray, lstm_h=None, lstm_c=None):
        """GPU half: the ONE batched net forward. Returns (actions, None, None)."""
        if obs.shape[0] == 0:
            return np.empty(0, dtype=np.int64), None, None
        with torch.no_grad():
            logits = self.net(torch.as_tensor(obs, device=self.device))
            if self.deterministic:
                actions = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits / self.temperature, dim=-1)
                actions = torch.multinomial(probs, 1, generator=self._gen).squeeze(-1)
        return actions.cpu().numpy().astype(np.int64), None, None

    def scatter(self, items, gathered: dict, actions, lstm_h=None, lstm_c=None) -> list:
        """CPU half: split the actions per item."""
        plans, off = [], 0
        for n in gathered["counts"]:
            n = int(n)
            plans.append(actions[off : off + n])
            off += n
        return plans

    def plan(self, items: Sequence[PlanItem]) -> list:
        gathered = self.gather(items)
        actions, _, _ = self.forward(gathered["obs"])
        return self.scatter(items, gathered, actions)

    def apply(self, items: Sequence[PlanItem], plans: list) -> None:
        for (sim, ids), actions in zip(items, plans):
            sim.step_dynamics(actions, ids)
