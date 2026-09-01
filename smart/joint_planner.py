"""smart_joint planner: the joint net as a rollout role.

Same contract as ``smart.planner`` and the same rolling history, but the unit of
computation is a SCENE. That is the whole point -- the model attends across
agents -- and it is also the cost:

``sim/parallel.py`` shards a rollout by shuttling one flat ``[rows, obs_dim]``
matrix through shared memory, so a planner without ``obs_dim`` is refused by name
and must run single process (``--workers 0``). This planner is in exactly the
position ``ctrl_sim`` is in, and it gives up the sharded-rollout result the
per-agent ``smart`` planner was designed around. Keep both: ``smart`` is the one
that can go inside DDPO.

It predicts an action for EVERY agent in the scene and applies only the ones its
role drives, which is what SMART does and what makes the other agents' encoded
motion available as context.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from sim.planners.base import Planner, PlanItem
from sim.world import SimScene

from .joint_net import KEYS, MAX_AGENTS, load_net
from .joint_observation import POSE_SLOTS, build_scene, self_features


class JointSMARTPlanner(Planner):
    batched_across_scenes = True
    # deliberately no obs_dim: sim.parallel must refuse this planner loudly
    # rather than silently re-batching a per-scene input.

    def __init__(self, planner_cfg, *, role: str, device: str | None = None):
        super().__init__(planner_cfg, role=role, device=device)
        self.net = load_net(planner_cfg)
        self.device = device or str(next(self.net.parameters()).device)
        self.deterministic = bool(self._require("deterministic"))
        self.temperature = float(self._require("temperature"))
        self.recurrent = False
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(self._require("seed")))

    def _poses_for(self, sim: SimScene) -> dict:
        store = getattr(sim, "_joint_poses", None)
        if store is None:
            store = sim._joint_poses = {}
        state = store.get(self.role)
        if state is None or state["poses"].shape[0] != sim.n:
            state = {"poses": np.zeros((sim.n, POSE_SLOTS, 3), np.float32), "filled": 0}
            store[self.role] = state
        return state

    def _advance(self, sim: SimScene, state: dict) -> None:
        p = state["poses"]
        p[:, :-1] = p[:, 1:]
        p[:, -1, 0], p[:, -1, 1], p[:, -1, 2] = sim.x, sim.y, sim.heading
        state["filled"] = min(state["filled"] + 1, POSE_SLOTS)

    def prime_history(self, sim: SimScene, poses: np.ndarray) -> None:
        state = self._poses_for(sim)
        k = poses.shape[1]
        state["poses"][:, POSE_SLOTS - k:] = poses
        state["filled"] = k

    def gather(self, items: Sequence[PlanItem]) -> dict:
        scenes = []
        for sim, _ in items:
            state = self._poses_for(sim)
            self._advance(sim, state)
            sf = self_features(
                np.hypot(sim.vx, sim.vy), sim.width, sim.length, sim.ptype,
                (sim.collision_state > 0).astype(np.float32), sim.removed.astype(np.float32),
            )
            pose = np.stack([sim.x, sim.y, sim.heading], axis=1)
            scenes.append(build_scene(
                pose, sf, state["poses"], POSE_SLOTS - state["filled"],
                sim.seg_mid, sim.seg_dir, sim.seg_half_len,
            ))
        return {"scenes": scenes,
                "counts": np.array([len(ids) for _, ids in items], dtype=np.int64)}

    def forward(self, gathered, lstm_h=None, lstm_c=None):
        scenes = gathered["scenes"]
        if not scenes:
            return np.zeros((0, MAX_AGENTS), np.int64), None, None
        batch = {k: torch.from_numpy(np.stack([s[k] for s in scenes])).to(self.device)
                 for k in KEYS}
        with torch.no_grad():
            logits = self.net(batch)
            if self.deterministic:
                a = logits.argmax(-1)
            else:
                p = torch.softmax(logits / self.temperature, dim=-1)
                a = torch.multinomial(p.reshape(-1, p.shape[-1]), 1,
                                      generator=self._gen).reshape(p.shape[:-1])
        return a.cpu().numpy().astype(np.int64), None, None

    def scatter(self, items, gathered, actions, lstm_h=None, lstm_c=None) -> list:
        """Take, per scene, the rows this role actually drives."""
        return [actions[s][np.minimum(ids, MAX_AGENTS - 1)]
                for s, (_, ids) in enumerate(items)]

    def plan(self, items: Sequence[PlanItem]) -> list:
        g = self.gather(items)
        a, _, _ = self.forward(g)
        return self.scatter(items, g, a)

    def apply(self, items: Sequence[PlanItem], plans: list) -> None:
        for (sim, ids), actions in zip(items, plans):
            sim.step_dynamics(actions, ids)
