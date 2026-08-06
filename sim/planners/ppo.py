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
    def __init__(self, planner_cfg, *, role: str, device: str | None = None):
        super().__init__(planner_cfg, role=role, device=device)
        from nets.selfplay_drive.net import load_net

        self.net = load_net(planner_cfg)
        # `device` (the caller's rollout device) wins when given; otherwise the
        # net's own resolved device is authoritative.
        self.device = device or str(next(self.net.parameters()).device)
        self.deterministic = bool(self._require("deterministic"))
        self.recurrent = self.net.recurrent

    def _lstm_state_for(self, sim: SimScene) -> dict[str, torch.Tensor]:
        states = getattr(sim, "_planner_lstm_states", None)
        if states is None:
            states = sim._planner_lstm_states = {}
        state = states.get(self.role)
        if state is None or state["lstm_h"].shape[0] != sim.n:
            state = self.net.initial_state(sim.n, device=self.device)
            states[self.role] = state
        return state

    def plan(self, items: Sequence[PlanItem]) -> list:
        obs_list = [sim.compute_obs(ids) for sim, ids in items]
        total = sum(ob.shape[0] for ob in obs_list)
        if total == 0:
            return [np.empty(0, dtype=np.int64) for _ in items]

        obs = torch.as_tensor(np.concatenate(obs_list), device=self.device)
        if self.recurrent:
            h_list, c_list = [], []
            for sim, ids in items:
                state = self._lstm_state_for(sim)
                h_list.append(state["lstm_h"][ids])
                c_list.append(state["lstm_c"][ids])
            batch_state = {
                "lstm_h": torch.cat(h_list, dim=0),
                "lstm_c": torch.cat(c_list, dim=0),
            }
            actions = self.net.act(
                obs, state=batch_state, deterministic=self.deterministic
            ).cpu().numpy()
            off = 0
            for sim, ids in items:
                n_ctrl = len(ids)
                state = self._lstm_state_for(sim)
                state["lstm_h"][ids] = batch_state["lstm_h"][off : off + n_ctrl]
                state["lstm_c"][ids] = batch_state["lstm_c"][off : off + n_ctrl]
                off += n_ctrl
        else:
            actions = self.net.act(obs, deterministic=self.deterministic).cpu().numpy()

        plans = []
        off = 0
        for ob in obs_list:
            n_ctrl = ob.shape[0]
            plans.append(actions[off : off + n_ctrl])
            off += n_ctrl
        return plans

    def apply(self, items: Sequence[PlanItem], plans: list) -> None:
        for (sim, ids), actions in zip(items, plans):
            sim.step_dynamics(actions, ids)
