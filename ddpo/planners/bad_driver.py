"""bad_driver planner: frozen recurrent PufferDrive net + numpy classic dynamics.

``plan`` gathers the driven agents' observations across scenes, batches them
through the frozen ``bad_driver`` network (the Drive policy port in
``planner/selfplay_drive/planner.py``) to discrete actions; ``apply``
integrates them with the bicycle ``step_dynamics`` model. Checkpoint / net
arch / determinism come from ``cfgs/planner/bad_driver.yaml``.

One instance drives one role's agents (see ``ddpo.planners.ROLES``); the LSTM
carry is kept per (scene, role), so several bad_driver instances -- or the same
config reused for every role -- never share or clobber recurrent state. The
loaded net itself is cached per resolved config, so identical role configs
share one frozen network.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..pufferdrive_sim import SimScene
from .base import Planner, PlanItem, SimulatorConfig


class BadDriverPlanner(Planner):
    def __init__(self, planner_cfg, params: SimulatorConfig, *, role: str, device: str | None = None):
        super().__init__(planner_cfg, params, role=role, device=device)
        from planner.selfplay_drive.planner import load_planner, load_planner_config

        self.planner = load_planner(planner_cfg)
        self.planner_config = load_planner_config(planner_cfg)
        self.device = device or str(next(self.planner.parameters()).device)
        det = planner_cfg.get("deterministic", None)
        self.deterministic = (
            bool(self.planner_config.deterministic) if det is None else bool(det)
        )
        self.recurrent = bool(getattr(self.planner, "recurrent", False))

    def _lstm_state_for(self, sim: SimScene) -> dict[str, torch.Tensor]:
        states = getattr(sim, "_planner_lstm_states", None)
        if states is None:
            states = sim._planner_lstm_states = {}
        state = states.get(self.role)
        if state is None or state["lstm_h"].shape[0] != sim.n:
            state = self.planner.initial_state(sim.n, device=self.device)
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
            actions = self.planner.act(
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
            actions = self.planner.act(obs, deterministic=self.deterministic).cpu().numpy()

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
