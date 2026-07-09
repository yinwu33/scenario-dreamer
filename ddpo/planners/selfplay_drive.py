"""selfplay_drive planner: frozen PufferDrive net + numpy classic dynamics.

This is the original DDPO rollout policy. Each step the controlled agents'
observations are gathered, batched through the frozen ``selfplay_drive`` network
to discrete actions, and integrated with the bicycle ``step_dynamics`` model.
"""

from __future__ import annotations

import numpy as np
import torch

from ..pufferdrive_sim import SimScene
from .base import NumpyPlanner, SimulatorConfig, register_planner


@register_planner("selfplay_drive")
class SelfplayDrivePlanner(NumpyPlanner):
    def __init__(self, planner_cfg, params: SimulatorConfig, *, device: str | None = None):
        super().__init__(planner_cfg, params, device=device)
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
        state = getattr(sim, "_planner_lstm_state", None)
        if state is None or state["lstm_h"].shape[0] != sim.n:
            state = self.planner.initial_state(sim.n, device=self.device)
            sim._planner_lstm_state = state
        return state

    def _advance(self, sims: list[SimScene], active: list[int]) -> None:
        obs_list = []
        ctrl_list = []
        for s in active:
            obs_s = sims[s].compute_obs()
            obs_list.append(obs_s)
            ctrl_list.append(sims[s].controlled.copy())
        total = sum(ob.shape[0] for ob in obs_list)
        if total == 0:
            return

        obs = torch.as_tensor(np.concatenate(obs_list), device=self.device)
        if self.recurrent:
            h_list, c_list = [], []
            for s, ctrl in zip(active, ctrl_list):
                state = self._lstm_state_for(sims[s])
                h_list.append(state["lstm_h"][ctrl])
                c_list.append(state["lstm_c"][ctrl])
            batch_state = {
                "lstm_h": torch.cat(h_list, dim=0),
                "lstm_c": torch.cat(c_list, dim=0),
            }
            actions = self.planner.act(
                obs, state=batch_state, deterministic=self.deterministic
            ).cpu().numpy()
            off = 0
            for s, ctrl in zip(active, ctrl_list):
                n_ctrl = len(ctrl)
                state = self._lstm_state_for(sims[s])
                state["lstm_h"][ctrl] = batch_state["lstm_h"][off : off + n_ctrl]
                state["lstm_c"][ctrl] = batch_state["lstm_c"][off : off + n_ctrl]
                off += n_ctrl
        else:
            actions = self.planner.act(obs, deterministic=self.deterministic).cpu().numpy()

        off = 0
        for s, ob in zip(active, obs_list):
            n_ctrl = ob.shape[0]
            sims[s].step_dynamics(actions[off : off + n_ctrl])
            off += n_ctrl
