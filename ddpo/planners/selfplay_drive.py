"""selfplay_drive planner: frozen PufferDrive net + numpy classic dynamics.

This is the original DDPO rollout policy. Each step the controlled agents'
observations are gathered, batched through the frozen ``selfplay_drive`` network
to discrete actions, and integrated with the bicycle ``step_dynamics`` model.
"""

from __future__ import annotations

import numpy as np
import torch

from ..pufferdrive_sim import SimScene
from .base import NumpyPlanner, RolloutParams, register_planner


@register_planner("selfplay_drive")
class SelfplayDrivePlanner(NumpyPlanner):
    def __init__(self, planner_cfg, params: RolloutParams, *, device: str | None = None):
        super().__init__(planner_cfg, params, device=device)
        from planner.selfplay_drive.planner import load_planner, load_planner_config

        self.planner = load_planner()
        self.device = device or str(next(self.planner.parameters()).device)
        det = planner_cfg.get("deterministic", None)
        self.deterministic = (
            bool(load_planner_config().deterministic) if det is None else bool(det)
        )

    def _advance(self, sims: list[SimScene], active: list[int]) -> None:
        obs_list = [sims[s].compute_obs() for s in active]
        obs = torch.as_tensor(np.concatenate(obs_list), device=self.device)
        actions = self.planner.act(obs, deterministic=self.deterministic).cpu().numpy()
        off = 0
        for s, ob in zip(active, obs_list):
            n_ctrl = ob.shape[0]
            sims[s].step_dynamics(actions[off : off + n_ctrl])
            off += n_ctrl
