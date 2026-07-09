"""puffer_drive planner: PufferDrive's vectorized C environment.

Wraps ``native_pufferdrive.PufferBackend`` (the frozen net + C sim hot path) and
fills the init-state validity metrics the C backend does not produce via
``static_metrics.add_static_metrics``.
"""

from __future__ import annotations

import torch

from ..interfaces import GeneratedScenes
from ..pufferdrive_sim import load_sim_config
from .base import SimulatorConfig, RolloutPlanner, RolloutResult, register_planner
from .static_metrics import add_static_metrics


@register_planner("puffer_drive")
class PufferDrivePlanner(RolloutPlanner):
    def __init__(self, planner_cfg, params: SimulatorConfig, *, device: str | None = None):
        from planner.selfplay_drive.planner import load_planner, load_planner_config

        from ..native_pufferdrive import PufferBackend

        self.params = params
        self.sim_cfg = load_sim_config()
        self.planner = load_planner()
        self.device = device or str(next(self.planner.parameters()).device)
        det = planner_cfg.get("deterministic", None)
        deterministic = (
            bool(load_planner_config().deterministic) if det is None else bool(det)
        )
        root = planner_cfg.get("pufferdrive_root", None)
        self.backend = PufferBackend(
            planner=self.planner,
            device=self.device,
            deterministic=deterministic,
            sim_steps=params.sim_steps,
            seed=params.seed,
            pufferdrive_root=root,
        )

    @torch.no_grad()
    def rollout(
        self, scenes: GeneratedScenes, *, record_trajectories: bool = False
    ) -> RolloutResult:
        metrics = self.backend.rollout(scenes, record_trajectories=record_trajectories)
        add_static_metrics(
            scenes,
            metrics,
            sim_cfg=self.sim_cfg,
            goal_offlane_threshold=self.params.goal_offlane_threshold,
            goal_onroad_threshold=self.params.goal_onroad_threshold,
            gen_invalid=self.params.gen_invalid,
        )
        trajectories = metrics.pop("trajectories", None)
        return RolloutResult(metrics=metrics, trajectories=trajectories)
