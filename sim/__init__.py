"""Scenario rollout: drive a batch of scenes with per-role planners and measure.

This is the shared simulation layer. It is deliberately independent of the
diffusion / RL stack: ``ddpo`` uses it to score generated adversaries, and the
SUT x traffic benchmark uses it to score planners, but nothing here knows about
either. The seams are:

  * ``scenes.GeneratedScenes`` -- what a scene *source* produces (a DDPO policy,
    the real-log loader, ...);
  * ``planners.Planner``       -- what drives one role's agents (idm, ppo_*);
  * ``hooks.MetricHook``       -- what gets measured while stepping;
  * ``runner.RolloutRunner``   -- the step loop that binds the three together.

Consumers assemble scores from ``runner.RolloutResult.metrics``; the rollout
itself never computes a reward.
"""

from __future__ import annotations

from .runner import ROLES, RolloutResult, RolloutRunner, SimulatorConfig
from .scenes import GeneratedScenes, single_adv_local_idx

__all__ = [
    "ROLES",
    "GeneratedScenes",
    "RolloutResult",
    "RolloutRunner",
    "SimulatorConfig",
    "single_adv_local_idx",
]
