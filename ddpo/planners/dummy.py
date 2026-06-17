"""dummy planner: deterministic rule-based goal-seeking rollout.

Each controlled agent translates straight toward its goal at its generated spawn
speed, with no network and no acceleration/steering integration (see
``SimScene.step_goal_seek``). This makes the reward a (near-)deterministic
function of the generated init state, which isolates the effect of DDPO scene-
init training from the variance of a learned planner.
"""

from __future__ import annotations

from ..pufferdrive_sim import SimScene
from .base import NumpyPlanner, register_planner


@register_planner("dummy")
class DummyPlanner(NumpyPlanner):
    def _advance(self, sims: list[SimScene], active: list[int]) -> None:
        for s in active:
            sims[s].step_goal_seek()
