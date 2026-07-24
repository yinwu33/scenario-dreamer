"""Per-role rollout planners for DDPO reward evaluation.

``RolloutRunner`` steps the scenes with one ``Planner`` per role (``sut`` /
``env`` / ``adv``; ``BadDriverPlanner`` is the only implementation) and fires
the metric hooks injected by the reward layer. Roles are composed per flow via
``planner@ddpo.planner.<role>: <name>``.
"""

from __future__ import annotations

from .base import (
    ROLES,
    Planner,
    RolloutResult,
    RolloutRunner,
    SimulatorConfig,
    build_planner,
    to_puffer_agent_types,
)
from .bad_driver import BadDriverPlanner

__all__ = [
    "ROLES",
    "BadDriverPlanner",
    "Planner",
    "RolloutResult",
    "RolloutRunner",
    "SimulatorConfig",
    "build_planner",
    "to_puffer_agent_types",
]
