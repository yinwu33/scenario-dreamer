"""Per-role rollout planners for DDPO reward evaluation.

``RolloutRunner`` steps the scenes with one ``Planner`` per role (``sut`` /
``env`` / ``adv``) and fires the metric hooks injected by the caller. Roles are
composed per flow via ``planner@<...>.planner.<role>: <name>``; the
implementations are ``BadDriverPlanner`` (frozen PufferDrive net) and
``IDMPlanner`` (rule-based, lane-graph routed).
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
from .idm import IDMPlanner

__all__ = [
    "ROLES",
    "BadDriverPlanner",
    "IDMPlanner",
    "Planner",
    "RolloutResult",
    "RolloutRunner",
    "SimulatorConfig",
    "build_planner",
    "to_puffer_agent_types",
]
