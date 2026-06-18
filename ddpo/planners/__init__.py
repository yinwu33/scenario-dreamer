"""Pluggable rollout planners for DDPO reward evaluation.

Importing this package registers the built-in planners (``dummy``,
``centerline_dummy``, ``selfplay_drive``, ``bad_driver``, ``puffer_drive``);
pick one by name via ``build_planner``.
"""

from __future__ import annotations

from .base import (
    NumpyPlanner,
    RolloutParams,
    RolloutPlanner,
    RolloutResult,
    build_planner,
    register_planner,
)

# Side-effect imports: register the built-in planners.
from . import dummy, centerline_dummy, selfplay_drive, bad_driver, puffer_drive  # noqa: E402,F401

__all__ = [
    "NumpyPlanner",
    "RolloutParams",
    "RolloutPlanner",
    "RolloutResult",
    "build_planner",
    "register_planner",
]
