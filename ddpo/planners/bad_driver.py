"""bad_driver planner: recurrent PufferDrive policy + numpy classic dynamics."""

from __future__ import annotations

from .base import register_planner
from .selfplay_drive import SelfplayDrivePlanner


@register_planner("bad_driver")
class BadDriverPlanner(SelfplayDrivePlanner):
    """Uses the same Drive rollout wrapper with a bad_driver checkpoint/config."""

