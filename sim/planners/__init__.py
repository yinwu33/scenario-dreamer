"""Planners: interchangeable policies that drive one rollout role's agents.

All of them implement ``base.Planner`` and are selected by ``name`` from
``cfgs/planner/<name>.yaml``, so the system under test, the background traffic
and the adversary are configured the same way and can be swapped for each
other freely (see ``sim.runner.ROLES``).

Two implementations back the current roster:

  * ``PPOPlanner`` -- frozen recurrent PufferDrive PPO net. The ``ppo_*`` family
    (``ppo_aggressive`` / ``ppo_normal`` / ``ppo_caution``) all map here: same
    checkpoint, different ``conditioning.collision_factor``, i.e. different
    driving styles rather than different policies.
  * ``IDMPlanner`` -- rule-based IDM + pure pursuit on a lane-graph route.
  * ``CtRLSimPlanner`` -- the frozen CtRL-Sim decision transformer. Its ``tilt``
    knob biases the predicted return-to-go, so ``ctrl_sim`` is ordinary traffic
    and ``ctrl_sim_adv`` is the behavior-driven adversary. It is the one planner
    that does not use the shared accel/steer table (see its module docstring).
"""

from __future__ import annotations

from .base import CONDITIONING_FIELDS, PlanItem, Planner, parse_conditioning, require
from .ctrl_sim import CtRLSimPlanner
from .idm import IDMPlanner
from .ppo import PPOPlanner

# Planner name (== the cfgs/planner/<name>.yaml stem) -> implementation.
# Several names may share an implementation: a "planner" here is a *configured*
# policy, so two conditioning presets of one net are two planners. Adding a
# variant means one yaml plus one line here.
PLANNER_REGISTRY: dict[str, type[Planner]] = {
    "ppo_aggressive": PPOPlanner,
    "ppo_normal": PPOPlanner,
    "ppo_caution": PPOPlanner,
    "idm": IDMPlanner,
    "ctrl_sim": CtRLSimPlanner,
    "ctrl_sim_adv": CtRLSimPlanner,
}

__all__ = [
    "CONDITIONING_FIELDS",
    "CtRLSimPlanner",
    "IDMPlanner",
    "PLANNER_REGISTRY",
    "PPOPlanner",
    "PlanItem",
    "Planner",
    "build_planner",
    "parse_conditioning",
    "require",
]


def build_planner(planner_cfg, *, role: str, device: str | None = None) -> Planner:
    """Instantiate the planner named by ``planner_cfg['name']`` for one role.

    ``planner_cfg`` is any mapping with a ``name`` key (an OmegaConf node from
    ``cfgs/planner/<name>.yaml`` or a plain dict). Unknown names raise, so a
    stale config fails loudly instead of silently falling back to a default.
    """
    name = str(require(planner_cfg, "<unnamed>", "name"))
    if name not in PLANNER_REGISTRY:
        raise ValueError(
            f"unknown planner {name!r}; available: {sorted(PLANNER_REGISTRY)}"
        )
    return PLANNER_REGISTRY[name](planner_cfg, role=role, device=device)
