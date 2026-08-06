"""The planner contract: one policy driving one role's agents.

A ``Planner`` is a two-phase policy over an agent subset. ``plan`` decides
actions from the current world state WITHOUT mutating it; ``apply`` integrates
them. ``sim.runner.RolloutRunner`` plans every role first and applies
afterwards, so no role ever observes another role's same-step movement.

Every planner implements exactly this interface and receives exactly its own
``cfgs/planner/<name>.yaml`` node, so any planner can drive any role: the SUT
and the traffic are the same kind of object, and swapping them is pure config
composition (``planner@planner.sut`` / ``planner@planner.env``). Nothing in
this module -- or in the runner -- knows which role is "the one being measured";
that asymmetry lives entirely in the hooks, which score the ego.

Planners are handed the ``SimScene`` they act on, so dynamics parameters (dt,
the discrete action table, goal behaviour) come from that scene rather than
from a separate config object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from omegaconf import OmegaConf

from ..world import SimScene

# Per-agent conditioning obs slots of the frozen PufferDrive policy family
# (trailing ego features; see SimScene). The values are policy inputs -- how
# defensively THIS role's policy drives -- so they live in each role's planner
# yaml (``conditioning:``), not in the shared rollout dynamics.
CONDITIONING_FIELDS = ("collision_factor", "offroad_factor", "lane_width")

# One plan/apply work unit: the scene sim + the agent ids this role currently
# drives in it (a subset of ``sim.controlled``).
PlanItem = tuple[SimScene, np.ndarray]


def parse_conditioning(role: str, cfg):
    """Validate one role's ``conditioning`` node.

    Returns ``None`` (``conditioning: null``: the policy ignores the
    conditioning obs, its agents' slots stay 0) or a dict mapping each field to
    a float (fixed value) or a (lo, hi) pair (per-agent uniform sample).
    Strict: exactly the CONDITIONING_FIELDS keys.
    """
    if cfg is None:
        return None
    m = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
    unknown = sorted(set(m) - set(CONDITIONING_FIELDS))
    missing = sorted(set(CONDITIONING_FIELDS) - set(m))
    if unknown or missing:
        raise ValueError(
            f"planner.{role}.conditioning: unknown keys {unknown}, missing keys {missing}; "
            f"expected exactly {list(CONDITIONING_FIELDS)}"
        )
    out = {}
    for key in CONDITIONING_FIELDS:
        v = m[key]
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(
                    f"planner.{role}.conditioning.{key}: a range must be [lo, hi], got {v!r}"
                )
            out[key] = (float(v[0]), float(v[1]))
        else:
            out[key] = float(v)
    return out


def require(planner_cfg, name: str, key: str, path: str | None = None):
    """Read a required key out of a planner yaml node, or raise.

    Planner configs are complete by contract: a missing key is a broken config,
    not an invitation to substitute a default. Every planner reads its settings
    through this so a typo fails at construction instead of silently changing
    how an agent drives. ``path`` names the key in the error message when the
    node is nested (``route.spacing`` rather than ``spacing``).
    """
    shown = path or key
    if key not in planner_cfg:
        raise KeyError(
            f"cfgs/planner/{name}.yaml is missing required key {shown!r}"
        )
    value = planner_cfg[key]
    if value is None:
        raise ValueError(
            f"cfgs/planner/{name}.yaml: {shown!r} must have a value, got null"
        )
    return value


class Planner(ABC):
    """Two-phase action policy for one role's agents.

    ``plan`` must not mutate world state (planner-internal state such as an
    LSTM carry is fine): the runner calls every role's ``plan`` from the same
    pre-step state before any ``apply`` runs.

    Subclasses keep this constructor signature so ``build_planner`` can build
    any of them for any role.
    """

    def __init__(self, planner_cfg, *, role: str, device: str | None = None):
        self.cfg = planner_cfg
        self.name = str(require(planner_cfg, "<unnamed>", "name"))
        self.role = role

    def _require(self, key: str):
        """Required setting from this planner's own yaml node."""
        return require(self.cfg, self.name, key)

    @abstractmethod
    def plan(self, items: Sequence[PlanItem]) -> list:
        """Decide actions for every ``(sim, agent_ids)`` item; return them aligned."""

    @abstractmethod
    def apply(self, items: Sequence[PlanItem], plans: list) -> None:
        """Integrate the actions previously returned by ``plan`` (in place)."""
