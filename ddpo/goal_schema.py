"""Single source of truth for agent-state layout constants.

The diffusion policies operate on normalized latents; reward and rendering use
the same first 9 physical columns:

``[x, y, speed, cos, sin, length, width, goal_x, goal_y]``.

Different DDPO generators share the same first 9 columns and goal columns, but
different latent extensions (e.g. fixed-map adds parking and still keeps the
vehicle-type block).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSchema:
    """Layout for the normalized agent latent.

    All indices are absolute in the latent vector and describe one-hot blocks in
    absolute index coordinates, not relative to any other layout.
    """

    agent_latent_dim: int
    goal_dims: tuple[int, int]
    size_dims: tuple[int, int]
    type_dims: tuple[int, int, int]
    vehicle_type_id: int = 0
    parking_dims: tuple[int, int] | None = None


DM_GOAL_AGENT_SCHEMA = AgentSchema(
    agent_latent_dim=12,
    goal_dims=(7, 8),
    size_dims=(5, 6),
    type_dims=(9, 10, 11),
    vehicle_type_id=0,
    parking_dims=None,
)


DM_FIXED_MAP_AGENT_GOAL_SCHEMA = AgentSchema(
    agent_latent_dim=14,
    goal_dims=(7, 8),
    size_dims=(5, 6),
    type_dims=(11, 12, 13),
    vehicle_type_id=0,
    parking_dims=(9, 10),
)


# Backwards-compatible module-level constants for existing dm_goal imports.
GOAL_DIMS = DM_GOAL_AGENT_SCHEMA.goal_dims
GOAL_SLICE = slice(DM_GOAL_AGENT_SCHEMA.goal_dims[0], DM_GOAL_AGENT_SCHEMA.goal_dims[-1] + 1)
SIZE_DIMS = DM_GOAL_AGENT_SCHEMA.size_dims
TYPE_DIMS = DM_GOAL_AGENT_SCHEMA.type_dims
VEHICLE_TYPE_ID = DM_GOAL_AGENT_SCHEMA.vehicle_type_id

# --- thresholds -------------------------------------------------------------
MIN_DISTANCE_TO_GOAL = 2.0   # metres; goal within this of spawn => parked/static


# --- FOV-frame (de)normalisation --------------------------------------------
# dm_goal stores agent x/y and goal_x/goal_y in [-1, 1] over a square FOV window
# centred on the SDC. These convert to/from physical metres (centre = 0); plain
# arithmetic only, so they work on Python floats, numpy arrays, and torch tensors.
def fov_unnormalize(v, fov):
    """[-1, 1] FOV coordinate -> physical metres (centre = 0)."""
    return (v + 1.0) / 2.0 * fov - fov / 2.0


def fov_normalize(p, fov):
    """Physical metres (centre = 0) -> [-1, 1] FOV coordinate."""
    return 2.0 * ((p + fov / 2.0) / fov) - 1.0
