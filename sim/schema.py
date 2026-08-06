"""Single source of truth for agent-state layout constants.

The diffusion policies operate on normalized latents; reward and rendering use
the same first 9 physical columns:

``[x, y, speed, cos, sin, length, width, goal_x, goal_y]``.
"""

from __future__ import annotations

# Agent latent layout: absolute indices into the normalized agent vector, with
# the one-hot type block at the tail.
AGENT_LATENT_DIM = 12
GOAL_DIMS = (7, 8)
GOAL_SLICE = slice(GOAL_DIMS[0], GOAL_DIMS[-1] + 1)
SIZE_DIMS = (5, 6)
TYPE_DIMS = (9, 10, 11)
VEHICLE_TYPE_ID = 0

# --- thresholds -------------------------------------------------------------
MIN_DISTANCE_TO_GOAL = 2.0   # metres; goal within this of spawn => parked/static
