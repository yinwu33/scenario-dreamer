"""Single source of truth for the dm_goal agent-state schema.

Two agent-state layouts share the same goal columns:
  * decoded / physical (9 dims), as carried by ``GeneratedScenes`` and the numpy
    sim:  ``[x, y, speed, cos, sin, length, width, goal_x, goal_y]``
  * normalized diffusion latent (12 dims), as sampled by the dm_goal policy:
    ``[x, y, speed, cos, sin, length, width, goal_x, goal_y, type_onehot(3)]``
The first 9 columns coincide, so ``GOAL_DIMS`` / ``GOAL_SLICE`` / ``SIZE_DIMS``
index both layouts.

NOTE on type encodings (do NOT conflate): ``TYPE_DIMS`` / ``VEHICLE_TYPE_ID``
below describe the latent one-hot block. The numpy sim uses a different scheme
(class ids ``TYPE_VEHICLE/PEDESTRIAN/CYCLIST = 1/2/3`` in ``pufferdrive_sim``).

This module is a leaf (no ``ddpo`` imports) so the sim / policy / viz can all
import it without circular imports.
"""

from __future__ import annotations

# --- column layout (shared by the 9-dim physical and 12-dim latent vectors) ---
GOAL_DIMS = (7, 8)            # goal_x, goal_y
GOAL_SLICE = slice(7, 9)
SIZE_DIMS = (5, 6)           # length, width

# --- latent-only (12-dim) type one-hot block --------------------------------
TYPE_DIMS = (9, 10, 11)
VEHICLE_TYPE_ID = 0          # index WITHIN TYPE_DIMS that means "vehicle"

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
