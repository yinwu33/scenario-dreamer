"""Agent-type conversions at the planner/simulator boundary."""

from __future__ import annotations

import numpy as np

from ..pufferdrive_sim import TYPE_CYCLIST, TYPE_VEHICLE


def to_puffer_agent_types(agent_types) -> np.ndarray:
    """Convert dataset/model ids (0 veh, 1 ped, 2 cyc) to PufferDrive ids.

    ``GeneratedScenes.agent_types`` deliberately keeps the model-side convention.
    PufferDrive observations and collision logic use entity ids 1..3, so planners
    convert at the boundary before constructing a sim scene or planner metrics.
    """
    return (
        np.asarray(agent_types, dtype=np.int64) + 1
    ).clip(TYPE_VEHICLE, TYPE_CYCLIST)
