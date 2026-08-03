"""**Runtime** (goal-side) processing of v2 scene records.

The offline half of the pipeline (``utils/goal_preprocess.py``) freezes everything the
original Scenario Dreamer preprocessing froze: which agents are in the scene, their
7-dimensional states, and the lane tensors. Everything *goal*-specific stays here so it
can be changed without regenerating 100+ GB of pickles:

* which timestep the goal is taken from (:func:`compute_goals`),
* the ``[goal_x, goal_y]`` columns appended to the agent states,
* goal-derived flags (clipped / parked / off-road),
* any goal-driven agent filtering (off by default -- see :func:`select_agents`).

:func:`prepare_scene` is the single entry point shared by the dataset
(``datasets/waymo/dataset_ae_goal_waymo.py``), the metrics ground-truth path
(``metrics.py``) and the preprocessing debug visualiser, so the three cannot drift.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from utils.goal_preprocess import V2_DATA_VERSION, VEHICLE_TYPE_INDEX

#: ``last_clipped`` -- last state that is both valid and inside the FOV box. This is the
#: definition the v1 pipeline baked in as ``clipped_final_states``.
#: ``last_raw``     -- last valid state, wherever it ended up (may be far outside the FOV).
GOAL_MODES = ("last_clipped", "last_raw")


def _last_true_index(mask: np.ndarray) -> np.ndarray:
    """Return the index of last valid, or -1 if all invalid.
    Inputs:
        mask: [N, T]
    Outputs:
        last_indices: [N]
    """
    any_true = mask.any(axis=1)
    last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    return np.where(any_true, last, -1)


def compute_goals(traj, traj_valid, clip_valid):
    """Derive per-agent goals from the stored trajectories.

    Inputs:
        traj: [N, T, 5] array of agent states (x, y, speed, cosθ, sinθ).
        traj_valid: [N, T] boolean array of valid states (FOV crop + closest-N cap).
        clip_valid: [N, T] boolean array of valid states that are also inside the FOV box (FOV crop only).

    Outputs:
        goal_xy: [N, 2] array of goal positions (x, y), (0, 0) for invalid agents.
        goal_valid: [N] boolean array of valid goals (True if the agent has a valid goal).
        clip_t: [N] array of timesteps at which the goal is taken (last valid timestep inside the FOV box, or -1 if invalid).
        goal_clipped: [N] boolean array indicating whether the goal was clipped (True if the agent's trajectory was clipped inside the FOV box).
    """
    raw_t = _last_true_index(traj_valid)  # the last valid timestep for whole time
    clip_t = _last_true_index(
        clip_valid
    )  # the last valid timestep that is also inside the FOV box

    goal_valid = clip_t >= 0
    rows = np.arange(len(clip_t))
    goal_xy = traj[rows, np.maximum(clip_t, 0), :2].astype(np.float32)
    goal_xy[~goal_valid] = 0.0
    # goal_clipped = (clip_t != raw_t) & goal_valid

    return goal_xy, goal_valid, clip_t


def point_to_polyline_dist(query_xy: np.ndarray, polylines: np.ndarray) -> np.ndarray:
    """Exact distance from each query point to the nearest lane polyline *segment*.

    Inputs:
        query_xy: [N, 2] array of query points (x, y).
        polylines: [M, L, 2] array of lane polylines (x, y), where M is the number of lanes and L is the number of points per lane.
    Outputs:
        dists: [N] array of distances from each query point to the nearest lane polyline segment.
    """
    query_xy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    polylines = np.asarray(polylines, dtype=np.float32)
    if len(query_xy) == 0:
        return np.zeros((0,), dtype=np.float32)
    if polylines.size == 0:
        return np.full(len(query_xy), np.inf, dtype=np.float32)

    starts = polylines[:, :-1, :].reshape(-1, 2)
    ends = polylines[:, 1:, :].reshape(-1, 2)
    seg = ends - starts
    seg_len_sq = np.maximum((seg**2).sum(axis=-1), 1e-12)

    rel = query_xy[:, None, :] - starts[None, :, :]
    t = np.clip((rel * seg[None, :, :]).sum(axis=-1) / seg_len_sq[None, :], 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * seg[None, :, :]
    return (
        np.linalg.norm(query_xy[:, None, :] - closest, axis=-1)
        .min(axis=1)
        .astype(np.float32)
    )


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    value = getattr(cfg, key, default)
    return default if value is None else value


def select_agents(
    record: Dict[str, Any],
    goal_valid: np.ndarray,
    goal_xy: np.ndarray,
    offroad_threshold: float,
    max_num_agents: int = 30,
) -> np.ndarray:
    """Runtime agent filter. Returns the kept indices (the ego is never dropped).

    Inputs:
        record: v2 data item
        goal_valid: [N] boolean array of valid goals (True if the agent has a valid goal).
        goal_xy: [N, 2] array of goal positions (x, y), (0, 0) for invalid agents.
        offroad_threshold: distance threshold for considering an agent off-road (meters).
        max_num_agents: maximum number of agents to keep (ego is always kept, then the closest N-1 agents are kept).

    Outputs:
        keep_idx: [M] array of indices of the kept agents (M <= max_num_agents).
    """
    num_agents = len(record["agent_states"])
    keep_idx = np.arange(num_agents)
    if num_agents == 0:
        return keep_idx

    types = np.asarray(record["agent_types"])
    is_vehicle = types[:, VEHICLE_TYPE_INDEX].astype(bool)
    drop = np.zeros(num_agents, dtype=bool)

    # drop agents whose starting position if offroad
    start_dist_to_road = point_to_polyline_dist(
        np.asarray(record["agent_states"])[:, :2], record["road_points"]
    )
    drop |= (start_dist_to_road > offroad_threshold) & is_vehicle

    # drop agents whose goal position is offroad
    goal_dist_to_road = point_to_polyline_dist(goal_xy, record["road_points"])
    drop |= (goal_dist_to_road > offroad_threshold) & is_vehicle & goal_valid

    # drop agents with invalid goals
    drop |= ~goal_valid

    # keep ego
    drop[0] = False

    keep_idx = keep_idx[~drop]

    if len(keep_idx) > max_num_agents:
        # states are already ordered ego-first then by distance to the origin
        keep_idx = keep_idx[:max_num_agents]

    return keep_idx


def prepare_scene(
    record: Dict[str, Any],
    cfg: Any,
) -> Dict[str, Any]:
    """v1 to v2 converter.

    Inputs:
        record: v1 data item
        cfg

    Outputs:
        dict:
            agent_states: [M, 9] array of agent states (x, y, speed, cosθ, sinθ, length, width, goal_x, goal_y)
            agent_types: [M, 4] array of agent type one-hot vectors
            goal_xy: [M, 2] array of goal positions (x, y)
            goal_valid: [M] boolean array of valid goals
            goal_timestep: [M] array of timesteps at which the goal is taken
            goal_dist: [M] array of distances from agent to goal
    """
    traj = np.asarray(record["trajectory"], dtype=np.float32)
    traj_valid = np.asarray(record["trajectory_valid"], dtype=bool)
    traj_clip_valid = np.asarray(record["trajectory_clip_valid"], dtype=bool)

    goal_xy, goal_valid, goal_timestep = compute_goals(
        traj, traj_valid, traj_clip_valid
    )
    keep_agent_idx = select_agents(
        record,
        goal_valid,
        goal_xy,
        offroad_threshold=cfg.offroad_threshold,
        max_num_agents=cfg.max_num_agents,
    )

    agent_states = np.asarray(record["agent_states"], dtype=np.float32)[keep_agent_idx]
    agent_states = np.concatenate([agent_states, goal_xy[keep_agent_idx]], axis=-1)

    goal_dist = np.linalg.norm(
        agent_states[:, :2] - agent_states[:, -2:], axis=-1
    ).astype(np.float32)

    return {
        "agent_states": agent_states,
        "agent_types": np.asarray(record["agent_types"], dtype=np.float32)[
            keep_agent_idx
        ],
        "goal_xy": goal_xy[keep_agent_idx],
        "goal_valid": goal_valid[keep_agent_idx],
        "goal_timestep": goal_timestep[keep_agent_idx],
        "goal_dist": goal_dist.astype(np.float32),
    }
