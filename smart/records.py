"""Read the Waymo goal records into the arrays this package trains on.

Source is ``data/advscene_preprocess_waymo/{train,val}``: one pickle per WOMD
scenario, carrying the FULL 91-step trajectories rather than just the initial
snapshot. Those pickles are UNFILTERED -- the off-road removal and the 30-agent
cap live at load time in ``utils.goal_runtime.select_agents``, not on disk -- so
this module applies that same filter. The agent set a model trains on then
matches the agent set ``SimScene`` will contain at rollout time, which is the
whole point: a traffic model trained on agents the simulator would have dropped
learns a distribution it never sees again.

Lifecycle is matched the same way. ``trajectory_clip_valid`` marks the steps an
agent is both tracked and inside the 64 m FOV box, which is exactly the
condition ``SimScene.remove_out_of_bounds`` enforces, so it is what gates a
transition here.

The ego (row 0) is kept and trained on like any other agent. It is a logged
human driver too, and at rollout time the only difference is which planner holds
its reins.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from sim.schema import MIN_DISTANCE_TO_GOAL
from utils.goal_runtime import compute_goals, select_agents

# Matches cfgs/ae_goal/dataset.yaml and cfgs/dataset/waymo_base.yaml.
OFFROAD_THRESHOLD = 1.5
MAX_NUM_AGENTS = 30

# [x, y, heading, signed_speed] -- the state the shared action table integrates.
STATE_DIM = 4


def load_scene(path: str | Path) -> dict:
    """One scene as plain arrays, on the simulator's agent set.

    Returns ``state`` [A, T, 4], ``valid`` [A, T], ``length`` / ``width`` [A],
    ``types`` [A, 3] one-hot, ``lanes`` [L, P, 2], and the scenario id.
    """
    with open(path, "rb") as fh:
        record = pickle.load(fh)

    traj = np.asarray(record["trajectory"], dtype=np.float32)
    traj_valid = np.asarray(record["trajectory_valid"], dtype=bool)
    clip_valid = np.asarray(record["trajectory_clip_valid"], dtype=bool)
    goal_xy, goal_valid, _ = compute_goals(traj, traj_valid, clip_valid)
    keep = select_agents(
        record, goal_valid, goal_xy,
        offroad_threshold=OFFROAD_THRESHOLD, max_num_agents=MAX_NUM_AGENTS,
    )

    traj, clip_valid = traj[keep], clip_valid[keep]
    x, y, vx, vy, yaw = (traj[:, :, i].astype(np.float64) for i in range(5))
    v_dot_h = vx * np.cos(yaw) + vy * np.sin(yaw)
    signed_speed = np.copysign(np.hypot(vx, vy), v_dot_h)

    states = np.asarray(record["agent_states"], dtype=np.float32)[keep]

    # Parked / moving, by the repo's own definition (sim.schema.MIN_DISTANCE_TO_GOAL,
    # the same 2 m that splits the parked/moving conditioning bucket): total
    # displacement between an agent's first and last valid step. About 30% of
    # agents are parked and they contribute 35% of the labelled transitions, all of
    # them the trivial "stay put" target -- which both floods the loss and, if the
    # two are averaged together, flatters every displacement metric.
    moving = np.zeros(len(traj), dtype=bool)
    for a in range(len(traj)):
        idx = np.flatnonzero(clip_valid[a])
        if len(idx) >= 2:
            span = np.hypot(x[a, idx[-1]] - x[a, idx[0]], y[a, idx[-1]] - y[a, idx[0]])
            moving[a] = span >= MIN_DISTANCE_TO_GOAL
    return {
        "scenario_id": record["scenario_id"],
        "state": np.stack([x, y, yaw, signed_speed], axis=-1),
        "valid": clip_valid,
        # Same floor SimScene applies (sim/world.py), so a degenerate box cannot
        # divide by ~0 in the bicycle yaw rate and cannot make the training set
        # disagree with the simulator about an agent's size.
        "length": np.maximum(states[:, 5], 0.5).astype(np.float64),
        "width": np.maximum(states[:, 6], 0.5).astype(np.float64),
        "moving": moving,
        "types": np.asarray(record["agent_types"], dtype=np.float32)[keep],
        "lanes": np.asarray(record["road_points"], dtype=np.float32),
        # route-planning planners (idm) need lane connectivity; stored, not derived
        "lane_edges": np.asarray(record["edge_index_lane_to_lane"]),
        "lane_conn": np.asarray(record["road_connection_types"]),
    }


def transitions(scene: dict) -> dict:
    """Flatten a scene into consecutive (state, next_state) pairs.

    A pair is kept only when BOTH steps are valid, so a track that blinks out and
    returns contributes no transition across the gap.
    """
    state, valid = scene["state"], scene["valid"]
    pair = valid[:, :-1] & valid[:, 1:]
    agent_idx, step_idx = np.nonzero(pair)
    return {
        "state": state[agent_idx, step_idx],
        "next_state": state[agent_idx, step_idx + 1],
        "length": scene["length"][agent_idx],
        "width": scene["width"][agent_idx],
        "agent_idx": agent_idx,
        "step_idx": step_idx,
    }


def scene_paths(split: str, root: str | Path = "data/advscene_preprocess_waymo") -> list[Path]:
    return sorted(Path(root, split).glob("*.pkl"))
