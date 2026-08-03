"""Shared **offline** preprocessing for the goal-augmented Waymo pipeline (v2 schema).

Both producers of the v2 dataset call :func:`build_record`:

* ``data_processing/waymo/preprocess_waymo_selfplay_dataset.py`` -- tfrecord -> v2
  (the official path; run wherever the raw Waymo data lives)
* ``scripts/tmp_convert_goal_v1_to_v2.py`` -- v1 selfplay pickles -> v2
  (a throwaway path so the existing 105 GB of v1 data can be reused today)

Routing every field through one function is the point: the two producers must not
drift, or models trained on converted data would not match models trained on
regenerated data.

Offline / runtime split
-----------------------
**Offline** is exactly what the original Scenario Dreamer autoencoder preprocessing
does (``datasets/waymo/dataset_autoencoder_waymo.py::get_data`` slow path):

    frame selection -> ego-frame transform -> FOV crop -> closest ``max_num_agents``
    -> off-road *vehicle* removal -> modify_agent_states -> lane-graph tensors

The resulting record is a drop-in for the original autoencoder's preprocessed pickle:
same key names, same layouts, ``agent_states`` is 7-dimensional.

**Runtime** is everything goal-specific -- picking the goal timestep, appending the
``[goal_x, goal_y]`` columns, goal distance / parked / clipped flags, and any future
goal-driven agent filtering. See ``datasets/waymo/dataset_ae_goal_waymo.py``. The
trajectories needed for that live in the record's goal block.

Off-road removal detail
-----------------------
Mirrors the original (``dataset_autoencoder_waymo.py:477-521``) exactly:

* only **vehicles** are dropped; pedestrians and cyclists are always kept,
* the ego (row 0) is never dropped,
* distance is measured to the FOV lane centerlines **resampled to 1000 points**
  (``upsample_lane_num_points``), before the ``max_num_lanes`` cap,
* and it runs *after* the closest-``max_num_agents`` cap, so the agent-count
  distribution matches the original.

``agent_road_dist`` is stored, so a *stricter* threshold can still be applied at
runtime without regenerating; a looser one cannot (those agents are gone).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from cfgs.config import NON_PARTITIONED
from utils.data_helpers import modify_agent_states
from utils.lane_graph_helpers import resample_polyline
from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph

V2_DATA_VERSION = 2

LANE_CONNECTION_TYPES = {"none": 0, "pred": 1, "succ": 2, "left": 3, "right": 4, "self": 5}
#: index of the vehicle class in the 3-way one-hot used by the selfplay/goal pickles.
#: NOTE: the *original* pipeline stores a 5-way one-hot and uses column 1 for vehicles;
#: here vehicles are column 0. Copying the original mask expression verbatim would
#: silently treat pedestrians as vehicles.
VEHICLE_TYPE_INDEX = 0


@dataclass
class V2Config:
    """Offline preprocessing knobs. Defaults reproduce the original pipeline."""

    map_range: float = 64.0  # square FOV side length, in metres
    max_num_agents: int = 30
    max_num_lanes: int = 100
    num_points_per_lane: int = 20  # lane resolution fed to the model
    upsample_lane_num_points: int = 1000  # lane resolution used for the off-road test
    lane_polyline_points: int = 50  # lane resolution kept for runtime geometry queries
    offroad_threshold: float = 1.5  # metres; <=0 disables off-road removal
    num_lane_connection_types: int = 6
    traj_dtype: str = "float32"


# --------------------------------------------------------------------------- lanes


def dense_lane_points(lanes: Dict[Any, np.ndarray], num_points: int) -> np.ndarray:
    """Concatenate every lane, resampled to ``num_points``, into one ``(P, 2)`` cloud.

    Matches the original off-road test, which measures distance against the
    1000-point upsampled FOV lanes (all of them -- before the ``max_num_lanes`` cap).
    """
    chunks = []
    for lane in lanes.values():
        lane = np.asarray(lane, dtype=np.float32)
        if len(lane) == 0:
            continue
        if len(lane) == 1:
            chunks.append(np.repeat(lane[:1], num_points, axis=0))
        else:
            chunks.append(resample_polyline(lane, num_points=num_points).astype(np.float32))
    if not chunks:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def min_dist_to_points(query_xy: np.ndarray, pts: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Distance from each query point to the nearest point in ``pts``.

    Chunked over ``pts`` because the lane cloud can reach a few hundred thousand
    points, which would otherwise allocate a multi-hundred-MB pairwise matrix.
    """
    query_xy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return np.full(len(query_xy), np.inf, dtype=np.float32)
    best = np.full(len(query_xy), np.inf, dtype=np.float32)
    for start in range(0, len(pts), chunk):
        block = pts[start:start + chunk]
        d = np.linalg.norm(query_xy[:, None, :] - block[None, :, :], axis=-1).min(axis=1)
        np.minimum(best, d.astype(np.float32), out=best)
    return best


def build_lane_tensors(lane_graph: Dict[str, Any], cfg: V2Config) -> Optional[Dict[str, Any]]:
    """Lane-graph dict -> the tensors the model consumes.

    Lanes are ordered by distance to the origin and capped at ``max_num_lanes``,
    exactly as the original ``get_road_points_adj`` does. ``lane_polylines`` carries
    the same lanes in the same order at a higher resolution, so runtime geometry
    queries (e.g. is this goal off-road?) stay possible once v1 is deleted.
    """
    lanes = lane_graph.get("lanes", {})
    if len(lanes) == 0:
        return None

    resampled, polylines, idx_to_id, id_to_idx = [], [], {}, {}
    for idx, lane_id in enumerate(lanes):
        lane = np.asarray(lanes[lane_id], dtype=np.float32)
        if len(lane) == 1:
            sampled = np.repeat(lane[:1], cfg.num_points_per_lane, axis=0)
            poly = np.repeat(lane[:1], cfg.lane_polyline_points, axis=0)
        else:
            sampled = resample_polyline(lane, num_points=cfg.num_points_per_lane)
            poly = resample_polyline(lane, num_points=cfg.lane_polyline_points)
        resampled.append(sampled.astype(np.float32))
        polylines.append(poly.astype(np.float32))
        idx_to_id[idx] = lane_id
        id_to_idx[lane_id] = idx

    resampled = np.asarray(resampled, dtype=np.float32)
    polylines = np.asarray(polylines, dtype=np.float32)
    num_lanes = min(len(resampled), cfg.max_num_lanes)
    dist_to_origin = np.linalg.norm(resampled, axis=-1).min(1)
    closest = np.argsort(dist_to_origin)[:num_lanes]
    resampled = resampled[closest]
    polylines = polylines[closest]

    idx_to_new_idx = {old: new for new, old in enumerate(closest)}
    new_idx_to_idx = {new: old for new, old in enumerate(closest)}

    adjacency = {
        key: np.zeros((num_lanes, num_lanes), dtype=np.float32)
        for key in ("pre_pairs", "suc_pairs", "left_pairs", "right_pairs")
    }
    for new_i in range(num_lanes):
        lane_id = idx_to_id[new_idx_to_idx[new_i]]
        for key in adjacency:
            for other_id in lane_graph.get(key, {}).get(lane_id, []):
                old_idx = id_to_idx.get(other_id)
                if old_idx in idx_to_new_idx:
                    adjacency[key][new_i, idx_to_new_idx[old_idx]] = 1.0

    edge_index_lane_to_lane = get_edge_index_complete_graph(num_lanes).numpy()
    road_connection_types = build_road_connection_types(
        edge_index_lane_to_lane,
        adjacency["pre_pairs"],
        adjacency["suc_pairs"],
        adjacency["left_pairs"],
        adjacency["right_pairs"],
    )

    return {
        "road_points": resampled,
        "lane_polylines": polylines,
        "edge_index_lane_to_lane": edge_index_lane_to_lane,
        "road_connection_types": road_connection_types,
        "num_lanes": int(num_lanes),
    }


def build_road_connection_types(edge_index_lane_to_lane, pre_adj, suc_adj, left_adj, right_adj):
    """One-hot lane-to-lane relation per edge, in the edge-index column order."""
    num_edges = edge_index_lane_to_lane.shape[1]
    out = np.zeros((num_edges, len(LANE_CONNECTION_TYPES)), dtype=np.float32)
    for i in range(num_edges):
        src = int(edge_index_lane_to_lane[0, i])
        dst = int(edge_index_lane_to_lane[1, i])
        if src == dst:
            name = "self"
        elif pre_adj[dst, src]:
            name = "pred"
        elif suc_adj[dst, src]:
            name = "succ"
        elif left_adj[dst, src]:
            name = "left"
        elif right_adj[dst, src]:
            name = "right"
        else:
            name = "none"
        out[i, LANE_CONNECTION_TYPES[name]] = 1.0
    return out

# --------------------------------------------------------------------------- record


def build_record(
    *,
    idx: int,
    scenario_id: str,
    source_file: str,
    source_record_index: int,
    scene_timestep: int,
    sdc_track_index: int,
    normalize: Dict[str, Any],
    map_id: int,
    agent_states_raw: np.ndarray,
    agent_types: np.ndarray,
    agent_ids: np.ndarray,
    agent_track_indices: np.ndarray,
    trajectory: np.ndarray,
    trajectory_valid: np.ndarray,
    trajectory_clip_valid: np.ndarray,
    lane_graph: Dict[str, Any],
    extras: Optional[Dict[str, Any]] = None,
    cfg: V2Config = None,
) -> Optional[Dict[str, Any]]:
    """Assemble one v2 scene record, or ``None`` for a scene the original would drop.

    Parameters
    ----------
    agent_states_raw : ``(N, 7)`` ``[x, y, vx, vy, yaw, length, width]`` in the ego
        frame, ego at row 0, others sorted by distance to the origin.
    agent_types : ``(N, 3)`` one-hot ``{vehicle: 0, pedestrian: 1, cyclist: 2}``.
    trajectory : ``(N, 91, 5)`` ``[x, y, vx, vy, yaw]`` in the ego frame -- the goal
        block. ``trajectory_valid`` is raw Waymo validity; ``trajectory_clip_valid``
        additionally requires the state to be inside the FOV box (this is what the
        "clipped" goal is derived from at runtime).
    lane_graph : compact lane graph, ``{"lanes": {id: (P_i, 2)}, "pre_pairs": ..., ...}``
        already expressed in the ego frame and cropped to the FOV.
    """
    cfg = cfg or V2Config()

    lane_tensors = build_lane_tensors(lane_graph, cfg)
    if lane_tensors is None:
        return None

    agent_states_raw = np.asarray(agent_states_raw, dtype=np.float32)
    agent_types = np.asarray(agent_types, dtype=np.float32)
    if len(agent_states_raw) == 0:
        return None

    dense = dense_lane_points(lane_graph.get("lanes", {}), cfg.upsample_lane_num_points)
    agent_road_dist = min_dist_to_points(agent_states_raw[:, :2], dense)


    # [x, y, vx, vy, yaw, l, w] -> [x, y, speed, cos, sin, l, w], as the original does
    agent_states = modify_agent_states(agent_states_raw)
    agent_types = agent_types
    num_agents = int(len(agent_states))
    num_lanes = lane_tensors["num_lanes"]

    traj_dtype = np.dtype(cfg.traj_dtype)
    record = {
        # "data_version": V2_DATA_VERSION,
        # ---- identity / provenance (the only way back to the raw Waymo scenario
        # once v1 is deleted) ----
        "idx": int(idx),
        "scenario_id": scenario_id,
        "source_file": source_file,
        "source_record_index": int(source_record_index),
        "scene_timestep": int(scene_timestep),
        "lg_type": NON_PARTITIONED,
        # "sdc_track_index": int(sdc_track_index),
        # "normalize": normalize,
        # "map_id": int(map_id),
        # ---- original-schema block (drop-in for the baseline autoencoder dataset) ----
        "num_agents": num_agents,
        "num_lanes": num_lanes,
        "agent_states": agent_states.astype(np.float32),
        "agent_types": agent_types.astype(np.float32),
        "road_points": lane_tensors["road_points"],
        "road_connection_types": lane_tensors["road_connection_types"],
        "edge_index_lane_to_lane": lane_tensors["edge_index_lane_to_lane"],
        "edge_index_agent_to_agent": get_edge_index_complete_graph(num_agents).numpy(),
        "edge_index_lane_to_agent": get_edge_index_bipartite(num_lanes, num_agents).numpy(),
        # ---- goal block (consumed at runtime) ----
        "trajectory": np.asarray(trajectory, dtype=np.float32).astype(traj_dtype),
        "trajectory_valid": np.asarray(trajectory_valid, dtype=bool),
        "trajectory_clip_valid": np.asarray(trajectory_clip_valid, dtype=bool),
        # ---- filtering provenance ----
        # "agent_ids": np.asarray(agent_ids, dtype=np.int64)[keep],
        # "agent_track_indices": np.asarray(agent_track_indices, dtype=np.int64)[keep],
        # "ego_index": 0,
        # "agent_road_dist": agent_road_dist[keep].astype(np.float32),
        # "offroad_threshold_offline": float(cfg.offroad_threshold),
        # "max_num_agents_offline": int(cfg.max_num_agents),
        # ---- geometry kept for runtime queries / simulation ----
        # "lane_polylines": lane_tensors["lane_polylines"],
    }

    if extras:
        record.update(extras)
    return record
