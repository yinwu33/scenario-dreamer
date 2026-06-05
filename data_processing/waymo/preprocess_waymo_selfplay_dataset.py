import argparse
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

from cfgs.config import NON_PARTITIONED
from utils.geometry import apply_se2_transform, normalize_angle
from utils.lane_graph_helpers import get_compact_lane_graph, resample_polyline
from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph


DATASET_ROOT = "/mnt/disk/data/public/waymo/motion_v_1_3_1/scenario/"
_MAP_RANGE = 64.0
_NUM_POINTS_PER_LANE = 20
_MAX_NUM_LANES = 100
_ERR_VAL = -1e4
_WAYMO_OBJECT_STR = {
    scenario_pb2.Track.TYPE_UNSET: "unset",
    scenario_pb2.Track.TYPE_VEHICLE: "vehicle",
    scenario_pb2.Track.TYPE_PEDESTRIAN: "pedestrian",
    scenario_pb2.Track.TYPE_CYCLIST: "cyclist",
    scenario_pb2.Track.TYPE_OTHER: "other",
}
_SELFPLAY_OBJECT_TYPES = {"vehicle": 0, "pedestrian": 1, "cyclist": 2}
_LANE_CONNECTION_TYPES = {"none": 0, "pred": 1, "succ": 2, "left": 3, "right": 4, "self": 5}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw Waymo TFRecords into SDC-centered selfplay data."
    )
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--max", type=int, default=None, help="Maximum saved scenarios for debugging.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of TFRecord files to process in parallel.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--viz", action="store_true", help="Write debug GIFs next to the output split.")
    return parser.parse_args()


def _default_waymo_folder(split):
    dataset_root = DATASET_ROOT
    if dataset_root is None:
        raise ValueError("DATASET_ROOT must be set when --output-dir is omitted or raw data is read.")

    split_dir = "training" if split == "train" else "validation"
    return Path(dataset_root) / split_dir


def _default_output_dir(split):
    dataset_root = os.environ.get("DATASET_ROOT")
    if dataset_root is None:
        raise ValueError("DATASET_ROOT must be set when --output-dir is omitted.")
    return Path(dataset_root) / "scenario_dreamer_selfplay_waymo" / split


def _rotation_from_sdc_yaw(yaw):
    return (np.pi / 2) - yaw


def _to_local_positions(xy, center, rotation):
    return apply_se2_transform(
        coordinates=xy,
        translation=center.reshape(1, 1, 2),
        yaw=rotation,
    )


def _to_local_vectors(xy, rotation):
    return apply_se2_transform(
        coordinates=xy,
        translation=np.zeros((1, 1, 2), dtype=xy.dtype),
        yaw=rotation,
    )


def _in_map_range(xy):
    radius = _MAP_RANGE / 2.0
    return np.logical_and(np.abs(xy[..., 0]) <= radius, np.abs(xy[..., 1]) <= radius)


def _object_type_name(track):
    return _WAYMO_OBJECT_STR.get(track.object_type, "other")


def _object_type_onehot(type_name):
    onehot = np.zeros(len(_SELFPLAY_OBJECT_TYPES), dtype=np.float32)
    if type_name in _SELFPLAY_OBJECT_TYPES:
        onehot[_SELFPLAY_OBJECT_TYPES[type_name]] = 1.0
    return onehot


def _lane_connection_type_onehot(type_name):
    onehot = np.zeros(len(_LANE_CONNECTION_TYPES), dtype=np.float32)
    onehot[_LANE_CONNECTION_TYPES[type_name]] = 1.0
    return onehot


def _extract_track_states(track):
    states = []
    valid = []
    for state in track.states:
        if state.valid:
            states.append(
                [
                    state.center_x,
                    state.center_y,
                    state.velocity_x,
                    state.velocity_y,
                    state.heading,
                    state.length,
                    state.width,
                    1.0,
                ]
            )
            valid.append(True)
        else:
            states.append([_ERR_VAL, _ERR_VAL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            valid.append(False)

    return np.asarray(states, dtype=np.float32), np.asarray(valid, dtype=bool)


def _localize_states(global_states, valid, center, rotation):
    local_states = np.array(global_states, copy=True)
    if valid.any():
        valid_positions = global_states[valid, :2][None, :, :]
        valid_velocities = global_states[valid, 2:4][None, :, :]
        local_states[valid, :2] = _to_local_positions(valid_positions, center, rotation)[0]
        local_states[valid, 2:4] = _to_local_vectors(valid_velocities, rotation)[0]
        local_states[valid, 4] = normalize_angle(global_states[valid, 4] + rotation)
    return local_states


def _last_valid_state(states, valid):
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return np.zeros(states.shape[-1], dtype=np.float32), False, -1
    idx = int(valid_indices[-1])
    return states[idx].copy(), True, idx


def _polyline_from_map_points(points):
    return np.asarray([[p.x, p.y] for p in points], dtype=np.float32)


def _polygon_from_map_points(points):
    return np.asarray([[p.x, p.y] for p in points], dtype=np.float32)


def _get_lane_neighbor_ids(neighbors):
    ids = []
    for neighbor in neighbors:
        if hasattr(neighbor, "feature_id"):
            ids.append(int(neighbor.feature_id))
    return ids


def _empty_lane_graph():
    return {
        "lanes": {},
        "pre_pairs": {},
        "suc_pairs": {},
        "left_pairs": {},
        "right_pairs": {},
    }


def _get_road_points_adj(lane_graph):
    if len(lane_graph["lanes"]) == 0:
        return (
            np.zeros((0, _NUM_POINTS_PER_LANE, 2), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            0,
        )

    resampled_lanes = []
    idx_to_id = {}
    id_to_idx = {}
    for idx, lane_id in enumerate(lane_graph["lanes"]):
        lane = lane_graph["lanes"][lane_id]
        if len(lane) == 1:
            sampled = np.repeat(lane[:1], _NUM_POINTS_PER_LANE, axis=0)
        else:
            sampled = resample_polyline(lane, num_points=_NUM_POINTS_PER_LANE)
        resampled_lanes.append(sampled.astype(np.float32))
        idx_to_id[idx] = lane_id
        id_to_idx[lane_id] = idx

    resampled_lanes = np.asarray(resampled_lanes, dtype=np.float32)
    num_lanes = min(len(resampled_lanes), _MAX_NUM_LANES)
    dist_to_origin = np.linalg.norm(resampled_lanes, axis=-1).min(1)
    closest_lane_idxs = np.argsort(dist_to_origin)[:num_lanes]
    resampled_lanes = resampled_lanes[closest_lane_idxs]

    idx_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(closest_lane_idxs)}
    new_idx_to_idx = {new_idx: old_idx for new_idx, old_idx in enumerate(closest_lane_idxs)}

    pre_adj = np.zeros((num_lanes, num_lanes), dtype=np.float32)
    suc_adj = np.zeros((num_lanes, num_lanes), dtype=np.float32)
    left_adj = np.zeros((num_lanes, num_lanes), dtype=np.float32)
    right_adj = np.zeros((num_lanes, num_lanes), dtype=np.float32)

    for new_idx_i in range(num_lanes):
        lane_id = idx_to_id[new_idx_to_idx[new_idx_i]]
        for other_id in lane_graph["pre_pairs"].get(lane_id, []):
            old_idx = id_to_idx.get(other_id)
            if old_idx in idx_to_new_idx:
                pre_adj[new_idx_i, idx_to_new_idx[old_idx]] = 1.0
        for other_id in lane_graph["suc_pairs"].get(lane_id, []):
            old_idx = id_to_idx.get(other_id)
            if old_idx in idx_to_new_idx:
                suc_adj[new_idx_i, idx_to_new_idx[old_idx]] = 1.0
        for other_id in lane_graph["left_pairs"].get(lane_id, []):
            old_idx = id_to_idx.get(other_id)
            if old_idx in idx_to_new_idx:
                left_adj[new_idx_i, idx_to_new_idx[old_idx]] = 1.0
        for other_id in lane_graph["right_pairs"].get(lane_id, []):
            old_idx = id_to_idx.get(other_id)
            if old_idx in idx_to_new_idx:
                right_adj[new_idx_i, idx_to_new_idx[old_idx]] = 1.0

    return resampled_lanes, pre_adj, suc_adj, left_adj, right_adj, num_lanes


def _build_road_connection_types(edge_index_lane_to_lane, pre_adj, suc_adj, left_adj, right_adj):
    road_connection_types = []
    for i in range(edge_index_lane_to_lane.shape[1]):
        src = int(edge_index_lane_to_lane[0, i])
        dst = int(edge_index_lane_to_lane[1, i])
        if src == dst:
            conn_type = "self"
        elif pre_adj[dst, src]:
            conn_type = "pred"
        elif suc_adj[dst, src]:
            conn_type = "succ"
        elif left_adj[dst, src]:
            conn_type = "left"
        elif right_adj[dst, src]:
            conn_type = "right"
        else:
            conn_type = "none"
        road_connection_types.append(_lane_connection_type_onehot(conn_type))
    return np.asarray(road_connection_types, dtype=np.float32)


def _extract_map(scenario, center, rotation):
    lane_graph = _empty_lane_graph()
    road_edges = []
    road_edge_masks = []
    crosswalks = []
    stop_signs = []

    for feature in scenario.map_features:
        if feature.HasField("lane"):
            polyline = _polyline_from_map_points(feature.lane.polyline)
            if len(polyline) < 2:
                continue
            local = _to_local_positions(polyline[None, :, :], center, rotation)[0]
            in_range = _in_map_range(local)
            if not in_range.any():
                continue
            lane_id = int(feature.id)
            lane_graph["lanes"][lane_id] = local[in_range].astype(np.float32)
            lane_graph["pre_pairs"][lane_id] = [int(lid) for lid in feature.lane.entry_lanes]
            lane_graph["suc_pairs"][lane_id] = [int(lid) for lid in feature.lane.exit_lanes]
            lane_graph["left_pairs"][lane_id] = _get_lane_neighbor_ids(feature.lane.left_neighbors)
            lane_graph["right_pairs"][lane_id] = _get_lane_neighbor_ids(feature.lane.right_neighbors)

        elif feature.HasField("road_edge"):
            polyline = _polyline_from_map_points(feature.road_edge.polyline)
            if len(polyline) == 0:
                continue
            local = _to_local_positions(polyline[None, :, :], center, rotation)[0]
            in_range = _in_map_range(local)
            if not in_range.any():
                continue
            points = local[in_range]
            if len(points) == 1:
                sampled = np.repeat(points[:1], _NUM_POINTS_PER_LANE, axis=0)
            else:
                sampled = resample_polyline(points, num_points=_NUM_POINTS_PER_LANE)
            sampled_mask = np.ones(_NUM_POINTS_PER_LANE, dtype=bool)
            road_edges.append(sampled)
            road_edge_masks.append(sampled_mask)

        elif feature.HasField("crosswalk"):
            polygon = _polygon_from_map_points(feature.crosswalk.polygon)
            if len(polygon) == 0:
                continue
            local = _to_local_positions(polygon[None, :, :], center, rotation)[0]
            in_range = _in_map_range(local)
            if in_range.any():
                crosswalks.append(local[in_range].astype(np.float32))

        elif feature.HasField("stop_sign"):
            point = np.asarray([[feature.stop_sign.position.x, feature.stop_sign.position.y]], dtype=np.float32)
            local = _to_local_positions(point[None, :, :], center, rotation)[0, 0]
            if _in_map_range(local):
                stop_signs.append(local.astype(np.float32))

    lane_ids = set(lane_graph["lanes"].keys())
    for pair_key in ("pre_pairs", "suc_pairs", "left_pairs", "right_pairs"):
        for lane_id in list(lane_graph[pair_key].keys()):
            lane_graph[pair_key][lane_id] = [other_id for other_id in lane_graph[pair_key][lane_id] if other_id in lane_ids]

    if len(lane_graph["lanes"]) == 0:
        road_points, pre_adj, suc_adj, left_adj, right_adj, num_lanes = _get_road_points_adj(lane_graph)
        edge_index_lane_to_lane = get_edge_index_complete_graph(num_lanes).numpy()
        road_connection_types = np.zeros((0, len(_LANE_CONNECTION_TYPES)), dtype=np.float32)
        compact_lane_graph = lane_graph
    else:
        compact_lane_graph = get_compact_lane_graph({"lane_graph": lane_graph})
        road_points, pre_adj, suc_adj, left_adj, right_adj, num_lanes = _get_road_points_adj(compact_lane_graph)
        edge_index_lane_to_lane = get_edge_index_complete_graph(num_lanes).numpy()
        road_connection_types = _build_road_connection_types(
            edge_index_lane_to_lane,
            pre_adj,
            suc_adj,
            left_adj,
            right_adj,
        )

    map_data = {
        "road_points": road_points,
        "road_point_masks": np.ones((num_lanes, _NUM_POINTS_PER_LANE), dtype=bool),
        "edge_index_lane_to_lane": edge_index_lane_to_lane,
        "road_connection_types": road_connection_types,
        "num_lanes": num_lanes,
        "lane_graph": compact_lane_graph,
        "road_edges": np.stack(road_edges, axis=0) if road_edges else np.zeros((0, _NUM_POINTS_PER_LANE, 2), dtype=np.float32),
        "road_edge_masks": np.stack(road_edge_masks, axis=0) if road_edge_masks else np.zeros((0, _NUM_POINTS_PER_LANE), dtype=bool),
        "crosswalks": crosswalks,
        "stop_signs": np.stack(stop_signs, axis=0) if stop_signs else np.zeros((0, 2), dtype=np.float32),
    }
    return map_data


def _process_scenario(scenario, source_file, record_index, output_path, viz_dir=None):
    current_t = int(scenario.current_time_index)
    sdc_idx = int(scenario.sdc_track_index)
    if sdc_idx < 0 or sdc_idx >= len(scenario.tracks):
        return False

    sdc_track = scenario.tracks[sdc_idx]
    if current_t < 0 or current_t >= len(sdc_track.states):
        return False
    sdc_state = sdc_track.states[current_t]
    if not sdc_state.valid:
        return False

    center = np.asarray([sdc_state.center_x, sdc_state.center_y], dtype=np.float32)
    rotation = _rotation_from_sdc_yaw(sdc_state.heading)

    map_data = _extract_map(scenario, center, rotation)
    if map_data["num_lanes"] == 0:
        return False

    agent_ids = []
    agent_track_indices = []
    agent_types = []
    agent_type_names = []
    global_trajectories = []
    local_trajectories = []
    trajectory_valid = []
    clipped_valid = []
    raw_global_final_states = []
    raw_final_states = []
    raw_final_valid = []
    raw_final_timesteps = []
    clipped_final_states = []
    clipped_final_valid = []
    clipped_final_timesteps = []
    current_local_states = []

    for track_index, track in enumerate(scenario.tracks):
        type_name = _object_type_name(track)
        if type_name not in _SELFPLAY_OBJECT_TYPES:
            continue
        if current_t >= len(track.states) or not track.states[current_t].valid:
            continue

        global_states, valid = _extract_track_states(track)
        local_states = _localize_states(global_states, valid, center, rotation)
        current_local = local_states[current_t]
        if not _in_map_range(current_local[:2]):
            continue

        in_range = _in_map_range(local_states[:, :2])
        clip_mask = np.logical_and(valid, in_range)
        raw_global_final, _, _ = _last_valid_state(global_states, valid)
        raw_final, raw_ok, raw_t = _last_valid_state(local_states, valid)
        clipped_final, clipped_ok, clipped_t = _last_valid_state(local_states, clip_mask)

        agent_ids.append(int(track.id) if track.id else track_index)
        agent_track_indices.append(track_index)
        agent_types.append(_object_type_onehot(type_name))
        agent_type_names.append(type_name)
        global_trajectories.append(global_states)
        local_trajectories.append(local_states)
        trajectory_valid.append(valid)
        clipped_valid.append(clip_mask)
        raw_global_final_states.append(raw_global_final)
        raw_final_states.append(raw_final)
        raw_final_valid.append(raw_ok)
        raw_final_timesteps.append(raw_t)
        clipped_final_states.append(clipped_final)
        clipped_final_valid.append(clipped_ok)
        clipped_final_timesteps.append(clipped_t)
        current_local_states.append(current_local)

    if len(agent_ids) == 0:
        return False

    current_local_states = np.asarray(current_local_states, dtype=np.float32)
    non_ego_order = np.argsort(np.linalg.norm(current_local_states[:, :2], axis=-1))
    ego_candidates = np.where(np.asarray(agent_track_indices, dtype=np.int64) == sdc_idx)[0]
    if len(ego_candidates) == 0:
        return False
    ego_unordered_idx = int(ego_candidates[0])
    order = np.concatenate(
        [
            np.asarray([ego_unordered_idx], dtype=np.int64),
            non_ego_order[non_ego_order != ego_unordered_idx],
        ],
        axis=0,
    )

    def ordered_array(values, dtype=None):
        arr = np.asarray(values, dtype=dtype)
        return arr[order]

    ordered_track_indices = ordered_array(agent_track_indices, dtype=np.int64)
    ego_matches = np.where(ordered_track_indices == sdc_idx)[0]

    scenario_id = scenario.scenario_id or Path(source_file).stem + f"_{record_index}"
    raw_file_name = f"{Path(source_file).name}_{record_index}"
    to_pickle = {
        "idx": record_index,
        "scenario_id": scenario_id,
        "source_file": source_file,
        "source_record_index": record_index,
        "scene_timestep": current_t,
        "map_range": _MAP_RANGE,
        "map_radius": _MAP_RANGE / 2.0,
        "sdc_track_index": sdc_idx,
        "map_id": 0,
        "normalize": {
            "center": center,
            "yaw": np.float32(sdc_state.heading),
            "rotation": np.float32(rotation),
        },
        "num_agents": len(agent_ids),
        "num_lanes": len(map_data["road_points"]),
        "agent_ids": ordered_array(agent_ids, dtype=np.int64),
        "agent_track_indices": ordered_track_indices,
        "ego_index": int(ego_matches[0]),
        "agent_types": ordered_array(agent_types, dtype=np.float32),
        "agent_type_names": [agent_type_names[i] for i in order],
        "agent_states": ordered_array(current_local_states, dtype=np.float32)[:, :-1],
        "raw_global_trajectory": ordered_array(global_trajectories, dtype=np.float32),
        "local_trajectory": ordered_array(local_trajectories, dtype=np.float32),
        "trajectory_valid": ordered_array(trajectory_valid, dtype=bool),
        "clipped_trajectory": ordered_array(local_trajectories, dtype=np.float32),
        "clipped_valid": ordered_array(clipped_valid, dtype=bool),
        "raw_global_final_states": ordered_array(raw_global_final_states, dtype=np.float32),
        "raw_final_states": ordered_array(raw_final_states, dtype=np.float32),
        "raw_final_valid": ordered_array(raw_final_valid, dtype=bool),
        "raw_final_timesteps": ordered_array(raw_final_timesteps, dtype=np.int64),
        "clipped_final_states": ordered_array(clipped_final_states, dtype=np.float32),
        "clipped_final_valid": ordered_array(clipped_final_valid, dtype=bool),
        "clipped_final_timesteps": ordered_array(clipped_final_timesteps, dtype=np.int64),
        "road_points": map_data["road_points"],
        "road_point_masks": map_data["road_point_masks"],
        "edge_index_lane_to_lane": map_data["edge_index_lane_to_lane"],
        "edge_index_lane_to_agent": get_edge_index_bipartite(map_data["num_lanes"], len(agent_ids)).numpy(),
        "edge_index_agent_to_agent": get_edge_index_complete_graph(len(agent_ids)).numpy(),
        "road_connection_types": map_data["road_connection_types"],
        "lg_type": NON_PARTITIONED,
        "map": map_data,
    }

    with open(output_path / f"{raw_file_name}.pkl", "wb") as f:
        pickle.dump(to_pickle, f, protocol=pickle.HIGHEST_PROTOCOL)

    if viz_dir is not None:
        _write_debug_gif(to_pickle, viz_dir / f"{raw_file_name}.gif")

    return True


def _write_debug_gif(data, gif_path):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Polygon
    except Exception as exc:
        print(f"[viz] Skipping {gif_path.name}: {exc}")
        return

    trajectories = data["local_trajectory"]
    valid = data["clipped_valid"]
    agent_type_names = data["agent_type_names"]
    ego_index = data["ego_index"]
    num_steps = trajectories.shape[1]
    radius = data["map_radius"]

    fig, ax = plt.subplots(figsize=(5, 5))

    def agent_color(agent_idx):
        if agent_idx == ego_index:
            return "red"
        type_name = agent_type_names[agent_idx]
        if type_name == "vehicle":
            return "blue"
        if type_name == "cyclist":
            return "yellow"
        if type_name == "pedestrian":
            return "purple"
        return "black"

    def bbox_corners(state):
        x, y, heading, length, width = state[0], state[1], state[4], state[5], state[6]
        local = np.asarray(
            [
                [-length / 2.0, -width / 2.0],
                [-length / 2.0, width / 2.0],
                [length / 2.0, width / 2.0],
                [length / 2.0, -width / 2.0],
            ],
            dtype=np.float32,
        )
        rot = np.asarray(
            [
                [np.cos(heading), -np.sin(heading)],
                [np.sin(heading), np.cos(heading)],
            ],
            dtype=np.float32,
        )
        return local @ rot.T + np.asarray([x, y], dtype=np.float32)

    def draw(frame):
        ax.clear()
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{data['scenario_id']} t={frame}")
        for lane in data["road_points"]:
            ax.plot(lane[:, 0], lane[:, 1], color="0.75", linewidth=0.7)
        for edge in data["map"]["road_edges"]:
            ax.plot(edge[:, 0], edge[:, 1], color="0.45", linewidth=0.7)
        ax.scatter([0.0], [0.0], s=18, color="red")
        for a in range(data["num_agents"]):
            mask = valid[a, : frame + 1]
            if mask.any():
                pts = trajectories[a, : frame + 1, :2][mask]
                ax.plot(pts[:, 0], pts[:, 1], linewidth=1.0)
            if valid[a, frame]:
                color = agent_color(a)
                corners = bbox_corners(trajectories[a, frame])
                ax.add_patch(
                    Polygon(
                        corners,
                        closed=True,
                        facecolor=color,
                        edgecolor="black",
                        linewidth=0.5,
                        alpha=0.45,
                    )
                )
            goal = data["clipped_final_states"][a]
            if data["clipped_final_valid"][a]:
                ax.scatter(goal[0], goal[1], marker="x", s=18, color="black")

    animation = FuncAnimation(fig, draw, frames=num_steps, interval=120)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(gif_path, writer=PillowWriter(fps=8))
    plt.close(fig)


def _process_file_worker(args):
    filename, output_path, viz_dir, max_scenarios, reserved_counter, saved_counter, counter_lock = args
    local_saved = 0
    dataset = tf.data.TFRecordDataset(str(filename), compression_type="")
    for record_index, data in enumerate(dataset):
        if max_scenarios is not None:
            with counter_lock:
                if reserved_counter.value >= max_scenarios:
                    return local_saved
                reserved_counter.value += 1

        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data.numpy())
        if _process_scenario(scenario, filename.name, record_index, output_path, viz_dir):
            local_saved += 1
            if max_scenarios is not None:
                with counter_lock:
                    saved_counter.value += 1
        elif max_scenarios is not None:
            with counter_lock:
                reserved_counter.value -= 1
    return local_saved


def main():
    args = _parse_args()
    raw_folder = _default_waymo_folder(args.split)
    output_path = Path(args.output_dir) / args.split if args.output_dir else _default_output_dir(args.split)
    output_path.mkdir(parents=True, exist_ok=True)
    viz_dir = output_path / "viz" if args.viz else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_folder.glob("*.tfrecord*"))
    if not files:
        raise FileNotFoundError(f"No TFRecord files found in {raw_folder}")

    num_workers = max(1, min(args.num_workers, len(files)))
    if num_workers == 1:
        reserved_value = mp.Value("i", 0)
        saved_value = mp.Value("i", 0)
        lock = mp.Lock()
        saved = 0
        for filename in tqdm(files, desc=f"selfplay-{args.split}"):
            saved += _process_file_worker((filename, output_path, viz_dir, args.max, reserved_value, saved_value, lock))
            if args.max is not None and reserved_value.value >= args.max:
                print(f"Saved {saved_value.value} scenarios to {output_path}")
                return
    else:
        manager = mp.Manager()
        reserved_value = manager.Value("i", 0)
        saved_value = manager.Value("i", 0)
        lock = manager.Lock()
        worker_args = [(filename, output_path, viz_dir, args.max, reserved_value, saved_value, lock) for filename in files]
        saved = 0
        with mp.Pool(processes=num_workers) as pool:
            for local_saved in tqdm(
                pool.imap_unordered(_process_file_worker, worker_args),
                total=len(worker_args),
                desc=f"selfplay-{args.split}",
            ):
                saved += local_saved
                if args.max is not None and reserved_value.value >= args.max:
                    pool.terminate()
                    pool.join()
                    print(f"Saved {saved_value.value} scenarios to {output_path}")
                    return

    print(f"Saved {saved} scenarios to {output_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
