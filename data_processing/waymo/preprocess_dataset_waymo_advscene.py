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

from utils.geometry import apply_se2_transform, normalize_angle
from utils.goal_preprocess import V2Config, build_record
from utils.lane_graph_helpers import get_compact_lane_graph, resample_polyline


DATASET_ROOT = os.environ.get("WAYMO_SCENARIO_ROOT")
_MAP_RANGE = 64.0
_NUM_POINTS_PER_LANE = 20
_ERR_VAL = -1e4
_WAYMO_OBJECT_STR = {
    scenario_pb2.Track.TYPE_UNSET: "unset",
    scenario_pb2.Track.TYPE_VEHICLE: "vehicle",
    scenario_pb2.Track.TYPE_PEDESTRIAN: "pedestrian",
    scenario_pb2.Track.TYPE_CYCLIST: "cyclist",
    scenario_pb2.Track.TYPE_OTHER: "other",
}
_SELFPLAY_OBJECT_TYPES = {"vehicle": 0, "pedestrian": 1, "cyclist": 2}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw Waymo TFRecords into SDC-centered goal (v2) scenes."
    )
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--max", type=int, default=None, help="Maximum saved scenarios for debugging.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of TFRecord files to process in parallel.")
    parser.add_argument("--waymo-dir", type=str, default=None,
                        help="Directory holding the raw scenario TFRecords "
                             "(default: $WAYMO_SCENARIO_ROOT/{training,validation}).")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--viz", action="store_true", help="Write debug GIFs next to the output split.")
    # ---- offline filtering, aligned with the original Scenario Dreamer pipeline ----
    parser.add_argument("--map-range", type=float, default=_MAP_RANGE, help="Square FOV side length in metres.")
    parser.add_argument("--max-num-agents", type=int, default=30)
    parser.add_argument("--max-num-lanes", type=int, default=100)
    parser.add_argument("--offroad-threshold", type=float, default=1.5,
                        help="Drop non-ego vehicles further than this from a lane centerline "
                             "(the original uses 1.5 m). Pass a value <= 0 to keep every agent; "
                             "a looser value here keeps the runtime threshold adjustable at the "
                             "cost of the record no longer matching the baseline agent set.")
    parser.add_argument("--traj-dtype", choices=["float32", "float16"], default="float32",
                        help="Storage dtype of the goal-block trajectories. float16 halves their "
                             "size at ~3 cm resolution near the FOV edge.")
    return parser.parse_args()


def _config_from_args(args):
    return V2Config(
        map_range=args.map_range,
        max_num_agents=args.max_num_agents,
        max_num_lanes=args.max_num_lanes,
        num_points_per_lane=_NUM_POINTS_PER_LANE,
        offroad_threshold=args.offroad_threshold,
        traj_dtype=args.traj_dtype,
    )


def _default_waymo_folder(split, waymo_dir=None):
    if waymo_dir is not None:
        return Path(waymo_dir)
    if DATASET_ROOT is None:
        raise ValueError("Pass --waymo-dir or set WAYMO_SCENARIO_ROOT to the raw scenario directory.")

    split_dir = "training" if split == "train" else "validation"
    return Path(DATASET_ROOT) / split_dir


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


def _in_map_range(xy, radius):
    return np.logical_and(np.abs(xy[..., 0]) <= radius, np.abs(xy[..., 1]) <= radius)


def _object_type_name(track):
    return _WAYMO_OBJECT_STR.get(track.object_type, "other")


def _object_type_onehot(type_name):
    onehot = np.zeros(len(_SELFPLAY_OBJECT_TYPES), dtype=np.float32)
    if type_name in _SELFPLAY_OBJECT_TYPES:
        onehot[_SELFPLAY_OBJECT_TYPES[type_name]] = 1.0
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



def _extract_map(scenario, center, rotation, radius):
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
            in_range = _in_map_range(local, radius)
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
            in_range = _in_map_range(local, radius)
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
            in_range = _in_map_range(local, radius)
            if in_range.any():
                crosswalks.append(local[in_range].astype(np.float32))

        elif feature.HasField("stop_sign"):
            point = np.asarray([[feature.stop_sign.position.x, feature.stop_sign.position.y]], dtype=np.float32)
            local = _to_local_positions(point[None, :, :], center, rotation)[0, 0]
            if _in_map_range(local, radius):
                stop_signs.append(local.astype(np.float32))

    lane_ids = set(lane_graph["lanes"].keys())
    for pair_key in ("pre_pairs", "suc_pairs", "left_pairs", "right_pairs"):
        for lane_id in list(lane_graph[pair_key].keys()):
            lane_graph[pair_key][lane_id] = [other_id for other_id in lane_graph[pair_key][lane_id] if other_id in lane_ids]

    # The lane tensors themselves (road_points / connections / edge index) are built by
    # utils.goal_preprocess.build_record, so the tfrecord path and the v1->v2
    # conversion path cannot produce different geometry from the same lane graph.
    compact_lane_graph = lane_graph
    if len(lane_graph["lanes"]) > 0:
        compact_lane_graph = get_compact_lane_graph({"lane_graph": lane_graph})

    return {
        "lane_graph": compact_lane_graph,
        "extras": {
            "road_edges": np.stack(road_edges, axis=0) if road_edges else np.zeros((0, _NUM_POINTS_PER_LANE, 2), dtype=np.float32),
            "road_edge_masks": np.stack(road_edge_masks, axis=0) if road_edge_masks else np.zeros((0, _NUM_POINTS_PER_LANE), dtype=bool),
            "crosswalks": crosswalks,
            "stop_signs": np.stack(stop_signs, axis=0) if stop_signs else np.zeros((0, 2), dtype=np.float32),
        },
    }


def _process_scenario(scenario, source_file, record_index, output_path, cfg, viz_dir=None):
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

    radius = cfg.map_range / 2.0
    center = np.asarray([sdc_state.center_x, sdc_state.center_y], dtype=np.float32)
    rotation = _rotation_from_sdc_yaw(sdc_state.heading)

    map_data = _extract_map(scenario, center, rotation, radius)
    if len(map_data["lane_graph"]["lanes"]) == 0:
        return False

    agent_ids = []
    agent_track_indices = []
    agent_types = []
    local_trajectories = []
    trajectory_valid = []
    trajectory_clip_valid = []
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
        if not _in_map_range(current_local[:2], radius):
            continue

        in_range = _in_map_range(local_states[:, :2], radius)

        agent_ids.append(int(track.id) if track.id else track_index)
        agent_track_indices.append(track_index)
        agent_types.append(_object_type_onehot(type_name))
        # [x, y, vx, vy, yaw] is everything the goal block needs; length/width are
        # constant per agent and already carried by agent_states.
        local_trajectories.append(local_states[:, :5])
        trajectory_valid.append(valid)
        trajectory_clip_valid.append(np.logical_and(valid, in_range))
        current_local_states.append(current_local)

    if len(agent_ids) == 0:
        return False

    # ego first, then the remaining agents by distance to the origin -- the ordering
    # utils.goal_preprocess.select_agents assumes.
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

    scenario_id = scenario.scenario_id or Path(source_file).stem + f"_{record_index}"
    raw_file_name = f"{Path(source_file).name}_{record_index}"

    record = build_record(
        idx=record_index,
        scenario_id=scenario_id,
        source_file=source_file,
        source_record_index=record_index,
        scene_timestep=current_t,
        sdc_track_index=sdc_idx,
        normalize={
            "center": center,
            "yaw": np.float32(sdc_state.heading),
            "rotation": np.float32(rotation),
        },
        map_id=0,
        agent_states_raw=ordered_array(current_local_states, dtype=np.float32)[:, :-1],
        agent_types=ordered_array(agent_types, dtype=np.float32),
        agent_ids=ordered_array(agent_ids, dtype=np.int64),
        agent_track_indices=ordered_array(agent_track_indices, dtype=np.int64),
        trajectory=ordered_array(local_trajectories, dtype=np.float32),
        trajectory_valid=ordered_array(trajectory_valid, dtype=bool),
        trajectory_clip_valid=ordered_array(trajectory_clip_valid, dtype=bool),
        lane_graph=map_data["lane_graph"],
        extras=map_data["extras"],
        cfg=cfg,
    )
    if record is None:
        return False

    with open(output_path / f"{raw_file_name}.pkl", "wb") as f:
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)

    if viz_dir is not None:
        _write_debug_gif(record, viz_dir / f"{raw_file_name}.gif")

    return True


def _write_debug_gif(data, gif_path):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Polygon
    except Exception as exc:
        print(f"[viz] Skipping {gif_path.name}: {exc}")
        return

    from utils.goal_runtime import compute_goals

    trajectories = np.asarray(data["trajectory"], dtype=np.float32)
    valid = data["trajectory_clip_valid"]
    # length/width are constant per agent and live in agent_states, not the trajectory
    sizes = np.asarray(data["agent_states"], dtype=np.float32)[:, 5:7]
    type_ids = np.argmax(np.asarray(data["agent_types"]), axis=1)
    ego_index = data["ego_index"]
    num_steps = trajectories.shape[1]
    radius = float(data["map_range"]) / 2.0
    goal_xy, goal_valid, _, _ = compute_goals(data)

    fig, ax = plt.subplots(figsize=(5, 5))

    def agent_color(agent_idx):
        if agent_idx == ego_index:
            return "red"
        return {0: "blue", 1: "purple", 2: "yellow"}.get(int(type_ids[agent_idx]), "black")

    def bbox_corners(state, size):
        x, y, heading = state[0], state[1], state[4]
        length, width = size
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
        for edge in data.get("road_edges", []):
            ax.plot(edge[:, 0], edge[:, 1], color="0.45", linewidth=0.7)
        ax.scatter([0.0], [0.0], s=18, color="red")
        for a in range(data["num_agents"]):
            mask = valid[a, : frame + 1]
            if mask.any():
                pts = trajectories[a, : frame + 1, :2][mask]
                ax.plot(pts[:, 0], pts[:, 1], linewidth=1.0)
            if valid[a, frame]:
                color = agent_color(a)
                corners = bbox_corners(trajectories[a, frame], sizes[a])
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
            if goal_valid[a]:
                ax.scatter(goal_xy[a, 0], goal_xy[a, 1], marker="x", s=18, color="black")

    animation = FuncAnimation(fig, draw, frames=num_steps, interval=120)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(gif_path, writer=PillowWriter(fps=8))
    plt.close(fig)


def _process_file_worker(args):
    filename, output_path, viz_dir, max_scenarios, cfg, reserved_counter, saved_counter, counter_lock = args
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
        if _process_scenario(scenario, filename.name, record_index, output_path, cfg, viz_dir):
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
    cfg = _config_from_args(args)
    raw_folder = _default_waymo_folder(args.split, args.waymo_dir)
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
            saved += _process_file_worker((filename, output_path, viz_dir, args.max, cfg, reserved_value, saved_value, lock))
            if args.max is not None and reserved_value.value >= args.max:
                print(f"Saved {saved_value.value} scenarios to {output_path}")
                return
    else:
        manager = mp.Manager()
        reserved_value = manager.Value("i", 0)
        saved_value = manager.Value("i", 0)
        lock = manager.Lock()
        worker_args = [(filename, output_path, viz_dir, args.max, cfg, reserved_value, saved_value, lock) for filename in files]
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
