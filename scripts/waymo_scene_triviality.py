"""How many Waymo scenes are trivially uncritical?

Measures the two numbers the AdvScene motivation rests on:

1. the ego barely moves over the 9.1 s scenario (static waiting behaviour), and
2. no other agent is anywhere near the ego.

Source is ``data/advscene_preprocess_waymo/{train,val}``: one record per WOMD
scenario, SDC-centred at ``scene_timestep``, carrying the full 91-step
trajectories. The off-road removal and the 30-agent cap are *not* baked into
those records -- they are applied at load time by
``utils.goal_runtime.select_agents`` -- so the on-disk agent set is everything
Waymo tracks that is valid at the scene timestep and inside the +-32 m FOV box.
Both views are reported: the unfiltered one (what "the Waymo dataset" means) and
the filtered one (what AdvScene actually generates into).

``--raw-tfrecord`` reruns the same statistics straight off the Waymo protos, to
quantify what the FOV crop costs.
"""

import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.goal_preprocess import VEHICLE_TYPE_INDEX
from utils.goal_runtime import compute_goals, select_agents

PREPROCESS_DIR = REPO_ROOT / "data" / "advscene_preprocess_waymo"
#: official WOMD scenario counts; the preprocessing dropped nothing, so any other
#: count means the split on disk is incomplete.
EXPECTED_NUM_SCENES = {"train": 486995, "val": 44097}

DISP_THRESHOLDS = (1.0, 2.0, 5.0, 10.0, 20.0)
SPEED_THRESHOLDS = (0.5, 1.0, 2.0)
#: FOV half-width is 32 m, so a threshold above it would query agents that were
#: never stored.
GAP_THRESHOLDS = (5.0, 10.0, 15.0, 20.0, 30.0)

OFFROAD_THRESHOLD = 1.5  # cfgs/ae_goal/dataset.yaml
MAX_NUM_AGENTS = 30  # cfgs/dataset/waymo_base.yaml

COLUMNS = (
    "net_disp",
    "path_len",
    "max_speed",
    "gap_init_all",
    "gap_init_veh",
    "gap_init_model",
    "gap_any_all",
    "gap_any_veh",
    "gap_any_model",
    "num_agents",
)


def _nearest_at_init(states, others):
    """Centre-to-centre distance from the ego (at the origin) to the closest of
    ``others`` at the scene timestep. ``inf`` when there is nobody."""
    if len(others) == 0:
        return np.inf
    return float(np.linalg.norm(states[others, :2], axis=1).min())


def _nearest_over_horizon(traj, valid, others):
    """Same distance, minimised over every step where both agents are valid."""
    if len(others) == 0:
        return np.inf
    both = valid[0][None, :] & valid[others]
    dist = np.linalg.norm(traj[others, :, :2] - traj[0:1, :, :2], axis=-1)
    return float(np.where(both, dist, np.inf).min())


def scene_stats(path):
    with open(path, "rb") as f:
        record = pickle.load(f)

    traj = np.asarray(record["trajectory"], dtype=np.float32)
    valid = np.asarray(record["trajectory_valid"], dtype=bool)
    states = np.asarray(record["agent_states"], dtype=np.float32)
    num_agents = len(states)

    ego_xy = traj[0, :, :2]
    steps = np.where(valid[0])[0]
    net_disp = float(np.linalg.norm(ego_xy[steps[-1]] - ego_xy[steps[0]]))
    path_len = float(np.linalg.norm(np.diff(ego_xy[steps], axis=0), axis=1).sum())
    max_speed = float(np.linalg.norm(traj[0, steps, 2:4], axis=1).max())

    is_vehicle = np.asarray(record["agent_types"])[:, VEHICLE_TYPE_INDEX].astype(bool)
    goal_xy, goal_valid, _ = compute_goals(
        traj, valid, np.asarray(record["trajectory_clip_valid"], dtype=bool)
    )
    keep = select_agents(
        record,
        goal_valid,
        goal_xy,
        offroad_threshold=OFFROAD_THRESHOLD,
        max_num_agents=MAX_NUM_AGENTS,
    )
    vehicles = np.where(is_vehicle)[0]
    others = {
        "all": np.arange(1, num_agents),
        "veh": vehicles[vehicles != 0],
        "model": keep[keep != 0],
    }

    return (
        net_disp,
        path_len,
        max_speed,
        *(_nearest_at_init(states, others[k]) for k in ("all", "veh", "model")),
        *(_nearest_over_horizon(traj, valid, others[k]) for k in ("all", "veh", "model")),
        float(num_agents),
    )


def collect(split, workers, limit):
    files = sorted(glob.glob(str(PREPROCESS_DIR / split / "*.pkl")))
    if limit is None:
        assert len(files) == EXPECTED_NUM_SCENES[split], (
            f"{split}: found {len(files)} scenes, expected {EXPECTED_NUM_SCENES[split]}"
        )
    else:
        files = files[::max(1, len(files) // limit)][:limit]

    if workers > 1:
        with Pool(workers) as pool:
            rows = pool.map(scene_stats, files, chunksize=64)
    else:
        rows = [scene_stats(p) for p in files]
    return np.asarray(rows, dtype=np.float64)


def summarize(rows, split):
    """Every number this script reports, as flat provenance-carrying entries."""
    col = {name: rows[:, i] for i, name in enumerate(COLUMNS)}
    n = len(rows)
    entries = []

    def add(metric, definition, agent_set, threshold, value):
        entries.append(
            {
                "metric": metric,
                "definition": definition,
                "agent_set": agent_set,
                "threshold_m": threshold,
                "split": split,
                "denominator": n,
                "value_pct": round(100.0 * float(value), 2),
            }
        )

    for th in DISP_THRESHOLDS:
        add("ego_static", "|ego(last valid) - ego(first valid)| over the 91-step scenario",
            "ego", th, np.mean(col["net_disp"] < th))
        add("ego_static_path", "summed step-to-step ego path length over the 91-step scenario",
            "ego", th, np.mean(col["path_len"] < th))
    for th in SPEED_THRESHOLDS:
        add("ego_slow", "max ego speed over the 91-step scenario (m/s)",
            "ego", th, np.mean(col["max_speed"] < th))

    for agent_set, key in (("all", "all"), ("vehicles", "veh"), ("model_view", "model")):
        for th in GAP_THRESHOLDS:
            add("isolated_init",
                "no other agent within the threshold of the ego at the scene timestep (centre-to-centre)",
                agent_set, th, np.mean(col[f"gap_init_{key}"] > th))
            add("isolated_horizon",
                "no other agent within the threshold of the ego at any step of the 9.1 s (centre-to-centre)",
                agent_set, th, np.mean(col[f"gap_any_{key}"] > th))

    medians = {
        "net_disp_m": round(float(np.median(col["net_disp"])), 2),
        "path_len_m": round(float(np.median(col["path_len"])), 2),
        "num_agents": float(np.median(col["num_agents"])),
        "gap_init_all_m": round(float(np.median(col["gap_init_all"])), 2),
    }
    return entries, medians


def render(entries, medians, split):
    def pick(metric, agent_set):
        return {e["threshold_m"]: e["value_pct"] for e in entries
                if e["metric"] == metric and e["agent_set"] == agent_set}

    n = entries[0]["denominator"]
    out = [f"### {split} ({n} scenes)", ""]
    out.append("| ego moves less than | net displacement | path length | max speed below | scenes |")
    out.append("| --- | --- | --- | --- | --- |")
    net, path = pick("ego_static", "ego"), pick("ego_static_path", "ego")
    slow = pick("ego_slow", "ego")
    speeds = list(SPEED_THRESHOLDS) + [None] * (len(DISP_THRESHOLDS) - len(SPEED_THRESHOLDS))
    for th, sp in zip(DISP_THRESHOLDS, speeds):
        cell = f"{sp} m/s | {slow[sp]:.2f}%" if sp is not None else " | "
        out.append(f"| {th:g} m | {net[th]:.2f}% | {path[th]:.2f}% | {cell} |")
    out += ["", "| no other agent within | at scene init | over the full 9.1 s |",
            "| --- | --- | --- |"]
    for agent_set in ("all", "vehicles", "model_view"):
        init, horizon = pick("isolated_init", agent_set), pick("isolated_horizon", agent_set)
        for th in GAP_THRESHOLDS:
            out.append(f"| {th:g} m ({agent_set}) | {init[th]:.2f}% | {horizon[th]:.2f}% |")
    out += ["", f"medians: {json.dumps(medians)}", ""]
    return "\n".join(out)


def run_raw(tfrecord, limit):
    """Same statistics from the Waymo protos, with no FOV crop and no validity
    requirement, to measure what the preprocessing costs."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    net_disp, gap_all, gap_veh = [], [], []
    for i, raw in enumerate(tf.data.TFRecordDataset(tfrecord, compression_type="")):
        if i >= limit:
            break
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(raw.numpy())
        t, sdc = scenario.current_time_index, scenario.sdc_track_index

        ego = scenario.tracks[sdc]
        xy = np.array([[s.center_x, s.center_y] for s in ego.states], dtype=np.float64)
        steps = np.where([s.valid for s in ego.states])[0]
        net_disp.append(np.linalg.norm(xy[steps[-1]] - xy[steps[0]]))

        dists, veh_dists = [], []
        for k, track in enumerate(scenario.tracks):
            if k == sdc or not track.states[t].valid:
                continue
            state = track.states[t]
            d = float(np.hypot(state.center_x - xy[t][0], state.center_y - xy[t][1]))
            dists.append(d)
            if track.object_type == scenario_pb2.Track.TYPE_VEHICLE:
                veh_dists.append(d)
        gap_all.append(min(dists) if dists else np.inf)
        gap_veh.append(min(veh_dists) if veh_dists else np.inf)

    net_disp, gap_all, gap_veh = map(np.asarray, (net_disp, gap_all, gap_veh))
    print(f"\nraw tfrecord: {tfrecord} ({len(net_disp)} scenarios, no FOV crop)")
    for th in DISP_THRESHOLDS:
        print(f"  ego net displacement < {th:>4g} m : {100 * np.mean(net_disp < th):5.2f}%")
    for th in GAP_THRESHOLDS:
        print(f"  no agent within {th:>4g} m at init : {100 * np.mean(gap_all > th):5.2f}%"
              f"   (vehicles only: {100 * np.mean(gap_veh > th):5.2f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evenly spaced subsample per split, for a quick look.")
    parser.add_argument("--out", type=str, default=None,
                        help="Directory for table.md and PROVENANCE.json.")
    parser.add_argument("--raw-tfrecord", type=str, default=None,
                        help="Run the cross-check on raw Waymo protos instead.")
    args = parser.parse_args()

    if args.raw_tfrecord is not None:
        run_raw(args.raw_tfrecord, args.limit or 400)
        return

    splits = ["train", "val"] if args.split == "both" else [args.split]
    rows = {s: collect(s, args.workers, args.limit) for s in splits}
    if len(splits) > 1:
        rows["train+val"] = np.concatenate([rows["train"], rows["val"]], axis=0)

    entries, tables = [], []
    for split, data in rows.items():
        split_entries, medians = summarize(data, split)
        entries += split_entries
        tables.append(render(split_entries, medians, split))
    table = "\n".join(tables)
    print(table)

    if args.out is None:
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table.md").write_text(table + "\n")
    (out_dir / "PROVENANCE.json").write_text(json.dumps({
        "source": str(PREPROCESS_DIR),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          cwd=REPO_ROOT, text=True).strip(),
        "scene_timestep": "WOMD current_time_index (10), SDC-centred",
        "fov_box_half_width_m": 32.0,
        "distance": "centre-to-centre, not bumper-to-bumper",
        "agent_sets": {
            "all": "every vehicle/pedestrian/cyclist valid at the scene timestep inside the FOV box",
            "vehicles": "the vehicle subset of the above",
            "model_view": f"after utils.goal_runtime.select_agents "
                          f"(off-road threshold {OFFROAD_THRESHOLD} m, cap {MAX_NUM_AGENTS}) "
                          f"-- the set AdvScene generates into",
        },
        "caveat_isolated_horizon": "over-states emptiness: an agent that enters the FOV box "
                                   "only after the scene timestep is absent from the record. "
                                   "The isolated_init numbers carry no such bias.",
        "entries": entries,
    }, indent=2) + "\n")
    print(f"\nwrote {out_dir}/table.md and {out_dir}/PROVENANCE.json")


if __name__ == "__main__":
    main()
