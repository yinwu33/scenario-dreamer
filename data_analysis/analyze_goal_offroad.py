#!/usr/bin/env python3
"""Analyze off-lane goal frequency in the preprocessed Waymo goal dataset.

The statistic mirrors the DDPO reward's goal validity check:
  * use agents with ``clipped_final_valid`` goals;
  * apply the same closest-to-origin ``max_num_agents`` truncation as
    ``WaymoDatasetDMGoal``;
  * exclude parking/static agents whose goal is within ``parking_threshold`` of
    their spawn;
  * mark a moving goal as off-lane when its min distance to any lane centerline
    segment is greater than ``goal_offlane_threshold``.

Example:
    python3 data_analysis/analyze_goal_offroad.py \
        --preprocess-dir "$DATASET_ROOT/scene_goal_preprocess_waymo" \
        --split train \
        --goal-offlane-threshold 1.0
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import random
from dataclasses import dataclass

import numpy as np


@dataclass
class SceneStats:
    path: str
    valid_goals: int
    parking_goals: int
    moving_goals: int
    finite_moving_goals: int
    offroad_goals: int
    no_lane_distance_goals: int
    mean_moving_dist: float
    p95_moving_dist: float
    max_moving_dist: float

    @property
    def offroad_frac(self) -> float:
        return self.offroad_goals / self.moving_goals if self.moving_goals else 0.0

    @property
    def finite_offroad_frac(self) -> float:
        if self.finite_moving_goals == 0:
            return 0.0
        return self.offroad_goals / self.finite_moving_goals


def default_preprocess_dir() -> str:
    dataset_root = os.environ.get("DATASET_ROOT")
    if dataset_root:
        return os.path.join(dataset_root, "scene_goal_preprocess_waymo")
    return "data/scene_goal_preprocess_waymo"


def min_dist_to_lane_centerline(points: np.ndarray, lanes: np.ndarray) -> np.ndarray:
    """Min Euclidean distance from each point [M,2] to lane polyline segments."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float32))
    lanes = np.asarray(lanes, dtype=np.float32)

    starts = []
    ends = []
    for poly in lanes:
        valid = np.isfinite(poly).all(axis=1)
        p = poly[valid, :2]
        if p.shape[0] < 2:
            continue
        starts.append(p[:-1])
        ends.append(p[1:])

    if not starts:
        return np.full(points.shape[0], np.inf, dtype=np.float32)

    a = np.concatenate(starts, axis=0)
    b = np.concatenate(ends, axis=0)
    ab = b - a
    denom = np.maximum((ab * ab).sum(axis=-1), 1e-9)
    ap = points[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(axis=-1) / denom[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    d = points[:, None, :] - proj
    return np.sqrt((d * d).sum(axis=-1)).min(axis=1)


def analyze_scene(
    path: str,
    *,
    goal_offlane_threshold: float,
    parking_threshold: float,
    max_num_agents: int,
) -> tuple[SceneStats | None, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    valid_goal_mask = np.asarray(data["clipped_final_valid"], dtype=bool)
    if valid_goal_mask.sum() == 0:
        return None, np.zeros(0, dtype=np.float32)

    agent_states = np.asarray(data["agent_states"], dtype=np.float32)[valid_goal_mask]
    goals = np.asarray(data["clipped_final_states"], dtype=np.float32)[valid_goal_mask, :2]

    if len(agent_states) > max_num_agents:
        # Match WaymoDatasetDMGoal.get_data before normalization.
        dist_to_origin = np.linalg.norm(agent_states[:, :2], axis=-1)
        keep = np.argsort(dist_to_origin)[:max_num_agents]
        agent_states = agent_states[keep]
        goals = goals[keep]

    valid_goals = int(len(agent_states))
    parking_dist = np.linalg.norm(goals - agent_states[:, :2], axis=-1)
    is_parking = parking_dist < parking_threshold
    moving = ~is_parking
    moving_goals = int(moving.sum())

    if moving_goals == 0:
        stats = SceneStats(
            path=path,
            valid_goals=valid_goals,
            parking_goals=int(is_parking.sum()),
            moving_goals=0,
            finite_moving_goals=0,
            offroad_goals=0,
            no_lane_distance_goals=0,
            mean_moving_dist=float("nan"),
            p95_moving_dist=float("nan"),
            max_moving_dist=float("nan"),
        )
        return stats, np.zeros(0, dtype=np.float32)

    road_points = np.asarray(data["road_points"], dtype=np.float32)
    if "num_lanes" in data:
        road_points = road_points[: int(data["num_lanes"])]
    moving_dist = min_dist_to_lane_centerline(goals[moving], road_points)
    finite = np.isfinite(moving_dist)
    # Reward-compatible definition: non-finite distances do not count as offroad,
    # but remain visible through no_lane_distance_goals below.
    offroad = finite & (moving_dist > goal_offlane_threshold)
    finite_dist = moving_dist[finite]

    stats = SceneStats(
        path=path,
        valid_goals=valid_goals,
        parking_goals=int(is_parking.sum()),
        moving_goals=moving_goals,
        finite_moving_goals=int(finite.sum()),
        offroad_goals=int(offroad.sum()),
        no_lane_distance_goals=int((~finite).sum()),
        mean_moving_dist=float(finite_dist.mean()) if finite_dist.size else float("nan"),
        p95_moving_dist=float(np.percentile(finite_dist, 95)) if finite_dist.size else float("nan"),
        max_moving_dist=float(finite_dist.max()) if finite_dist.size else float("nan"),
    )
    return stats, finite_dist.astype(np.float32)


def pct(numer: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{100.0 * numer / denom:.2f}%"


def write_scene_csv(path: str, rows: list[SceneStats]) -> None:
    fieldnames = [
        "path",
        "valid_goals",
        "parking_goals",
        "moving_goals",
        "finite_moving_goals",
        "offroad_goals",
        "no_lane_distance_goals",
        "offroad_frac",
        "finite_offroad_frac",
        "mean_moving_dist",
        "p95_moving_dist",
        "max_moving_dist",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "path": r.path,
                "valid_goals": r.valid_goals,
                "parking_goals": r.parking_goals,
                "moving_goals": r.moving_goals,
                "finite_moving_goals": r.finite_moving_goals,
                "offroad_goals": r.offroad_goals,
                "no_lane_distance_goals": r.no_lane_distance_goals,
                "offroad_frac": r.offroad_frac,
                "finite_offroad_frac": r.finite_offroad_frac,
                "mean_moving_dist": r.mean_moving_dist,
                "p95_moving_dist": r.p95_moving_dist,
                "max_moving_dist": r.max_moving_dist,
            })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess-dir", default=default_preprocess_dir())
    ap.add_argument("--split", default="train", help="train, val, or test")
    ap.add_argument("--goal-offlane-threshold", type=float, default=1.0)
    ap.add_argument("--parking-threshold", type=float, default=2.0)
    ap.add_argument("--max-num-agents", type=int, default=30)
    ap.add_argument("--max-files", type=int, default=0, help="0 means all files")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--save-scene-csv", default=None)
    args = ap.parse_args()

    pattern = os.path.join(args.preprocess_dir, args.split, "*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no pkl files found: {pattern}")
    if args.max_files and args.max_files < len(files):
        random.seed(args.seed)
        files = random.sample(files, args.max_files)

    scene_rows: list[SceneStats] = []
    all_finite_distances = []
    skipped = 0
    failed = 0

    for i, path in enumerate(files, start=1):
        try:
            stats, finite_dist = analyze_scene(
                path,
                goal_offlane_threshold=args.goal_offlane_threshold,
                parking_threshold=args.parking_threshold,
                max_num_agents=args.max_num_agents,
            )
        except Exception as exc:
            failed += 1
            print(f"[warn] failed {path}: {exc}")
            continue
        if stats is None:
            skipped += 1
            continue
        scene_rows.append(stats)
        if finite_dist.size:
            all_finite_distances.append(finite_dist)
        if i % 5000 == 0:
            print(f"processed {i}/{len(files)} files...")

    if not scene_rows:
        raise RuntimeError("no scenes with valid goals were analyzed")

    total_valid = sum(r.valid_goals for r in scene_rows)
    total_parking = sum(r.parking_goals for r in scene_rows)
    total_moving = sum(r.moving_goals for r in scene_rows)
    total_finite_moving = sum(r.finite_moving_goals for r in scene_rows)
    total_offroad = sum(r.offroad_goals for r in scene_rows)
    total_no_lane_distance = sum(r.no_lane_distance_goals for r in scene_rows)
    scenes_with_moving = sum(r.moving_goals > 0 for r in scene_rows)
    scenes_with_offroad = sum(r.offroad_goals > 0 for r in scene_rows)
    scenes_with_no_lane_distance = sum(r.no_lane_distance_goals > 0 for r in scene_rows)

    if all_finite_distances:
        dist = np.concatenate(all_finite_distances)
    else:
        dist = np.zeros(0, dtype=np.float32)

    print("\nGoal Off-Lane Dataset Analysis")
    print(f"preprocess_dir: {args.preprocess_dir}")
    print(f"split: {args.split}")
    print(f"files_seen: {len(files)}")
    print(f"scenes_analyzed: {len(scene_rows)}  skipped_no_valid_goal: {skipped}  failed: {failed}")
    print(f"goal_offlane_threshold_m: {args.goal_offlane_threshold}")
    print(f"parking_threshold_m: {args.parking_threshold}")
    print(f"max_num_agents: {args.max_num_agents}")
    print()
    print(f"valid_goals_total: {total_valid}")
    print(f"parking_goals_excluded: {total_parking} ({pct(total_parking, total_valid)} of valid)")
    print(f"moving_goals_denominator: {total_moving}")
    print()
    print(
        "offroad_moving_goals_reward_style: "
        f"{total_offroad}/{total_moving} ({pct(total_offroad, total_moving)})"
    )
    print(
        "offroad_moving_goals_finite_distance_only: "
        f"{total_offroad}/{total_finite_moving} ({pct(total_offroad, total_finite_moving)})"
    )
    print(
        "moving_goals_without_lane_distance: "
        f"{total_no_lane_distance}/{total_moving} ({pct(total_no_lane_distance, total_moving)})"
    )
    print(
        "scenes_with_any_offroad_moving_goal: "
        f"{scenes_with_offroad}/{scenes_with_moving} ({pct(scenes_with_offroad, scenes_with_moving)})"
    )
    print(
        "scenes_with_any_missing_lane_distance: "
        f"{scenes_with_no_lane_distance}/{len(scene_rows)} ({pct(scenes_with_no_lane_distance, len(scene_rows))})"
    )

    if dist.size:
        print()
        print("moving goal -> nearest lane centerline distance, parking excluded, finite only:")
        for q in (50, 75, 90, 95, 99):
            print(f"  p{q}: {np.percentile(dist, q):.3f} m")
        print(f"  mean: {dist.mean():.3f} m")
        print(f"  max: {dist.max():.3f} m")

    top = sorted(scene_rows, key=lambda r: (r.offroad_frac, r.offroad_goals), reverse=True)[: args.top_k]
    if top:
        print()
        print(f"Top {len(top)} scenes by offroad fraction:")
        for r in top:
            rel = os.path.relpath(r.path, args.preprocess_dir)
            print(
                f"  {rel}  offroad={r.offroad_goals}/{r.moving_goals} "
                f"({100.0 * r.offroad_frac:.1f}%)  parking={r.parking_goals}/{r.valid_goals} "
                f"p95_dist={r.p95_moving_dist:.2f}m"
            )

    if args.save_scene_csv:
        write_scene_csv(args.save_scene_csv, scene_rows)
        print(f"\nwrote scene csv: {args.save_scene_csv}")


if __name__ == "__main__":
    main()
