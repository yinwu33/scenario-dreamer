#!/usr/bin/env python3
"""Create side-by-side GIFs of dataset trajectories vs frozen planner rollout.

This bypasses AE/LDM/DDPO entirely. For each preprocessed Waymo goal pkl, it uses:
  * dataset original local trajectories from ``local_trajectory`` / ``clipped_valid``;
  * original current agent states, original clipped final goals, and original map
    as input to the in-repo PufferDrive planner rollout.

The resulting GIF lets you answer whether bad goals/rollouts are already present
in the raw goal dataset, or introduced by the generator/DDPO path.

Example:
    python3 data_analysis/compare_original_vs_planner_rollout.py \
        --preprocess-dir data/scene_goal_preprocess_waymo \
        --split val \
        --num-scenes 4
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def default_preprocess_dir() -> str:
    """Return the default Waymo goal preprocess directory.

    If ``DATASET_ROOT`` is set, the preprocess directory is resolved relative
    to it; otherwise the repository-local data path is used.
    """
    dataset_root = os.environ.get("DATASET_ROOT")
    if dataset_root:
        return os.path.join(dataset_root, "scene_goal_preprocess_waymo")
    return "data/scene_goal_preprocess_waymo"


def default_device() -> str:
    """Return ``cuda`` when PyTorch can see a GPU, otherwise ``cpu``."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def raw_agent_to_model_state(agent_states: np.ndarray) -> np.ndarray:
    """Convert raw agent state columns to the planner model state layout.

    Input rows are interpreted as ``[x, y, vx, vy, yaw, length, width]`` and
    output rows are ``[x, y, speed, cos(yaw), sin(yaw), length, width]``.
    """
    out = np.zeros_like(agent_states[:, :7], dtype=np.float32)
    out[:, :2] = agent_states[:, :2]
    out[:, 2] = np.sqrt(agent_states[:, 2] ** 2 + agent_states[:, 3] ** 2)
    out[:, 3] = np.cos(agent_states[:, 4])
    out[:, 4] = np.sin(agent_states[:, 4])
    out[:, 5:7] = agent_states[:, 5:7]
    return out


def select_training_agents(data: dict, max_num_agents: int, require_ego_valid_goal: bool) -> np.ndarray:
    """Select agents with valid clipped final goals for planner evaluation.

    The selection is capped to ``max_num_agents`` by keeping the valid agents
    nearest the current local origin. If requested, the local ego agent must
    have a valid clipped final goal.
    """
    valid_goal = np.asarray(data["clipped_final_valid"], dtype=bool)
    if require_ego_valid_goal and (len(valid_goal) == 0 or not bool(valid_goal[0])):
        # idx 0 is ego
        raise ValueError("ego/local agent 0 has no valid clipped final goal")
    idx = np.nonzero(valid_goal)[0]
    if len(idx) == 0:
        raise ValueError("scene has no valid clipped final goals")
    if len(idx) > max_num_agents:
        current = np.asarray(data["agent_states"], dtype=np.float32)[idx]
        keep_local = np.argsort(np.linalg.norm(current[:, :2], axis=-1))[:max_num_agents]
        idx = idx[keep_local]
    return idx.astype(np.int64)


def build_generated_scene(data: dict, selected: np.ndarray, device: str) -> tuple[GeneratedScenes, np.ndarray, np.ndarray]:
    """Build a single ``GeneratedScenes`` planner input from preprocessed data.

    Returns the scene object plus numpy copies of the model-layout agent states
    and integer agent types used by rendering.
    """
    import torch

    from ddpo.interfaces import GeneratedScenes

    current_raw = np.asarray(data["agent_states"], dtype=np.float32)[selected]
    goals = np.asarray(data["clipped_final_states"], dtype=np.float32)[selected, :2]
    agent_states = np.concatenate([raw_agent_to_model_state(current_raw), goals], axis=-1).astype(np.float32)
    agent_types = np.asarray(data["agent_types"], dtype=np.float32)[selected].argmax(axis=-1).astype(np.int64)

    n_lanes = int(data.get("num_lanes", len(data["road_points"])))
    lanes = np.asarray(data["road_points"], dtype=np.float32)[:n_lanes]
    scenes = GeneratedScenes(
        agent_states=torch.as_tensor(agent_states, device=device),
        agent_types=torch.as_tensor(agent_types, device=device),
        agent_scene_idx=torch.zeros(len(agent_states), dtype=torch.long, device=device),
        lane_polylines=torch.as_tensor(lanes, device=device),
        num_scenes=1,
        meta={"lane_scene_idx": torch.zeros(len(lanes), dtype=torch.long, device=device)},
    )
    return scenes, agent_states, agent_types


def _fill_invalid(values: np.ndarray, valid: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Forward-fill invalid trajectory samples for stable render bounds.

    Invalid frames are still hidden later via the respawn mask; this function
    only prevents missing samples from distorting plot limits.
    """
    out = values.copy()
    n, t, d = out.shape
    for a in range(n):
        last = fallback[a].astype(np.float32)
        first_valid = np.nonzero(valid[a])[0]
        if len(first_valid):
            last = out[a, first_valid[0]].astype(np.float32)
        for k in range(t):
            if valid[a, k]:
                last = out[a, k].astype(np.float32)
            else:
                out[a, k] = last
    return out


def build_original_traj(data: dict, selected: np.ndarray) -> dict:
    """Build a renderer-compatible trajectory dict from dataset trajectories.

    The returned arrays use the frame-major layout expected by
    ``render_rollout_frames`` and mark invalid dataset frames as respawned so
    they are omitted from the visualization.
    """
    traj = np.asarray(data["local_trajectory"], dtype=np.float32)[selected]
    valid = np.asarray(data.get("clipped_valid", data["trajectory_valid"]), dtype=bool)[selected]
    current = np.asarray(data["agent_states"], dtype=np.float32)[selected]

    x = traj[:, :, 0]
    y = traj[:, :, 1]
    heading = traj[:, :, 4]
    filled = _fill_invalid(
        np.stack([x, y, heading], axis=-1),
        valid,
        current[:, [0, 1, 4]],
    )
    out = {
        "x": filled[:, :, 0].T,
        "y": filled[:, :, 1].T,
        "heading": filled[:, :, 2].T,
        "length": current[:, 5].astype(np.float32),
        "width": current[:, 6].astype(np.float32),
        # Reuse respawn masking to hide invalid dataset frames.
        "respawn": (~valid).T,
        "done": np.zeros(traj.shape[1], dtype=bool),
    }
    return out


def original_parking_mask(data: dict, selected: np.ndarray, parking_threshold: float) -> np.ndarray:
    """Return a boolean mask for agents whose clipped goal is near the start.

    Agents are considered parking or stationary when their current position and
    clipped final goal are closer than ``parking_threshold`` meters.
    """
    current = np.asarray(data["agent_states"], dtype=np.float32)[selected, :2]
    goals = np.asarray(data["clipped_final_states"], dtype=np.float32)[selected, :2]
    return np.linalg.norm(goals - current, axis=-1) < parking_threshold


def parking_agent_colors(is_parking: np.ndarray) -> list[str | None]:
    """Return render colors that mark parking agents in black."""
    return ["black" if bool(v) else None for v in is_parking]


def pad_frames(frames: np.ndarray, target_len: int) -> np.ndarray:
    """Pad a frame stack by repeating the last frame to ``target_len``.

    Freezing on the last frame lets both sides of the comparison advance one
    simulation step per GIF frame even when their original lengths differ.
    """
    if len(frames) >= target_len:
        return frames
    if len(frames) == 0:
        raise ValueError("empty frame array")
    pad = np.repeat(frames[-1:], target_len - len(frames), axis=0)
    return np.concatenate([frames, pad])


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Concatenate two frame stacks horizontally after length and height sync."""
    t = max(len(left), len(right))
    left = pad_frames(left, t)
    right = pad_frames(right, t)
    h = min(left.shape[1], right.shape[1])
    w_left, w_right = left.shape[2], right.shape[2]
    left = left[:, :h, :w_left]
    right = right[:, :h, :w_right]
    return np.concatenate([left, right], axis=2)


def safe_name(path: str) -> str:
    """Return a filesystem-safe output stem derived from ``path``."""
    stem = Path(path).stem
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)


def choose_files(args) -> list[str]:
    """Return input pickle files requested by CLI arguments.

    Explicit ``--files`` are used as-is. Otherwise files are loaded from
    ``<preprocess_dir>/<split>/*.pkl`` and optionally sampled with ``--seed``.
    """
    if args.files:
        return args.files
    pattern = os.path.join(args.preprocess_dir, args.split, "*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no pkl files found: {pattern}")
    if args.num_scenes and args.num_scenes < len(files):
        random.seed(args.seed)
        files = random.sample(files, args.num_scenes)
    return files


def main() -> None:
    """Parse CLI arguments and write original-vs-planner comparison GIFs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess-dir", default=default_preprocess_dir())
    ap.add_argument("--split", default="val")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--num-scenes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="data_analysis/rollout_compare")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--sim-steps", type=int, default=91)
    ap.add_argument("--max-num-agents", type=int, default=32)
    ap.add_argument("--goal-offlane-threshold", type=float, default=1.0)
    ap.add_argument("--goal-offlane-penalty", type=float, default=5.0)
    ap.add_argument("--parking-threshold", type=float, default=2.0)
    ap.add_argument("--parking-mismatch-penalty", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=91)
    ap.add_argument("--require-ego-valid-goal", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    from ddpo.reward import PufferDriveReward
    from ddpo.viz import render_rollout_frames, save_gif

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = choose_files(args)

    reward = PufferDriveReward(
        sim_steps=args.sim_steps,
        deterministic=True,
        goal_offlane_threshold=args.goal_offlane_threshold,
        goal_offlane_penalty=args.goal_offlane_penalty,
        parking_mismatch_penalty=args.parking_mismatch_penalty,
        seed=args.seed,
    )

    wrote = 0
    for path in files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        try:
            selected_idx = select_training_agents(data, args.max_num_agents, args.require_ego_valid_goal)
        except ValueError as exc:
            print(f"[skip] {path}: {exc}")
            continue

        scenes, agent_states, agent_types = build_generated_scene(data, selected_idx, args.device)
        metrics = reward.evaluate(scenes, record_trajectories=True)
        planner_traj = metrics["trajectories"][0]
        original_traj = build_original_traj(data, selected_idx)
        is_parking = original_parking_mask(data, selected_idx, args.parking_threshold)
        agent_colors = parking_agent_colors(is_parking)
        lanes = np.asarray(data["road_points"], dtype=np.float32)[: int(data.get("num_lanes", len(data["road_points"])))]

        title_base = data.get("scenario_id", Path(path).stem)
        original_frames = render_rollout_frames(
            original_traj,
            lanes,
            agent_states=agent_states,
            agent_types=agent_types,
            agent_colors=agent_colors,
            reward=None,
            title=f"dataset original {title_base} parking={int(is_parking.sum())}/{len(is_parking)}",
            max_frames=args.max_frames,
        )
        planner_frames = render_rollout_frames(
            planner_traj,
            lanes,
            agent_states=agent_states,
            agent_types=agent_types,
            agent_colors=agent_colors,
            reward=metrics["reward"][0],
            ego_collision=metrics["ego_collision"][0] > 0,
            ego_offroad=metrics["ego_offroad"][0] > 0,
            init_invalid=metrics["init_invalid"][0] > 0,
            ego_min_ttc=metrics["ego_min_ttc"][0],
            goal_offlane_frac=metrics["goal_offlane_frac"][0],
            parking_mismatch_frac=metrics["parking_mismatch_frac"][0],
            title=f"planner rollout {title_base} parking=black",
            max_frames=args.max_frames,
        )

        frames = side_by_side(original_frames, planner_frames)
        out_path = out_dir / f"{safe_name(path)}_original_vs_planner.gif"
        save_gif(frames, str(out_path), fps=args.fps)
        wrote += 1
        print(
            f"[wrote] {out_path}  reward={metrics['reward'][0]:+.3f} "
            f"collision={int(metrics['ego_collision'][0] > 0)} "
            f"goal_offlane={metrics['goal_offlane_frac'][0]:.3f} "
            f"parking={int(is_parking.sum())}/{len(is_parking)} "
            f"agents={len(selected_idx)}"
        )

    print(f"wrote {wrote} comparison gif(s) to {out_dir}")


if __name__ == "__main__":
    main()
