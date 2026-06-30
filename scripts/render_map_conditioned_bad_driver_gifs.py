"""Render side-by-side bad_driver rollout GIFs for map-conditioned artifacts.

Reads the artifacts produced by ``run_map_conditioned_bad_driver_table.py`` and
renders one comparison GIF per scene:

    original | base_diffusion_full | base_diffusion_one | ddpo_diffusion

The rendering path intentionally reuses ``ddpo.viz.render_rollout_frames`` and
``ddpo.viz.save_gif`` so the output matches DDPO eval media.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SCRATCH_ROOT", "data")
os.environ.setdefault("DATASET_ROOT", "data")
os.environ.setdefault("PROJECT_ROOT", str(ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from critical_scene.benchmark import _build_reward, _compose_cfg
from critical_scene.schema import assert_same_map
from ddpo.interfaces import GeneratedScenes
from ddpo.viz import CONTROL_COLOR, render_rollout_frames, save_gif

SOURCES = ("original", "base_diffusion_full", "base_diffusion_one", "ddpo_diffusion")
COMPONENT_KEYS = (
    "criticality",
    "r_ttc",
    "r_approach",
    "r_risk",
    "r_collision",
    "constraint",
    "c_lane",
    "c_parking",
    "c_trivial",
    "ego_adv_init_dist",
    "ego_adv_min_dist_warmup",
    "ego_collision_time",
)


def _as_cpu_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value).detach().cpu()


def _slice_scenes(scenes: GeneratedScenes, start: int, end: int) -> GeneratedScenes:
    a_idx = _as_cpu_tensor(scenes.agent_scene_idx).long()
    l_idx = _as_cpu_tensor(scenes.meta["lane_scene_idx"]).long()
    a_sel = (a_idx >= start) & (a_idx < end)
    l_sel = (l_idx >= start) & (l_idx < end)

    meta = {"lane_scene_idx": l_idx[l_sel] - start}
    for key in ("gen_agent_mask", "gt_parking_mask"):
        if key in scenes.meta:
            meta[key] = _as_cpu_tensor(scenes.meta[key]).bool()[a_sel]

    return GeneratedScenes(
        agent_states=_as_cpu_tensor(scenes.agent_states)[a_sel],
        agent_types=_as_cpu_tensor(scenes.agent_types).long()[a_sel],
        agent_scene_idx=a_idx[a_sel] - start,
        lane_polylines=_as_cpu_tensor(scenes.lane_polylines)[l_sel],
        num_scenes=end - start,
        meta=meta,
    )


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _pad_frames(frames: np.ndarray, target_len: int) -> np.ndarray:
    if len(frames) >= target_len:
        return frames
    if len(frames) == 0:
        raise ValueError("empty frame stack")
    pad = np.repeat(frames[-1:], target_len - len(frames), axis=0)
    return np.concatenate([frames, pad], axis=0)


def _concat_frames(frame_stacks: list[np.ndarray]) -> np.ndarray:
    target_t = max(len(frames) for frames in frame_stacks)
    padded = [_pad_frames(frames, target_t) for frames in frame_stacks]
    height = min(frames.shape[1] for frames in padded)
    return np.concatenate([frames[:, :height] for frames in padded], axis=2)


def _scene_kwargs(
    *,
    source: str,
    split: str,
    scene_id: int,
    local_idx: int,
    scenes: GeneratedScenes,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    agent_scene_idx = _to_numpy(scenes.agent_scene_idx).astype(np.int64)
    states = _to_numpy(scenes.agent_states)
    types = _to_numpy(scenes.agent_types)
    a_sel = agent_scene_idx == local_idx

    agent_colors = None
    gen_agent_mask = scenes.meta.get("gen_agent_mask")
    if gen_agent_mask is not None:
        gen_agent_s = _to_numpy(gen_agent_mask).astype(bool)[a_sel]
        agent_colors = [
            CONTROL_COLOR if (agent_i > 0 and bool(is_gen_agent)) else None
            for agent_i, is_gen_agent in enumerate(gen_agent_s)
        ]

    components = {
        key: metrics[key][local_idx]
        for key in COMPONENT_KEYS
        if key in metrics
    }
    return {
        "agent_states": states[a_sel],
        "agent_types": types[a_sel],
        "agent_colors": agent_colors,
        "reward": metrics["reward"][local_idx],
        "ego_collision": metrics["ego_collision"][local_idx] > 0,
        "ego_offroad": metrics["ego_offroad"][local_idx] > 0,
        "init_invalid": metrics["init_invalid"][local_idx] > 0,
        "ego_min_ttc": metrics["ego_min_ttc"][local_idx],
        "goal_offlane_frac": metrics["goal_offlane_frac"][local_idx],
        "parking_mismatch_frac": metrics["parking_mismatch_frac"][local_idx],
        "components": components,
        "title": f"{split} {source} scene={scene_id}",
    }


def _metric_row(split: str, source: str, scene_id: int, metrics: dict[str, Any], local_idx: int) -> dict[str, Any]:
    row = {"split": split, "source": source, "scene_id": scene_id}
    for key in (
        "reward",
        "ego_collision",
        "ego_offroad",
        "init_invalid",
        "reached_goal",
        "ego_min_ttc",
        "goal_offlane_frac",
        "parking_mismatch_frac",
        "ego_adv_min_dist",
        "gen_agent_is_parked",
    ):
        if key in metrics:
            value = float(metrics[key][local_idx])
            row[key] = value if np.isfinite(value) else "nan"
    return row


def _load_split_artifacts(out_dir: Path, split: str, sources: tuple[str, ...], num_scenes: int):
    artifacts = {}
    for source in sources:
        path = out_dir / "artifacts" / split / f"{source}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        artifact = torch.load(path, map_location="cpu")
        artifact["artifact_path"] = str(path)
        artifacts[source] = artifact

    reference = artifacts[sources[0]]["metadata"]
    for source in sources[1:]:
        assert_same_map(reference, artifacts[source]["metadata"])
    if len(reference["scene_ids"]) < num_scenes:
        raise ValueError(f"{split} has only {len(reference['scene_ids'])} scenes, requested {num_scenes}")
    return artifacts


def _render_split(
    *,
    out_dir: Path,
    gif_dir: Path,
    split: str,
    sources: tuple[str, ...],
    num_scenes: int,
    reward,
    max_frames: int,
    fps: int,
    force: bool,
) -> list[dict[str, Any]]:
    artifacts = _load_split_artifacts(out_dir, split, sources, num_scenes)
    scene_ids = [int(v) for v in artifacts[sources[0]]["metadata"]["scene_ids"][:num_scenes]]

    split_scenes: dict[str, GeneratedScenes] = {}
    split_metrics: dict[str, dict[str, Any]] = {}
    for source in sources:
        scenes = _slice_scenes(artifacts[source]["scenes"], 0, num_scenes)
        split_scenes[source] = scenes
        print(f"[rollout] split={split} source={source} scenes=0..{num_scenes - 1}", flush=True)
        split_metrics[source] = reward.evaluate(scenes, record_trajectories=True)

    rows: list[dict[str, Any]] = []
    split_out = gif_dir / split
    split_out.mkdir(parents=True, exist_ok=True)
    for local_idx, scene_id in enumerate(scene_ids):
        out_path = split_out / f"scene_{scene_id:04d}_compare.gif"
        if out_path.exists() and not force:
            print(f"[gif] skip existing {out_path}", flush=True)
        else:
            frame_stacks = []
            for source in sources:
                scenes = split_scenes[source]
                metrics = split_metrics[source]
                lanes = _to_numpy(scenes.lane_polylines)
                lane_scene_idx = _to_numpy(scenes.meta["lane_scene_idx"]).astype(np.int64)
                frames = render_rollout_frames(
                    metrics["trajectories"][local_idx],
                    lanes[lane_scene_idx == local_idx],
                    max_frames=max_frames,
                    **_scene_kwargs(
                        source=source,
                        split=split,
                        scene_id=scene_id,
                        local_idx=local_idx,
                        scenes=scenes,
                        metrics=metrics,
                    ),
                )
                frame_stacks.append(frames)

            save_gif(_concat_frames(frame_stacks), str(out_path), fps=fps)
            print(f"[gif] wrote {out_path}", flush=True)

        for source in sources:
            row = _metric_row(split, source, scene_id, split_metrics[source], local_idx)
            row["gif"] = str(out_path)
            rows.append(row)
    return rows


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_critical_scene_map_conditioned_dm_goal")
    ap.add_argument("--out-dir", default="data/critical_scene/map_conditioned_bad_driver_2000")
    ap.add_argument("--gif-dir", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    ap.add_argument("--num-scenes", type=int, default=10)
    ap.add_argument("--planner-device", default="cpu")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=90)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    gif_dir = Path(args.gif_dir) if args.gif_dir else out_dir / "gifs"
    sources = tuple(args.sources)

    cfg_root = _compose_cfg(args.config_name, "bad_driver", [])
    OmegaConf.set_struct(cfg_root.ddpo.planner, False)
    cfg_root.ddpo.planner.device = args.planner_device
    OmegaConf.set_struct(cfg_root.ddpo.planner, True)
    reward = _build_reward(cfg_root)

    rows: list[dict[str, Any]] = []
    for split in args.splits:
        rows.extend(
            _render_split(
                out_dir=out_dir,
                gif_dir=gif_dir,
                split=split,
                sources=sources,
                num_scenes=int(args.num_scenes),
                reward=reward,
                max_frames=int(args.max_frames),
                fps=int(args.fps),
                force=bool(args.force),
            )
        )

    manifest = gif_dir / "manifest.csv"
    _write_manifest(manifest, rows)
    print(f"[manifest] wrote {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
