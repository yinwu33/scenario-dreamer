#!/usr/bin/env python
"""Render paper qualitative samples as four-source GIF/PNG directories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from critical_scene.ldm_adv_eval import (
    build_reward,
    compose_eval_cfg,
    prepare_ldm_cfg,
    slice_payload,
)
from utils.viz import render_rollout, render_rollout_frames
from render_scene_init_gifs import agent_colors


SOURCES = (
    ("original", "Log"),
    ("proximity_adv", "Log + proximity adversary"),
    ("base_gen", "AdvScene-base"),
    ("ddpo_gen", "AdvScene-RL"),
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table-dir", default="data/critical_scene/table_main_v3")
    p.add_argument("--out-dir", default="data/critical_scene/qualitative_2x4")
    p.add_argument("--cell", required=True)
    p.add_argument("--sut", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--slots", type=int, nargs="+", required=True)
    p.add_argument("--start-index", type=int, required=True)
    p.add_argument("--config-name", default="config_ldm_adv_ddpo")
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=90)
    return p.parse_args()


def _load_artifacts(path: Path) -> dict[str, dict]:
    return {
        source: torch.load(path / f"{source}.pt", map_location="cpu", weights_only=False)
        for source, _ in SOURCES
    }


def _scalar(metrics: dict, key: str):
    value = float(metrics[key][0])
    return value if np.isfinite(value) else None


def _pad_frames(frames: np.ndarray, length: int) -> np.ndarray:
    if len(frames) == length:
        return frames
    return np.concatenate([frames, np.repeat(frames[-1:], length - len(frames), axis=0)])


def main() -> int:
    args = _parse()
    cell_dir = Path(args.table_dir) / args.cell
    artifacts = _load_artifacts(cell_dir / "artifacts")
    cfg = compose_eval_cfg(args.config_name, [
        f"planner@ddpo.planner.sut={args.sut}",
        f"planner@ddpo.planner.env={args.env}",
        f"planner@ddpo.planner.adv={args.env}",
        "ddpo/reward=hierarchical_v3",
        *args.override,
    ])
    ldm_cfg = prepare_ldm_cfg(cfg)
    reward = build_reward(cfg, ldm_cfg, num_workers=0, batch_size=64)
    out_root = Path(args.out_dir)

    for offset, slot in enumerate(args.slots):
        sample_index = args.start_index + offset
        sample_dir = out_root / f"sample_{sample_index:03d}_{args.cell}_slot{slot:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        frames_by_source = {}
        png_by_source = {}
        method_metadata = {}

        for source, label in SOURCES:
            blob = artifacts[source]
            scenes = slice_payload(blob["payload"], slot, slot + 1)
            metrics = reward.evaluate(scenes, record_trajectories=True)
            trajectory = metrics["trajectories"][0]
            lanes = scenes.lane_polylines
            lanes = lanes.numpy() if isinstance(lanes, torch.Tensor) else np.asarray(lanes)
            kwargs = dict(
                agent_states=scenes.agent_states.numpy(),
                agent_types=scenes.agent_types.numpy(),
                agent_colors=agent_colors(scenes),
                reward=float(metrics["reward"][0]),
                ego_collision=bool(metrics["ego_collision"][0] > 0),
                init_invalid=bool(metrics["init_invalid"][0] > 0),
                ego_min_ttc=float(metrics["ego_min_ttc"][0]),
                title=label,
            )
            stem = f"{args.cell}_slot{slot:04d}_{source}_rollout"
            gif_path = sample_dir / f"{stem}.gif"
            png_path = sample_dir / f"{stem}.png"

            frames = render_rollout_frames(
                trajectory, lanes, max_frames=args.max_frames, annotate=False, **kwargs
            )
            imageio.mimsave(gif_path, list(frames), fps=args.fps, loop=0)
            fig = render_rollout(
                trajectory, lanes, final_boxes_only=True, annotate=False, **kwargs
            )
            fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            frames_by_source[source] = frames
            png_by_source[source] = np.asarray(Image.open(png_path).convert("RGB"))
            method_metadata[source] = {
                "display_name": label,
                "gif": gif_path.name,
                "png": png_path.name,
                "num_agents": int(scenes.agent_states.shape[0]),
                "reward": _scalar(metrics, "reward"),
                "ego_collision": bool(metrics["ego_collision"][0] > 0),
                "ego_fault_collision": bool(metrics["ego_fault_collision"][0] > 0),
                "collision_time_s": _scalar(metrics, "ego_collision_time"),
                "ego_min_ttc_s": _scalar(metrics, "ego_min_ttc"),
                "init_invalid": bool(metrics["init_invalid"][0] > 0),
                "reached_goal": bool(metrics["reached_goal"][0] > 0),
            }

        max_length = max(len(frames) for frames in frames_by_source.values())
        row_gif = np.concatenate([
            _pad_frames(frames_by_source[source], max_length) for source, _ in SOURCES
        ], axis=2)
        imageio.mimsave(sample_dir / "row_preview.gif", list(row_gif), fps=args.fps, loop=0)
        row_png = np.concatenate([png_by_source[source] for source, _ in SOURCES], axis=1)
        Image.fromarray(row_png).save(sample_dir / "row_preview.png")

        metadata = {
            "sample_id": sample_dir.name,
            "pool_slot": slot,
            "dataset_scene_idx": int(artifacts["ddpo_gen"]["metadata"]["dataset_scene_idx"][slot]),
            "table_dir": args.table_dir,
            "cell": args.cell,
            "planner": {"sut": args.sut, "env": args.env, "adv": args.env},
            "reward": "hierarchical_v3",
            "row_preview_gif": "row_preview.gif",
            "row_preview_png": "row_preview.png",
            "methods": method_metadata,
            "note": (
                "proximity_adv is Log plus one inserted agent; original/proximity share "
                "a real scene, while base_gen/ddpo_gen share a generated base scene."
            ),
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[qualitative] wrote {sample_dir}", flush=True)

    reward.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
