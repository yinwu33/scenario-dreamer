"""Render 2x2 comparison GIFs for the four ldm_adv evaluation sources.

For each requested scene (pool slot of a run produced by
scripts/run_ldm_adv_ppo_table.py) the four sources are rolled out under
the same ppo simulator used for the tables and tiled into one GIF:

    top-left:  original             top-right:  original_ddpo_adv
    bottom-left: base_gen           bottom-right: ddpo_gen

Left column = real map, right/bottom cells carry the generated adversary drawn
in green. Each cell's status line shows reward / collision / TTC exactly like
the DDPO training visuals.

Usage (env vars from scripts/define_env_variables.sh must be set):
  .venv/bin/python scripts/render_ldm_adv_2x2_gifs.py \
      --out-dir data/critical_scene/ldm_adv_ppo_eval_1000 --scenes 0 1 2 3
  .venv/bin/python scripts/render_ldm_adv_2x2_gifs.py \
      --out-dir data/critical_scene/ldm_adv_ppo_eval_1000 --top-collisions 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from critical_scene.ldm_adv_eval import (
    build_reward,
    compose_eval_cfg,
    prepare_ldm_cfg,
    slice_payload,
)
from ddpo.viz import render_rollout_frames, save_gif

# 2x2 grid order: (row, col) -> source
GRID = (
    ("original", "original_ddpo_adv"),
    ("base_gen", "ddpo_gen"),
)
SOURCES_2X2 = tuple(s for row in GRID for s in row)


def _load_artifacts(artifact_dir: Path) -> dict[str, dict]:
    out = {}
    for source in SOURCES_2X2:
        path = artifact_dir / f"{source}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing artifact {path}; run the table script first")
        out[source] = torch.load(path, map_location="cpu", weights_only=False)
    return out


def _agent_colors(scenes) -> list | None:
    """Green for the generated adversary, defaults elsewhere."""
    mask = scenes.meta.get("gen_agent_mask")
    if mask is None:
        return None
    mask = mask.numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    return ["tab:green" if m else None for m in mask]


def _render_source_frames(reward, payload, slot: int, source: str, max_frames: int) -> np.ndarray:
    scenes = slice_payload(payload, slot, slot + 1)
    metrics = reward.evaluate(scenes, record_trajectories=True)
    lanes = scenes.lane_polylines
    lanes = lanes.numpy() if isinstance(lanes, torch.Tensor) else np.asarray(lanes)
    states = scenes.agent_states.numpy()
    types = scenes.agent_types.numpy()
    return render_rollout_frames(
        metrics["trajectories"][0],
        lanes,
        agent_states=states,
        agent_types=types,
        agent_colors=_agent_colors(scenes),
        reward=float(metrics["reward"][0]),
        ego_collision=bool(metrics["ego_collision"][0] > 0),
        init_invalid=bool(metrics["init_invalid"][0] > 0),
        ego_min_ttc=float(metrics["ego_min_ttc"][0]),
        title=source,
        max_frames=max_frames,
    )


def _pad_frames(frames: np.ndarray, length: int) -> np.ndarray:
    if frames.shape[0] >= length:
        return frames
    pad = np.repeat(frames[-1:], length - frames.shape[0], axis=0)
    return np.concatenate([frames, pad], axis=0)


def _tile_2x2(frames_by_source: dict[str, np.ndarray]) -> np.ndarray:
    length = max(f.shape[0] for f in frames_by_source.values())
    padded = {s: _pad_frames(f, length) for s, f in frames_by_source.items()}
    rows = [np.concatenate([padded[a], padded[b]], axis=2) for a, b in GRID]
    return np.concatenate(rows, axis=1)


def _top_collision_slots(out_dir: Path, k: int) -> list[int]:
    """Slots where ddpo_gen collided, most interesting (highest reward) first."""
    import csv

    path = out_dir / "benchmark" / "ddpo_gen" / "per_scene.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the benchmark first")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    hits = [r for r in rows if float(r["ego_collision"]) > 0]
    hits.sort(key=lambda r: -float(r["reward"]))
    return [int(r["pool_slot"]) for r in hits[:k]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_critical_scene_ldm_adv_ddpo")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--out-dir", default="data/critical_scene/ldm_adv_ppo_eval_1000")
    ap.add_argument("--scenes", nargs="*", type=int, default=None, help="pool slots to render")
    ap.add_argument(
        "--top-collisions",
        type=int,
        default=0,
        help="render the K ddpo_gen collision scenes with the highest reward instead of --scenes",
    )
    ap.add_argument("--gif-dir", default=None, help="default: <out-dir>/media")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=90)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    gif_dir = Path(args.gif_dir) if args.gif_dir else out_dir / "media"
    gif_dir.mkdir(parents=True, exist_ok=True)

    if args.top_collisions > 0:
        slots = _top_collision_slots(out_dir, args.top_collisions)
        if not slots:
            print("no ddpo_gen collision scenes found")
            return 0
    else:
        slots = args.scenes if args.scenes else [0, 1, 2, 3]

    cfg_root = compose_eval_cfg(args.config_name, args.overrides)
    ldm_cfg = prepare_ldm_cfg(cfg_root)
    reward = build_reward(cfg_root, ldm_cfg)
    artifacts = _load_artifacts(out_dir / "artifacts")

    for slot in slots:
        frames_by_source = {
            source: _render_source_frames(
                reward, artifacts[source]["payload"], slot, source, args.max_frames
            )
            for source in SOURCES_2X2
        }
        gif = _tile_2x2(frames_by_source)
        ds_idx = artifacts["original"]["metadata"]["dataset_scene_idx"][slot]
        path = gif_dir / f"scene_slot{slot:05d}_ds{ds_idx}.gif"
        save_gif(gif, str(path), fps=args.fps)
        print(f"[gif] wrote {path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
