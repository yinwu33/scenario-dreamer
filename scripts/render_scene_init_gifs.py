"""Tile all eight scene-initialization methods of one table cell into GIFs.

Every source in a cell is built on the SAME template slots, so slot i is the
same underlying map and traffic across all eight; tiling them is a paired,
like-for-like comparison of what each initialization does to one scene.

Slots are picked by contrast: scenes the recorded initialization survives but a
generated one does not, which is where the methods actually differ.

  .venv/bin/python scripts/render_scene_init_gifs.py \
      --cell idm-idm --sut idm --env idm --count 4
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
import matplotlib.pyplot as plt

import imageio.v2 as imageio
import numpy as np
import torch

from critical_scene.ldm_adv_eval import (
    build_reward,
    compose_eval_cfg,
    prepare_ldm_cfg,
    slice_payload,
)
from ddpo.viz import render_rollout, render_rollout_frames

# Reading order matches the table rows.
GRID = (
    (("original", "Log"),
     ("proximity_adv", "Log + proximity"),
     ("base_gen_uncond_bok1", "AdvScene (uncond)"),
     ("base_gen_uncond_bok32", "AdvScene (uncond, Bo32)")),
    (("base_gen", "AdvScene (Base)"),
     ("base_gen_bok32", "AdvScene (Base, Bo32)"),
     ("ddpo_gen", "AdvScene (RL)"),
     ("original_ddpo_adv", "AdvScene (RL) + Log")),
)
SOURCES = tuple(s for row in GRID for s, _ in row)


def load_artifacts(artifact_dir: Path, sources=SOURCES) -> dict[str, dict]:
    out = {}
    for source in sources:
        path = artifact_dir / f"{source}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing artifact {path}")
        out[source] = torch.load(path, map_location="cpu", weights_only=False)["payload"]
    return out


def agent_colors(scenes):
    mask = scenes.meta.get("gen_agent_mask")
    if mask is None:
        return None
    mask = mask.numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    return ["tab:green" if m else None for m in mask]


def pick_slots(reward, payloads, n_probe: int, count: int) -> list[int]:
    """Slots the recorded scene survives but the most aggressive source does not."""
    probe = {}
    for source in ("original", "base_gen_bok32", "ddpo_gen"):
        m = reward.evaluate(slice_payload(payloads[source], 0, n_probe))
        probe[source] = np.asarray(m["ego_collision"]) > 0
    contrast = (~probe["original"]) & (probe["base_gen_bok32"] | probe["ddpo_gen"])
    slots = np.nonzero(contrast)[0].tolist()
    print(f"[gif] {len(slots)} contrast slots in the first {n_probe}")
    return slots[:count]


def render(
    reward,
    payloads,
    slot: int,
    max_frames: int,
    *,
    annotate: bool,
    source_only: str | None = None,
    png_path: Path | None = None,
) -> np.ndarray:
    frames = {}
    cells = [c for row in GRID for c in row]
    if source_only is not None:
        cells = [c for c in cells if c[0] == source_only]
    for source, label in cells:
        scenes = slice_payload(payloads[source], slot, slot + 1)
        m = reward.evaluate(scenes, record_trajectories=True)
        lanes = scenes.lane_polylines
        trajectory = m["trajectories"][0]
        render_kwargs = dict(
            agent_states=scenes.agent_states.numpy(),
            agent_types=scenes.agent_types.numpy(),
            agent_colors=agent_colors(scenes),
            reward=float(m["reward"][0]),
            ego_collision=bool(m["ego_collision"][0] > 0),
            init_invalid=bool(m["init_invalid"][0] > 0),
            ego_min_ttc=float(m["ego_min_ttc"][0]),
            title=label,
        )
        frames[source] = render_rollout_frames(
            trajectory,
            lanes.numpy() if isinstance(lanes, torch.Tensor) else np.asarray(lanes),
            max_frames=max_frames,
            annotate=annotate,
            **render_kwargs,
        )
        if png_path is not None:
            fig = render_rollout(
                trajectory,
                lanes.numpy() if isinstance(lanes, torch.Tensor) else np.asarray(lanes),
                final_boxes_only=True,
                annotate=False,
                **render_kwargs,
            )
            fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
    if source_only is not None:
        return frames[source_only]
    length = max(f.shape[0] for f in frames.values())
    pad = lambda f: f if f.shape[0] >= length else np.concatenate(
        [f, np.repeat(f[-1:], length - f.shape[0], axis=0)], axis=0)
    rows = [np.concatenate([pad(frames[s]) for s, _ in row], axis=2) for row in GRID]
    return np.concatenate(rows, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-dir", default="data/critical_scene/table_main_20260830")
    ap.add_argument("--cell", default="idm-idm")
    ap.add_argument("--sut", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--config-name", default="config_ldm_adv_ddpo")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--probe", type=int, default=64, help="slots to scan for contrast")
    ap.add_argument("--slots", nargs="*", type=int, default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=90)
    ap.add_argument(
        "--no-title",
        action="store_true",
        help="omit titles, timestamps, and rollout metrics; render only the plots",
    )
    ap.add_argument(
        "--png",
        action="store_true",
        help="also save a static PNG with final-frame boxes and full trajectories",
    )
    ap.add_argument(
        "--source",
        choices=SOURCES,
        default=None,
        help="render one source per GIF instead of the 2x4 comparison",
    )
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cell_dir = Path(args.table_dir) / args.cell
    out_dir = Path(args.out_dir) if args.out_dir else cell_dir / "media"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_root = compose_eval_cfg(args.config_name, [
        f"planner@ddpo.planner.sut={args.sut}",
        f"planner@ddpo.planner.env={args.env}",
        f"planner@ddpo.planner.adv={args.env}",
        *args.override,
    ])
    ldm_cfg = prepare_ldm_cfg(cfg_root)
    reward = build_reward(cfg_root, ldm_cfg, num_workers=0, batch_size=64)

    sources = (args.source,) if args.source else SOURCES
    payloads = load_artifacts(cell_dir / "artifacts", sources)
    slots = args.slots or pick_slots(reward, payloads, args.probe, args.count)
    if not slots:
        print("[gif] no contrast slots found; pass --slots explicitly")
        return 1

    for slot in slots:
        suffix = f"{args.source}_rollout" if args.source else "scene_init_2x4"
        stem = f"{args.cell}_slot{slot:04d}_{suffix}"
        tiled = render(
            reward,
            payloads,
            slot,
            args.max_frames,
            annotate=not args.no_title,
            source_only=args.source,
            png_path=out_dir / f"{stem}.png" if args.png else None,
        )
        path = out_dir / f"{stem}.gif"
        imageio.mimsave(path, list(tiled), fps=args.fps, loop=0)
        print(f"[gif] wrote {path}  ({tiled.shape[0]} frames)")
    reward.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
