#!/usr/bin/env python
"""Render representative GIFs for the PDM paper-table cells.

The scene artifacts are planner-independent.  This script re-scores a small
candidate prefix with PDM as the SUT, selects a balanced mix of outcomes, then
re-rolls only the selected scenes with trajectory recording enabled.
"""

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

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from cfgs.config import CONFIG_PATH
from critical_scene.ldm_adv_eval import (
    cat_payloads,
    payload_to_scenes,
    scenes_to_payload,
    slice_payload,
)
from critical_scene.planner_matrix_eval import (
    build_runner,
    evaluate_scenes,
    select_gif_scenes,
)
from ddpo.viz import CONTROL_COLOR, render_rollout_frames, save_gif


TRAFFIC_ARTIFACTS = {
    "idm": "ppo-idm",
    "ppo_aggressive": "ppo-ppo_aggressive",
    "ppo_normal": "ppo-ppo_norm",
    "ppo_caution": "ppo-ppo_caution",
}
SOURCES = ("original", "proximity_adv", "base_gen")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--table-dir",
        default="data/critical_scene/table_main_20260830",
    )
    p.add_argument("--traffic", nargs="+", choices=tuple(TRAFFIC_ARTIFACTS), default=list(TRAFFIC_ARTIFACTS))
    p.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--candidates", type=int, default=128)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=50)
    return p.parse_args()


def _compose_cfg(traffic: str):
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base=None):
        return compose(
            config_name="config_planner_matrix",
            overrides=[
                "planner@planner.sut=pdm",
                f"planner@planner.env={traffic}",
                f"planner@planner.adv={traffic}",
            ],
        )


def _selected_scenes(payload: dict, slots: list[int]):
    chunks = [scenes_to_payload(slice_payload(payload, slot, slot + 1)) for slot in slots]
    return payload_to_scenes(cat_payloads(chunks))


def _outcome(metrics: dict[str, np.ndarray], scene: int) -> str:
    if metrics["ego_collision_any"][scene] > 0:
        return "collision"
    if metrics["ego_offroad_proxy"][scene] > 0:
        return "offroad"
    if metrics["reached_goal"][scene] > 0:
        return "reached"
    return "timeout"


def _render_selected(
    runner,
    cfg,
    scenes,
    slots: list[int],
    dataset_ids: list[int],
    traffic: str,
    source: str,
    out_dir: Path,
    *,
    fps: int,
    max_frames: int,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, trajectories = evaluate_scenes(runner, cfg, scenes, record_trajectories=True)
    if trajectories is None:
        raise RuntimeError("trajectory recording returned no trajectories")

    states = scenes.agent_states.detach().cpu().numpy()
    types = scenes.agent_types.detach().cpu().numpy()
    agent_scene = scenes.agent_scene_idx.detach().cpu().numpy()
    lanes = scenes.lane_polylines.detach().cpu().numpy()
    lane_scene = scenes.meta["lane_scene_idx"].detach().cpu().numpy()
    gen_mask = scenes.meta.get("gen_agent_mask")
    if gen_mask is not None:
        gen_mask = gen_mask.detach().cpu().numpy()

    rows = []
    for scene, slot in enumerate(slots):
        a_sel = agent_scene == scene
        colors = None
        if gen_mask is not None:
            colors = [CONTROL_COLOR if generated else None for generated in gen_mask[a_sel]]
        outcome = _outcome(metrics, scene)
        filename = f"{outcome}_slot{slot:04d}_dataset{dataset_ids[scene]}.gif"
        frames = render_rollout_frames(
            trajectories[scene],
            lanes[lane_scene == scene],
            agent_states=states[a_sel],
            agent_types=types[a_sel],
            agent_colors=colors,
            ego_collision=bool(metrics["ego_collision_any"][scene] > 0),
            ego_offroad=bool(metrics["ego_offroad_proxy"][scene] > 0),
            title=(
                f"PDM / {traffic} / {source}  slot={slot} dataset={dataset_ids[scene]}  "
                f"goal={metrics['ego_goal_dist'][scene]:.1f}m"
            ),
            max_frames=max_frames,
        )
        save_gif(frames, str(out_dir / filename), fps=fps)
        row = {
            "traffic": traffic,
            "source": source,
            "slot": slot,
            "dataset_scene_idx": dataset_ids[scene],
            "outcome": outcome,
            "reached_goal": int(metrics["reached_goal"][scene]),
            "ego_collision_any": int(metrics["ego_collision_any"][scene]),
            "ego_offroad_proxy": int(metrics["ego_offroad_proxy"][scene]),
            "ego_goal_dist": float(metrics["ego_goal_dist"][scene]),
            "path": str(out_dir / filename),
        }
        rows.append(row)
        print(f"[gif] {traffic}/{source} {scene + 1}/{len(slots)} {filename}", flush=True)
    return rows


def _write_index(table_dir: Path, rows: list[dict]) -> None:
    root = table_dir / "pdm_gifs"
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# PDM rollout GIFs",
        "",
        "Red = PDM SUT, green = inserted adversary, blue/purple = other traffic.",
    ]
    for traffic in TRAFFIC_ARTIFACTS:
        traffic_rows = [row for row in rows if row["traffic"] == traffic]
        if not traffic_rows:
            continue
        lines.extend(["", f"## Traffic: {traffic}"])
        for source in SOURCES:
            source_rows = [row for row in traffic_rows if row["source"] == source]
            if not source_rows:
                continue
            lines.extend(["", f"### {source}", ""])
            for row in source_rows:
                rel = Path(row["path"]).relative_to(root)
                lines.append(
                    f"- [{row['outcome']} · slot {row['slot']} · dataset {row['dataset_scene_idx']}]({rel.as_posix()})"
                )
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse()
    if args.count <= 0 or args.candidates < args.count:
        raise ValueError("require candidates >= count > 0")

    table_dir = Path(args.table_dir)
    all_rows = []
    for traffic in args.traffic:
        cfg = _compose_cfg(traffic)
        runner = build_runner(cfg)
        min_ego_drive = float(cfg.benchmark.min_ego_drive)
        artifact_dir = table_dir / TRAFFIC_ARTIFACTS[traffic] / "artifacts"
        for source in args.sources:
            blob = torch.load(artifact_dir / f"{source}.pt", map_location="cpu", weights_only=False)
            payload = blob["payload"]
            candidate_count = min(args.candidates, int(payload["num_scenes"]))
            candidates = slice_payload(payload, 0, candidate_count)
            metrics, _ = evaluate_scenes(runner, cfg, candidates)
            slots = select_gif_scenes(metrics, args.count, min_ego_drive=min_ego_drive)
            if len(slots) != args.count:
                raise RuntimeError(
                    f"{traffic}/{source}: selected {len(slots)} scenes, expected {args.count}"
                )
            dataset_ids = [int(blob["metadata"]["dataset_scene_idx"][slot]) for slot in slots]
            selected = _selected_scenes(payload, slots)
            out_dir = table_dir / "pdm_gifs" / traffic / source
            rows = _render_selected(
                runner,
                cfg,
                selected,
                slots,
                dataset_ids,
                traffic,
                source,
                out_dir,
                fps=args.fps,
                max_frames=args.max_frames,
            )
            (out_dir / "selection.json").write_text(
                json.dumps(rows, indent=2) + "\n", encoding="utf-8"
            )
            all_rows.extend(rows)
            _write_index(table_dir, all_rows)

    print(f"[gif] wrote {len(all_rows)} GIFs under {table_dir / 'pdm_gifs'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
