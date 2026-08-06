#!/usr/bin/env python
"""Run one cell (or several) of the SUT x scene-initialization benchmark.

A cell is (ego planner, traffic planner, scene source). The planner axes are the
rollout's role axes, so a cell is selected purely by composing
``planner@planner.sut`` / ``planner@planner.env``; nothing in the harness knows
which planner it is running.

Examples
--------
The first table cell (IDM ego, IDM traffic, real log scenes)::

    python scripts/run_planner_matrix.py --sut idm --env idm \
        --num-scenes 1000 --out-dir data/critical_scene/planner_matrix_log

Add the PPO control row to the same table (bad_driver IS a PPO policy; its
``conditioning.collision_factor`` selects the driving style, 0 = aggressive,
2 = cautious)::

    python scripts/run_planner_matrix.py --sut idm --env bad_driver \
        --env-override planner.env.conditioning.collision_factor=0 \
        --num-scenes 1000 --out-dir data/critical_scene/planner_matrix_log

Reruns append to ``<out-dir>/table.{csv,md}``: each cell keeps its own
``benchmark/<cell>/`` directory and the table is rebuilt from every
``summary.json`` found there, so cells can be filled in one at a time.

Outputs::

    <out-dir>/benchmark/<sut>__<env>__<source>/per_scene.csv
    <out-dir>/benchmark/<sut>__<env>__<source>/summary.json
    <out-dir>/table.csv  and  <out-dir>/table.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfgs.config import CONFIG_PATH
from critical_scene.log_scenes import list_scene_files, load_log_scenes
from critical_scene.planner_matrix_eval import (
    SUMMARY_COLUMNS,
    build_runner,
    cell_label,
    concat_metrics,
    evaluate_scenes,
    render_cell_gifs,
    select_gif_scenes,
    summarize,
    write_json,
    write_per_scene_csv,
    write_table,
)

SOURCES = ("log",)


def compose_cfg(config_name: str, overrides: list[str]):
    with initialize_config_dir(config_dir=str(Path(CONFIG_PATH).resolve()), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, timeout=10
        ).stdout.strip()
    except Exception:
        return "unknown"


def select_indices(cfg, num_scenes: int, seed: int) -> list[int]:
    """Scene indices to evaluate: a seeded sample of the split, in index order.

    A contiguous prefix would be biased -- the files are sorted by tfrecord
    shard, so the first N scenes come from a handful of recording sessions.
    """
    files = list_scene_files(cfg.benchmark.preprocess_dir, cfg.benchmark.split)
    if not files:
        raise FileNotFoundError(
            f"no scenes under {cfg.benchmark.preprocess_dir}/{cfg.benchmark.split}"
        )
    n = min(int(num_scenes), len(files))
    rng = np.random.default_rng(int(seed))
    return sorted(rng.choice(len(files), size=n, replace=False).tolist())


def run_cell(cfg, args, indices: list[int]) -> tuple[dict[str, np.ndarray], list[int]]:
    """Roll every requested scene out in batches; returns metrics + scene ids."""
    runner = build_runner(cfg)
    batch = int(args.batch_size or cfg.benchmark.batch_size)
    files = list_scene_files(cfg.benchmark.preprocess_dir, cfg.benchmark.split)

    chunks, kept_all = [], []
    for start in range(0, len(indices), batch):
        part = indices[start : start + batch]
        print(
            f"[benchmark] scenes {start}..{start + len(part) - 1} / {len(indices)}",
            flush=True,
        )
        scenes, kept = load_log_scenes(
            cfg.benchmark.preprocess_dir,
            cfg.benchmark.split,
            part,
            cfg.dataset,
            files=files,
        )
        metrics, _ = evaluate_scenes(runner, cfg, scenes)
        chunks.append(metrics)
        kept_all.extend(kept)
    return concat_metrics(chunks), kept_all


def rebuild_table(out_dir: Path) -> None:
    """Rebuild table.{csv,md} from every summary.json under benchmark/."""
    summaries: dict[str, dict[str, float]] = {}
    for summary_path in sorted((out_dir / "benchmark").glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        # Rebuild the label from its parts rather than trusting the stored
        # ``cell`` string, so a label-format change applies to old runs too.
        cell = cell_label(payload["sut"], payload["env"], payload["source"])
        summaries[cell] = {c: payload["summary"].get(c) for c in SUMMARY_COLUMNS}
        # Surface the rollout lifecycle each row was produced under: cells
        # accumulate across runs and two lifecycles are not comparable.
        summaries[cell]["goal_behavior"] = (
            payload.get("planner", {}).get("sim", {}).get("goal_behavior", "?")
        )
    if summaries:
        write_table(out_dir, summaries)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sut", default="idm", help="ego planner (cfgs/planner/<name>.yaml)")
    p.add_argument("--env", default="idm", help="traffic planner (cfgs/planner/<name>.yaml)")
    p.add_argument("--source", default="log", choices=SOURCES, help="scene initialization")
    p.add_argument("--split", default=None, help="dataset split (default: config)")
    p.add_argument("--num-scenes", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=0, help="seeds the scene sample")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--config-name", default="config_planner_matrix")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="extra hydra override, repeatable (e.g. planner.env.conditioning.collision_factor=0)",
    )
    p.add_argument(
        "--gif",
        type=int,
        default=0,
        metavar="N",
        help="after the run, re-roll N scenes with trajectory recording and save "
        "one GIF each to <cell-dir>/gifs (collisions and failures first)",
    )
    p.add_argument("--gif-fps", type=int, default=10)
    args = p.parse_args()

    overrides = [f"planner@planner.sut={args.sut}", f"planner@planner.env={args.env}"]
    if args.split:
        overrides.append(f"benchmark.split={args.split}")
    overrides += list(args.override)
    cfg = compose_cfg(args.config_name, overrides)

    cell = cell_label(args.sut, args.env, args.source)
    indices = select_indices(cfg, args.num_scenes, args.seed)
    print(f"[benchmark] cell {cell}: {len(indices)} scenes from {cfg.benchmark.split}", flush=True)

    metrics, kept = run_cell(cfg, args, indices)
    summary = summarize(metrics, min_ego_drive=float(cfg.benchmark.min_ego_drive))

    metadata = {
        "cell": cell,
        "sut": args.sut,
        "env": args.env,
        "source": args.source,
        "split": str(cfg.benchmark.split),
        "seed": int(args.seed),
        "num_requested": len(indices),
        "dataset_scene_idx": kept,
        "planner": OmegaConf.to_container(cfg.planner, resolve=True),
        "simulator": OmegaConf.to_container(cfg.simulator, resolve=True),
        "overrides": overrides,
        "git_commit": git_commit(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    cell_dir = args.out_dir / "benchmark" / f"{args.sut}__{args.env}__{args.source}"
    write_per_scene_csv(cell_dir / "per_scene.csv", cell=cell, metadata=metadata, metrics=metrics)
    write_json(cell_dir / "summary.json", {**metadata, "summary": summary})
    print(f"[benchmark] wrote {cell_dir}", flush=True)

    for key in SUMMARY_COLUMNS:
        print(f"  {key:36s} {summary.get(key)}")
    rebuild_table(args.out_dir)

    if args.gif > 0:
        picked = select_gif_scenes(
            metrics, args.gif, min_ego_drive=float(cfg.benchmark.min_ego_drive)
        )
        if not picked:
            print("[gif] no driving-ego scenes to render", flush=True)
            return
        # `picked` indexes the concatenated results; map back to dataset scene ids.
        scene_ids = [kept[s] for s in picked]
        scenes, _ = load_log_scenes(
            cfg.benchmark.preprocess_dir,
            cfg.benchmark.split,
            scene_ids,
            cfg.dataset,
            files=list_scene_files(cfg.benchmark.preprocess_dir, cfg.benchmark.split),
        )
        paths = render_cell_gifs(
            build_runner(cfg), cfg, scenes, scene_ids, cell_dir / "gifs", fps=args.gif_fps
        )
        print(f"[gif] wrote {len(paths)} GIFs to {cell_dir / 'gifs'}", flush=True)
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
