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

Add a PPO traffic column to the same table. The ppo_* variants are one frozen
checkpoint at different ``conditioning.collision_factor`` values (0 = aggressive,
2 = cautious), so each is just another planner name::

    python scripts/run_planner_matrix.py --sut idm --env ppo_aggressive \
        --num-scenes 1000 --out-dir data/critical_scene/planner_matrix_log

The third axis is where the scenes come from. ``--source log`` reads the
preprocessed Waymo pickles; any other name is a generated-sample cache declared
in ``benchmark.gen_dirs``, and adding a checkpoint's cache to the table is one
yaml line::

    python scripts/run_planner_matrix.py --sut idm --env idm \
        --source ldm_adv_base \
        --num-scenes 1000 --out-dir data/critical_scene/planner_matrix_ldm_adv_base

Cross-source rows are a DISTRIBUTION-level comparison, not a paired one:
unconditional samples correspond to no particular log scene, and their egos have
nearer goals, so read success rates next to ``ego_goal_dist_mean`` (see
``critical_scene.gen_scenes``).

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfgs.config import CONFIG_PATH
from critical_scene.gen_scenes import list_gen_scene_files, load_gen_scenes
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

LOG_SOURCE = "log"
GIF_FPS = 10


def compose_cfg(config_name: str, overrides: list[str]):
    with initialize_config_dir(
        config_dir=str(Path(CONFIG_PATH).resolve()), version_base=None
    ):
        return compose(config_name=config_name, overrides=overrides)


@dataclass
class SceneSource:
    """One scene-initialization axis value: where scenes come from, and how.

    ``load`` hides the difference between the two producers behind the
    ``GeneratedScenes`` contract, so nothing downstream branches on the source --
    the same property that lets the planner axes be pure config composition.
    """

    name: str  # the table's third axis value
    origin: str  # human-readable provenance, stored per scene
    files: list[str]  # scene files, indexed by the sampled indices
    load: Callable[[Sequence[int]], tuple[object, list[int]]]


def build_source(cfg, args) -> SceneSource:
    """Resolve ``--source`` against the config.

    ``log`` is the preprocessed-Waymo loader; every other name must be a key of
    ``benchmark.gen_dirs``, i.e. a cached batch of samples from some generative
    checkpoint. Unknown names fail here with the available set rather than
    somewhere inside a loader.
    """
    if args.source == LOG_SOURCE:
        preprocess_dir, split = str(cfg.benchmark.preprocess_dir), str(
            cfg.benchmark.split
        )
        files = list_scene_files(preprocess_dir, split)
        if not files:
            raise FileNotFoundError(f"no scenes under {preprocess_dir}/{split}")
        return SceneSource(
            name=LOG_SOURCE,
            origin=split,
            files=files,
            load=lambda part: load_log_scenes(
                preprocess_dir, split, part, cfg.dataset, files=files
            ),
        )

    gen_dirs = OmegaConf.to_container(cfg.benchmark.gen_dirs, resolve=True) or {}
    if args.source not in gen_dirs:
        raise KeyError(
            f"unknown --source {args.source!r}; expected {LOG_SOURCE!r} or one of the "
            f"generated caches in benchmark.gen_dirs: {sorted(gen_dirs) or '(none)'}"
        )
    sample_dir = str(gen_dirs[args.source])
    files = list_gen_scene_files(sample_dir)
    if not files:
        raise FileNotFoundError(f"no cached samples under {sample_dir}")
    return SceneSource(
        name=args.source,
        origin=Path(sample_dir).name,
        files=files,
        load=lambda part: load_gen_scenes(sample_dir, part, files=files),
    )


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def select_indices(source: SceneSource, num_scenes: int, seed: int) -> list[int]:
    """Scene indices to evaluate: a seeded sample of the source, in index order.

    A contiguous prefix would be biased -- log files are sorted by tfrecord
    shard, so the first N scenes come from a handful of recording sessions, and
    a generated cache is ordered by sampling batch.
    """
    n = min(int(num_scenes), len(source.files))
    rng = np.random.default_rng(int(seed))
    return sorted(rng.choice(len(source.files), size=n, replace=False).tolist())


def run_cell(
    cfg, args, source: SceneSource, indices: list[int]
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Roll every requested scene out in batches; returns metrics + scene ids."""
    runner = build_runner(cfg)
    batch = int(args.batch_size or cfg.benchmark.batch_size)

    chunks, kept_all = [], []
    for start in range(0, len(indices), batch):
        part = indices[start : start + batch]
        print(
            f"[benchmark] scenes {start}..{start + len(part) - 1} / {len(indices)}",
            flush=True,
        )
        scenes, kept = source.load(part)
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
        # Older summaries predate the "adv" field; default to env, same as the
        # implicit behaviour before --adv existed.
        adv = payload.get("adv", payload["env"])
        cell = cell_label(payload["sut"], payload["env"], adv, payload["source"])
        summaries[cell] = {c: payload["summary"].get(c) for c in SUMMARY_COLUMNS}
        # Surface the rollout lifecycle each row was produced under: cells
        # accumulate across runs and two lifecycles are not comparable.
        summaries[cell]["goal_behavior"] = (
            payload.get("planner", {}).get("sim", {}).get("goal_behavior", "?")
        )
    if summaries:
        write_table(out_dir, summaries)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sut", default="idm", help="ego planner (cfgs/planner/<name>.yaml)"
    )
    p.add_argument(
        "--env", default="idm", help="traffic planner (cfgs/planner/<name>.yaml)"
    )
    p.add_argument(
        "--adv", default=None, help="adversary planner, use env planner by default"
    )
    p.add_argument(
        "--source",
        default=LOG_SOURCE,
        help="scene initialization: 'log', or a generated cache named in "
        "benchmark.gen_dirs (e.g. ldm_adv_base)",
    )
    p.add_argument(
        "--split", default=None, help="dataset split, log source only (default: config)"
    )
    p.add_argument("--num-scenes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0, help="seeds the scene sample")
    p.add_argument("--out-dir", default=Path("output/planner_matrix"))
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
    args = p.parse_args()

    adv = args.adv or args.env
    overrides = [
        f"planner@planner.sut={args.sut}",
        f"planner@planner.env={args.env}",
        f"planner@planner.adv={adv}",
    ]
    if args.split:
        overrides.append(f"benchmark.split={args.split}")
    overrides += list(args.override)
    cfg = compose_cfg(args.config_name, overrides)

    source = build_source(cfg, args)
    cell = cell_label(args.sut, args.env, adv, source.name)
    indices = select_indices(source, args.num_scenes, args.seed)
    print(
        f"[benchmark] cell {cell}: {len(indices)} scenes from {source.origin}",
        flush=True,
    )

    metrics, kept = run_cell(cfg, args, source, indices)
    summary = summarize(metrics, min_ego_drive=float(cfg.benchmark.min_ego_drive))

    metadata = {
        "cell": cell,
        "sut": args.sut,
        "env": args.env,
        "adv": adv,
        "source": source.name,
        # Which slice of that source: the split name for log scenes, the cache
        # directory for generated ones. Recorded per scene in the CSV.
        "split": source.origin,
        "seed": int(args.seed),
        "num_requested": len(indices),
        "dataset_scene_idx": kept,
        "planner": OmegaConf.to_container(cfg.planner, resolve=True),
        "simulator": OmegaConf.to_container(cfg.simulator, resolve=True),
        "overrides": overrides,
        "git_commit": git_commit(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    cell_dir_name = f"{args.sut}__{args.env}__{args.source}"
    if adv != args.env:
        cell_dir_name += f"__adv-{adv}"
    cell_dir = args.out_dir / "benchmark" / cell_dir_name
    write_per_scene_csv(
        cell_dir / "per_scene.csv", cell=cell, metadata=metadata, metrics=metrics
    )
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
        scenes, _ = source.load(scene_ids)
        paths = render_cell_gifs(
            build_runner(cfg), cfg, scenes, scene_ids, cell_dir / "gifs", fps=GIF_FPS
        )
        print(f"[gif] wrote {len(paths)} GIFs to {cell_dir / 'gifs'}", flush=True)
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
