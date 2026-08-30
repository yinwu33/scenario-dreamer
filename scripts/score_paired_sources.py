#!/usr/bin/env python
"""Score ldm_adv_eval's paired scene artifacts with the PLANNER-QUALITY metrics.

`critical_scene.ldm_adv_eval` generates the three scene sources the results table
compares -- ``original`` / ``base_gen`` / ``ddpo_gen`` -- from the SAME template
pool slots, same base-scene latents and same initial adversary noise. That
pairing is what makes the table's three rows a controlled comparison, and it is
worth keeping.

What is NOT reusable is how that script scores them. It benchmarks through
``ddpo.reward.RewardModel``, whose ``ego_collision`` is the ADVERSARIAL notion:
ego vs the generated adversary only (``EgoCollisionHook``: ``if adv < 0: return``).
On the ``original`` source there is no generated adversary, so that column is a
structural 0.00 -- not a measurement -- and on the generated sources it counts a
strictly narrower event than the table's other rows, which come from the planner
benchmark's ego-vs-ANY-vehicle hook. Mixing the two in one table compares
different quantities, which is the defect this script exists to avoid.

So: paired scenes from ldm_adv_eval, metrics from planner_matrix_eval.

    python scripts/score_paired_sources.py \
        --artifacts data/critical_scene/table_main/idm-idm/artifacts \
        --sut idm --env idm --out data/critical_scene/table_main/idm-idm/scored.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir

from cfgs.config import CONFIG_PATH
from critical_scene.ldm_adv_eval import payload_to_scenes, slice_payload
from critical_scene.planner_matrix_eval import (
    build_runner,
    concat_metrics,
    evaluate_scenes,
    summarize,
)

SOURCES = ("original", "base_gen", "ddpo_gen", "original_ddpo_adv")
# The columns the results table reports, on the driving-ego subset.
COLUMNS = (
    ("Succ.", "reached_goal_rate_driving"),
    ("Off.", "ego_offroad_rate_driving"),
    ("Coll.", "ego_collision_rate_driving"),
    ("Coll._f", "ego_fault_collision_rate_driving"),
)


def _parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--artifacts", required=True, help="<out-dir>/artifacts from run_ldm_adv_ppo_table")
    p.add_argument("--sut", required=True, help="ego planner (the table row)")
    p.add_argument("--env", required=True, help="traffic planner (the table column)")
    p.add_argument("--adv", default=None, help="planner driving the generated adversary (default: --env)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", default=None, help="write the markdown table here")
    p.add_argument("--sources", nargs="+", default=list(SOURCES))
    return p.parse_args()


def main() -> int:
    args = _parse()
    adv = args.adv or args.env
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base=None):
        cfg = compose(
            config_name="config_planner_matrix",
            overrides=[
                f"planner@planner.sut={args.sut}",
                f"planner@planner.env={args.env}",
                f"planner@planner.adv={adv}",
            ],
        )
    runner = build_runner(cfg, num_workers=int(args.workers), batch_size=int(args.batch_size))
    min_ego_drive = float(cfg.benchmark.min_ego_drive)

    summaries = {}
    art = Path(args.artifacts)
    for source in args.sources:
        blob = torch.load(art / f"{source}.pt", map_location="cpu", weights_only=False)
        payload = blob["payload"]
        n = int(payload["num_scenes"])
        chunks = []
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            scenes = slice_payload(payload, start, end)
            metrics, _ = evaluate_scenes(runner, cfg, scenes)
            chunks.append(metrics)
            print(f"[score] {source} {end}/{n}", flush=True)
        summaries[source] = summarize(concat_metrics(chunks), min_ego_drive=min_ego_drive)
    if args.workers:
        # Rollout workers outlive the script as orphans otherwise.
        runner.close()

    header = f"| source | n_driving | " + " | ".join(c[0] for c in COLUMNS) + " |"
    lines = [
        f"cell: SUT={args.sut}  traffic={args.env}  adv={adv}",
        "metrics: planner-quality (ego vs ANY vehicle), driving-ego subset",
        "",
        header,
        "|---|---:|" + "---:|" * len(COLUMNS),
    ]
    for source in args.sources:
        s = summaries[source]
        vals = " | ".join(f"{100.0 * float(s[k]):.2f}" for _, k in COLUMNS)
        lines.append(f"| {source} | {int(s['num_driving_ego'])} | {vals} |")
    text = "\n".join(lines)
    print("\n" + text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )
        print(f"\n[score] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
