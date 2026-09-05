#!/usr/bin/env python
"""Score paired scene artifacts with the ADVERSARIAL metrics (ego vs the adversary).

The companion `score_paired_sources.py` reports planner quality, whose collision
is ego-vs-ANY-vehicle. This script reports the same two collision notions in the
scope the DDPO reward actually optimizes: ego vs the GENERATED adversary.
`ego_fault_collision` uses the front-face/moving-ego predicate in sim/world.py,
so every pre-2026-09 number for that column is stale.

`original` has no generated adversary, so both of its rates are a structural 0.00
rather than a measurement.

    .venv/bin/python scripts/score_adv_sources.py \
        --artifacts <dir with <source>.pt> --sut ppo_normal --env ppo_normal \
        --sources base_gen ddpo_gen --workers 16 --batch-size 128 --out <path>.md
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

import numpy as np
import torch

from critical_scene.ldm_adv_eval import (
    benchmark_payload,
    build_reward,
    compose_eval_cfg,
    prepare_ldm_cfg,
    summarize,
)

# (header, summary key, scale) -- 100.0 renders a rate as a percentage, 1.0
# leaves minTTC in seconds. It is a mean over FINITE values, so a source with no
# generated adversary (``original``) has no near miss at all and reports nan.
COLUMNS = (
    ("Succ.", "reached_goal_rate_driving", 100.0),
    ("Off.", "ego_offroad_rate_driving", 100.0),
    ("Coll.", "ego_collision_rate_driving", 100.0),
    ("Coll._f", "ego_fault_collision_rate_driving", 100.0),
    ("minTTC", "ego_min_ttc_mean_driving", 1.0),
)


def _parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--artifacts", required=True)
    p.add_argument("--sut", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--adv", default=None, help="default: --env")
    p.add_argument("--config-name", default="config_ldm_adv_ddpo")
    p.add_argument("--reward", required=True,
                   help="cfgs/ddpo/reward/<name>.yaml to score with. Required: the "
                        "entrypoint config carries a reward of its own, so omitting "
                        "this silently scored every artifact under that default "
                        "(hierarchical_v2) whatever the run was trained with, and "
                        "wrote its reward/tier into scored_adv.npz under the wrong name.")
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--out", required=True, help="markdown table; .json and .npz go beside it")
    return p.parse_args()


def main() -> int:
    args = _parse()
    adv = args.adv or args.env
    cfg_root = compose_eval_cfg(args.config_name, [
        f"ddpo/reward={args.reward}",
        f"planner@ddpo.planner.sut={args.sut}",
        f"planner@ddpo.planner.env={args.env}",
        f"planner@ddpo.planner.adv={adv}",
        *args.override,
    ])
    ldm_cfg = prepare_ldm_cfg(cfg_root)
    reward = build_reward(
        cfg_root, ldm_cfg, num_workers=int(args.workers), batch_size=int(args.batch_size)
    )
    min_ego_drive = float(cfg_root.ddpo.min_ego_drive)

    art, out = Path(args.artifacts), Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summaries, per_scene = {}, {}
    for source in args.sources:
        blob = torch.load(art / f"{source}.pt", map_location="cpu", weights_only=False)
        metrics = benchmark_payload(
            reward, blob["payload"], batch_size=int(args.batch_size), label=source
        )
        summaries[source] = summarize(metrics, min_ego_drive=min_ego_drive)
        # Same scenes across sources, so a paired test needs no second rollout.
        for key, arr in metrics.items():
            per_scene[f"{source}/{key}"] = arr
    if args.workers:
        # Rollout workers outlive the script as orphans otherwise.
        reward.close()

    lines = [
        f"cell: SUT={args.sut}  traffic={args.env}  adv={adv}",
        f"reward: {args.reward}",
        "metrics: adversarial (ego vs the GENERATED ADVERSARY), driving-ego subset",
        "fault uses the front-face/moving-ego predicate (sim/world.py)",
        "",
        "| source | n_driving | " + " | ".join(c[0] for c in COLUMNS) + " |",
        "|---|---:|" + "---:|" * len(COLUMNS),
    ]
    for source in args.sources:
        s = summaries[source]
        vals = " | ".join(f"{scale * float(s[k]):.2f}" for _, k, scale in COLUMNS)
        lines.append(f"| {source} | {int(s['num_driving_ego'])} | {vals} |")
    text = "\n".join(lines)
    print("\n" + text)
    out.write_text(text + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    np.savez_compressed(out.with_suffix(".npz"), **per_scene)
    print(f"\n[score] wrote {out} (+ .json, .npz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
