#!/usr/bin/env python
"""Assemble table_main_v3 from the per-cell scored_adv.json files.

All metrics are the ADVERSARIAL scope (ego vs the generated adversary), matching
what hierarchical_v3 optimizes. Cells that have not finished are rendered as
``--`` so the table can be inspected while the pipeline is still running.

best-of-K rows are absent on purpose: the v2-selected artifacts would understate
the baseline and re-selecting under v3 costs ~18 h. They must be added back.

    .venv/bin/python scripts/emit_table_main_v3.py [--latex]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "data/critical_scene/table_main_v3"

# Reorder or trim this list to change the table's metric columns.
METRICS = (
    ("Succ.", "reached_goal_rate_driving", 100.0),
    ("Coll.", "ego_collision_rate_driving", 100.0),
    ("Coll$_f$", "ego_fault_collision_rate_driving", 100.0),
    ("minTTC", "ego_min_ttc_mean_driving", 1.0),
)
ROWS = (
    ("Log", "original"),
    ("Log + proximity adv.", "proximity_adv"),
    ("AdvScene (Base)", "base_gen"),
    ("AdvScene (RL)", "ddpo_gen"),
    ("AdvScene (RL) + Log", "original_ddpo_adv"),
)
SUTS = (("IDM", "idm"), ("PDM", "pdm"), ("PPO$_{norm}$", "ppo"))
TRAFFIC = (("IDM", "idm"), ("PPO$_{norm}$", "ppo_norm"),
           ("PPO$_{aggr}$", "ppo_aggressive"), ("PPO$_{caut}$", "ppo_caution"))


def load() -> dict[str, dict]:
    out = {}
    for d in sorted(CELLS.glob("*/scored_adv.json")):
        out[d.parent.name] = json.loads(d.read_text(encoding="utf-8"))
    return out


def cell_value(data, cell, source, key, scale) -> str:
    s = data.get(cell, {}).get(source)
    if s is None:
        return "--"
    v = scale * float(s[key])
    return "--" if v != v else f"{v:.2f}"   # nan (no adversary) renders as --


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    data = load()
    print(f"% {len(data)}/12 cells present: {', '.join(sorted(data)) or 'none'}\n")

    sep = " & " if args.latex else " | "
    end = " \\\\" if args.latex else " |"
    for sut_label, sut in SUTS:
        print(f"{'% ' if args.latex else ''}=== SUT = {sut_label} ===")
        head = ["method"] + [f"{t} {m}" for _, t in TRAFFIC for m, _, _ in METRICS]
        print(("" if args.latex else "| ") + sep.join(head) + end)
        if not args.latex:
            print("|---|" + "---:|" * (len(head) - 1))
        for label, source in ROWS:
            vals = [
                cell_value(data, f"{sut}-{traffic}", source, key, scale)
                for _, traffic in TRAFFIC
                for _, key, scale in METRICS
            ]
            print(("" if args.latex else "| ") + sep.join([label] + vals) + end)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
