#!/usr/bin/env python
"""Render the best-of-$K$ comparison out of an ``eval_rollout.py`` output root.

``eval_rollout.py``'s own digest prints four rates and a finite-only ``minTTC``.
This table replaces that ``minTTC`` with the threshold counts (a scene where the
ego never approached is ``+inf``, so a mean over finite values is taken over a
different subset per row and rows are not comparable), and splits ``Succ.``
three ways, because ``path_conflict.skip_rollout`` decides what the column means:

    Succ.     every scene actually rolled out. Needs the root to have been
              scored with ``path_conflict.skip_rollout=false``.
    Succ._c   restricted to scenes whose ego/adv chords conflict.
    Succ._s   ``reached_goal AND path_conflict`` -- what a root scored with the
              reward yaml's own ``skip_rollout: true`` reports, since a retired
              scene never moves and books ``reached_goal = 0``.

Rows are paired by scene stem, so ``--ref`` gets a McNemar exact test on the
threshold column beside each rate.

    .venv/bin/python scripts/emit_bok_table.py --root data/final/cache/scenario_bok
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from paired_ckpt_test import mcnemar

COLUMNS = (
    ("Succ.", "succ"),
    ("Succ._c", "succ_conflict"),
    ("Succ._s", "succ_skip"),
    ("Coll.", "collision"),
    ("Coll._f", "collision_ego_fault"),
    ("TTC<3s", "ttc_lt_3s"),
    ("TTC<1.5s", "ttc_lt_1p5s"),
)


def load(root: Path, name: str) -> dict[str, np.ndarray]:
    z = np.load(root / name / "metrics.npz", allow_pickle=True)
    conflict = z["path_conflict"].astype(bool)
    ttc = z["ego_min_ttc"]
    return {
        "stems": np.array([str(s) for s in z["scenario"]]),
        "succ": z["reached_goal"].astype(bool),
        "succ_conflict": z["reached_goal"].astype(bool)[conflict],
        "succ_skip": z["reached_goal"].astype(bool) & conflict,
        "collision": z["ego_collision"].astype(bool),
        "collision_ego_fault": z["ego_fault_collision"].astype(bool),
        "ttc_lt_3s": ttc < 3.0,
        "ttc_lt_1p5s": ttc < 1.5,
    }


def order_key(name: str):
    """base_null family, then base_cond family, then everything else.

    Within a family the plain cache comes first and the budgets ascend, so the
    curve reads down the table."""
    family = ("base_null", 0) if name.startswith("base_null") else \
             ("base_cond", 1) if name.startswith("base_cond") else (name, 2)
    budget = int(name.rsplit("_bok", 1)[1]) if "_bok" in name else 0
    return family[1], family[0], budget


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ref", default="main_ppo-ppo_norm",
                    help="column each row is McNemar-tested against")
    ap.add_argument("--test-metric", default="ttc_lt_3s")
    ap.add_argument("--out", default=None, help="default: <root>/bok_table.md")
    args = ap.parse_args()

    root = Path(args.root)
    names = sorted((d.name for d in root.iterdir() if (d / "metrics.npz").exists()),
                   key=order_key)
    data = {n: load(root, n) for n in names}
    if args.ref not in data:
        raise SystemExit(f"--ref {args.ref!r} is not one of {names}")
    ref = data[args.ref]

    head = ("| source | n | " + " | ".join(c[0] for c in COLUMNS)
            + " | win | lose | p vs ref |")
    lines = [
        f"root: `{root}`   ref: `{args.ref}`   test: `{args.test_metric}`",
        "",
        "Rates are over every scene in the cache. `Succ._c` conditions on",
        "`path_conflict`; `Succ._s` is `reached_goal AND path_conflict`, the value a",
        "root scored with `skip_rollout: true` reports.",
        "",
        f"`win` counts scenes where the ROW has `{args.test_metric}` and `{args.ref}`",
        "does not, `lose` the reverse, and `p` is McNemar's exact test on those",
        "discordant pairs. Rows are paired by scene stem.",
        "",
        head,
        "|---|---:|" + "---:|" * (len(COLUMNS) + 3),
    ]
    for name in names:
        d = data[name]
        cells = " | ".join(f"{100.0 * d[k].mean():.2f}" for _, k in COLUMNS)
        if name == args.ref:
            tail = "-- | -- | --"
        else:
            keep = np.isin(d["stems"], ref["stems"])
            keep_ref = np.isin(ref["stems"], d["stems"])
            up, down, p = mcnemar(ref[args.test_metric][keep_ref], d[args.test_metric][keep])
            tail = f"{up} | {down} | {p:.2g}"
        lines.append(f"| {name} | {d['succ'].size} | {cells} | {tail} |")

    text = "\n".join(lines)
    out = Path(args.out) if args.out else root / "bok_table.md"
    out.write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"\n[bok] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
