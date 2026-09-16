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
    ("Valid", "valid"),
    ("Succ.", "succ"),
    ("Succ._c", "succ_conflict"),
    ("Succ._s", "succ_skip"),
    ("Coll.", "collision"),
    ("Coll._f", "collision_ego_fault"),
    ("TTC<3s", "ttc_lt_3s"),
    ("TTC<1.5s", "ttc_lt_1p5s"),
)


def load(root: Path, name: str, min_t: float) -> dict:
    """Per-scene indicators plus the DENOMINATOR each column is measured over.

    Every rate is conditional on the scene being valid -- the ego interpenetrating
    no vehicle at t=0 -- because such a scene has its collision, its TTC and its
    arrival decided before a planner acts. ``valid`` is the one column that is not,
    and ``Succ._c`` conditions on the chords conflicting as well, so the denominator
    travels with the column rather than being assumed uniform."""
    z = np.load(root / name / "metrics.npz", allow_pickle=True)
    if "init_ego_overlap_frac" not in z.files:
        raise SystemExit(f"{root / name}: no init_ego_overlap_frac; run "
                         "scripts/backfill_init_ego_overlap.py --roots "
                         f"{root.name} first")
    conflict = z["path_conflict"].astype(bool)
    valid = z["init_ego_overlap_frac"] == 0
    ttc = z["ego_min_ttc"]
    # A contact before min_t, in a scene that is otherwise valid, is a legally
    # placed adversary the ego had no room to react to -- a real outcome. Excluding
    # it is a robustness cut, not a second artifact filter, so min_t defaults to 0.
    late = z["ego_collision_time"] >= min_t
    vals = {
        "valid": valid,
        "succ": z["reached_goal"].astype(bool),
        "succ_conflict": z["reached_goal"].astype(bool),
        "succ_skip": z["reached_goal"].astype(bool) & conflict,
        "collision": (z["ego_collision"] > 0) & late,
        "collision_ego_fault": (z["ego_fault_collision"] > 0) & late,
        "ttc_lt_3s": ttc < 3.0,
        "ttc_lt_1p5s": ttc < 1.5,
    }
    masks = dict.fromkeys(vals, valid)
    masks["valid"] = np.ones_like(valid)
    masks["succ_conflict"] = valid & conflict
    return {"stems": np.array([str(s) for s in z["scenario"]]),
            "vals": vals, "masks": masks, "n": int(valid.size)}


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
    ap.add_argument("--collision-min-time", type=float, default=0.0,
                    help="restrict both collision columns to contacts at or after "
                         "this many seconds (0 = no restriction)")
    ap.add_argument("--out", default=None, help="default: <root>/bok_table.md")
    args = ap.parse_args()

    root = Path(args.root)
    names = sorted((d.name for d in root.iterdir() if (d / "metrics.npz").exists()),
                   key=order_key)
    data = {n: load(root, n, args.collision_min_time) for n in names}
    if args.ref not in data:
        raise SystemExit(f"--ref {args.ref!r} is not one of {names}")
    ref = data[args.ref]

    head = ("| source | n | " + " | ".join(c[0] for c in COLUMNS)
            + " | win | lose | p vs ref |")
    lines = [
        f"root: `{root}`   ref: `{args.ref}`   test: `{args.test_metric}`",
        "",
        "Every rate but `Valid` is measured on that row's VALID scenes -- the ego",
        "interpenetrates no vehicle at t=0 -- since a scene that starts in contact has",
        "its outcome decided before a planner acts. `Valid` is the share that were, so",
        "the rates are NOT an unconditional rate rescaled by it: the dropped scenes",
        "carry events. `Succ._c` conditions on `path_conflict` as well; `Succ._s` is",
        "`reached_goal AND path_conflict`, the value a root scored with",
        "`skip_rollout: true` reports.",
        (f"Both collision columns additionally require the contact at or after "
         f"{args.collision_min_time:g} s." if args.collision_min_time else
         "Collision columns carry no time restriction."),
        "",
        f"`win` counts scenes where the ROW has `{args.test_metric}` and `{args.ref}`",
        "does not, `lose` the reverse, and `p` is McNemar's exact test on those",
        "discordant pairs. Rows are paired by scene stem AND restricted to the scenes",
        "valid in BOTH rows, so the pairing survives the per-row denominators.",
        "",
        head,
        "|---|---:|" + "---:|" * (len(COLUMNS) + 3),
    ]
    for name in names:
        d = data[name]
        cells = " | ".join(f"{100.0 * d['vals'][k][d['masks'][k]].mean():.2f}"
                           for _, k in COLUMNS)
        if name == args.ref:
            tail = "-- | -- | --"
        else:
            # Align on the stems both rows carry FIRST, then keep the scenes valid
            # in BOTH. Masking each side independently would drop different scenes
            # from each and silently misalign the pairing.
            _, i_ref, i_row = np.intersect1d(ref["stems"], d["stems"],
                                             return_indices=True)
            both = (ref["masks"][args.test_metric][i_ref]
                    & d["masks"][args.test_metric][i_row])
            up, down, p = mcnemar(ref["vals"][args.test_metric][i_ref][both],
                                  d["vals"][args.test_metric][i_row][both])
            tail = f"{up} | {down} | {p:.2g}"
        lines.append(f"| {name} | {d['n']} | {cells} | {tail} |")

    text = "\n".join(lines)
    out = Path(args.out) if args.out else root / "bok_table.md"
    out.write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"\n[bok] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
