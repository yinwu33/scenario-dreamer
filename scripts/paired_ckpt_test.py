"""Paired significance test over a DDPO run's checkpoints.

Every checkpoint is scored on the SAME template slots, so the comparison against
the frozen base is paired: what matters is how many scenes FLIPPED outcome, not
the unpaired binomial error of each rate. McNemar's exact test on the discordant
pairs is the right statistic; the unpaired error is reported alongside to show
how much the pairing buys.

  .venv/bin/python scripts/paired_ckpt_test.py --run-dir <ddpo output dir>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy import stats

METRICS = (("Coll", "ego_collision_any"),
           ("Coll_f", "ego_fault_collision_any"),
           ("Succ", "reached_goal"))


def mcnemar(base: np.ndarray, other: np.ndarray) -> tuple[int, int, float]:
    """(base=0 -> other=1, base=1 -> other=0, two-sided exact p)."""
    b01 = int(np.sum((~base) & other))
    b10 = int(np.sum(base & (~other)))
    n = b01 + b10
    p = 1.0 if n == 0 else float(stats.binomtest(b01, n, 0.5).pvalue)
    return b01, b10, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--min-ego-drive", type=float, default=10.0)
    args = ap.parse_args()

    files = sorted(Path(args.run_dir, "per_scene").glob("it*.npz"))
    if not files:
        raise SystemExit("no per_scene/*.npz; run track_ckpt_metrics.py first")
    data = {int(f.stem[2:]): np.load(f) for f in files}
    its = sorted(data)
    base_it = its[0]
    print(f"[paired] baseline = it{base_it}, comparing {its[1:]}\n")

    # The driving subset is a property of the template, but reconstruction jitter
    # moves a handful of egos across the 10 m threshold, so intersect the masks:
    # a paired test may only use scenes that are in the subset for BOTH members.
    for label, key in METRICS:
        print(f"=== {label}")
        print(f"{'it':>5} {'rate':>7} {'d(pp)':>7} {'n_pair':>7} "
              f"{'flip+':>6} {'flip-':>6} {'p':>9}   {'unpaired SE':>11}")
        for it in its:
            a, b = data[base_it], data[it]
            mask = (a["ego_goal_dist"] >= args.min_ego_drive) & \
                   (b["ego_goal_dist"] >= args.min_ego_drive)
            x = a[key][mask] > 0
            y = b[key][mask] > 0
            rate = 100.0 * y.mean()
            delta = 100.0 * (y.mean() - x.mean())
            se = 100.0 * math.sqrt(y.mean() * (1 - y.mean()) / mask.sum())
            if it == base_it:
                print(f"{it:>5} {rate:>7.2f} {'--':>7} {mask.sum():>7} "
                      f"{'--':>6} {'--':>6} {'--':>9}   {se:>11.2f}")
                continue
            up, down, p = mcnemar(x, y)
            print(f"{it:>5} {rate:>7.2f} {delta:>+7.2f} {mask.sum():>7} "
                  f"{up:>6} {down:>6} {p:>9.4f}   {se:>11.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
