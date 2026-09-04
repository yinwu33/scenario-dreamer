"""Render the main results table from the per-cell ``scored.json`` records.

Same source of truth as ``scripts/emit_full_matrix_table.py`` (which this imports
the loader from), restricted to the ppo_normal traffic column so that behaviour
is held fixed and the rows differ only in how the scene was initialized.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.emit_full_matrix_table import MISSING, METRICS, fmt, load, raw

TRAFFIC = "ppo_norm"
SUTS = (("IDM", "idm"), (r"$\mathrm{PPO}_{\text{self-play}}$", "ppo"))
ROWS = (
    ("Log", "original", False),
    ("Log + proximity adversary", "proximity_adv", False),
    (None, None, None),                                   # rule
    ("AdvScene (uncond)", "base_gen_uncond_bok1", False),
    (r"AdvScene (uncond, Bo$32$)", "base_gen_uncond_bok32", True),
    ("AdvScene (Base)", "base_gen", False),
    (r"AdvScene (Base, best-of-$32$)", "base_gen_bok32", True),
    (None, None, None),
    (r"\textbf{AdvScene (RL)}", "ddpo_gen", False),
    (r"\textbf{AdvScene (RL) + Log}", "original_ddpo_adv", False),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-dir", default="data/critical_scene/table_main_20260830")
    ap.add_argument("--out", default="research/overleaf/things/tables/table_main.tex")
    args = ap.parse_args()

    data = load(Path(args.table_dir))
    heads = " & ".join(h for h, _ in METRICS)

    L = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Effectiveness of different scene initialization methods across systems",
        r"under test. Background traffic is the normal self-play PPO policy in every cell",
        r"and the inserted agent is driven by that same policy, so no row uses an",
        r"adversarial behavior model. Every row of a column is measured on the SAME",
        r"template scenes with the SAME metric definitions, on the driving-ego subset of",
        r"$1000$ scenes. Coll. counts any ego-vehicle contact regardless of fault;",
        r"Coll.$_{\text{f}}$ counts only contacts the ego drove into, so a stationary ego that",
        r"is rammed is never at fault. The best-of-$32$ row spends $32\times$ the sampling",
        r"budget of every other row and is shown for reference. Off. is a diagnostic rather",
        r"than a criticality metric, since a generator can raise it with implausible",
        r"geometry.}",
        r"\label{tab:main}",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{l *{%d}{cccc}}" % len(SUTS),
        r"\toprule",
        r"\multirow{2}{*}{Scene Initialization} &",
        " &\n".join(r"\multicolumn{4}{c}{SUT: %s}" % lbl for lbl, _ in SUTS) + r" \\",
    ]
    L += [r"\cmidrule(lr){%d-%d}" % (2 + 4 * i, 5 + 4 * i) for i in range(len(SUTS))]
    L.append("\n".join(f"& {heads}" for _ in SUTS) + r" \\")
    L += [r"\midrule", ""]

    for label, source, _ in ROWS:
        if label is None:
            L += [r"\midrule", ""]
            continue
        row = [label]
        for _, sut_key in SUTS:
            row.append("& " + " & ".join(fmt(raw(data, sut_key, TRAFFIC, source))))
        L += ["\n".join(row) + r" \\", ""]

    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    Path(args.out).write_text("\n".join(L))
    print(f"wrote {args.out}")
    for label, source, _ in ROWS:
        if label is None:
            continue
        vals = []
        for _, sut_key in SUTS:
            v = raw(data, sut_key, TRAFFIC, source)
            vals.append("  ".join(f"{x:6.2f}" for x in v) if v else MISSING)
        print(f"  {label:36s} " + " | ".join(vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
