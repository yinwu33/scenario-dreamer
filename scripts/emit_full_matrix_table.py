"""Render the full-matrix table: SUT x background traffic x scene initialization.

Reads the per-cell ``scored.json`` / ``scored_bok.json`` written by
``scripts/score_paired_sources.py`` and ``scripts/run_best_of_k.py``, which are
the authoritative per-cell records; ``PROVENANCE.json`` only carries the subset
of cells that fed the main table.

Cells that were never scored are printed as ``--`` rather than guessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUTS = (("IDM", "idm"), ("PDM", "pdm"), (r"$\mathrm{PPO}_{\text{norm}}$", "ppo"))
TRAFFIC = (
    ("IDM", "idm"),
    (r"$\mathrm{PPO}_{\text{aggr}}$", "ppo_aggressive"),
    (r"$\mathrm{PPO}_{\text{norm}}$", "ppo_norm"),
    (r"$\mathrm{PPO}_{\text{caut}}$", "ppo_caution"),
)
# (row label, scored.json source key, which file holds it)
ROWS = (
    ("Log", "original", "scored.json"),
    ("Log + proximity adversary", "proximity_adv", "scored.json"),
    ("AdvScene (uncond)", "base_gen_uncond_bok1", "scored_uncond.json"),
    (r"AdvScene (uncond, Bo$32$)", "base_gen_uncond_bok32", "scored_uncond.json"),
    ("AdvScene (Base)", "base_gen", "scored.json"),
    (r"AdvScene (Base, best-of-$32$)", "base_gen_bok32", "scored_bok.json"),
    ("AdvScene (RL)", "ddpo_gen", "scored.json"),
    ("AdvScene (RL) + Log", "original_ddpo_adv", "scored.json"),
)
# (column header, scored.json metric key)
METRICS = (
    (r"Succ. $\downarrow$", "reached_goal_rate_driving"),
    (r"Off.", "ego_offroad_rate_driving"),
    (r"Coll. $\uparrow$", "ego_collision_rate_driving"),
    (r"Coll.$_{\text{f}}$ $\uparrow$", "ego_fault_collision_rate_driving"),
)
MISSING = "$--$"


def load(root: Path):
    """{(sut, traffic): {source: {metric: value}}} over every scored cell."""
    out = {}
    for cell in root.iterdir():
        if not cell.is_dir() or "-" not in cell.name:
            continue
        sut, traffic = cell.name.split("-", 1)
        bucket = out.setdefault((sut, traffic), {})
        for fname in ("scored.json", "scored_bok.json", "scored_uncond.json"):
            path = cell / fname
            if path.exists():
                bucket.update(json.loads(path.read_text()))
    return out


def raw(data, sut, traffic, source):
    """The row's four metrics as percentages, or None when the cell is unscored."""
    rates = data.get((sut, traffic), {}).get(source)
    if rates is None:
        return None
    return [100.0 * float(rates[key]) for _, key in METRICS]


def fmt(values):
    return [MISSING] * len(METRICS) if values is None else [f"${v:.2f}$" for v in values]


def average(per_traffic):
    if any(v is None for v in per_traffic):
        return None
    return [sum(col) / len(col) for col in zip(*per_traffic)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-dir", default="data/critical_scene/table_main_20260830")
    ap.add_argument("--out", default="research/overleaf/things/tables/table_full_matrix.tex")
    args = ap.parse_args()

    data = load(Path(args.table_dir))
    n_groups = len(TRAFFIC) + 1
    heads = " & ".join(h for h, _ in METRICS)

    L = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Evaluation of different scene initialization methods.}",
        r"\label{tab:full_matrix}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll *{%d}{cccc}}" % n_groups,
        r"\toprule",
        r"\multirow{2}{*}{SUT} &",
        r"\multirow{2}{*}{Scene Initialization} &",
        " &\n".join(
            [r"\multicolumn{4}{c}{%s}" % lbl for lbl, _ in TRAFFIC]
            + [r"\multicolumn{4}{c}{\textit{Average}}"]
        ) + r" \\",
    ]
    L += [r"\cmidrule(lr){%d-%d}" % (3 + 4 * i, 6 + 4 * i) for i in range(n_groups)]
    L.append("&")
    L.append("\n".join(f"& {heads}" for _ in range(n_groups)) + r" \\")
    L.append(r"\midrule")
    L.append("")

    for si, (sut_label, sut_key) in enumerate(SUTS):
        if si:
            L += [r"\midrule", ""]
        L.append(r"\multirow{%d}{*}{%s}" % (len(ROWS), sut_label))
        for row_label, source, _ in ROWS:
            per_traffic = [raw(data, sut_key, tk, source) for _, tk in TRAFFIC]
            row = [f"& {row_label}"]
            for values in per_traffic + [average(per_traffic)]:
                row.append("& " + " & ".join(fmt(values)))
            L += ["\n".join(row) + r" \\", ""]

    L += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    Path(args.out).write_text("\n".join(L))
    print(f"wrote {args.out}")

    missing = [(s, t, src) for _, s in SUTS for _, t in TRAFFIC
               for _, src, _ in ROWS if raw(data, s, t, src) is None]
    print(f"unscored cells: {len(missing)} of {len(SUTS)*len(TRAFFIC)*len(ROWS)}")
    for s, t, src in missing:
        print(f"    {s}-{t}  {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
