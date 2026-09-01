"""Render the appendix full-matrix table: SUT x traffic x scene initialization.

Layout follows the earlier draft (SUT blocks on rows, traffic planners on column
groups), with two corrections:

  * SMART / CtRL-Sim are not SUT rows. ``smart.planner`` refuses the SUT role by
    design (``Succ.`` is a goal check and both models are goal-free), so IDM and
    PPO are the complete SUT set.
  * The AdvScene rows are protocol B: ONE checkpoint per SUT, the one fine-tuned
    against ppo_normal traffic, swept across every traffic column. Reading across
    the row is then a genuine traffic sweep of a fixed model. PROVENANCE only
    holds the diagonal (each cell scored with its own trio's checkpoint), so the
    off-diagonal cells are marked pending until the re-scoring runs.

The Log / proximity / AdvScene-base rows need no re-scoring: their scenes are
planner independent, so every cell is already in PROVENANCE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# column groups: (table label, PROVENANCE traffic key)
TRAFFIC = (
    (r"IDM", "idm"),
    (r"$\mathrm{PPO}_{\text{aggr}}$", "ppo_aggressive"),
    (r"$\mathrm{PPO}_{\text{norm}}$", "ppo_norm"),
    (r"$\mathrm{PPO}_{\text{caut}}$", "ppo_caution"),
)
SUTS = ((r"IDM", "idm"), (r"$\mathrm{PPO}_{\text{self-play}}$", "ppo"))

# (table label, PROVENANCE source, is the row tied to one DDPO checkpoint?)
ROWS = (
    ("Log", "original", False),
    ("Log + proximity adversary", "proximity_adv", False),
    ("AdvScene-base", "base_gen", False),
    (r"\textbf{AdvScene}", "ddpo_gen", True),
    (r"\textbf{Log + AdvScene adversary}", "original_ddpo_adv", True),
)
METRICS = ("Succ", "Off", "Coll", "Coll_f")
# Column header per metric, in METRICS order.
HEADS = (r"Succ. $\downarrow$", r"Off.", r"Coll. $\uparrow$", r"Coll.$_{\text{f}}$ $\uparrow$")

# Protocol B pins the adversary model per SUT to the ppo_normal-traffic run.
PINNED_TRAFFIC = "ppo_norm"
PENDING = r"$\cdot$"


def raw(prov, sut_key, traffic_key, source, pinned):
    """The cell's metric values, or None when the protocol leaves it unmeasured."""
    if pinned and traffic_key != PINNED_TRAFFIC:
        return None
    rates = prov[f"{sut_key}-{traffic_key}"]["rates_pct"][source]
    return [float(rates[m]) for m in METRICS]


def fmt(values):
    return [PENDING] * len(METRICS) if values is None else [f"${v:.2f}$" for v in values]


def average(per_traffic):
    """Unweighted mean over the traffic columns, or pending if any is missing."""
    if any(v is None for v in per_traffic):
        return None
    return [sum(col) / len(col) for col in zip(*per_traffic)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance",
                    default="data/critical_scene/table_main_20260830/PROVENANCE.json")
    ap.add_argument("--out", default="research/overleaf/things/tables/table_full_matrix.tex")
    ap.add_argument("--advscene-protocol", choices=("diagonal", "pinned"), default="pinned",
                    help="diagonal: each cell uses the checkpoint trained for its own "
                         "(sut, traffic) trio -- every cell is already measured, but reading "
                         "across the row compares different models. pinned: one checkpoint "
                         "per SUT (the ppo_normal-traffic run) swept across traffic, which "
                         "needs re-scoring for the off-diagonal cells.")
    args = ap.parse_args()

    blob = json.loads(Path(args.provenance).read_text())
    prov = blob["cells"]
    n = blob["protocol"]["num_scenes"]

    adv_note = (
        r"The two AdvScene rows pin the adversary to the checkpoint fine-tuned "
        r"against $\mathrm{PPO}_{\text{norm}}$ traffic for that SUT, so reading across "
        r"them is a traffic sweep of one fixed model; " + PENDING + r" marks cells whose "
        r"re-scoring has not been run."
        if args.advscene_protocol == "pinned" else
        r"In the two AdvScene rows each cell uses the checkpoint fine-tuned against that "
        r"cell's own (SUT, traffic) trio, so those rows are a diagonal of eight models each "
        r"evaluated on its training condition, NOT a sweep of one model. The other three "
        r"rows use planner-independent scenes and are a genuine traffic sweep."
    )

    L = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full matrix: scene initialization $\times$ background traffic, for each",
        r"system under test. Every cell is $%d$ validation scenes under the same rollout" % n,
        r"and metric definitions. Coll. counts any ego-vehicle contact regardless of fault;",
        r"Coll.$_{\text{f}}$ counts only contacts the ego drove into (its closing speed on the",
        r"other vehicle exceeds $0.5$ m/s), so a stationary ego that is rammed is never at",
        r"fault. IDM and PPO are the complete set of systems under test: the CtRL-Sim and",
        r"SMART traffic models are goal-free and Succ. is a goal check, so neither can",
        r"occupy an ego column comparably. " + adv_note,
        r"\textit{Average} is the unweighted mean over the four traffic columns, which",
        r"weights the four policies equally rather than by any real-world frequency.}",
        r"\label{tab:full_matrix}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll *{5}{cccc}}",
        r"\toprule",
        r"\multirow{2}{*}{SUT} &",
        r"\multirow{2}{*}{Scene Initialization} &",
        " &\n".join(
            [r"\multicolumn{4}{c}{%s}" % lbl for lbl, _ in TRAFFIC]
            + [r"\multicolumn{4}{c}{\textit{Average}}"]
        ) + r" \\",
    ]
    n_groups = len(TRAFFIC) + 1
    for i in range(n_groups):
        L.append(r"\cmidrule(lr){%d-%d}" % (3 + 4 * i, 6 + 4 * i))
    L.append("&")
    L.append("\n".join("& " + " & ".join(HEADS) for _ in range(n_groups)) + r" \\")
    L.append(r"\midrule")

    pinned_mode = args.advscene_protocol == "pinned"
    for si, (sut_label, sut_key) in enumerate(SUTS):
        if si:
            L.append(r"\midrule")
        L.append(r"\multirow{%d}{*}{%s}" % (len(ROWS), sut_label))
        for row_label, source, tied in ROWS:
            pinned = tied and pinned_mode
            per_traffic = [raw(prov, sut_key, tk, source, pinned) for _, tk in TRAFFIC]
            row = [f"& {row_label}"]
            for values in per_traffic:
                row.append("& " + " & ".join(fmt(values)))
            row.append("& " + " & ".join(fmt(average(per_traffic))))
            L.append("\n".join(row) + r" \\")
        L.append("")

    L += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]

    out = Path(args.out)
    out.write_text("\n".join(L))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
