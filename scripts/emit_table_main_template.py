#!/usr/bin/env python
"""Render the main table in the paper's transposed layout, under three conditions.

``scripts/emit_table_main.py`` owns the numbers; this owns one particular SHAPE of
them, the one `temp/table_main_final_template.tex` fixes: rows are
(traffic planner x scene-initialization method), columns are (ego planner x metric),
and a final *Overall* block averages the 12 (traffic x ego) cells of a row. Every
estimate, every resample and every denominator comes from the other module, so the
two renderings can never disagree about a cell.

Three files, differing only in what a rate is measured over:

    table_main_final_original.tex      every scene
    table_main_final_remove_collid.tex scenes whose ego overlaps nothing at t=0
    table_main_final_collid_1s.tex     the same, and both collision columns
                                       additionally require contact at >= 1 s

The first is re-derived through today's estimator rather than copied out of
``data/final/table_main/``, and ``--check-original`` asserts the two agree cell for
cell -- which is the proof that the gate is the only thing that moved.

Usage (env vars from scripts/define_env_variables.sh must be set)::

    .venv/bin/python scripts/emit_table_main_template.py --out temp
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from emit_table_main import (  # noqa: E402
    EGO,
    GATE,
    ROWS,
    TRAFFIC,
    boot_stats,
    build_clusters,
    cell_column,
    cell_estimate,
    load_summaries,
)

# Short row labels for the paper. The keys are ``emit_table_main.ROWS`` labels, so a
# row added there and not here fails loudly rather than rendering under its long name.
LABEL = {
    "Log (closest agent as adversary)": "Log",
    "SceneControl": "SceneControl",
    "Scenario Dreamer": "ScenarioDreamer",
    "AdvScene-Base (null)": r"AdvScene-Base (null)",
    "AdvScene-Base (cond)": r"AdvScene-Base (cond)",
    "AdvScene-RL (init scene)": r"AdvScene-RL (init scene)",
    "AdvScene-RL (init agent)": r"AdvScene-RL (init agent)",
    "AdvScene-RL (init adv)": r"AdvScene-RL (init adv)",
}

# (summary key, header, better direction). Four metrics, as the template has.
# ``Valid`` is deliberately NOT a column here -- the template has no room for it --
# so the caption carries the range instead; see ``valid_note``.
PLAIN = [("succ", r"Succ. $\downarrow$", "min"),
         ("collision", r"Coll. $\uparrow$", "max"),
         ("collision_ego_fault", r"Coll.$_{\text{ego}}$ $\uparrow$", "max"),
         ("ttc_lt_3s", r"TTC$_{<3s}$ $\uparrow$", "max")]
# Same four columns with the two collision keys swapped for their >= 1 s variants
# (``eval_rollout.EARLY_CONTACT_T``). Both, not just Coll._ego: they are the same
# event at two fault levels, and restricting one would stop the other containing it.
T1 = [(k + "_t1" if k.startswith("collision") else k, h, d) for k, h, d in PLAIN]

BASE_CAPTION = (r"Evaluation of safety-critical scenarios initialized using "
                r"different scene initialization methods on different driving "
                r"policies.")
GATE_CAPTION = (
    r"Every rate is measured on the VALID scenes of its own row: those whose ego "
    r"interpenetrates no vehicle at $t=0$, since such a scene has its collision, its "
    r"time-to-collision and its arrival decided before any planner acts. ")
T1_CAPTION = (
    r"Both collision columns additionally require the contact to have happened at "
    r"least $1\,$s after the start, so Coll.$_{\text{ego}}$ stays a subset of Coll. "
    r"This is a robustness cut rather than a second artifact filter: the validity "
    r"gate already removed the scenes that begin in contact, and what this removes "
    r"on top is a legally placed adversary the ego had no room to react to.")
SE_CAPTION = (
    r"Each cell is $\text{value}_{\pm\text{SE}}$, the standard deviation of the cell "
    r"over {n} bootstrap resamples of the 1000 evaluated scenes, drawn jointly across "
    r"a row's twelve cells so the \textit{Overall} block carries their correlation. "
    r"\textit{Overall} is the unweighted mean of those twelve cells. Bold is the best "
    r"method within each (ego $\times$ metric) column.")

TABLES = [("table_main_final_original.tex", "tab:main", None, PLAIN, ""),
          ("table_main_final_remove_collid.tex", "tab:main_valid", GATE, PLAIN,
           GATE_CAPTION),
          ("table_main_final_collid_1s.tex", "tab:main_valid_t1", GATE, T1,
           GATE_CAPTION + T1_CAPTION)]


def collect(summaries, clusters, owner, metrics, gate):
    """``cells[label][ego][traffic][key] = (value, sd)`` plus an ``Overall`` traffic.

    A row's twelve cells have to share one bootstrap cluster or their mean cannot be
    taken inside a resample; that is asserted rather than assumed."""
    out: dict = {}
    for label, src, template in ROWS:
        summary = summaries[src]
        d = summary["npz"]
        out[label] = {e: {} for e, _ in EGO}
        per_row: dict[str, list] = {}
        names = set()
        for ego, _ in EGO:
            for traffic, _ in TRAFFIC:
                col = cell_column(summary, template, ego, traffic)
                j = summary["checkpoint"].index(col)
                name = owner[(src, j)]
                names.add(name)
                pos = clusters[name]["col_pos"][j]
                cell = {}
                for key, _, _ in metrics:
                    value, _ = cell_estimate(d, key, j, gate)
                    draws = clusters[name]["draws"][key][:, pos]
                    cell[key] = (value, boot_stats(draws, name)["sd"])
                    per_row.setdefault(key, []).append(draws)
                out[label][ego][traffic] = cell
        if len(names) != 1:
            raise SystemExit(f"{label}: its twelve cells span clusters {sorted(names)}, "
                             "so an Overall cannot be taken under one resample")
        out[label]["Overall"] = {
            key: (float(np.mean([out[label][e][t][key][0]
                                 for e, _ in EGO for t, _ in TRAFFIC])),
                  float(np.mean(per_row[key], axis=0).std(ddof=1)))
            for key, _, _ in metrics}
    return out


def valid_note(summaries, gate) -> str:
    """The Valid column the template has no room for, compressed into a sentence."""
    if gate is None:
        return ""
    lo, hi, worst = 100.0, 0.0, ""
    for label, src, template in ROWS:
        summary = summaries[src]
        vals = [cell_estimate(summary["npz"], gate,
                              summary["checkpoint"].index(
                                  cell_column(summary, template, e, t)), gate)[0]
                for e, _ in EGO for t, _ in TRAFFIC]
        if min(vals) < lo:
            lo, worst = min(vals), LABEL[label]
        hi = max(hi, max(vals))
    return (rf"The gate costs little: the share of valid scenes runs "
            rf"{lo:.1f}--{hi:.1f}\% over all rows and cells, its minimum on {worst}. ")


def render(cells, metrics, *, label: str, caption: str) -> str:
    keys = [k for k, _, _ in metrics]
    span = len(keys)
    head = " & ".join(h for _ in EGO for _, h, _ in metrics)

    def block(traffic: str, traffic_label: str) -> list[str]:
        """One traffic band, or the Overall band when ``traffic`` is ``"Overall"``.

        Overall has no ego axis -- each metric spans the three ego columns -- so it
        is one comparison per metric rather than one per (ego, metric)."""
        overall = traffic == "Overall"
        get = ((lambda lab, ego, key: cells[lab]["Overall"][key]) if overall
               else (lambda lab, ego, key: cells[lab][ego][traffic][key]))
        best = {}
        for key, _, direction in metrics:
            for ego, _ in EGO:
                vals = [get(lab, ego, key)[0] for lab, _, _ in ROWS]
                best[(ego, key)] = (min if direction == "min" else max)(vals)
        lines = [rf"\multirow{{{len(ROWS)}}}{{*}}{{{traffic_label}}}", ""]
        for lab, _, _ in ROWS:
            ego0 = EGO[0][0]
            if overall:
                body = " & ".join(
                    rf"\multicolumn{{{len(EGO)}}}{{c}}"
                    rf"{{${_fmt(get(lab, ego0, key), best[(ego0, key)])}$}}"
                    for key, _, _ in metrics)
            else:
                body = " & ".join(
                    f"${_fmt(get(lab, ego, key), best[(ego, key)])}$"
                    for ego, _ in EGO for key, _, _ in metrics)
            lines += [rf"& {LABEL[lab]}", f"& {body} " + r"\\", ""]
        return lines

    body: list[str] = []
    for traffic, traffic_label in TRAFFIC:
        body += block(traffic, traffic_label)
        body.append(r"\midrule")
    body.append(r"\midrule")
    body += ["& & " + " & ".join(rf"\multicolumn{{{len(EGO)}}}{{c}}{{{h}}}"
                                 for _, h, _ in metrics) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + i * len(EGO)}-{2 + (i + 1) * len(EGO)}}}"
                     for i in range(span)), ""]
    body += block("Overall", r"\textit{Overall}")
    body = body[:-1]

    return "\n".join([
        r"% Generated by scripts/emit_table_main_template.py -- do not hand-edit.",
        r"% Rows = (pi_traffic, scene init); columns = pi_ego x metric.",
        r"% Overall = unweighted mean of the 12 (traffic x ego) cells of the row.",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{ll *{{{len(EGO)}}}{{{'c' * span}}}}}",
        r"\toprule",
        r"\multirow{3}{*}{$\textcolor{MyBlue}{\pi_{\mathrm{traffic}}}, "
        r"\textcolor{MyGreen}{\pi_{\mathrm{adv}}}$}&",
        r"\multirow{3}{*}{Scene Initialization} &",
        rf"\multicolumn{{{span * len(EGO)}}}{{c}}"
        r"{$\textcolor{MyRed}{\pi_{\mathrm{ego}}}$} \\",
        rf"\cmidrule(lr){{3-{2 + span * len(EGO)}}}",
        "",
        "& &",
        " &\n".join(rf"\multicolumn{{{span}}}{{c}}{{{n}}}" for _, n in EGO) + r" \\",
        "",
        "\n".join(rf"\cmidrule(lr){{{3 + i * span}-{2 + (i + 1) * span}}}"
                  for i in range(len(EGO))),
        "",
        "& & " + head + r" \\",
        r"\midrule",
        "",
        *body,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ])


def _fmt(cell: tuple[float, float], best: float) -> str:
    value, sd = cell
    txt = f"{value:.2f}"
    if abs(value - best) < 5e-3:
        txt = rf"\bm{{{txt}}}"
    return txt + rf"_{{\pm {sd:.2f}}}"


def check_original(cells, path: Path) -> None:
    """The unconditional table must reproduce the pre-gate csv exactly."""
    want = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["traffic"] != "Average":
                want[(r["method"], r["ego"], r["traffic"], r["metric"])] = float(r["value"])
    bad = []
    for (method, ego, traffic, key), v in want.items():
        if key not in {k for k, _, _ in PLAIN}:
            continue
        got = cells[method][ego][traffic][key][0]
        if abs(got - v) > 5e-9:
            bad.append(f"{method}/{ego}/{traffic}/{key}: {got} != {v}")
    if bad:
        raise SystemExit(f"unconditional rendering differs from {path}:\n" +
                         "\n".join(bad[:10]))
    print(f"[template] unconditional table reproduces {path} on {len(want)} cells")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/final/cache")
    ap.add_argument("--out", default="temp")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--check-original", default="data/final/table_main/table_main.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(Path(args.root))
    for fname, tex_label, gate, metrics, tail in TABLES:
        clusters, owner = build_clusters(summaries, args.bootstrap, args.boot_seed, gate)
        cells = collect(summaries, clusters, owner, metrics, gate)
        if gate is None and args.check_original:
            check_original(cells, Path(args.check_original))
        # Joined rather than concatenated: the parts each end differently and a
        # caption is one paragraph, so normalise the seams instead of hand-placing
        # a space in every one of them.
        caption = " ".join(part.strip() for part in
                           (BASE_CAPTION, tail, valid_note(summaries, gate),
                            SE_CAPTION.replace("{n}", str(args.bootstrap)))
                           if part.strip())
        (out / fname).write_text(render(cells, metrics, label=tex_label,
                                        caption=caption))
        print(f"[template] {out / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
