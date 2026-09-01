"""Render the scene-generation-quality table from score_scene_gen_table.py's JSON.

The AdvScene row aggregates the eight DDPO runs as mean +- std across MODELS
(spread over adversary configurations, not a seed variance). The lane-graph
columns are frozen: DDPO trains the adversary branch only, and the adversary is
downstream-only in FactorizedDiTBlock, so every row shares one base scene. They
are printed once per row and flagged in the caption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Frozen: determined entirely by the shared stage-1 base scene.
FROZEN = ("frechet_connectivity", "frechet_density", "frechet_reach", "frechet_convenience")
FROZEN_PAIRS = (("route_length_mean", "route_length_std"),
                ("endpoint_dist_mean", "endpoint_dist_std"))
AGENT = ("nearest_dist_jsd", "lat_dev_jsd", "ang_dev_jsd", "length_jsd", "width_jsd", "speed_jsd")
GOAL = ("goal_dist_jsd", "goal_lat_dev_jsd", "goal_ang_dev_jsd", "goal_offroad_rate")
OVERLAP = "collision_rate"


def ddpo_rows(rows: dict) -> list[str]:
    """Every scored row except the two reference rows is one DDPO run."""
    return [k for k in rows if k not in ("scenario_dreamer", "advscene_base")]


def agg(rows: dict, key: str) -> tuple[float, float]:
    v = np.array([rows[r][key] for r in ddpo_rows(rows)], dtype=float)
    return float(v.mean()), float(v.std(ddof=1))


def fmt(v: float) -> str:
    return f"${v:.2f}$"


def fmt_pm(m: float, s: float) -> str:
    """Two decimals on the value; enough on the spread to keep one significant
    digit, so a small cross-model spread does not print as 0.00."""
    dec = 2 if s >= 0.005 else (3 if s >= 0.0005 else 4)
    return f"${m:.2f}{{\\scriptstyle\\,\\pm\\,{s:.{dec}f}}}$"


def row_cells(rows: dict, name: str | None) -> list[str]:
    """One table row. ``name=None`` builds the aggregated AdvScene row."""
    def one(key):
        if name is not None:
            return rows[name].get(key)
        return agg(rows, key)[0]

    def spread(key):
        return None if name is not None else agg(rows, key)[1]

    cells = []
    for mean_key, std_key in FROZEN_PAIRS:
        cells.append(fmt_pm(one(mean_key), one(std_key)))
    for k in FROZEN:
        cells.append(fmt(one(k)))
    for k in AGENT + GOAL + (OVERLAP,):
        v = one(k)
        if v is None:
            cells.append("--")
        elif name is None:
            cells.append(fmt_pm(v, spread(k)))
        else:
            cells.append(fmt(v))
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="data/scene_gen_table/metrics.json")
    ap.add_argument("--out", default="research/overleaf/things/tables/table_scene_gen.tex")
    args = ap.parse_args()

    blob = json.loads(Path(args.metrics).read_text())
    rows = blob["rows"]
    n_gen, n_gt = blob["num_samples"], blob["num_gt_samples"]

    n_runs = len(ddpo_rows(rows))
    frozen_spread = max(abs(agg(rows, k)[1]) for k in FROZEN)
    frozen_note = ("identical across all generated rows to every digit reported"
                   if frozen_spread == 0.0
                   else f"max std across the {n_runs} runs: ${frozen_spread:.1e}$")
    base = rows["advscene_base"]
    off_gt = base["goal_offroad_rate_gt"]
    parked, parked_gt = base["parked_rate"], base["parked_rate_gt"]

    body = [
        ("Scenario Dreamer~\\cite{rowe2025scenario}", "$\\times$", row_cells(rows, "scenario_dreamer")),
        ("AdvScene-base", "\\checkmark", row_cells(rows, "advscene_base")),
        ("\\textbf{AdvScene} (ours)", "\\checkmark", row_cells(rows, None)),
    ]

    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        f"    \\caption{{Initial scene generation quality. All models generate ${n_gen}$ scenes",
        f"    unconditionally from the same layout prior and are scored against the same",
        f"    ${n_gt}$ reference scenes with the same metric code. Fr\\'echet distances are",
        "    computed on four urban planning statistics of the lane graph, not on the",
        "    features of a perception model; JSD is computed on agent attribute",
        "    distributions. AdvScene aggregates the",
        f"    {n_runs} DDPO runs as mean $\\pm$ std ACROSS MODELS, i.e. the spread over adversary",
        "    configurations rather than a seed variance. DDPO trains the adversary branch",
        "    only and the adversary is downstream-only in the denoiser, so all three",
        "    generated rows share one base scene: the route, endpoint and Fr\\'echet columns",
        "    are frozen by construction and carry no model-to-model spread",
        f"    ({frozen_note}). The adversary is decoded",
        "    jointly with the other agents, so it also perturbs its neighbours; the agent",
        "    columns therefore mix its own contribution with that indirect effect. Overlap is",
        "    the fraction of generated VEHICLES intersecting another at spawn, our cheapest",
        "    realism proxy. Goal metrics are only defined for models that generate goals;",
        f"    the reference set's own goal off-road rate is ${off_gt:.2f}\\%$ and its parked rate",
        f"    ${parked_gt:.2f}\\%$ against AdvScene-base's ${parked:.2f}\\%$.}}",
        "    \\label{tab:scene_gen}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{l c cc cccc cccccc cccc c}",
        "        \\toprule",
        "        \\multirow{2}{*}{Model}",
        "        & \\multirow{2}{*}{Goals}",
        "        & \\multicolumn{6}{c}{Lane graph (frozen, see caption)}",
        "        & \\multicolumn{6}{c}{Agent JSD $\\downarrow$}",
        "        & \\multicolumn{4}{c}{Goal quality $\\downarrow$}",
        "        & \\multirow{2}{*}{Overlap $\\downarrow$} \\\\",
        "        \\cmidrule(lr){3-8}",
        "        \\cmidrule(lr){9-14}",
        "        \\cmidrule(lr){15-18}",
        "        & &",
        "        Route Len.",
        "        & Endpoint Dist. $\\downarrow$",
        "        & Conn. $\\downarrow$",
        "        & Dens. $\\downarrow$",
        "        & Reach $\\downarrow$",
        "        & Conve. $\\downarrow$",
        "        & Near. Dist.",
        "        & Lat. Dev.",
        "        & Ang. Dev.",
        "        & Length",
        "        & Width",
        "        & Speed",
        "        & Dist.",
        "        & Lat. Dev.",
        "        & Ang. Dev.",
        "        & Off-road",
        "        & \\\\",
        "        \\midrule",
        "",
    ]
    for label, goals, cells in body:
        lines.append(f"        {label}")
        lines.append(f"        & {goals}")
        lines.append("        & " + " & ".join(cells[:2]))
        lines.append("        & " + " & ".join(cells[2:6]))
        lines.append("        & " + " & ".join(cells[6:12]))
        lines.append("        & " + " & ".join(cells[12:16]))
        lines.append(f"        & {cells[16]} \\\\")
        lines.append("")
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}%",
        "    }",
        "\\end{table*}",
        "",
    ]

    out = Path(args.out)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    for label, _, cells in body:
        print(f"{label:44s} " + "  ".join(c.replace('$','').replace('\\scriptstyle\\,\\pm\\,','+-') for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
