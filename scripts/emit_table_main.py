#!/usr/bin/env python
"""Build the main table: scene-initialization methods x planner combinations.

Two stages, deliberately separate. First every cell is resolved out of the
rollout summaries into ONE tidy csv (plus an npz that keeps exact dtypes and
inf). Then the LaTeX is rendered from that csv and nothing else. So a number in
the table can always be traced: table cell -> csv row -> the summary column it
names -> that column's per-scene ``metrics.npz``. Nothing is transcribed by hand.

Rows are (ego planner, scene-initialization method); columns are (traffic
planner, metric) plus an Average over the four traffic planners.

Which summary each method comes from:

    Log                       scenario_log/        log@<ego>-<traffic>
    SceneControl              scenario/            scenecontrol@<ego>-<traffic>
    AdvScene-Base null        scenario/            base_null@<ego>-<traffic>
    AdvScene-Base cond        scenario/            base@<ego>-<traffic>
    AdvScene-RL (init scene)  scenario/            main_<pair>
    AdvScene-RL (init agent)  scenario_init_agent/ main_<pair>
    AdvScene-RL (init adv)    scenario_init_adv/   main_<pair>

The three AdvScene-RL rows read the checkpoint trained FOR that cell, so each is
its own home fixture. The other four rows are one artifact scored under every
pair. Both are what the corresponding paper row claims.

Usage (env vars from scripts/define_env_variables.sh must be set)::

    .venv/bin/python scripts/emit_table_main.py --root data/final/cache \
        --out data/final/table_main

``--bootstrap N`` adds a second set of files carrying a spread per cell:
``table_main_pm.csv`` / ``.npz`` and two renderings, the full six metrics
(``table_main_pm.tex``) and a narrower four (``table_main_pm_narrow.tex``). The
spread is a bootstrap over the evaluated SCENES, not over generation seeds --
see ``build_clusters`` for why the resample is per scene set and shared across a
group's columns. The point-estimate files are re-derived and checked, never
rewritten, so both tables provably quote the same numbers.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

EGO = [("idm", "IDM"), ("pdm", "PDM"), ("ppo_normal", r"$\mathrm{PPO}_{\text{norm}}$")]
TRAFFIC = [("idm", "IDM"), ("ppo_aggressive", r"$\mathrm{PPO}_{\text{aggr}}$"),
           ("ppo_normal", r"$\mathrm{PPO}_{\text{norm}}$"),
           ("ppo_caution", r"$\mathrm{PPO}_{\text{caut}}$")]

# (metric key in summary.npz, header, better direction). ``None`` means the
# column is a diagnostic and is never bolded: off-road is not a criticality
# measure -- a generator can raise it with implausible geometry -- and here the
# scene sources do not even share a map distribution (SceneControl conditions on
# real val lanes, the AdvScene rows generate their own), so the proxy is not
# comparable down the column.
METRICS = [
    ("succ", r"Succ. $\downarrow$", "min"),
    ("offroad", r"Off.", None),
    ("collision", r"Coll. $\uparrow$", "max"),
    ("collision_ego_fault", r"Coll.$_{\text{ego}}$ $\uparrow$", "max"),
    ("ttc_lt_3s", r"TTC$_{<3s}$ $\uparrow$", "max"),
    ("ttc_lt_1p5s", r"TTC$_{<1.5s}$ $\uparrow$", "max"),
]

# The +- table is 1.9x the width of the plain one at the same font, and the plain
# one is already inside a \resizebox. This subset keeps it near the current width:
# ``offroad`` is the diagnostic column that is never bolded, and ``ttc_lt_1p5s``
# repeats ``ttc_lt_3s`` at a tighter threshold. Both tables come from the same csv.
NARROW = [m for m in METRICS
          if m[0] in ("succ", "collision", "collision_ego_fault", "ttc_lt_3s")]

# (row label, summary root, column template). ``{pair}`` is filled with the
# checkpoint trained for the cell; ``{ego}``/``{traffic}`` with the planners.
ROWS = [
    ("Log (closest agent as adversary)", "scenario_log", "log@{ego}-{traffic}"),
    ("SceneControl", "scenario", "scenecontrol@{ego}-{traffic}"),
    ("AdvScene-Base (null)", "scenario", "base_null@{ego}-{traffic}"),
    ("AdvScene-Base (cond)", "scenario", "base@{ego}-{traffic}"),
    ("AdvScene-RL (init scene)", "scenario", "{pair}"),
    ("AdvScene-RL (init agent)", "scenario_init_agent", "{pair}"),
    ("AdvScene-RL (init adv)", "scenario_init_adv", "{pair}"),
]


def load_summaries(root: Path) -> dict[str, dict]:
    out = {}
    for name in {r[1] for r in ROWS}:
        path = root / name / "summary.npz"
        if not path.exists():
            raise SystemExit(f"missing {path}; run the rollout batch for '{name}' first")
        d = np.load(path, allow_pickle=True)
        out[name] = {"npz": d, "checkpoint": list(d["checkpoint"]),
                     "sut": list(d["sut"]), "env": list(d["env"])}
    return out


def cell_column(summary: dict, template: str, ego: str, traffic: str) -> str:
    """The summary column a cell reads, resolved and checked to exist.

    ``{pair}`` is looked up by planner rather than by name so the cache's naming
    convention (``main_ppo-ppo_norm`` for ppo_normal/ppo_normal) never has to be
    reproduced here."""
    if "{pair}" in template:
        hits = [c for c, s, e in zip(summary["checkpoint"], summary["sut"], summary["env"])
                if c.startswith("main_") and s == ego and e == traffic]
        if len(hits) != 1:
            raise SystemExit(f"expected one main_* column for {ego}/{traffic}, got {hits}")
        return hits[0]
    col = template.format(ego=ego, traffic=traffic)
    if col not in summary["checkpoint"]:
        raise SystemExit(f"column {col!r} not in this summary")
    return col


def build_clusters(summaries: dict[str, dict], n_boot: int, seed: int) -> tuple[dict, dict]:
    """One resample of the SCENES per (summary root, finite-support) group.

    A root's columns do not all cover the same scenes: ``scenario_log`` stacks
    the 1000 log scenes and the 1000 generated ones into 2000 rows with
    complementary NaN support. And three roots reuse the same positional ids
    (``0_0``, ``0_1``, ...) for scenes that are NOT the same -- ``init_agent``
    and ``init_adv`` are the val scenes named by ``--val-index``, while
    ``scenario`` is drawn from the layout prior and corresponds to no val scene.
    So a draw is only ever shared inside one group, never across roots.

    Sharing it across the group's columns is the point: the Average column is a
    mean over four traffic planners scored on the SAME scenes, whose per-scene
    outcomes correlate at phi ~ 0.47, so resampling each column independently
    would understate its spread. Resampling scenes reproduces that correlation.

    The resample is done as multinomial counts rather than index gathering --
    identical in distribution, one matmul per (group, metric) instead of a
    [n_boot, n, cols] gather."""
    rng = np.random.default_rng(seed)
    clusters: dict[str, dict] = {}
    owner: dict[tuple[str, int], str] = {}
    # sorted, not ``summaries.items()``: ``load_summaries`` builds its dict from a
    # set of root names, so the iteration order varies with the interpreter's hash
    # seed. Unsorted, each run would hand a different draw to each cluster and
    # ``--boot-seed`` would not reproduce anything.
    for src, summary in sorted(summaries.items()):
        d = summary["npz"]
        groups: dict[bytes, list[int]] = {}
        for j in range(len(summary["checkpoint"])):
            masks = [np.isfinite(d[key][:, j]) for key, _, _ in METRICS]
            if not all(np.array_equal(m, masks[0]) for m in masks):
                raise SystemExit(f"{src}/{summary['checkpoint'][j]}: the six metrics "
                                 "do not agree on which scenes they cover")
            groups.setdefault(masks[0].tobytes(), []).append(j)
        for gi, (pat, cols) in enumerate(groups.items()):
            mask = np.frombuffer(pat, dtype=bool)
            n = int(mask.sum())
            name = f"{src}#{gi}"
            counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)
            draws = {}
            for key, _, _ in METRICS:
                x = d[key][mask][:, cols]
                # The table's point estimate is a nanmean over the whole column;
                # the bootstrap runs on the support. They have to be the same
                # number, or the +- would belong to a different quantity.
                point = 100.0 * np.nanmean(d[key][:, cols], axis=0)
                if not np.allclose(point, 100.0 * x.mean(axis=0), atol=1e-9, rtol=0):
                    raise SystemExit(f"{name}/{key}: support mean != column nanmean")
                draws[key] = 100.0 * (counts @ x) / n
            clusters[name] = {"root": src, "n": n, "col_pos": {c: i for i, c in enumerate(cols)},
                              "draws": draws}
            for c in cols:
                owner[(src, c)] = name
    return clusters, owner


def boot_stats(draws: np.ndarray, cluster: str) -> dict:
    """``sd`` is the standard error of the cell: the spread of the estimate over
    resampled scene sets. It is NOT a seed-to-seed spread, and the caption says so."""
    return {"sd": float(draws.std(ddof=1)), "boot_mean": float(draws.mean()),
            "ci_lo": float(np.percentile(draws, 2.5)),
            "ci_hi": float(np.percentile(draws, 97.5)), "cluster": cluster}


def build_records(summaries: dict[str, dict], clusters: dict | None = None,
                  owner: dict | None = None) -> list[dict]:
    """One record per (ego, method, traffic-or-Average, metric). Long form: it
    pivots into any table shape and cannot silently transpose."""
    records = []
    for ego, _ in EGO:
        for label, src, template in ROWS:
            summary = summaries[src]
            d = summary["npz"]
            per_traffic, per_traffic_boot, cell_clusters = {}, {}, set()
            for traffic, _ in TRAFFIC:
                col = cell_column(summary, template, ego, traffic)
                j = summary["checkpoint"].index(col)
                for key, _, _ in METRICS:
                    value = 100.0 * float(np.nanmean(d[key][:, j]))
                    per_traffic.setdefault(key, []).append(value)
                    rec = {"ego": ego, "method": label, "traffic": traffic,
                           "metric": key, "value": value,
                           "source": f"{src}/{col}",
                           "n": int(np.isfinite(d[key][:, j]).sum())}
                    if clusters:
                        name = owner[(src, j)]
                        cell_clusters.add(name)
                        draws = clusters[name]["draws"][key][:, clusters[name]["col_pos"][j]]
                        per_traffic_boot.setdefault(key, []).append(draws)
                        rec.update(boot_stats(draws, name))
                    records.append(rec)
            for key, _, _ in METRICS:
                rec = {"ego": ego, "method": label, "traffic": "Average",
                       "metric": key, "value": float(np.mean(per_traffic[key])),
                       "source": "mean of the four traffic planners", "n": ""}
                if clusters:
                    # The four traffic columns of a method live in one group, so
                    # the average is taken WITHIN a resampled scene set.
                    if len(cell_clusters) != 1:
                        raise SystemExit(f"{label}/{ego}: the four traffic columns span "
                                         f"{sorted(cell_clusters)}, so they cannot be averaged "
                                         "under one resample")
                    rec.update(boot_stats(np.mean(per_traffic_boot[key], axis=0),
                                          next(iter(cell_clusters))))
                records.append(rec)
    return records


PLAIN_FIELDS = ["ego", "method", "traffic", "metric", "value", "n", "source"]
PM_FIELDS = ["ego", "method", "traffic", "metric", "value", "sd", "boot_mean",
             "ci_lo", "ci_hi", "n", "cluster", "source"]


def csv_text(records: list[dict], fields: list[str]) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in records:
        row = {k: r[k] for k in fields if k in r}
        for k in ("value", "sd", "boot_mean", "ci_lo", "ci_hi"):
            if k in row:
                row[k] = f"{row[k]:.4f}"
        w.writerow(row)
    return out.getvalue()


def read_csv(path: Path) -> dict[tuple, float]:
    """The table is rendered from the csv, not from the arrays that made it."""
    with open(path, newline="") as f:
        return {(r["ego"], r["method"], r["traffic"], r["metric"]): float(r["value"])
                for r in csv.DictReader(f)}


def read_pm_csv(path: Path) -> tuple[dict[tuple, float], dict[tuple, float]]:
    """Same discipline for the +- table: values and spreads both come back out
    of the csv, so a rendered cell is always one csv row."""
    values, sds = {}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            k = (r["ego"], r["method"], r["traffic"], r["metric"])
            values[k], sds[k] = float(r["value"]), float(r["sd"])
    return values, sds


CAPTION = (
    r"Evaluation of different scene initialization methods on different "
    r"planner combinations. Collision is ego vs the generated adversary; "
    r"Coll.$_{\text{ego}}$ counts only contacts the ego's own front face made. "
    r"TTC columns count scenes whose minimum ego-to-adversary time-to-collision "
    r"fell below the threshold.")

# Only true of a rendering that actually carries the column.
OFFROAD_NOTE = (
    r" Off. is a diagnostic and is not bolded: the rows "
    r"do not share a map distribution, so the off-road proxy is not comparable "
    r"down a column.")


def render_tex(table: dict[tuple, float], *, metrics: list = METRICS,
               pm: dict[tuple, float] | None = None, src_csv: str = "table_main.csv",
               tex_label: str = "tab:full_matrix", caption_tail: str = "") -> str:
    cols = [t for t, _ in TRAFFIC] + ["Average"]
    span = len(metrics)
    head_groups = " &\n".join(
        rf"\multicolumn{{{span}}}{{c}}{{{name}}}"
        for name in [n for _, n in TRAFFIC] + [r"\textit{Average}"])
    cmid = "\n".join(rf"\cmidrule(lr){{{3 + i * span}-{2 + (i + 1) * span}}}"
                     for i in range(len(cols)))
    metric_head = " & ".join(h for _ in cols for _, h, _ in metrics)
    caption = CAPTION + (OFFROAD_NOTE if any(k == "offroad" for k, _, _ in metrics)
                         else "") + caption_tail

    body = []
    for ego, ego_label in EGO:
        body.append(rf"\multirow{{{len(ROWS)}}}{{*}}{{{ego_label}}}")
        body.append("")
        best = {}
        for traffic in cols:
            for key, _, direction in metrics:
                if direction is None:
                    continue
                vals = [table[(ego, label, traffic, key)] for label, _, _ in ROWS]
                best[(traffic, key)] = (min if direction == "min" else max)(vals)
        for label, _, _ in ROWS:
            cells = []
            for traffic in cols:
                for key, _, direction in metrics:
                    v = table[(ego, label, traffic, key)]
                    txt = f"{v:.2f}"
                    if direction is not None and abs(v - best[(traffic, key)]) < 5e-3:
                        txt = rf"\mathbf{{{txt}}}"
                    if pm is not None:
                        txt += rf"_{{\pm {pm[(ego, label, traffic, key)]:.2f}}}"
                    cells.append(f"${txt}$")
            body.append(f"& {label}")
            body.append("& " + " & ".join(cells) + r" \\")
            body.append("")
        body.append(r"\midrule")
    body = body[:-1]  # the last group ends with \bottomrule instead

    return "\n".join([
        rf"% Generated by scripts/emit_table_main.py from {src_csv} -- do not",
        r"% hand-edit: rerun the script instead, so every number keeps its trace",
        r"% back to a rollout summary column.",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{tex_label}}}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{ll *{{{len(cols)}}}{{{'c' * span}}}}}",
        r"\toprule",
        r"\multirow{3}{*}{$\mathrm{Planner}_{ego}$} &",
        r"\multirow{3}{*}{Scene Initialization} &",
        rf"\multicolumn{{{span * len(cols)}}}{{c}}{{$\mathrm{{Planner}}_{{traffic}} + "
        rf"\mathrm{{Planner}}_{{adv}}$}} \\",
        rf"\cmidrule(lr){{3-{2 + span * len(cols)}}}",
        "",
        "& &",
        head_groups + r" \\",
        "",
        cmid,
        "",
        "& & " + metric_head + r" \\",
        r"\midrule",
        "",
        *body,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])


def write_checked(path: Path, text: str) -> str:
    """Write, unless the file already says something else.

    ``--bootstrap`` re-derives the point-estimate files so the +- table can be
    checked against them cell for cell. They are deterministic, so reproducing
    them must be a no-op; if it is not, the summaries moved and the two tables
    would disagree. That is worth stopping on rather than overwriting."""
    # Bytes, not text: csv writes CRLF and ``read_text`` would translate it away,
    # so a text comparison reports a difference that is not there.
    data = text.encode()
    if path.exists():
        if path.read_bytes() == data:
            return "unchanged"
        raise SystemExit(f"{path} already holds different content -- the rollout "
                         "summaries changed since it was written. Move the old "
                         "directory aside instead of overwriting it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "written"


def npz_arrays(records: list[dict], fields: tuple[str, ...]) -> dict:
    return {k: np.array([r[k] for r in records]) for k in fields}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/final/cache")
    ap.add_argument("--out", default="data/final/table_main")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="scene-cluster bootstrap replicates; 0 emits only the plain table")
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)

    summaries = load_summaries(root)
    clusters, owner = ({}, {})
    if args.bootstrap:
        clusters, owner = build_clusters(summaries, args.bootstrap, args.boot_seed)
        for name, c in clusters.items():
            print(f"[boot] {name}: n={c['n']} scenes, {len(c['col_pos'])} columns")
    records = build_records(summaries, clusters, owner)

    plain = csv_text(records, PLAIN_FIELDS)
    print(f"[table] table_main.csv {write_checked(out / 'table_main.csv', plain)}")
    npz = out / "table_main.npz"
    if npz.exists():
        # A pure function of the records the csv above was just checked against,
        # so rewriting it could only change the zip's timestamps.
        print("[table] table_main.npz unchanged")
    else:
        np.savez_compressed(npz, **npz_arrays(
            records, ("ego", "method", "traffic", "metric", "value", "source")))
        print("[table] table_main.npz written")
    tex = render_tex(read_csv(out / "table_main.csv")) + "\n"
    print(f"[table] table_main.tex {write_checked(out / 'table_main.tex', tex)}")
    prov = json.dumps({
        "rows": [{"label": r[0], "summary": r[1], "column": r[2]} for r in ROWS],
        "metrics": [{"key": k, "header": h, "bold": d} for k, h, d in METRICS],
        "ego": [e for e, _ in EGO], "traffic": [t for t, _ in TRAFFIC],
    }, indent=1)
    print(f"[table] PROVENANCE.json {write_checked(out / 'PROVENANCE.json', prov)}")

    if not args.bootstrap:
        print(f"[table] {len(records)} records -> {out}/table_main.csv, .npz, .tex")
        return 0

    n_scenes = sorted({c["n"] for c in clusters.values()})
    tail = (rf" Each cell is $\text{{value}}_{{\pm\text{{SE}}}}$. The SE is the "
            rf"standard deviation of the cell over {args.bootstrap} bootstrap "
            rf"resamples of the {n_scenes[0]} evaluated SCENES, drawn jointly "
            r"across the four traffic planners so the Average column carries "
            r"their correlation. It is the sampling error of this scene set, "
            r"not a spread over generation seeds.")
    (out / "table_main_pm.csv").write_bytes(csv_text(records, PM_FIELDS).encode())
    np.savez_compressed(out / "table_main_pm.npz", **npz_arrays(
        records, ("ego", "method", "traffic", "metric", "value", "sd", "boot_mean",
                  "ci_lo", "ci_hi", "cluster", "source")))
    values, sds = read_pm_csv(out / "table_main_pm.csv")
    (out / "table_main_pm.tex").write_text(render_tex(
        values, pm=sds, src_csv="table_main_pm.csv", tex_label="tab:full_matrix_pm",
        caption_tail=tail) + "\n")
    (out / "table_main_pm_narrow.tex").write_text(render_tex(
        values, metrics=NARROW, pm=sds, src_csv="table_main_pm.csv",
        tex_label="tab:full_matrix_pm_narrow",
        caption_tail=tail + r" Off. and TTC$_{<1.5s}$ are dropped here for width; "
        r"both carry the same spread in Table~\ref{tab:full_matrix_pm}.") + "\n")
    (out / "PROVENANCE_pm.json").write_text(json.dumps({
        "n_boot": args.bootstrap, "seed": args.boot_seed,
        "statistic": "bootstrap SD of the cell = standard error over resampled scene sets",
        "resample_unit": "scene, drawn once per cluster and shared by that cluster's columns",
        "clusters": {n: {"root": c["root"], "num_scenes": c["n"],
                         "num_columns": len(c["col_pos"])} for n, c in clusters.items()},
        "tables": {"table_main_pm.tex": [k for k, _, _ in METRICS],
                   "table_main_pm_narrow.tex": [k for k, _, _ in NARROW]},
    }, indent=1))
    print(f"[table] {len(records)} records -> {out}/table_main_pm.csv, .npz, "
          ".tex, _narrow.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
