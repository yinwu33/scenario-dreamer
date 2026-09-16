#!/usr/bin/env python
"""Render the two scene-quality tables in the shape ``temp/*_before.tex`` fixes.

    agent -> temp/table_scene_gen_agent.tex
    lane  -> temp/table_scene_gen_lane.tex

Two stages, as in ``scripts/emit_table_main.py``: every cell is resolved into one
tidy csv, then the LaTeX is rendered from that csv and nothing else, so a number
traces back to a row of the metrics json and the cache that produced it.

Which scorer, and why it matters
--------------------------------
The numbers come from ``scripts/score_scene_gen_table.py``, i.e. from
``compute_lane_metrics`` + ``compute_agent_metrics`` + ``compute_goal_metrics``,
which is what produced the values already in the templates. NOT from
``eval_scene.py``: that one splits the agent statistics into normal-agents and
adversary and reports neither the goal JSDs, the collision rate, nor the route
and endpoint standard deviations, so its agent columns are a DIFFERENT quantity
and cannot be dropped into these tables. The agent columns here therefore POOL
every vehicle, the generated adversary included.

The two meanings of +- in these tables
--------------------------------------
They are not the same statistic and the captions say so:

  Route Length, Endpoint Dist.  ``*_std`` -- the spread ACROSS SCENES within one
                                sample set, a property of the generated
                                distribution itself.
  every other column            the spread ACROSS THE MODELS the row pools, i.e.
                                over adversary configurations. Neither a
                                sampling error nor a seed variance.

An AdvScene-RL cell in the two route columns keeps the first meaning, since all
twelve models share one stage-1 lane sample and the within-sample spread is the
quantity worth printing there.

Rows this script does not measure
---------------------------------
DriveSceneGen is carried verbatim from the templates and flagged ``lit`` in the
csv. ScenarioDreamer and SceneControl are measured from their generated caches.

Usage (env vars from scripts/define_env_variables.sh must be set)::

    .venv/bin/python scripts/emit_scene_gen_tables.py \
        --metrics data/final/scene_gen/metrics_full.json --out temp
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

PAIRS = ["idm-idm", "idm-ppo_aggressive", "idm-ppo_caution", "idm-ppo_norm",
         "pdm-idm", "pdm-ppo_aggressive", "pdm-ppo_caution", "pdm-ppo_norm",
         "ppo-idm", "ppo-ppo_aggressive", "ppo-ppo_caution", "ppo-ppo_norm"]

AGENT_COLS = [("nearest_dist_jsd", "Near. Dist."), ("lat_dev_jsd", "Lat. Dev."),
              ("ang_dev_jsd", "Ang. Dev."), ("length_jsd", "Length"),
              ("width_jsd", "Width"), ("speed_jsd", "Speed")]
GOAL_COLS = [("goal_dist_jsd", "Dist."), ("goal_lat_dev_jsd", "Lat. Dev."),
             ("goal_ang_dev_jsd", "Ang. Dev."), ("goal_offroad_rate", "Off-road")]
COLL_KEY, COLL_HEAD = "collision_rate", r"Collision Rate (\%) $\downarrow$"
AGENT_KEYS = [k for k, _ in AGENT_COLS + GOAL_COLS] + [COLL_KEY]

# (key, header, paired std key or None). The two paired columns print
# ``mean +- std`` where the std is the within-sample spread across scenes.
LANE_COLS = [("route_length_mean", "Route Length", "route_length_std"),
             ("endpoint_dist_mean", r"Endpoint Dist. $\downarrow$", "endpoint_dist_std"),
             ("frechet_connectivity", r"Conn. $\downarrow$", None),
             ("frechet_density", r"Dens. $\downarrow$", None),
             ("frechet_reach", r"Reach $\downarrow$", None),
             ("frechet_convenience", r"Conve. $\downarrow$", None)]
LANE_KEYS = [k for k, _, _ in LANE_COLS] + [s for _, _, s in LANE_COLS if s]

# Carried verbatim from temp/table_scene_gen_*_before.tex. Not scored here.
LIT = {
    "DriveSceneGen": {
        "agent": {"nearest_dist_jsd": 0.63, "lat_dev_jsd": 1.01, "ang_dev_jsd": 2.43,
                  "length_jsd": 58.86, "width_jsd": 54.51, "speed_jsd": 18.70,
                  "collision_rate": 0.2},
        "lane": {"route_length_mean": 41.61, "route_length_std": 18.61,
                 "endpoint_dist_mean": 0.01, "endpoint_dist_std": 0.00,
                 "frechet_connectivity": 4.53, "frechet_density": 1.18,
                 "frechet_reach": 0.64, "frechet_convenience": 5.58},
    },
}
PARAMS = {"DriveSceneGen": "-", "SceneControl": "-", "ScenarioDreamer": "$376M$",
          "AdvScene-Base": "$236M$", "AdvScene-RL (uncond)": "$236M$",
          "AdvScene-RL": "$236M$"}

# (model, label as printed, Goals column, source). ``source``: "lit" carries LIT,
# a list pools those metric-json rows.
ROWS = [
    ("DriveSceneGen", r"DriveSceneGen~\cite{sun2024drivescenegen}", r"$\times$", "lit"),
    ("SceneControl", r"SceneControl~\cite{lu2024scenecontrol}", r"$\checkmark$",
     ["scenecontrol/init_agent"]),
    ("ScenarioDreamer", r"ScenarioDreamer~\cite{rowe2025scenario}", r"$\checkmark$",
     ["scenario-dreamer/init_scene"]),
    # Base is the UNCONDITIONAL cache, by choice, and the two RL rows are the
    # same twelve policies sampled at the two protocols. AdvScene-RL (uncond)
    # shares the base row's protocol, so that pair differs by the RL stage and
    # nothing else; AdvScene-RL is sampled at cond_adv_ego, the protocol DDPO
    # trains under, so its goal columns carry the conditioning as well.
    ("AdvScene-Base", "AdvScene-Base", r"$\checkmark$", ["base_null/init_scene"]),
    ("AdvScene-RL (uncond)", "AdvScene-RL (uncond)", r"$\checkmark$",
     [f"main_{p}_null/init_scene" for p in PAIRS]),
    ("AdvScene-RL", "AdvScene-RL", r"$\checkmark$",
     [f"main_{p}/init_scene" for p in PAIRS]),
]
# SceneControl has no lane-connectivity head (nn_modules.dm.DM.decode_outputs reads
# the graph off its conditioning scene), so a free-map cache would be edgeless and
# every lane column is a function of that graph. Measured, not missing.
LANE_NA = {"SceneControl"}


def pooled(rows: dict, keys: list, wanted: list) -> dict[str, tuple[float, float]]:
    """(mean, spread across the pooled caches) per metric. A metric absent from
    any pooled cache is dropped rather than averaged over a subset."""
    missing = [k for k in keys if k not in rows]
    if missing:
        raise SystemExit(f"metrics json has no row for {missing}")
    out = {}
    for key in wanted:
        vals = [rows[k][key] for k in keys if key in rows[k]]
        if len(vals) != len(keys):
            continue
        v = np.array(vals, dtype=float)
        out[key] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    return out


def build_records(blob: dict) -> list[dict]:
    rows = blob["rows"]
    records = []
    for table, wanted in (("agent", AGENT_KEYS), ("lane", LANE_KEYS)):
        for model, label, goals, source in ROWS:
            if table == "lane" and model in LANE_NA:
                status, got = "na", {}
            elif source == "lit":
                status, got = "lit", {k: (v, 0.0) for k, v in LIT[model][table].items()}
            else:
                status, got = "ok", pooled(rows, source, wanted)
            for key in wanted:
                v, s = got.get(key, (None, None))
                records.append({
                    "table": table, "model": model, "label": label, "goals": goals,
                    "metric": key,
                    "status": status if key in got else
                              ("na" if status == "na" else "missing"),
                    "value": "" if v is None else f"{v:.6f}",
                    "spread": "" if s is None else f"{s:.6f}",
                    "n_caches": len(source) if isinstance(source, list) else 0,
                    "source": ";".join(source) if isinstance(source, list) else status})
    return records


FIELDS = ["table", "model", "label", "goals", "metric", "status", "value",
          "spread", "n_caches", "source"]


def csv_text(records: list[dict]) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(records)
    return out.getvalue()


def read_csv(path: Path) -> dict:
    """The tables are rendered from the csv, not from the arrays that made it."""
    with open(path, newline="") as f:
        return {(r["table"], r["model"], r["metric"]): r for r in csv.DictReader(f)}


# All AdvScene init_scene caches hold the SAME stage-1 lane sample, so a spread
# across them is float noise, not a measurement: 1e-4 on three Frechet columns and
# 1.3e-3 on the fourth, purely because that one is a shade less stable. Printing
# the fourth and suppressing the other three by a threshold reads as if density
# alone varied. None of them is printed.
NO_CROSS_MODEL_SPREAD = {"lane"}


def cell(table: dict, which: str, model: str, key: str, best: dict,
         std_key: str | None = None) -> str:
    """One rendered cell. ``--`` is 'this model has no such quantity' (a model
    without goals); ``$-$`` is 'not applicable', which for the lane table is the
    measured SceneControl result rather than a hole in the run."""
    rec = table.get((which, model, key))
    if rec is None or not rec["value"]:
        return "$-$" if rec is not None and rec["status"] == "na" else "--"
    v = float(rec["value"])
    txt = f"{v:.2f}"
    if key in best and abs(v - best[key]) < 5e-3:
        txt = rf"\mathbf{{{txt}}}"
    if std_key is not None:
        srec = table.get((which, model, std_key))
        if srec is not None and srec["value"]:
            return rf"${txt}{{\scriptstyle\,\pm\,{float(srec['value']):.2f}}}$"
        return f"${txt}$"
    if which not in NO_CROSS_MODEL_SPREAD and rec["spread"] and float(rec["spread"]) >= 5e-4:
        s = float(rec["spread"])
        # Keep one significant digit on a spread smaller than the value's own
        # precision, so a real but tiny spread does not print as exactly 0.00.
        return rf"${txt}{{\scriptstyle\,\pm\,{s:.{2 if s >= 5e-3 else 3}f}}}$"
    # Below the printed precision the spread is float noise across caches that
    # hold the same sample; printing "+-0.000" would dress that up as measured.
    return f"${txt}$"


def best_per_column(table: dict, which: str, keys: list) -> dict:
    """Lower is better only on the columns whose header carries an arrow. Route
    Length has none: it is an absolute quantity that wants to MATCH the log, not
    to be minimised, so a shorter one is not a better one. Only rows carrying a
    real value compete, so a placeholder never wins and a literature row does."""
    directed = {k for k, h, _ in LANE_COLS if r"\downarrow" in h}
    directed |= {k for k, _ in AGENT_COLS + GOAL_COLS} | {COLL_KEY}
    best = {}
    for key in keys:
        if key not in directed:
            continue
        vals = [float(table[(which, m, key)]["value"])
                for m, _, _, _ in ROWS
                if (which, m, key) in table
                and table[(which, m, key)]["status"] in ("ok", "lit")
                and table[(which, m, key)]["value"]]
        if vals:
            best[key] = min(vals)
    return best


HEADER_NOTE = [
    r"% Generated by scripts/emit_scene_gen_tables.py from table_scene_gen.csv --",
    r"% do not hand-edit: rerun the script instead. DriveSceneGen is carried",
    r"% from the template; every other numeric row is scored from a cache.",
]


def render_agent(table: dict) -> str:
    best = best_per_column(table, "agent", AGENT_KEYS)
    body = []
    for model, label, goals, _ in ROWS:
        body += [
            f"        {label}",
            f"        & {goals}",
            "        & " + " & ".join(cell(table, "agent", model, k, best)
                                      for k, _ in AGENT_COLS),
            "        & " + " & ".join(cell(table, "agent", model, k, best)
                                      for k, _ in GOAL_COLS),
            "        & " + cell(table, "agent", model, COLL_KEY, best) + r" \\",
            ""]
    return "\n".join([
        *HEADER_NOTE,
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \caption{Evaluation of agent initial state and goal quality on waymo "
        r"open motion test dataset. Agent columns pool every vehicle, the generated "
        r"adversary included. Both AdvScene-RL rows pool the twelve adversary "
        r"configurations and their $\pm$ is the spread across those models, which "
        r"is neither a sampling error nor a seed variance. The two differ only in "
        r"the sampling protocol: AdvScene-RL (uncond) uses the unconditional "
        r"protocol AdvScene-Base is drawn at, so that pair isolates the RL stage, "
        r"while AdvScene-RL is sampled at the conditioning its fine-tuning uses "
        r"and its goal columns therefore carry that conditioning as well. "
        r"ScenarioDreamer is the Base architecture retrained on the same goal "
        r"representation; it has neither conditioning nor a dedicated adversary "
        r"head.}",
        r"    \label{tab:scene_gen_agent}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{l c cccccc cccc c}",
        r"        \toprule",
        r"        \multirow{2}{*}{Model}",
        r"        & \multirow{2}{*}{Goals}",
        r"        & \multicolumn{6}{c}{Agent Initial State JSD $\downarrow$}",
        r"        & \multicolumn{4}{c}{Agent Goal JSD $\downarrow$}",
        rf"        & \multirow{{2}}{{*}}{{{COLL_HEAD}}} \\",
        r"        \cmidrule(lr){3-8}",
        r"        \cmidrule(lr){9-12}",
        r"        & &",
        *[f"        {'' if i == 0 else '& '}{h}"
          for i, (_, h) in enumerate(AGENT_COLS + GOAL_COLS)],
        r"        & \\",
        r"        \midrule",
        "",
        *body,
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        r"\end{table*}",
    ])


def render_lane(table: dict) -> str:
    best = best_per_column(table, "lane", [k for k, _, _ in LANE_COLS])
    body = []
    for model, label, _, _ in ROWS:
        body += [label, f"& {PARAMS[model]}"]
        body += [f"& {cell(table, 'lane', model, k, best, std)}"
                 for k, _, std in LANE_COLS]
        body[-1] += r" \\"
        body.append("")
    return "\n".join([
        *HEADER_NOTE,
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Evaluation of lane graph quality on waymo open motion test "
        r"dataset. The $\pm$ on Route Length and Endpoint Dist. is the spread "
        r"across scenes within one sample set, not a spread across models. "
        r"SceneControl is not applicable: \texttt{dm\_goal} has no "
        r"lane-connectivity head and reads the graph off its conditioning scene, "
        r"so a free-map sample carries an edgeless graph and every column here is "
        r"a function of that graph. The three AdvScene rows share one stage-1 "
        r"lane sample, since the RL stage trains the adversary branch only; "
        r"where their printed digits differ it is float noise in the "
        r"Fr\'echet estimator, not a measured difference. ScenarioDreamer is "
        r"the goal-aware Base architecture without conditioning or a dedicated "
        r"adversary head.}",
        r"\label{tab:scene_gen_lane}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l ccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{Model}",
        r"& \multirow{2}{*}{Num. Param.}",
        r"& \multirow{2}{*}{Route Length}",
        r"& \multirow{2}{*}{Endpoint Dist. $\downarrow$}",
        r"& \multicolumn{4}{c}{Urban Planning} \\",
        r"\cmidrule(lr){5-8}",
        r"& & & & ",
        *[f"{'' if i == 0 else '& '}{h}" + (r" \\" if i == len(LANE_COLS) - 3 else "")
          for i, (_, h, _) in enumerate(LANE_COLS[2:])],
        r"\midrule",
        "",
        *body,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="data/final/scene_gen/metrics_full.json")
    ap.add_argument("--out", default="temp")
    ap.add_argument("--csv", default="data/final/scene_gen/table_scene_gen.csv")
    args = ap.parse_args()

    blob = json.loads(Path(args.metrics).read_text())
    records = build_records(blob)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(csv_text(records).encode())

    table = read_csv(csv_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_scene_gen_agent.tex").write_text(render_agent(table) + "\n")
    (out / "table_scene_gen_lane.tex").write_text(render_lane(table) + "\n")
    print(f"[scene-gen] {len(records)} records -> {csv_path}")
    print(f"[scene-gen] {out}/table_scene_gen_agent.tex, table_scene_gen_lane.tex")
    print(f"[scene-gen] reference: {blob['num_gt_samples']} scenes, "
          f"{len(blob['rows'])} caches scored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
