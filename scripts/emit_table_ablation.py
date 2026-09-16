#!/usr/bin/env python
"""Render the PPO-PPO_norm ablation table in the shape of temp/table_ablation_template.tex.

Seven variants, one planner pair, three columns. Two of them are rollout rates and
carry the same scene-validity gate as the main table; the third is a generation
statistic and deliberately does NOT:

    Coll._ego   valid scenes only, and the contact at >= 1 s (--collision-min-time)
    TTC_<3s     valid scenes only
    JSD_agent   ALL scenes, read from the bootstrap JSONs

The asymmetry is the point. An invalid scene has no measurable OUTCOME -- its
collision and its TTC are decided before a planner acts -- but it is still a
generation outcome, so conditioning the realism column on it would let a model hide
the scenes it got wrong. Same reason ``path_conflict`` is not a gate.

Sources, one per row, each verified against ``init_overlap_frac`` by
``scripts/backfill_init_ego_overlap.py`` before it carried an ego column::

    w/o r_prox              scenario_reward/noprox.npz
    w/o r_ttc_ego           scenario_reward/nottc.npz
    Full = RL (cond)        scenario/main_ppo-ppo_norm/metrics.npz
    Base (uncond)           scenario/base_null@ppo_normal-ppo_normal/metrics.npz
    Base (cond)             scenario/base@ppo_normal-ppo_normal/metrics.npz
    RL (uncond)             scenario_rlnull/rollout_b128.npz

``scenario_rlnull/rollout.npz`` is NOT usable: its scenes are not the ones in
``scene/main_ppo-ppo_norm_null/init_scene`` any more (101 of 1000 differ), so it
cannot be re-gated. ``rollout_b128`` is also the batch-128 run, matching the main
table's protocol.

Usage (env vars from scripts/define_env_variables.sh must be set)::

    .venv/bin/python scripts/emit_table_ablation.py --out temp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

CACHE = ROOT / "data/final/cache"

# (row label, rollout npz, key prefix, JSD json, JSD row). ``None`` for a JSD row
# means the reference run, whose own spread the bootstrap files do not carry.
ROWS = [
    (r"w/o $r_{\mathrm{prox.}}$", "scenario_reward/noprox.npz", "init_scene/",
     "scenario_reward/bootstrap_jsd_reward.json", "Prox x"),
    (r"w/o $r_{\mathrm{ttc\_ego}}$", "scenario_reward/nottc.npz", "init_scene/",
     "scenario_reward/bootstrap_jsd_reward.json", "TTC x"),
    ("Full reward", "scenario/main_ppo-ppo_norm/metrics.npz", "",
     "scenario_rlnull/bootstrap_jsd_2x2.json", "RL + cond"),
    (None, None, None, None, None),   # \midrule between the two blocks
    ("AdvScene-Base (uncond)", "scenario/base_null@ppo_normal-ppo_normal/metrics.npz",
     "", "scenario_rlnull/bootstrap_jsd_2x2.json", "base + null"),
    ("AdvScene-Base (cond)", "scenario/base@ppo_normal-ppo_normal/metrics.npz", "",
     "scenario_rlnull/bootstrap_jsd_2x2.json", "base + cond"),
    ("AdvScene-RL (uncond)", "scenario_rlnull/rollout_b128.npz", "init_scene/",
     "scenario_rlnull/bootstrap_jsd_2x2.json", "RL + null"),
    ("AdvScene-RL (cond)", "scenario/main_ppo-ppo_norm/metrics.npz", "",
     "scenario_rlnull/bootstrap_jsd_2x2.json", "RL + cond"),
]

CAPTION = (
    r"Ablation studies on $\text{PPO}-\text{PPO}_{\text{norm}}$. "
    r"$\mathrm{Coll}_{ego}$ and $\mathrm{TTC}_{<3s}$ are measured on the VALID scenes "
    r"of their own row -- those whose ego interpenetrates no vehicle at $t=0$, since "
    r"such a scene has its outcome decided before any planner acts -- and "
    r"$\mathrm{Coll}_{ego}$ additionally requires the contact to have happened at "
    r"least $1\,$s after the start. $\mathrm{JSD}_{agent}$ is a property of the "
    r"generated scenes rather than of a rollout, so it is measured on ALL of them: "
    r"conditioning it would let a model hide the scenes it got wrong. Cells are "
    r"$\text{value}_{\pm\text{SE}}$; the rollout columns resample the 1000 evaluated "
    r"scenes {n} times, the $\mathrm{JSD}_{agent}$ column carries the 200-replicate "
    r"bootstrap of its own scoring pass. $\mathrm{Coll}_{ego}$ rests on 1--5 events "
    r"per row at this sample size and separates no two variants; read the ablation "
    r"off $\mathrm{TTC}_{<3s}$ (22--92 events) and $\mathrm{JSD}_{agent}$.")


def load(spec: str, prefix: str) -> dict[str, np.ndarray]:
    z = np.load(CACHE / spec, allow_pickle=True)
    return {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}


def rates(m: dict, min_t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(valid, Coll._ego, TTC<3s) as per-scene 0/1, before any averaging."""
    valid = m["init_ego_overlap_frac"] == 0
    fault = (m["ego_fault_collision"] > 0) & (m["ego_collision_time"] >= min_t)
    return valid, fault.astype(float), (m["ego_min_ttc"] < 3.0).astype(float)


def conditional(x: np.ndarray, valid: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    """Point estimate and bootstrap SE of ``mean(x | valid)``, in percent.

    The resample moves the DENOMINATOR as well: a draw with fewer valid scenes in it
    has to widen the rate, not merely shrink its numerator. Each row is its own scene
    set -- these seven runs generated seven different batches -- so unlike
    ``emit_table_main.build_clusters`` there is no cluster to share a draw across, and
    the counts are drawn per row."""
    point = 100.0 * float(x[valid].mean())
    draws = 100.0 * (counts @ (x * valid)) / (counts @ valid)
    return point, float(draws.std(ddof=1))


def render(rows: list, caption: str) -> str:
    best = {}
    for i, key in enumerate(("coll", "ttc", "jsd")):
        vals = [r[1][i][0] for r in rows if r[0] is not None]
        best[key] = min(vals) if key == "jsd" else max(vals)

    body = []
    for label, cells in rows:
        if label is None:
            body.append(r"\midrule")
            continue
        out = []
        for (v, sd), key, fmt in zip(cells, ("coll", "ttc", "jsd"), ("{:.2f}",) * 2 + ("{:.3f}",)):
            txt = fmt.format(v)
            if abs(v - best[key]) < 5e-4:
                txt = rf"\mathbf{{{txt}}}"
            out.append(rf"${txt}_{{\pm{sd:.3f}}}$" if key == "jsd"
                       else rf"${txt}_{{\pm{sd:.2f}}}$")
        body += [label, "& " + " & ".join(out) + r" \\", ""]
    body = body[:-1]

    return "\n".join([
        r"% Generated by scripts/emit_table_ablation.py -- do not hand-edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Variant",
        r"& $\mathrm{Coll}_{ego}\uparrow$",
        r"& $\mathrm{TTC}_{<3s}\uparrow$",
        r"& $\mathrm{JSD}_{agent}\downarrow$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="temp")
    ap.add_argument("--name", default="table_ablation_filled_coll_1s.tex")
    ap.add_argument("--collision-min-time", type=float, default=1.0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.boot_seed)
    rows = []
    for label, spec, prefix, jsd_file, jsd_row in ROWS:
        if label is None:
            rows.append((None, None))
            continue
        m = load(spec, prefix)
        if "init_ego_overlap_frac" not in m:
            raise SystemExit(f"{spec}: no init_ego_overlap_frac; run "
                             "scripts/backfill_init_ego_overlap.py first")
        valid, fault, ttc = rates(m, args.collision_min_time)
        n = len(valid)
        counts = rng.multinomial(n, np.full(n, 1.0 / n),
                                 size=args.bootstrap).astype(np.float64)
        jsd = json.loads((CACHE / jsd_file).read_text())["rows"][jsd_row]["agent"]
        rows.append((label, [conditional(fault, valid, counts),
                             conditional(ttc, valid, counts),
                             (jsd["point"]["mean"], jsd["se"]["mean"])]))
        print(f"[ablation] {label:26s} n_valid={int(valid.sum()):4d}/{n}  "
              f"Coll_ego {rows[-1][1][0][0]:5.2f}  TTC {rows[-1][1][1][0]:5.2f}  "
              f"JSD {rows[-1][1][2][0]:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / args.name).write_text(
        render(rows, CAPTION.replace("{n}", str(args.bootstrap))))
    print(f"[ablation] {out / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
