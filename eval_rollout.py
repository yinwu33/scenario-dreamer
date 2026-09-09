#!/usr/bin/env python
"""Roll out the scene caches ``generate_scene.py`` writes and report criticality.

One row is one scene cache under one planner pair. Collision is EGO vs the
GENERATED ADVERSARY -- the scope the DDPO reward optimises -- so a row measures
the adversary rather than the traffic at large. ``Coll._f`` splits out the share
the ego drove into, using the front-face/moving-ego predicate in ``sim/world.py``.

The log baseline needs an adversary to be scored in that scope: a log scene has
no generated agent, so ego-vs-adversary is a structural zero rather than a
measurement. ``--log`` therefore designates the non-ego agent that spawns
nearest the ego as the adversary. That is the whole rule -- no clearance
constraint and no insertion -- so the log row keeps its own agent count while a
generated row carries one extra vehicle, and part of any gap is density rather
than adversary quality. Say so in the caption.

The cache and the log row are addressed by the SAME ``--val-index``, and the
latent and preprocessed directories are the same sorted file list, so index i is
the same scene on both sides and the rows pair element-wise. Only ``init_scene``
caches break that: they come from the layout prior and correspond to no
particular log scene, so compare those at distribution level.

Usage (env vars from scripts/define_env_variables.sh must be set)::

    .venv/bin/python eval_rollout.py --caches data/scenes/* --log \
        --sut ppo_normal --env ppo_normal --reward hierarchical_v4 \
        --workers 16 --batch-size 128 --out data/scenes/rollout_ppo-ppo_norm.md
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from critical_scene.gen_scenes import list_gen_scene_files, load_gen_scenes
from critical_scene.ldm_adv_eval import (
    benchmark_payload,
    build_reward,
    compose_eval_cfg,
    prepare_ldm_cfg,
    scenes_to_payload,
    summarize,
)
from critical_scene.log_scenes import closest_agent_adv_idx, load_log_scenes

# Directory-name token -> planner name, following the pair convention in AGENTS.md.
_SUT = {"ppo": "ppo_normal", "idm": "idm", "pdm": "pdm"}
_ENV = {"idm": "idm", "ppo_norm": "ppo_normal",
        "ppo_caution": "ppo_caution", "ppo_aggressive": "ppo_aggressive"}


def pair_from_name(policy: str) -> tuple[str, str]:
    """The (sut, env) the checkpoint in ``policy`` was fine-tuned against.

    A ``base*`` cache has no pair of its own (it is the frozen model, under one
    conditioning protocol or another) and the ``kl_*`` arms are ppo-ppo_norm runs
    with the trust region varied, so both fall back to that pair."""
    if policy.startswith("base"):
        return "ppo_normal", "ppo_normal"
    name = policy.split("_", 1)[1] if policy.startswith(("main_", "kl_")) else policy
    name = name.split("_kl")[0]
    sut, _, env = name.partition("-")
    if sut not in _SUT or env not in _ENV:
        raise ValueError(f"cannot read a planner pair out of {policy!r}")
    return _SUT[sut], _ENV[env]


# (header, metric key, scale). Rates are unfiltered: the caches are addressed by
# index rather than by an ego-goal threshold, so every scene is in the denominator.
COLUMNS = (
    ("Succ.", "reached_goal", 100.0),
    ("Off.", "ego_offroad_proxy", 100.0),
    ("Coll.", "ego_collision", 100.0),
    ("Coll._f", "ego_fault_collision", 100.0),
)


def cache_payload(cache_dir: Path) -> tuple[dict, list[str]]:
    """Load a cache and check its adversary index against the recorded one.

    ``load_gen_scenes`` derives the adversary from the append-last ordering;
    ``generate_scene.py`` also writes it explicitly. They agree by construction,
    so a mismatch means the cache was written by something else and the rollout
    would silently score the wrong agent."""
    files = list_gen_scene_files(cache_dir)
    scenes, kept = load_gen_scenes(cache_dir, range(len(files)), files=files)
    derived = scenes.adv_local_idx.numpy()
    for slot, idx in enumerate(kept):
        with open(files[idx], "rb") as f:
            recorded = int(pickle.load(f)["adv_local_idx"])
        if derived[slot] >= 0 and derived[slot] != recorded:
            raise ValueError(
                f"{files[idx]}: recorded adv_local_idx {recorded} != append-last "
                f"{derived[slot]}; the cache does not follow the writer's ordering"
            )
    return scenes_to_payload(scenes), [Path(files[i]).stem for i in kept]


def log_payload(preprocess_dir: Path, indices: list[int], dataset_cfg) -> dict:
    """Log scenes with the ego's nearest neighbour designated as the adversary.

    The designation itself is ``closest_agent_adv_idx`` -- shared with
    ``scripts/make_closest_adv.py``, which writes the same row as a standalone
    artifact. This used to be an inline copy that took the nearest agent of any
    type while that script took the nearest vehicle, so the two produced a
    different adversary in 91 of 1000 val scenes.
    """
    scenes, _ = load_log_scenes(preprocess_dir, "val", indices, dataset_cfg)
    payload = scenes_to_payload(scenes)
    payload["adv_local_idx"] = torch.from_numpy(closest_agent_adv_idx(
        payload["agent_states"].numpy(),
        payload["agent_types"].numpy(),
        payload["agent_scene_idx"].numpy(),
        int(payload["num_scenes"]),
    ))
    return payload


# The five columns the cross-checkpoint summary carries, and the rollout metric
# behind each. ``ego_offroad`` is NOT one of them: the maps carry lane
# centerlines only, so the simulator's real off-road test never fires and that
# field is always 0 (sim/hooks.py). ``ego_offroad_proxy`` is the substitute.
SUMMARY_METRICS = (
    ("succ", "reached_goal"),
    ("collision", "ego_collision"),
    ("collision_ego_fault", "ego_fault_collision"),
    ("offroad", "ego_offroad_proxy"),
    ("minTTC_ego_to_adv", "ego_min_ttc"),
)
# Derived from minTTC rather than measured again: a threshold count is what the
# tables report, and deriving it here keeps the csv self-describing instead of
# making every consumer re-apply the cut. inf (the ego never approached) is
# below no threshold, which np.less gives for free.
TTC_THRESHOLDS = ((3.0, "ttc_lt_3s"), (1.5, "ttc_lt_1p5s"))


def rebuild_summary(args) -> int:
    """Rewrite the summary from the per-cache outputs already on disk.

    The rollouts are the expensive half and they are already persisted; a
    failure while assembling the matrix should cost a rerun of the assembly,
    not of the rollouts."""
    out_root = Path(args.out)
    per_cache = {}
    for d in sorted(p for p in out_root.iterdir() if (p / "metrics.npz").exists()):
        blob = np.load(d / "metrics.npz", allow_pickle=True)
        summary = json.loads((d / "summary.json").read_text())
        per_cache[d.name] = {
            "stems": [str(x) for x in blob["scenario"]],
            "metrics": {k: blob[k] for k in blob.files if k != "scenario"},
            "summary": summary, "sut": summary["sut"], "env": summary["env"],
        }
    if not per_cache:
        raise SystemExit(f"no per-cache metrics.npz under {out_root}")
    print(f"[rollout] rebuilding summary from {len(per_cache)} cached columns")
    write_summary(per_cache, out_root, args)
    return 0


def run_batch(args) -> int:
    """Roll out every cache under a root, each under its own planner pair.

    The pair is read from the cache's directory name, because that is what the
    checkpoint in it was fine-tuned against. That makes each column the
    checkpoint's own home fixture and NOT comparable with its neighbours: the
    planner pair moves ego collision far more than the generator does, so a
    column is 'this checkpoint against the traffic it was trained on', never
    'this checkpoint is better than that one'. The emitted markdown says so.

    Caches are grouped by pair so one RewardModel (and one set of rollout
    workers) serves every cache that shares it."""
    root, out_root = Path(args.caches_root), Path(args.out)
    caches = sorted(p for p in root.glob(f"*/{args.mode}") if (p / "manifest.json").exists())
    if not caches:
        raise SystemExit(f"no scene caches under {root}/*/{args.mode}")
    cache_of = {c.parent.name: c for c in caches}
    # A cache that lives outside the root, or under another mode. It carries no
    # planner pair in its name, so it is only usable with --all-pairs.
    for spec in args.extra_cache:
        name, _, path = spec.partition("=")
        extra = Path(path)
        if not (extra / "manifest.json").exists():
            raise SystemExit(f"--extra-cache {name}: no manifest.json in {extra}")
        if name in cache_of:
            raise SystemExit(f"--extra-cache {name}: the root already has that name")
        cache_of[name] = extra
    names = list(cache_of)
    if args.log:
        # A pseudo-cache: log scenes with the ego's nearest neighbour designated
        # as the adversary. It has no directory and no pair of its own, so it is
        # always scored under every pair.
        names.append("log")
    missing = set(args.all_pairs) - set(names)
    if missing:
        raise SystemExit(f"--all-pairs names no such cache: {sorted(missing)}")

    # The pairs a cache is scored under. A cache named by --all-pairs is scored
    # under EVERY pair the other caches bring, which is what makes it a control:
    # a baseline is only comparable with a checkpoint when both met the same
    # planners. Everything else keeps the single pair its name implies.
    own = {n: (args.sut or pair_from_name(n)[0], args.env or pair_from_name(n)[1])
           for n in names if n not in args.all_pairs and n != "log"}
    every = sorted(set(own.values()))
    if not every:
        raise SystemExit("every cache is in --all-pairs, so no planner pair is implied "
                         "by any name; nothing defines the pair set")
    pairs_of = {n: (every if (n in args.all_pairs or n == "log") else [own[n]])
                for n in names}
    jobs = [(n, pair) for n in names for pair in pairs_of[n]]
    label = {(n, pair): (f"{n}@{pair[0]}-{pair[1]}" if len(pairs_of[n]) > 1 else n)
             for n, pair in jobs}
    by_pair: dict[tuple, list[str]] = {}
    for n, pair in jobs:
        by_pair.setdefault(pair, []).append(n)
    print(f"[rollout] {len(names)} caches, {len(every)} planner pairs, {len(jobs)} rollouts")

    per_cache: dict[str, dict] = {}
    for pair in sorted(by_pair):
        sut, env = pair
        cfg_root = compose_eval_cfg(args.config_name, [
            f"ddpo/reward={args.reward}",
            f"planner@ddpo.planner.sut={sut}",
            f"planner@ddpo.planner.env={env}",
            f"planner@ddpo.planner.adv={env}",
            *args.override,
        ])
        ldm_cfg = prepare_ldm_cfg(cfg_root)
        reward = build_reward(cfg_root, ldm_cfg, num_workers=int(args.workers),
                              batch_size=int(args.batch_size))
        min_ego_drive = float(cfg_root.ddpo.min_ego_drive)
        for name in by_pair[pair]:
            tag = label[(name, pair)]
            out_dir = out_root / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[rollout] {tag}  sut={sut} env={env}", flush=True)
            if name == "log":
                indices = json.loads(Path(args.val_index).read_text())["scene_idx"]
                payload = log_payload(ROOT / "data" / "advscene_preprocess_waymo",
                                      indices, ldm_cfg.dataset)
                stems = [f"{i}_log" for i in indices]
            else:
                payload, stems = cache_payload(cache_of[name])
            metrics = benchmark_payload(reward, payload,
                                        batch_size=int(args.batch_size), label=tag)
            summary = summarize(metrics, min_ego_drive=min_ego_drive)
            summary.update(cache=name, sut=sut, env=env, mode=args.mode,
                           reward=args.reward, num_scenes=int(len(stems)))
            np.savez_compressed(out_dir / "metrics.npz", scenario=np.array(stems), **metrics)
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            per_cache[tag] = {"stems": stems, "metrics": metrics, "summary": summary,
                              "sut": sut, "env": env}
        if args.workers:
            reward.close()   # rollout workers outlive the process otherwise

    write_summary(per_cache, out_root, args)
    print(f"[rollout] done -> {out_root}")
    return 0


def _stem_key(stem: str):
    """Numeric where a segment is numeric, lexical where it is not.

    Cache stems are ``<i>_<batch>``, but the log pseudo-cache's are
    ``<val_idx>_log``; each segment is tagged so the two never compare an int
    against a str."""
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in stem.split("_"))


def write_summary(per_cache: dict, out_root: Path, args) -> None:
    """scenario x checkpoint matrices, plus a long table and a readable digest.

    Rows are aligned by the cache's scene STEM rather than by position: a scene
    that decodes with no lanes is dropped at load time, and that can happen in
    one cache and not another. A checkpoint missing a stem gets NaN there."""
    policies = sorted(per_cache)
    stems = sorted({s for p in policies for s in per_cache[p]["stems"]}, key=_stem_key)
    row_of = {s: i for i, s in enumerate(stems)}

    grids = {}
    for name, key in SUMMARY_METRICS:
        grid = np.full((len(stems), len(policies)), np.nan, dtype=np.float64)
        for col, policy in enumerate(policies):
            entry = per_cache[policy]
            for slot, stem in enumerate(entry["stems"]):
                grid[row_of[stem], col] = entry["metrics"][key][slot]
        grids[name] = grid

    for thr, name in TTC_THRESHOLDS:
        grids[name] = (grids["minTTC_ego_to_adv"] < thr).astype(np.float64)
        grids[name][np.isnan(grids["minTTC_ego_to_adv"])] = np.nan

    np.savez_compressed(
        out_root / "summary.npz",
        scenario=np.array(stems), checkpoint=np.array(policies),
        sut=np.array([per_cache[p]["sut"] for p in policies]),
        env=np.array([per_cache[p]["env"] for p in policies]),
        **grids,
    )

    names = [n for n, _ in SUMMARY_METRICS] + [n for _, n in TTC_THRESHOLDS]
    lines = ["scenario,checkpoint,sut,env," + ",".join(names)]
    for i, stem in enumerate(stems):
        for col, policy in enumerate(policies):
            vals = ",".join("" if np.isnan(grids[n][i, col]) else f"{grids[n][i, col]:g}"
                            for n in names)
            lines.append(f"{stem},{policy},{per_cache[policy]['sut']},"
                         f"{per_cache[policy]['env']},{vals}")
    (out_root / "summary.csv").write_text("\n".join(lines) + "\n")

    md = [
        f"# Rollout summary ({args.mode})",
        "",
        f"reward: `{args.reward}`  scenes: {len(stems)}  checkpoints: {len(policies)}",
        "",
        "Collision is ego vs the GENERATED ADVERSARY; fault uses the",
        "front-face/moving-ego predicate. Off-road is `ego_offroad_proxy` (the",
        "simulator's own off-road test never fires on these maps).",
        "",
        "COMPARE ONLY WITHIN A PLANNER PAIR. Each checkpoint is rolled out under",
        "the pair it was fine-tuned against, and the planner moves these rates far",
        "more than the generator does, so two rows with different sut/env say",
        "nothing about each other. Rows sharing a sut/env -- a checkpoint and the",
        "baselines carrying the same @sut-env tag -- are the comparable set.",
        "",
        "`minTTC` is a mean over FINITE values only; a scene where the ego never",
        "approached the adversary is +inf and is excluded from that mean, with",
        "the count of such scenes in `n_inf`.",
        "",
        "| checkpoint | sut | env | n | succ | collision | coll_ego_fault | offroad | minTTC | n_inf |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for col, policy in enumerate(policies):
        n = int(np.isfinite(grids["succ"][:, col]).sum())
        cells = []
        for name in names[:4]:
            v = grids[name][:, col]
            cells.append(f"{100.0 * np.nanmean(v):.2f}")
        ttc = grids["minTTC_ego_to_adv"][:, col]
        finite = ttc[np.isfinite(ttc)]
        n_inf = int(np.sum(np.isinf(ttc)))
        cells.append(f"{finite.mean():.2f}" if finite.size else "--")
        e = per_cache[policy]
        md.append(f"| {policy} | {e['sut']} | {e['env']} | {n} | " + " | ".join(cells) + f" | {n_inf} |")
    (out_root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_ldm_adv_ddpo")
    ap.add_argument("--caches", nargs="*", default=[])
    ap.add_argument("--log", action="store_true", help="add a log row from --val-index")
    ap.add_argument("--val-index", default="metadata/val1000.json")
    ap.add_argument("--caches-root",
                    help="score every <root>/<policy>/<mode> cache, each under the "
                         "planner pair its name implies, and write the "
                         "scenario x checkpoint summary")
    ap.add_argument("--mode", default="init_scene")
    ap.add_argument("--rebuild-summary", action="store_true",
                    help="skip the rollouts and rewrite the summary from the "
                         "per-cache metrics.npz already in --out")
    ap.add_argument("--extra-cache", nargs="*", default=[], metavar="NAME=PATH",
                    help="add a cache from outside --caches-root (another mode, "
                         "another generator). It has no pair in its name, so it "
                         "must also be named in --all-pairs.")
    ap.add_argument("--all-pairs", nargs="*", default=[],
                    help="cache names to score under EVERY planner pair in the "
                         "batch, not just the one their name implies -- the "
                         "baselines, so each checkpoint has a control that met "
                         "the same planners. Their columns are labelled "
                         "<name>@<sut>-<env>.")
    ap.add_argument("--sut", help="required unless --caches-root")
    ap.add_argument("--env", help="required unless --caches-root")
    ap.add_argument("--adv", default=None, help="default: --env")
    ap.add_argument("--reward", required=True,
                    help="cfgs/ddpo/reward/<name>.yaml to score with. Required: the "
                         "entrypoint config carries a reward of its own, so omitting "
                         "this scores every row under that default whatever the run "
                         "was trained with, and writes its reward/tier under the "
                         "wrong name.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--out", required=True, help="markdown table; .json and .npz go beside it")
    args = ap.parse_args()

    if args.rebuild_summary:
        return rebuild_summary(args)
    if args.caches_root:
        if args.caches:
            ap.error("--caches-root scores a whole root; it does not take --caches")
        return run_batch(args)
    if not args.caches and not args.log:
        ap.error("nothing to score: pass --caches, --log, or both")
    if not (args.sut and args.env):
        ap.error("--sut and --env are required unless --caches-root")
    adv = args.adv or args.env
    cfg_root = compose_eval_cfg(args.config_name, [
        f"ddpo/reward={args.reward}",
        f"planner@ddpo.planner.sut={args.sut}",
        f"planner@ddpo.planner.env={args.env}",
        f"planner@ddpo.planner.adv={adv}",
        *args.override,
    ])
    ldm_cfg = prepare_ldm_cfg(cfg_root)
    reward = build_reward(
        cfg_root, ldm_cfg, num_workers=int(args.workers), batch_size=int(args.batch_size)
    )
    min_ego_drive = float(cfg_root.ddpo.min_ego_drive)

    rows: dict[str, dict] = {}
    if args.log:
        indices = json.loads(Path(args.val_index).read_text())["scene_idx"]
        rows["log"] = {
            "payload": log_payload(
                ROOT / "data" / "advscene_preprocess_waymo", indices, ldm_cfg.dataset
            ),
            "manifest": {"mode": "log", "conditioning": "-", "ddpo_ckpt": None},
        }
    for cache in args.caches:
        cache_dir = Path(cache)
        rows[cache_dir.name] = {
            "payload": cache_payload(cache_dir)[0],
            "manifest": json.loads((cache_dir / "manifest.json").read_text()),
        }

    summaries, per_scene = {}, {}
    for name, row in rows.items():
        metrics = benchmark_payload(
            reward, row["payload"], batch_size=int(args.batch_size), label=name
        )
        summaries[name] = summarize(metrics, min_ego_drive=min_ego_drive)
        summaries[name]["mode"] = row["manifest"]["mode"]
        summaries[name]["ego_fault_collision_rate"] = float(np.mean(metrics["ego_fault_collision"]))
        for key, arr in metrics.items():
            per_scene[f"{name}/{key}"] = arr
    if args.workers:
        # Rollout workers outlive the script as orphans otherwise.
        reward.close()

    lines = [
        f"cell: SUT={args.sut}  traffic={args.env}  adv={adv}",
        f"reward: {args.reward}",
        "metrics: ego vs the GENERATED ADVERSARY (log: the ego's nearest neighbour)",
        "fault uses the front-face/moving-ego predicate (sim/world.py)",
        "rates are over all scenes; n_driving is reported for reference only",
        "",
        "| row | mode | n | " + " | ".join(c[0] for c in COLUMNS) + " | minTTC | n_driving |",
        "|---|---|---:|" + "---:|" * (len(COLUMNS) + 2),
    ]
    for name, s in summaries.items():
        cells = [
            f"{100.0 * s['reached_goal_rate']:.2f}",
            f"{100.0 * s['ego_offroad_rate']:.2f}",
            f"{100.0 * s['ego_collision_rate']:.2f}",
            f"{100.0 * s['ego_fault_collision_rate']:.2f}",
        ]
        lines.append(
            f"| {name} | {s['mode']} | {int(s['num_scenes'])} | " + " | ".join(cells)
            + f" | {s['ego_min_ttc_mean']:.2f} | {int(s['num_driving_ego'])} |"
        )
    text = "\n".join(lines)
    print("\n" + text)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    np.savez_compressed(out.with_suffix(".npz"), **per_scene)
    print(f"\n[rollout] wrote {out} (+ .json, .npz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
