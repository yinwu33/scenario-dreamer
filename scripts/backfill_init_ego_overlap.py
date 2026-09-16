#!/usr/bin/env python
"""One-time backfill of ``init_ego_overlap_frac`` into ``data/final/cache``.

The eval scene-validity gate needs the ego's spawn overlap, which ``InitOverlapHook``
only started emitting on 2026-09-15. Re-running the 133 rollouts to get it would cost
~80 min and would also re-derive 40 columns that are already correct, so instead this
replays each cache through the SAME hook with ``sim_steps=0``: the scenes are built
and ``before_rollout`` runs, the step loop does not. Nothing here re-implements the
geometry -- it is ``RewardModel.evaluate`` with the loop length set to zero.

The correctness argument is the check, not the construction. The same replay also
recomputes ``init_overlap_frac`` (the adversary's spawn overlap, recorded back when
the rollout ran), and it must come back EXACTLY equal per scene. That pins the scene
loading, the agent ordering, the pedestrian exclusion and the box geometry all at
once; if the replay reproduced a different t=0 state, this is where it shows.

The t=0 state does not depend on the planner pair, so a scene source is replayed once
and broadcast to the (up to 12) columns that share it, joined by scene stem.

    .venv/bin/python scripts/backfill_init_ego_overlap.py --dry-run
    .venv/bin/python scripts/backfill_init_ego_overlap.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from critical_scene.ldm_adv_eval import (  # noqa: E402
    benchmark_payload,
    build_reward,
    compose_eval_cfg,
    prepare_ldm_cfg,
)
from eval_rollout import cache_payload, log_payload  # noqa: E402

# Root -> the --mode its caches were scored under (from each root's summary.md).
MODE = {"scenario": "init_scene", "scenario_init_agent": "init_agent",
        "scenario_init_adv": "init_adv", "scenario_log": "init_scene",
        "scenario_bok": "init_scene"}
# Scene caches that do not sit at data/final/cache/scene/<column>/<mode>: the
# --extra-cache one, and the best-of-K root's two plain baselines, which are the
# main root's caches re-scored under a different column name.
OUTSIDE = {"scenecontrol": Path("data/final/scenecontrol/init_agent"),
           "base_cond": Path("data/final/cache/scene/base/init_scene"),
           "base_null": Path("data/final/cache/scene/base_null/init_scene")}
KEY = "init_ego_overlap_frac"

# The ablation rollouts were written as ONE flat npz per run, keyed ``<mode>/<metric>``,
# rather than as a directory per column. Same replay, same check, different container.
# (npz path relative to the repo, key prefix, scene cache).
FLAT = [
    ("data/final/cache/scenario_reward/noprox.npz", "init_scene",
     "data/final/cache/scene/reward_ppo-ppo_norm_noprox/init_scene"),
    ("data/final/cache/scenario_reward/nottc.npz", "init_scene",
     "data/final/cache/scene/reward_ppo-ppo_norm_nottc/init_scene"),
    ("data/final/cache/scenario_rlnull/rollout_b128.npz", "init_scene",
     "data/final/cache/scene/main_ppo-ppo_norm_null/init_scene"),
    ("data/final/cache/scenario_rlnull/rollout.npz", "init_scene",
     "data/final/cache/scene/main_ppo-ppo_norm_null/init_scene"),
]


def scene_source(column: str) -> str:
    """``base@idm-idm`` -> ``base``; ``main_idm-idm`` -> itself. The part before
    ``@`` is the cache, the part after is the planner pair it was scored under,
    and the pair cannot move an agent at t=0."""
    return column.split("@", 1)[0]


def source_dir(src: str, mode: str) -> Path:
    if src in OUTSIDE:
        return ROOT / OUTSIDE[src]
    # The best-of-K budgets are their own generated caches, under their own root.
    stem = "scene_bok" if "_bok" in src else "scene"
    return ROOT / "data/final/cache" / stem / src / mode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="data/final/cache")
    ap.add_argument("--config-name", default="config_ldm_adv_ddpo")
    ap.add_argument("--reward", default="hierarchical_v4")
    ap.add_argument("--val-index", default="metadata/val1000.json")
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--roots", default="all",
                    help=f"comma-separated subset of {sorted(MODE)}, or 'none'")
    ap.add_argument("--skip-flat", action="store_true",
                    help="skip the flat ablation npz in FLAT")
    args = ap.parse_args()

    root = ROOT / args.cache_root
    # (source, mode) -> the columns that share it. Sorted so a rerun replays in
    # the same order and its log is diffable.
    jobs: dict[tuple[str, str], list[Path]] = {}
    roots = ({} if args.roots == "none" else MODE if args.roots == "all"
             else {r: MODE[r] for r in args.roots.split(",")})
    for sub, mode in roots.items():
        for d in sorted(p for p in (root / sub).iterdir() if (p / "metrics.npz").exists()):
            jobs.setdefault((scene_source(d.name), mode), []).append(d)
    print(f"[backfill] {sum(len(v) for v in jobs.values())} columns, "
          f"{len(jobs)} distinct scene sources")

    # sim_steps=0: build the scenes, run before_rollout, skip the step loop. The
    # pair is irrelevant to t=0 but a rollout still needs one, so every source is
    # replayed under idm/idm -- the pair that loads no checkpoint.
    cfg_root = compose_eval_cfg(args.config_name, [
        f"ddpo/reward={args.reward}",
        "planner@ddpo.planner.sut=idm",
        "planner@ddpo.planner.env=idm",
        "planner@ddpo.planner.adv=idm",
        "ddpo.simulator.sim_steps=0",
    ])
    ldm_cfg = prepare_ldm_cfg(cfg_root)
    reward = build_reward(cfg_root, ldm_cfg, num_workers=0, batch_size=args.batch_size)

    written = 0
    for (src, mode), dirs in sorted(jobs.items()):
        if src == "log":
            indices = json.loads((ROOT / args.val_index).read_text())["scene_idx"]
            payload = log_payload(ROOT / "data" / "advscene_preprocess_waymo",
                                  indices, ldm_cfg.dataset)
            stems = [f"{i}_log" for i in indices]
        else:
            payload, stems = cache_payload(source_dir(src, mode))
        m = benchmark_payload(reward, payload, batch_size=args.batch_size)
        ego = dict(zip(stems, m[KEY]))
        adv = dict(zip(stems, m["init_overlap_frac"]))

        for d in dirs:
            blob = np.load(d / "metrics.npz", allow_pickle=True)
            recorded = {k: blob[k] for k in blob.files}
            got = np.array([adv[s] for s in recorded["scenario"]], dtype=np.float32)
            if not np.array_equal(got, recorded["init_overlap_frac"]):
                bad = int((got != recorded["init_overlap_frac"]).sum())
                raise SystemExit(
                    f"{d}: replay reproduced a DIFFERENT init_overlap_frac on {bad} "
                    "scenes, so its t=0 state is not the one that was rolled out; "
                    "the ego column would be wrong too. Nothing written."
                )
            recorded[KEY] = np.array([ego[s] for s in recorded["scenario"]],
                                     dtype=np.float32)
            print(f"[backfill] {d.relative_to(root)}  n={len(recorded['scenario'])}  "
                  f"ego_overlap>0 {100.0 * (recorded[KEY] > 0).mean():.2f}%  "
                  f"adv_overlap>0 {100.0 * (recorded['init_overlap_frac'] > 0).mean():.2f}%"
                  + ("  [dry-run]" if args.dry_run else ""))
            if not args.dry_run:
                np.savez_compressed(d / "metrics.npz", **recorded)
            written += 1

    for rel, prefix, scene in ([] if args.skip_flat else FLAT):
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"{path}: no such ablation rollout")
        payload, _ = cache_payload(ROOT / scene)
        m = benchmark_payload(reward, payload, batch_size=args.batch_size)
        blob = np.load(path, allow_pickle=True)
        recorded = {k: blob[k] for k in blob.files}
        want = recorded[f"{prefix}/init_overlap_frac"]
        got = m["init_overlap_frac"][:len(want)]
        if not np.array_equal(got, want):
            print(f"[backfill] SKIP {rel}: replaying {scene} reproduces a different "
                  f"init_overlap_frac on {int((got != want).sum())} of {len(want)} scenes, "
                  "so that cache is not the one it was rolled out on")
            continue
        recorded[f"{prefix}/{KEY}"] = m[KEY][:len(want)].astype(np.float32)
        print(f"[backfill] {rel}  n={len(want)}  "
              f"ego_overlap>0 {100.0 * (recorded[f'{prefix}/{KEY}'] > 0).mean():.2f}%"
              + ("  [dry-run]" if args.dry_run else ""))
        if not args.dry_run:
            np.savez_compressed(path, **recorded)
        written += 1

    print(f"[backfill] {'checked' if args.dry_run else 'wrote'} {written} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
