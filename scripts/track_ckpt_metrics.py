"""Score a DDPO run's checkpoints on both axes, loading the reference set once.

Per checkpoint:
  criticality  -- generate the ddpo_gen scene source on the shared template pool
                  and roll it out under the run's planner trio, reporting the same
                  planner-quality rates the tables use (driving-ego subset).
  scene quality-- generate unconditionally from the layout prior and report the
                  agent-attribute JSDs plus spawn overlap.

Only the agent-side scene metrics are computed: the lane columns are frozen (DDPO
trains the adversary branch only), so every checkpoint shares them, and the goal
metrics need an un-memoisable ground-truth pass per row.

  .venv/bin/python scripts/track_ckpt_metrics.py \
      --run-dir data/critical_scene/critical_scene_ddpo_ldm_adv_ddim_ppo-ppo_norm_fixedkl_v2_hier_v2 \
      --sut ppo_normal --env ppo_normal
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from tqdm import tqdm

from cfgs.config import CONFIG_PATH
from critical_scene.ldm_adv_eval import (
    build_policy,
    build_pool,
    cat_payloads,
    compose_eval_cfg,
    slice_payload,
    generate_chunk,
    prepare_ldm_cfg,
)
from critical_scene.planner_matrix_eval import (
    build_runner,
    concat_metrics,
    evaluate_scenes,
    summarize,
)
from scripts.gen_scene_gen_samples import build_prior_batches, stage_one, stage_two
from scripts.score_scene_gen_table import load_generated, load_reference, memoise_reference_stats
from utils import metrics_helpers as mh


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in run_dir.glob("*_[0-9][0-9][0-9][0-9][0-9].ckpt"):
        out.append((int(re.search(r"_(\d{5})\.ckpt$", p.name).group(1)), p))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sut", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--config-name", default="config_ldm_adv_ddpo")
    ap.add_argument("--num-scenes", type=int, default=1000)
    ap.add_argument("--chunk-size", type=int, default=32)  # must match run_ldm_adv_ppo_table:
                    # generate_chunk seeds per chunk, so the chunk size changes which
                    # scenes come out and breaks pairing with the table artifacts
    ap.add_argument("--scene-gen-scenes", type=int, default=1000)
    ap.add_argument("--scene-gen-batch", type=int, default=128)
    ap.add_argument("--num-gt-samples", type=int, default=43658)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--scratch", default=None, help="dir for the generated scene pickles")
    ap.add_argument("--with-base", action="store_true",
                    help="also score the frozen base checkpoint as iteration 0. Its "
                         "criticality must reproduce the table's base_gen row exactly, "
                         "so this doubles as an end-to-end check of the protocol.")
    ap.add_argument("--criticality-only", action="store_true",
                    help="skip the scene-quality axis (and its one-off reference load)")
    ap.add_argument("--iters", nargs="*", type=int, default=None,
                    help="only these checkpoint iterations")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpts = checkpoints(run_dir)
    if not ckpts:
        raise SystemExit(f"no numbered checkpoints under {run_dir}")
    if args.iters:
        ckpts = [(it, p) for it, p in ckpts if it in set(args.iters)]
    print(f"[track] {len(ckpts)} checkpoints: {[it for it, _ in ckpts]}")

    overrides = [f"planner@ddpo.planner.sut={args.sut}",
                 f"planner@ddpo.planner.env={args.env}",
                 f"planner@ddpo.planner.adv={args.env}"]
    cfg_root = compose_eval_cfg(args.config_name, overrides)
    ldm_cfg = prepare_ldm_cfg(cfg_root)
    base_ckpt = str(cfg_root.ddpo.ldm_adv_ckpt)
    if args.with_base:
        ckpts = [(0, Path(base_ckpt))] + ckpts
        print(f"[track] iteration 0 = frozen base ({base_ckpt})")

    # ---- criticality harness (same planner trio, same driving-ego subset)
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base=None):
        pm_cfg = compose(config_name="config_planner_matrix", overrides=[
            f"planner@planner.sut={args.sut}",
            f"planner@planner.env={args.env}",
            f"planner@planner.adv={args.env}",
        ])
    runner = build_runner(pm_cfg, num_workers=args.workers, batch_size=128)
    pool = build_pool(cfg_root, ldm_cfg, split="val",
                      pool_size=args.num_scenes, device=args.device)
    base_policy = build_policy(cfg_root, ldm_cfg, ckpt=base_ckpt, device=args.device)

    # ---- scene-quality harness (prior-mode layouts + one reference pass)
    lit = prior_batches = base_latents = reference = None
    if not args.criticality_only:
        from models.scenario_dreamer_ldm_adv import ScenarioDreamerLDMAdv
        lit = ScenarioDreamerLDMAdv.load_from_checkpoint(
            base_ckpt, cfg=ldm_cfg, cfg_ae=cfg_root.ae_goal).to(args.device).eval()
        torch.manual_seed(0)
        prior_batches = build_prior_batches(lit, cfg_root, args.scene_gen_scenes, args.scene_gen_batch)
        base_latents = stage_one(lit, prior_batches, 0, args.device)
        reference = load_reference(
            ldm_cfg.dataset,
            ROOT / "metadata" / "waymo_goal_val_eval_set.pkl",
            ROOT / "data" / "advscene_preprocess_waymo" / "val",
            args.num_gt_samples,
        )
        memoise_reference_stats(reference)
        print(f"[track] reference: {len(reference)} scenes")

    scratch = Path(args.scratch or (run_dir / "track_scenes"))
    per_scene_dir = run_dir / "per_scene"
    per_scene_dir.mkdir(parents=True, exist_ok=True)
    slots = [list(range(s, min(s + args.chunk_size, args.num_scenes)))
             for s in range(0, args.num_scenes, args.chunk_size)]
    out_path = Path(args.out or (run_dir / "track_metrics.json"))
    results = {}

    for it, ckpt in ckpts:
        print(f"\n[track] ===== iteration {it}  ({ckpt.name})")
        policy = build_policy(cfg_root, ldm_cfg, ckpt=str(ckpt), device=args.device)

        chunks = []
        for cid, sl in enumerate(tqdm(slots, desc=f"gen ddpo_gen it{it}")):
            out = generate_chunk(base_policy=base_policy, ddpo_policy=policy, pool=pool,
                                 ldm_cfg=ldm_cfg, slots=sl, seed=0, chunk_id=cid,
                                 device=args.device, sources=("ddpo_gen",))
            chunks.append(out["ddpo_gen"])
            pool._cache.clear()
        # Same slicing as scripts/score_paired_sources.py: the PPO planner is
        # batched across scenes, so which scenes share a batch is part of the
        # protocol and the numbers are only comparable if the batching matches.
        payload = cat_payloads(chunks)
        n = int(payload["num_scenes"])
        parts = []
        for start in range(0, n, 128):
            part, _ = evaluate_scenes(runner, pm_cfg, slice_payload(payload, start, min(start + 128, n)))
            parts.append(part)
        per_scene = concat_metrics(parts)
        crit = summarize(per_scene, min_ego_drive=float(pm_cfg.benchmark.min_ego_drive))
        # Per-scene outcomes for paired testing: every checkpoint is scored on the
        # SAME template slots, so a difference of a few tenths of a point can only
        # be judged against how many scenes actually flipped, not against the
        # unpaired binomial error of each rate on its own.
        np.savez_compressed(
            per_scene_dir / f"it{it:05d}.npz",
            **{k: np.asarray(per_scene[k]) for k in
               ("ego_goal_dist", "reached_goal", "ego_collision_any",
                "ego_fault_collision_any", "ego_offroad_proxy")},
        )

        agent = {}
        if not args.criticality_only:
            cache_dir = scratch / f"it{it:05d}"
            stage_two(lit, policy, prior_batches, base_latents, seed=0, device=args.device,
                      batch_size=args.scene_gen_batch, cache_dir=cache_dir)
            samples = load_generated(cache_dir, args.scene_gen_scenes)
            agent = mh.compute_agent_metrics(samples=samples, gt_samples=reference)

        results[it] = {
            "ckpt": ckpt.name,
            "criticality": {k: float(v) for k, v in crit.items()},
            "scene_quality": {k: float(v) for k, v in agent.items()},
        }
        c, q = results[it]["criticality"], results[it]["scene_quality"]
        tail = ("" if not q else
                f" | overlap={q['collision_rate']:.2f} speedJSD={q['speed_jsd']:.3f} "
                f"nearJSD={q['nearest_dist_jsd']:.3f}")
        print(f"[track] it{it}: n={c['num_driving_ego']} "
              f"Succ={100*c['reached_goal_rate_driving']:.2f} "
              f"Off={100*c['ego_offroad_rate_driving']:.2f} "
              f"Coll={100*c['ego_collision_rate_driving']:.2f} "
              f"Coll_f={100*c['ego_fault_collision_rate_driving']:.2f}" + tail)
        out_path.write_text(json.dumps(results, indent=2))
        del policy
        torch.cuda.empty_cache()

    print(f"\n[track] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
