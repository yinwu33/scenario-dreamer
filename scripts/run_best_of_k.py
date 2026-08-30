#!/usr/bin/env python
"""Best-of-$K$ sampling from the FROZEN base generator, on the evaluation pool.

The results table's strongest naive baseline: draw $K$ adversaries from the
pretrained generator in the same context, simulate all of them, and keep the most
critical one. Since one AdvScene sample and one base sample cost the same, this is
what a practitioner would do without training anything, and it is the curve a
learned method has to beat at matched compute.

This differs from ``scripts/headroom_probe.py``, which answers the same question
on the TRAINING pool as a pre-training diagnostic. Here the pool, the split, the
seeds and the payload schema are the evaluation harness's, so the selected draws
drop straight into ``scripts/score_paired_sources.py`` next to the other rows.

Draw 0 reuses ``run_ldm_adv_ppo_table``'s adversary seed, so the candidate set
literally contains the ``base_gen`` sample and the two rows are paired. The
curve's $k=1$ point is not that draw but the expectation over a single random
draw (a uniform average of all $K$), which estimates the same quantity with less
variance.

Selection is by the DDPO reward -- the objective a practitioner would optimize --
and the reported numbers come from the independent planner-quality pass, so the
row is not scored on the quantity that chose it.

    python scripts/run_best_of_k.py --config-name config_ldm_adv_ddpo \
        --overrides planner@ddpo.planner.sut=idm ... \
        --out-dir data/critical_scene/table_main_20260830/idm-ppo_norm -k 32
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from math import comb
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from critical_scene.ldm_adv_eval import (
    _seed_all,
    build_metadata,
    build_policy,
    build_pool,
    build_reward,
    benchmark_payload,
    cat_payloads,
    compose_eval_cfg,
    make_generated_cond,
    prepare_ldm_cfg,
    sample_base_scene_latents,
    scenes_to_payload,
    slice_payload,
    write_json,
)
from critical_scene.metrics_common import ego_goal_dist


def _parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config-name", default="config_ldm_adv_ddpo")
    p.add_argument("--overrides", nargs="*", default=[])
    p.add_argument("--out-dir", required=True)
    p.add_argument("-k", "--num-draws", type=int, default=32)
    p.add_argument("--num-scenes", type=int, default=1000)
    p.add_argument("--split", default="val")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--benchmark-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--base-ckpt", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _draw_seed(seed: int, chunk_id: int, i: int, k: int) -> int:
    """Draw 0 is run_ldm_adv_ppo_table's ``base_gen`` seed; the rest are fresh."""
    if i == 0:
        return seed * 1_000_003 + 2000 + chunk_id
    return seed * 1_000_003 + 3_000_000 + chunk_id * k + i


def _ladder(n_draws: int) -> list[int]:
    """Budgets to report: 1, 2, 4, ... up to the number of draws."""
    out, k = [], 1
    while k <= n_draws:
        out.append(k)
        k *= 2
    return out


def _selection_weights(k: int, n_draws: int) -> np.ndarray:
    """P(the rank-j draw is the max-reward one of a random k-subset), j ascending."""
    denom = comb(n_draws, k)
    return np.array(
        [comb(j - 1, k - 1) / denom if j >= k else 0.0 for j in range(1, n_draws + 1)]
    )


def _curve(reward: np.ndarray, collision: np.ndarray, keep: np.ndarray) -> dict:
    """Exact expected outcome of reward-selecting the best of k, per k."""
    n_draws = reward.shape[1]
    order = np.argsort(reward[keep], axis=1)          # ascending reward
    r_sorted = np.take_along_axis(reward[keep], order, axis=1)
    c_sorted = np.take_along_axis(collision[keep], order, axis=1)

    out = {}
    k = 1
    while k <= n_draws:
        w = _selection_weights(k, n_draws)
        out[str(k)] = {
            "reward": float(np.mean(r_sorted @ w)),
            "ego_collision_rate": float(np.mean(c_sorted @ w)),
        }
        k *= 2
    return out


def main() -> int:
    args = _parse()
    k = int(args.num_draws)
    cfg_root = compose_eval_cfg(args.config_name, args.overrides)
    ldm_cfg = prepare_ldm_cfg(cfg_root)
    base_ckpt = args.base_ckpt or str(cfg_root.ddpo.ldm_adv_ckpt)

    out_dir = Path(args.out_dir)
    chunk_dir = out_dir / "artifacts" / f"base_gen_bok{k}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n = int(args.num_scenes)
    size = int(args.chunk_size)
    chunk_slots = [list(range(s, min(s + size, n))) for s in range(0, n, size)]

    pool = build_pool(cfg_root, ldm_cfg, split=args.split, pool_size=n, device=args.device)
    policy = None
    reward = build_reward(cfg_root, ldm_cfg, num_workers=int(args.workers),
                          batch_size=int(args.benchmark_batch_size))

    for chunk_id, slots in enumerate(chunk_slots):
        path = chunk_dir / f"chunk_{chunk_id:05d}.pt"
        if path.exists():
            print(f"[bok] skip existing chunk {chunk_id}", flush=True)
            continue
        if policy is None:
            policy = build_policy(cfg_root, ldm_cfg, ckpt=base_ckpt, device=args.device)

        cond = pool.batch_from_indices(slots)
        _seed_all(args.seed * 1_000_003 + 1000 + chunk_id, args.device)
        x_agent, x_lane = sample_base_scene_latents(policy, cond)
        gen_cond = make_generated_cond(policy, cond, x_agent, x_lane)

        payloads = []
        for i in range(k):
            _seed_all(_draw_seed(int(args.seed), chunk_id, i, k), args.device)
            with torch.no_grad():
                scenes, _ = policy.sample(gen_cond)
            payloads.append(scenes_to_payload(scenes))
        print(f"[bok] chunk {chunk_id + 1}/{len(chunk_slots)}: sampled {k} draws", flush=True)

        # One rollout over every draw at once. A single chunk is far too small to
        # feed the worker pool (32 scenes is 4 shard blocks), so the draws are
        # concatenated into one k*chunk-scene batch and split by the benchmark's
        # own batching, which keeps every worker busy.
        metrics = benchmark_payload(
            reward, cat_payloads(payloads),
            batch_size=int(args.benchmark_batch_size),
            label=f"chunk {chunk_id + 1}/{len(chunk_slots)}",
        )
        R = metrics["reward"].reshape(k, len(slots)).T
        C = metrics["ego_collision"].reshape(k, len(slots)).T

        # Keep the selection for every budget on the doubling ladder, not just
        # the largest: the paper's claim is a CROSSOVER ("one AdvScene sample is
        # worth K base samples"), which can only be read off if each K is scored
        # with the same planner-quality metrics as the AdvScene row. Budget k
        # uses the first k draws, i.e. one honest run of "sample k, keep the
        # best", and costs nothing extra because the rollouts are already done.
        selected = {}
        for budget in _ladder(k):
            best = R[:, :budget].argmax(axis=1)
            selected[budget] = cat_payloads([
                scenes_to_payload(slice_payload(payloads[int(b)], s, s + 1))
                for s, b in enumerate(best)
            ])
        torch.save(
            {
                "payloads": selected,
                "slots": slots,
                "dataset_scene_idx": [int(pool.resolved_scene_idx[s]) for s in slots],
                "reward": R,
                "ego_collision": C,
            },
            path,
        )
        pool._cache.clear()
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    reward.close()
    del policy
    gc.collect()

    blobs = [torch.load(chunk_dir / f"chunk_{i:05d}.pt", map_location="cpu", weights_only=False)
             for i in range(len(chunk_slots))]
    merged = None
    for budget in _ladder(k):
        merged = cat_payloads([b["payloads"][budget] for b in blobs])
        metadata = build_metadata(
            source=f"base_gen_bok{budget}",
            config_name=args.config_name,
            overrides=list(args.overrides),
            split=args.split,
            seed=int(args.seed),
            slots=[s for b in blobs for s in b["slots"]],
            resolved_scene_idx=[i for b in blobs for i in b["dataset_scene_idx"]],
            base_ckpt=base_ckpt,
            ddpo_ckpt="(none: best-of-k draws the frozen base model)",
            cfg_root=cfg_root,
        )
        metadata["num_draws"] = budget
        merged_path = out_dir / "artifacts" / f"base_gen_bok{budget}.pt"
        torch.save({"payload": merged, "metadata": metadata}, merged_path)
        print(f"[bok] wrote {merged_path}", flush=True)

    R = np.concatenate([b["reward"] for b in blobs], axis=0)
    C = np.concatenate([b["ego_collision"] for b in blobs], axis=0)
    keep = ego_goal_dist(merged) >= float(cfg_root.ddpo.min_ego_drive)
    curve = _curve(R, C, keep)
    write_json(out_dir / f"bok{k}_curve.json", {
        "num_draws": k,
        "num_scenes": int(R.shape[0]),
        "num_driving_ego": int(keep.sum()),
        "selection": "max DDPO reward within the k-subset",
        "curve": curve,
    })
    print(json.dumps(curve, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
