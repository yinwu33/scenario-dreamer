#!/usr/bin/env python
"""Headroom probe: best-of-N sampling from the FROZEN ldm_adv base checkpoint.

Answers "what reward / collision rate could DDPO ever reach?" before spending
training compute: DDPO + KL-to-base can only sharpen the base model's own
distribution, so if best-of-N sampling from the base cannot find critical
adversaries in a context, no amount of fine-tuning anchored to that base will
either. Conversely a large best-of-N vs mean gap means the ceiling is high and
the problem is optimization, not capacity.

For M contexts (same conditioning pool + adv_cond_target + validity gates as
DDPO training), draw N adversaries each from the base model, roll every sample
out with the composed planner trio, and report:

  * best-of-k curves for k = 1, 2, 4, ... N: expected max reward over k draws
    (exact order-statistics estimator) and P(>= 1 adv-ego collision within k)
    (hypergeometric, per context, averaged);
  * the "attackable context" fraction: contexts where at least one of the N
    samples collides / is an ego-fault collision / clears a reward threshold.

Planner roles are composed exactly like DDPO training, so a pair is selected
on the command line with no code change:

    python scripts/headroom_probe.py --sut idm --env idm --adv idm \
        --num-contexts 64 --samples-per-context 64

Read-only: no training, no wandb; results go to a JSON + stdout table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cfgs.config import CONFIG_PATH


# Per-scene metric columns collected from RewardModel.evaluate.
_METRIC_KEYS = (
    "reward",
    "ego_collision",
    "ego_fault_collision",
    "r_risk",
    "r_ttc",
    "r_approach",
    "ego_min_ttc",
    "ego_adv_min_dist_warmup",
    "c_invalid",
    "init_invalid",
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sut", default="idm", help="planner name for the ego role")
    p.add_argument("--env", default="idm", help="planner name for background traffic")
    p.add_argument("--adv", default="idm", help="planner name driving the adversary")
    p.add_argument("--num-contexts", type=int, default=64, help="M distinct conditioning scenes")
    p.add_argument("--samples-per-context", type=int, default=64, help="N base-model draws per scene")
    p.add_argument("--chunk-scenes", type=int, default=128, help="max scenes per sampling forward")
    p.add_argument("--split", default=None,
                   help="dataset split for the conditioning pool (default: the config's train_split)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward-threshold", type=float, default=0.5,
                   help="per-context 'high reward reachable' threshold for the summary")
    p.add_argument("--out", default=None,
                   help="output JSON path (default: data/headroom_probe/<sut>-<env>-<adv>.json)")
    p.add_argument("--workers", type=int, default=0,
                   help="shard each rollout across N worker processes (sim.parallel); "
                        "bit-exact, so this is a pure throughput knob")
    p.add_argument("--override", action="append", default=[],
                   help="extra hydra overrides (repeatable)")
    return p.parse_args()


def _compose_cfg(args):
    overrides = [
        f"planner@ddpo.planner.sut={args.sut}",
        f"planner@ddpo.planner.env={args.env}",
        f"planner@ddpo.planner.adv={args.adv}",
        f"experiment.planner_name={args.sut}-{args.env}",
        *( [f"ddpo.train_split={args.split}"] if args.split else [] ),
        # The probe never touches the training run's output_dir, but resolve it
        # to something inert anyway.
        "experiment.output_dir=${scratch_root}/critical_scene/headroom_probe",
        *args.override,
    ]
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        return compose(config_name="config_ldm_adv_ddpo", overrides=overrides)


def _best_of_k_mean_max(rewards_sorted: np.ndarray, k: int) -> float:
    """E[max of k draws without replacement] from one context's n sorted rewards."""
    n = len(rewards_sorted)
    denom = comb(n, k)
    # P(max is the j-th smallest) = C(j-1, k-1) / C(n, k), j = 1..n
    acc = 0.0
    for j in range(k, n + 1):
        acc += comb(j - 1, k - 1) / denom * float(rewards_sorted[j - 1])
    return acc


def _hit_prob_within_k(n: int, hits: int, k: int) -> float:
    """P(>= 1 hit among k draws without replacement | `hits` of n samples hit)."""
    if hits <= 0:
        return 0.0
    if hits >= n:
        return 1.0
    return 1.0 - comb(n - hits, k) / comb(n, k)


def main():
    args = _parse_args()
    t0 = time.time()
    cfg_root = _compose_cfg(args)
    cfg = cfg_root.ddpo
    device = cfg.device
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    # Reuse the exact DDPO construction path so the probe measures the same
    # base checkpoint, conditioning targets, validity gates and reward.
    from ddpo.train_loop import _build_policy_and_pool, _build_gen_invalid
    from ddpo.reward import RewardModel, build_reward_config
    from sim.runner import SimulatorConfig

    model_type, policy, pool, eval_dataset_cfg = _build_policy_and_pool(cfg_root, cfg, device)
    reward = RewardModel(
        planner_cfg=cfg.planner,
        simulator_cfg=SimulatorConfig(
            seed=int(args.seed),
            gen_invalid=_build_gen_invalid(cfg, eval_dataset_cfg),
            **OmegaConf.to_container(cfg.simulator, resolve=True),
        ),
        reward_cfg=build_reward_config(cfg.reward),
        num_workers=int(args.workers),
        train_batch_size=int(args.chunk_scenes),
    )

    M = int(args.num_contexts)
    N = int(args.samples_per_context)
    if len(pool) < M:
        raise SystemExit(f"pool has only {len(pool)} contexts, need {M}")
    slot_rng = np.random.default_rng(args.seed)
    slots = np.sort(slot_rng.choice(len(pool), size=M, replace=False))

    groups_per_chunk = max(1, int(args.chunk_scenes) // N)
    per_scene: dict[str, list[np.ndarray]] = {k: [] for k in _METRIC_KEYS}
    done = 0
    for lo in range(0, M, groups_per_chunk):
        chunk = slots[lo : lo + groups_per_chunk]
        idx = np.repeat(chunk, N)
        cond = pool.batch_from_indices(idx)
        scenes, _ = policy.sample(cond)
        metrics = reward.evaluate(scenes)
        for k in _METRIC_KEYS:
            per_scene[k].append(np.asarray(metrics[k], dtype=np.float64))
        done += len(chunk)
        print(
            f"[probe] contexts {done}/{M} "
            f"({done * N} rollouts, {time.time() - t0:.0f}s elapsed)",
            flush=True,
        )

    data = {k: np.concatenate(v).reshape(M, N) for k, v in per_scene.items()}

    # ---- per-context aggregates -------------------------------------------
    rewards = data["reward"]
    coll = data["ego_collision"] > 0
    fault = data["ego_fault_collision"] > 0
    near_miss = data["r_risk"] > 0.5
    valid = (data["c_invalid"] <= 0) & (data["init_invalid"] <= 0)

    coll_hits = coll.sum(axis=1)
    fault_hits = fault.sum(axis=1)
    near_hits = near_miss.sum(axis=1)
    thr_hits = (rewards >= args.reward_threshold).sum(axis=1)

    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= N]
    rewards_sorted = np.sort(rewards, axis=1)
    curve = []
    for k in ks:
        best_r = float(np.mean([_best_of_k_mean_max(rewards_sorted[m], k) for m in range(M)]))
        p_coll = float(np.mean([_hit_prob_within_k(N, int(coll_hits[m]), k) for m in range(M)]))
        p_fault = float(np.mean([_hit_prob_within_k(N, int(fault_hits[m]), k) for m in range(M)]))
        p_near = float(np.mean([_hit_prob_within_k(N, int(near_hits[m]), k) for m in range(M)]))
        curve.append({
            "k": k,
            "mean_best_reward": best_r,
            "p_collision": p_coll,
            "p_ego_fault": p_fault,
            "p_near_miss": p_near,
        })

    ttc = data["ego_min_ttc"]
    finite_ttc = np.isfinite(ttc)
    summary = {
        "pair": f"sut={args.sut} env={args.env} adv={args.adv}",
        "num_contexts": M,
        "samples_per_context": N,
        "seed": int(args.seed),
        "split": str(cfg.train_split),
        "base_mean_reward": float(rewards.mean()),
        "base_collision_rate": float(coll.mean()),
        "base_ego_fault_rate": float(fault.mean()),
        "base_near_miss_rate": float(near_miss.mean()),
        "base_valid_rate": float(valid.mean()),
        "base_min_ttc_mean": float(ttc[finite_ttc].mean()) if finite_ttc.any() else None,
        "base_adv_min_dist_mean": float(np.nanmean(
            np.where(np.isfinite(data["ego_adv_min_dist_warmup"]),
                     data["ego_adv_min_dist_warmup"], np.nan)
        )),
        "frac_contexts_with_collision": float((coll_hits > 0).mean()),
        "frac_contexts_with_ego_fault": float((fault_hits > 0).mean()),
        "frac_contexts_with_near_miss": float((near_hits > 0).mean()),
        f"frac_contexts_reward_ge_{args.reward_threshold}": float((thr_hits > 0).mean()),
        "collision_hits_per_context_hist": np.bincount(
            coll_hits.astype(int), minlength=N + 1
        ).tolist(),
        "best_of_k": curve,
        "per_context": {
            "pool_slot": slots.tolist(),
            "scene_idx": [int(pool.resolved_scene_idx.get(int(s), -1)) for s in slots],
            "collision_hits": coll_hits.astype(int).tolist(),
            "ego_fault_hits": fault_hits.astype(int).tolist(),
            "near_miss_hits": near_hits.astype(int).tolist(),
            "max_reward": rewards.max(axis=1).tolist(),
            "mean_reward": rewards.mean(axis=1).tolist(),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }

    out = Path(args.out or f"data/headroom_probe/{args.sut}-{args.env}-{args.adv}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n=== headroom probe: {summary['pair']} | {M} contexts x {N} samples ===")
    print(f"base (k=1): reward={summary['base_mean_reward']:.3f} "
          f"coll={summary['base_collision_rate']:.3f} "
          f"fault={summary['base_ego_fault_rate']:.3f} "
          f"near_miss={summary['base_near_miss_rate']:.3f} "
          f"valid={summary['base_valid_rate']:.3f} "
          f"adv_dist={summary['base_adv_min_dist_mean']:.2f}m")
    print(f"contexts with >=1 in {N} samples: "
          f"collision {summary['frac_contexts_with_collision']:.1%}, "
          f"ego-fault {summary['frac_contexts_with_ego_fault']:.1%}, "
          f"near-miss {summary['frac_contexts_with_near_miss']:.1%}, "
          f"reward>={args.reward_threshold} "
          f"{summary[f'frac_contexts_reward_ge_{args.reward_threshold}']:.1%}")
    print(f"{'k':>4} {'E[best reward]':>15} {'P(collision)':>13} {'P(ego-fault)':>13} {'P(near-miss)':>13}")
    for row in curve:
        print(f"{row['k']:>4} {row['mean_best_reward']:>15.3f} {row['p_collision']:>13.3f} "
              f"{row['p_ego_fault']:>13.3f} {row['p_near_miss']:>13.3f}")
    print(f"\nwrote {out} ({summary['elapsed_s']}s)")
    # Shut the rollout workers down; without this they outlive the probe as
    # orphans holding their shared-memory blocks.
    reward.close()


if __name__ == "__main__":
    main()
