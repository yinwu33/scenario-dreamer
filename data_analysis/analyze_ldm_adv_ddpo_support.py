#!/usr/bin/env python3
"""Measure whether frozen ldm_adv has high-reward support for DDPO.

This script answers the practical DDPO question:

  Does the base diffusion policy already sample a non-trivial critical-scene
  tail under the same conditioning / reward configuration used by DDPO?

It composes ``cfgs/config_critical_scene_ldm_adv_ddpo.yaml``, builds the same
LDMAdv policy, conditioning pools, and PufferDrive reward used by training, then
draws K independent base samples per conditioning scene. Outputs:

  * per-scene CSV: reward tail statistics for every evaluated context;
  * top-samples CSV: high reward / suspicious samples for inspection;
  * aggregate JSON: machine-readable split summaries;
  * markdown report: DDPO feasibility conclusion.

Run from the repository root, usually with:

    source scripts/define_env_variables.sh
    .venv/bin/python data_analysis/analyze_ldm_adv_ddpo_support.py \
        --splits train val --num-scenes 1000 --samples-per-scene 16

Use ``--num-scenes 0`` to evaluate the full configured pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _default_env() -> None:
    """Mirror scripts/define_env_variables.sh when the caller forgot to source it."""
    os.environ.setdefault("PROJECT_ROOT", str(REPO_ROOT))
    os.environ.setdefault("SCRATCH_ROOT", "data")
    os.environ.setdefault("DATASET_ROOT", os.environ["SCRATCH_ROOT"])
    os.environ.setdefault("CONFIG_PATH", str(REPO_ROOT / "cfgs"))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    pythonpath = os.environ.get("PYTHONPATH", "")
    if str(REPO_ROOT) not in pythonpath.split(":"):
        os.environ["PYTHONPATH"] = f"{REPO_ROOT}:{pythonpath}" if pythonpath else str(REPO_ROOT)


def _finite_mean(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def _finite_min(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.min()) if v.size else float("nan")


def _finite_max(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else float("nan")


def _finite_percentile(values: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else float("nan")


def _rate(mask: np.ndarray) -> float:
    m = np.asarray(mask)
    return float(m.mean()) if m.size else float("nan")


def _scalar(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _json_dump(path: Path, obj: Any) -> None:
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(val) for k, val in v.items()}
        if isinstance(v, list):
            return [clean(val) for val in v]
        if isinstance(v, tuple):
            return [clean(val) for val in v]
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, (np.floating, np.integer)):
            return clean(v.item())
        if isinstance(v, float):
            return v if math.isfinite(v) else None
        return v

    path.write_text(json.dumps(clean(obj), indent=2, sort_keys=True), encoding="utf-8")


def _load_cfg(config_name: str, overrides: list[str]):
    _default_env()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_dir = os.environ.get("CONFIG_PATH", str(REPO_ROOT / "cfgs"))
    with initialize_config_dir(version_base=None, config_dir=str(Path(config_dir).resolve())):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def _choose_device(requested: str, cfg_device: str) -> str:
    if requested == "config":
        device = str(cfg_device)
    elif requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return device


def _build_reward(cfg):
    from ddpo.reward import PufferDriveReward

    return PufferDriveReward(
        planner_cfg=cfg.get("planner", None),
        sim_steps=cfg.sim_steps,
        deterministic=cfg.get("planner_deterministic", None),
        ttc_tau=cfg.get("ttc_tau", 3.0),
        init_overlap_margin=cfg.get("init_overlap_margin", 0.0),
        init_overlap_penalty=cfg.get("init_overlap_penalty", 1.0),
        init_overlap_gate_lo=cfg.get("init_overlap_gate_lo", 0.02),
        init_overlap_gate_hi=cfg.get("init_overlap_gate_hi", 0.20),
        goal_offlane_threshold=cfg.get("goal_offlane_threshold", 3.0),
        goal_onroad_threshold=cfg.get("goal_onroad_threshold", 2.0),
        goal_offlane_penalty=cfg.get("goal_offlane_penalty", 0.5),
        parking_mismatch_penalty=cfg.get("parking_mismatch_penalty", 0.5),
        min_dist_coef=cfg.get("min_dist_coef", 0.0),
        min_dist_dmax=cfg.get("min_dist_dmax", 20.0),
        gen_agent_parking_penalty=cfg.get("gen_agent_parking_penalty", 0.0),
        risk_coef=cfg.get("risk_coef", 1.0),
        approach_d_safe=cfg.get("approach_d_safe", 6.0),
        approach_d_scale=cfg.get("approach_d_scale", 2.0),
        approach_close_delta=cfg.get("approach_close_delta", 2.0),
        approach_close_scale=cfg.get("approach_close_scale", 1.0),
        approach_warmup_time=cfg.get("approach_warmup_time", 0.5),
        lane_soft=cfg.get("lane_soft", 0.5),
        collision_enabled=cfg.get("collision_enabled", False),
        collision_coef=cfg.get("collision_coef", 0.0),
        collision_warmup=cfg.get("collision_warmup", 0.75),
        collision_window=cfg.get("collision_window", 0.5),
        trivial_collision_t=cfg.get("trivial_collision_t", 0.75),
        trivial_collision_penalty=cfg.get("trivial_collision_penalty", 0.5),
        seed=cfg.seed,
        backend=cfg.get("reward_backend", "numpy"),
        pufferdrive_root=cfg.get("pufferdrive_root", None),
    )


def _build_split_pool(model_type: str, eval_dataset_cfg, cfg, split: str, device: str):
    from ddpo.conditioning import ConditioningPool, LDMAdvConditioningPool, LDMGoalConditioningPool
    from datasets.waymo.dataset_dm_fixed_map_agent_goal_waymo import WaymoDatasetDMFixedMapAgentGoal

    if model_type == "ldm_adv":
        return LDMAdvConditioningPool(
            eval_dataset_cfg,
            split_name=split,
            pool_size=cfg.pool_size,
            device=device,
            seed=cfg.seed,
            min_ego_drive=cfg.get("min_ego_drive", 10.0),
            prune_base_to_ego=cfg.get("prune_base_to_ego", False),
            adv_cond_target=cfg.get("adv_cond_target", None),
        )
    if model_type == "ldm_goal":
        return LDMGoalConditioningPool(
            eval_dataset_cfg,
            split_name=split,
            pool_size=cfg.pool_size,
            device=device,
            seed=cfg.seed,
        )
    pool_kwargs = {
        "control_agent_num": cfg.get("control_agent_num", -1),
        "ego_goal_override": cfg.get("ego_goal_override", None),
    }
    if model_type == "dm_fixed_map_agent_goal":
        pool_kwargs["dataset_cls"] = WaymoDatasetDMFixedMapAgentGoal
    return ConditioningPool(
        eval_dataset_cfg,
        split_name=split,
        pool_size=cfg.pool_size,
        device=device,
        seed=cfg.seed,
        **pool_kwargs,
    )


def _load_policy_checkpoint(policy, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=policy.device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    if any(k.startswith("diff_model.") for k in sd):
        sd = {k[len("diff_model.") :]: v for k, v in sd.items() if k.startswith("diff_model.")}
    missing, unexpected = policy.net.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] policy checkpoint missing {len(missing)} keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"[warn] policy checkpoint unexpected {len(unexpected)} keys, e.g. {unexpected[:3]}")


def _select_pool_indices(pool_len: int, num_scenes: int, seed: int, mode: str) -> list[int]:
    if num_scenes == 0 or num_scenes >= pool_len:
        return list(range(pool_len))
    rng = random.Random(seed)
    if mode == "first":
        return list(range(num_scenes))
    return sorted(rng.sample(range(pool_len), num_scenes))


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _adv_attributes(scenes) -> dict[str, np.ndarray]:
    states = _to_numpy(scenes.agent_states)
    types = _to_numpy(scenes.agent_types).astype(np.int64)
    scene_idx = _to_numpy(scenes.agent_scene_idx).astype(np.int64)
    gen_agent_mask = scenes.meta.get("gen_agent_mask")
    if gen_agent_mask is None:
        gen_agent_mask = np.zeros(len(states), dtype=bool)
    else:
        gen_agent_mask = _to_numpy(gen_agent_mask).astype(bool)

    rows: list[dict[str, float]] = []
    for s in range(int(scenes.num_scenes)):
        global_idx = np.nonzero(scene_idx == s)[0]
        local_gen_agent = [
            g for local_i, g in enumerate(global_idx)
            if local_i > 0 and gen_agent_mask[g]
        ]
        if not local_gen_agent:
            rows.append({
                "adv_type": -1,
                "adv_x": float("nan"),
                "adv_y": float("nan"),
                "adv_speed": float("nan"),
                "adv_length": float("nan"),
                "adv_width": float("nan"),
                "adv_goal_dist": float("nan"),
                "adv_dist_to_ego": float("nan"),
                "adv_in_fov": False,
                "adv_small_vehicle": False,
            })
            continue
        g = local_gen_agent[0]
        ego = states[global_idx[0]]
        adv = states[g]
        goal_dist = float(np.hypot(adv[7] - adv[0], adv[8] - adv[1]))
        dist_to_ego = float(np.hypot(adv[0] - ego[0], adv[1] - ego[1]))
        in_fov = bool(abs(float(adv[0])) <= 32.0 and abs(float(adv[1])) <= 32.0)
        small_vehicle = bool(types[g] == 0 and (float(adv[5]) < 3.0 or float(adv[6]) < 1.5))
        rows.append({
            "adv_type": int(types[g]),
            "adv_x": float(adv[0]),
            "adv_y": float(adv[1]),
            "adv_speed": float(adv[2]),
            "adv_length": float(adv[5]),
            "adv_width": float(adv[6]),
            "adv_goal_dist": goal_dist,
            "adv_dist_to_ego": dist_to_ego,
            "adv_in_fov": in_fov,
            "adv_small_vehicle": small_vehicle,
        })

    out: dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        out[key] = np.asarray([r[key] for r in rows])
    return out


def _sample_fields(metrics: dict[str, np.ndarray], attrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    fields = {
        "reward": metrics["reward"],
        "r_ttc": metrics["r_ttc"],
        "r_approach": metrics["r_approach"],
        "r_risk": metrics["r_risk"],
        "criticality": metrics["criticality"],
        "constraint": metrics["constraint"],
        "c_lane": metrics["c_lane"],
        "c_trivial": metrics["c_trivial"],
        "ego_collision": metrics["ego_collision"],
        "ego_fault_collision": metrics.get("ego_fault_collision", np.zeros_like(metrics["reward"])),
        "init_invalid": metrics["init_invalid"],
        "goal_offlane_frac": metrics["goal_offlane_frac"],
        "gen_agent_is_parked": metrics.get("gen_agent_is_parked", np.zeros_like(metrics["reward"])),
        "ego_min_ttc": metrics["ego_min_ttc"],
        "ego_adv_min_dist": metrics["ego_adv_min_dist"],
        "ego_adv_init_dist": metrics["ego_adv_init_dist"],
        "ego_adv_min_dist_warmup": metrics["ego_adv_min_dist_warmup"],
    }
    fields.update(attrs)
    return {k: np.asarray(v) for k, v in fields.items()}


def _scene_stats_from_samples(
    split: str,
    policy_name: str,
    pool_indices: list[int],
    sample_fields: dict[str, np.ndarray],
    samples_per_scene: int,
    high_reward_threshold: float,
) -> list[dict[str, Any]]:
    n = len(pool_indices)
    rows = []
    reward = sample_fields["reward"].reshape(n, samples_per_scene)
    r_risk = sample_fields["r_risk"].reshape(n, samples_per_scene)
    init_invalid = sample_fields["init_invalid"].reshape(n, samples_per_scene)
    collision = sample_fields["ego_collision"].reshape(n, samples_per_scene)
    offlane = sample_fields["goal_offlane_frac"].reshape(n, samples_per_scene)
    parking = sample_fields["gen_agent_is_parked"].reshape(n, samples_per_scene)
    small = sample_fields["adv_small_vehicle"].reshape(n, samples_per_scene).astype(bool)
    min_dist = sample_fields["ego_adv_min_dist"].reshape(n, samples_per_scene)
    approach = sample_fields["r_approach"].reshape(n, samples_per_scene)
    ttc = sample_fields["r_ttc"].reshape(n, samples_per_scene)
    constraint = sample_fields["constraint"].reshape(n, samples_per_scene)
    adv_len = sample_fields["adv_length"].reshape(n, samples_per_scene)
    adv_width = sample_fields["adv_width"].reshape(n, samples_per_scene)

    for i, pool_idx in enumerate(pool_indices):
        rows.append({
            "split": split,
            "policy": policy_name,
            "pool_idx": int(pool_idx),
            "n_samples": int(samples_per_scene),
            "reward_mean": _finite_mean(reward[i]),
            "reward_max": _finite_max(reward[i]),
            "reward_p50": _finite_percentile(reward[i], 50),
            "reward_p90": _finite_percentile(reward[i], 90),
            "reward_p95": _finite_percentile(reward[i], 95),
            "positive_rate": _rate(reward[i] > 0.0),
            "high_reward_rate": _rate(reward[i] > high_reward_threshold),
            "near_miss_rate": _rate(r_risk[i] > 0.5),
            "collision_rate": _rate(collision[i] > 0.0),
            "init_invalid_rate": _rate(init_invalid[i] > 0.0),
            "goal_offlane_rate": _rate(offlane[i] > 0.0),
            "gen_agent_parking_rate": _rate(parking[i] > 0.0),
            "small_vehicle_rate": _rate(small[i]),
            "ego_adv_min_dist_mean": _finite_mean(min_dist[i]),
            "ego_adv_min_dist_min": _finite_min(min_dist[i]),
            "r_approach_mean": _finite_mean(approach[i]),
            "r_ttc_mean": _finite_mean(ttc[i]),
            "constraint_mean": _finite_mean(constraint[i]),
            "adv_length_mean": _finite_mean(adv_len[i]),
            "adv_width_mean": _finite_mean(adv_width[i]),
        })
    return rows


def _condition_stats(
    split: str,
    pool_indices: list[int],
    metrics: dict[str, np.ndarray],
    attrs: dict[str, np.ndarray],
    high_reward_threshold: float,
) -> list[dict[str, Any]]:
    fields = _sample_fields(metrics, attrs)
    rows = []
    for i, pool_idx in enumerate(pool_indices):
        rows.append({
            "split": split,
            "policy": "no_adv",
            "pool_idx": int(pool_idx),
            "n_samples": 1,
            "reward_mean": float(fields["reward"][i]),
            "reward_max": float(fields["reward"][i]),
            "reward_p50": float(fields["reward"][i]),
            "reward_p90": float(fields["reward"][i]),
            "reward_p95": float(fields["reward"][i]),
            "positive_rate": float(fields["reward"][i] > 0.0),
            "high_reward_rate": float(fields["reward"][i] > high_reward_threshold),
            "near_miss_rate": float(fields["r_risk"][i] > 0.5),
            "collision_rate": float(fields["ego_collision"][i] > 0.0),
            "init_invalid_rate": float(fields["init_invalid"][i] > 0.0),
            "goal_offlane_rate": float(fields["goal_offlane_frac"][i] > 0.0),
            "gen_agent_parking_rate": float(fields["gen_agent_is_parked"][i] > 0.0),
            "small_vehicle_rate": float(fields["adv_small_vehicle"][i]),
            "ego_adv_min_dist_mean": float(fields["ego_adv_min_dist"][i]),
            "ego_adv_min_dist_min": float(fields["ego_adv_min_dist"][i]),
            "r_approach_mean": float(fields["r_approach"][i]),
            "r_ttc_mean": float(fields["r_ttc"][i]),
            "constraint_mean": float(fields["constraint"][i]),
            "adv_length_mean": float(fields["adv_length"][i]),
            "adv_width_mean": float(fields["adv_width"][i]),
        })
    return rows


def _flatten_sample_rows(
    split: str,
    policy_name: str,
    pool_indices: list[int],
    sample_fields: dict[str, np.ndarray],
    samples_per_scene: int,
) -> list[dict[str, Any]]:
    rows = []
    total = len(pool_indices) * samples_per_scene
    pool_for_sample = np.repeat(np.asarray(pool_indices), samples_per_scene)
    sample_id = np.tile(np.arange(samples_per_scene), len(pool_indices))
    for j in range(total):
        row = {
            "split": split,
            "policy": policy_name,
            "pool_idx": int(pool_for_sample[j]),
            "sample_id": int(sample_id[j]),
        }
        for k, v in sample_fields.items():
            val = v[j]
            if isinstance(val, np.bool_):
                row[k] = bool(val)
            elif np.issubdtype(np.asarray(v).dtype, np.integer):
                row[k] = int(val)
            else:
                row[k] = float(val)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _scalar(row.get(k)) for k in fieldnames})


def _aggregate_scene_rows(rows: list[dict[str, Any]], high_reward_threshold: float) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["split"], row["policy"])].append(row)

    out = {}
    for (split, policy), group in sorted(by_key.items()):
        reward_mean = np.asarray([r["reward_mean"] for r in group], dtype=np.float64)
        reward_max = np.asarray([r["reward_max"] for r in group], dtype=np.float64)
        high_rate = np.asarray([r["high_reward_rate"] for r in group], dtype=np.float64)
        summary = {
            "n_scenes": len(group),
            "high_reward_threshold": high_reward_threshold,
            "scene_with_positive_sample_rate": _rate(np.asarray([r["positive_rate"] > 0.0 for r in group])),
            "scene_with_high_reward_sample_rate": _rate(reward_max > high_reward_threshold),
            "mean_sample_high_reward_rate": _finite_mean(high_rate),
            "reward_mean_mean": _finite_mean(reward_mean),
            "reward_mean_p50": _finite_percentile(reward_mean, 50),
            "reward_mean_p95": _finite_percentile(reward_mean, 95),
            "reward_max_mean": _finite_mean(reward_max),
            "reward_max_p50": _finite_percentile(reward_max, 50),
            "reward_max_p90": _finite_percentile(reward_max, 90),
            "reward_max_p95": _finite_percentile(reward_max, 95),
            "reward_max_p99": _finite_percentile(reward_max, 99),
            "near_miss_rate_mean": _finite_mean([r["near_miss_rate"] for r in group]),
            "collision_rate_mean": _finite_mean([r["collision_rate"] for r in group]),
            "init_invalid_rate_mean": _finite_mean([r["init_invalid_rate"] for r in group]),
            "goal_offlane_rate_mean": _finite_mean([r["goal_offlane_rate"] for r in group]),
            "gen_agent_parking_rate_mean": _finite_mean([r["gen_agent_parking_rate"] for r in group]),
            "small_vehicle_rate_mean": _finite_mean([r["small_vehicle_rate"] for r in group]),
            "ego_adv_min_dist_min_p50": _finite_percentile(
                [r["ego_adv_min_dist_min"] for r in group], 50
            ),
            "ego_adv_min_dist_min_p10": _finite_percentile(
                [r["ego_adv_min_dist_min"] for r in group], 10
            ),
        }
        out[f"{split}/{policy}"] = summary
    return out


def _aggregate_samples(sample_rows: list[dict[str, Any]], high_reward_threshold: float) -> dict[str, Any]:
    out = {}
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        by_key[(row["split"], row["policy"])].append(row)
    for (split, policy), group in sorted(by_key.items()):
        reward = np.asarray([r["reward"] for r in group], dtype=np.float64)
        high = reward > high_reward_threshold
        invalid = np.asarray([r["init_invalid"] > 0.0 for r in group])
        offlane = np.asarray([r["goal_offlane_frac"] > 0.0 for r in group])
        parking = np.asarray([r["gen_agent_is_parked"] > 0.0 for r in group])
        small = np.asarray([bool(r["adv_small_vehicle"]) for r in group])
        collision = np.asarray([r["ego_collision"] > 0.0 for r in group])
        fault_collision = np.asarray([r["ego_fault_collision"] > 0.0 for r in group])
        near_miss = np.asarray([r["r_risk"] > 0.5 for r in group])
        artifact = invalid | offlane | parking | small
        clean_high = high & (~artifact)
        out[f"{split}/{policy}"] = {
            "n_samples": len(group),
            "reward_mean": _finite_mean(reward),
            "reward_p50": _finite_percentile(reward, 50),
            "reward_p90": _finite_percentile(reward, 90),
            "reward_p95": _finite_percentile(reward, 95),
            "reward_p99": _finite_percentile(reward, 99),
            "reward_max": _finite_max(reward),
            "positive_sample_rate": _rate(reward > 0.0),
            "high_reward_sample_rate": _rate(high),
            "clean_high_reward_sample_rate": _rate(clean_high),
            "artifact_sample_rate": _rate(artifact),
            "small_vehicle_sample_rate": _rate(small),
            "artifact_given_high_reward_rate": _rate(artifact[high]) if high.any() else float("nan"),
            "small_vehicle_given_high_reward_rate": _rate(small[high]) if high.any() else float("nan"),
            "near_miss_given_high_reward_rate": _rate(near_miss[high]) if high.any() else float("nan"),
            "collision_given_high_reward_rate": _rate(collision[high]) if high.any() else float("nan"),
            "fault_collision_given_high_reward_rate": _rate(fault_collision[high]) if high.any() else float("nan"),
            "near_miss_given_clean_high_reward_rate": _rate(near_miss[clean_high]) if clean_high.any() else float("nan"),
            "collision_given_clean_high_reward_rate": _rate(collision[clean_high]) if clean_high.any() else float("nan"),
        }
    return out


def _feasibility(summary: dict[str, Any], split: str = "train") -> tuple[str, list[str]]:
    key = f"{split}/base"
    if key not in summary["sample_summary"] or key not in summary["scene_summary"]:
        return "unknown", [f"missing {key} summary"]
    ss = summary["sample_summary"][key]
    cs = summary["scene_summary"][key]
    sample_tail = ss["high_reward_sample_rate"]
    scene_tail = cs["scene_with_high_reward_sample_rate"]
    artifact_high = ss["artifact_given_high_reward_rate"]
    invalid = cs["init_invalid_rate_mean"]
    small = cs["small_vehicle_rate_mean"]

    reasons = [
        f"base high-reward sample rate={sample_tail:.4f}",
        f"base scene coverage with any high sample={scene_tail:.4f}",
        f"artifact among high reward={artifact_high if artifact_high is not None else float('nan'):.4f}",
        f"init-invalid mean={invalid:.4f}",
        f"small-vehicle mean={small:.4f}",
    ]
    if sample_tail >= 0.01 and scene_tail >= 0.10 and (math.isnan(artifact_high) or artifact_high <= 0.30):
        return "favorable", reasons
    if sample_tail >= 0.002 and scene_tail >= 0.03 and (math.isnan(artifact_high) or artifact_high <= 0.50):
        return "conditional", reasons
    if small > 0.02 or (not math.isnan(artifact_high) and artifact_high > 0.50):
        return "blocked_by_artifacts", reasons
    return "weak_base_support", reasons


def _write_report(path: Path, cfg, args, summary: dict[str, Any], selected: dict[str, int]) -> None:
    verdict, reasons = _feasibility(summary, split="train" if "train/base" in summary["sample_summary"] else args.splits[0])
    lines = [
        "# LDMAdv DDPO Support Analysis",
        "",
        "## Run",
        f"- config: `{args.config_name}`",
        f"- splits: `{', '.join(args.splits)}`",
        f"- selected scenes: `{selected}`",
        f"- samples per scene: `{args.samples_per_scene}`",
        f"- device: `{args.device_resolved}`",
        f"- sampler: `{cfg.ddpo.get('sampler', 'ddpm')}`",
        f"- high reward threshold: `{args.high_reward_threshold}`",
        "",
        "## Conclusion",
        f"- ddpo_feasibility: `{verdict}`",
    ]
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend(["", "## Split Summary"])
    for key, ss in summary["sample_summary"].items():
        cs = summary["scene_summary"].get(key, {})
        lines.extend([
            f"### {key}",
            f"- n_samples: `{ss.get('n_samples')}`",
            f"- high_reward_sample_rate: `{ss.get('high_reward_sample_rate'):.6f}`",
            f"- clean_high_reward_sample_rate: `{ss.get('clean_high_reward_sample_rate', float('nan')):.6f}`",
            f"- positive_sample_rate: `{ss.get('positive_sample_rate'):.6f}`",
            f"- reward_p95 / p99 / max: `{ss.get('reward_p95'):.4f}` / `{ss.get('reward_p99'):.4f}` / `{ss.get('reward_max'):.4f}`",
            f"- scene_with_high_reward_sample_rate: `{cs.get('scene_with_high_reward_sample_rate', float('nan')):.6f}`",
            f"- artifact_given_high_reward_rate: `{ss.get('artifact_given_high_reward_rate', float('nan')):.6f}`",
            f"- near_miss_given_high_reward_rate: `{ss.get('near_miss_given_high_reward_rate', float('nan')):.6f}`",
            f"- collision_given_high_reward_rate: `{ss.get('collision_given_high_reward_rate', float('nan')):.6f}`",
            f"- near_miss_rate_mean: `{cs.get('near_miss_rate_mean', float('nan')):.6f}`",
            f"- collision_rate_mean: `{cs.get('collision_rate_mean', float('nan')):.6f}`",
            f"- init_invalid_rate_mean: `{cs.get('init_invalid_rate_mean', float('nan')):.6f}`",
            f"- goal_offlane_rate_mean: `{cs.get('goal_offlane_rate_mean', float('nan')):.6f}`",
            f"- gen_agent_parking_rate_mean: `{cs.get('gen_agent_parking_rate_mean', float('nan')):.6f}`",
            f"- small_vehicle_rate_mean: `{cs.get('small_vehicle_rate_mean', float('nan')):.6f}`",
            "",
        ])
    lines.extend([
        "## Interpretation Rules",
        "- `favorable`: base diffusion already has a usable high-reward tail; DDPO should mainly increase its probability.",
        "- `conditional`: base support exists but is sparse; DDPO may work only with careful KL/reward/artifact controls.",
        "- `weak_base_support`: base almost never samples high reward; DDPO will likely leave the base manifold.",
        "- `blocked_by_artifacts`: high reward is dominated by invalid/offlane/parked/small-vehicle artifacts.",
        "",
        "## Config Snapshot",
        "```yaml",
        OmegaConf.to_yaml(cfg.ddpo),
        "```",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_split(policy, pool, reward, split: str, indices: list[int], args) -> tuple[list[dict], list[dict], list[dict]]:
    scene_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    start_time = time.time()
    total = len(indices)
    chunk_contexts = max(1, int(args.chunk_contexts))

    for start in range(0, total, chunk_contexts):
        chunk_idx = indices[start : start + chunk_contexts]
        repeated = np.repeat(np.asarray(chunk_idx, dtype=np.int64), args.samples_per_scene).tolist()
        cond = pool.batch_from_indices(repeated)

        torch.manual_seed(int(args.seed) + start)
        scenes, _ = policy.sample(cond, use_reference=(args.policy == "base"))
        metrics = reward.evaluate(scenes)
        attrs = _adv_attributes(scenes)
        fields = _sample_fields(metrics, attrs)
        scene_rows.extend(
            _scene_stats_from_samples(
                split,
                args.policy,
                chunk_idx,
                fields,
                args.samples_per_scene,
                args.high_reward_threshold,
            )
        )
        sample_rows.extend(_flatten_sample_rows(split, args.policy, chunk_idx, fields, args.samples_per_scene))

        if args.include_conditioning:
            cond_context = pool.batch_from_indices(chunk_idx)
            cond_scenes = policy.conditioning_scenes(cond_context)
            cond_metrics = reward.evaluate(cond_scenes)
            cond_attrs = _adv_attributes(cond_scenes)
            condition_rows.extend(
                _condition_stats(split, chunk_idx, cond_metrics, cond_attrs, args.high_reward_threshold)
            )

        done = min(start + chunk_contexts, total)
        elapsed = time.time() - start_time
        print(f"[{split}] {done}/{total} contexts, elapsed={elapsed:.1f}s")

    return scene_rows, sample_rows, condition_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_critical_scene_ldm_adv_ddpo")
    ap.add_argument("--override", dest="overrides", action="append", default=[], help="Hydra override, repeatable")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--num-scenes", type=int, default=256, help="0 means all pool scenes")
    ap.add_argument("--samples-per-scene", type=int, default=8)
    ap.add_argument("--chunk-contexts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selection", choices=["random", "first"], default="random")
    ap.add_argument("--device", choices=["config", "auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--policy", choices=["base", "current"], default="base")
    ap.add_argument("--policy-ckpt", default=None, help="Optional DDPO checkpoint for --policy current")
    ap.add_argument("--include-conditioning", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--high-reward-threshold", type=float, default=0.3)
    ap.add_argument("--top-k-samples", type=int, default=200)
    ap.add_argument("--output-dir", default="data_analysis/ldm_adv_ddpo_support")
    args = ap.parse_args()

    cfg_root = _load_cfg(args.config_name, args.overrides)
    cfg = cfg_root.ddpo
    args.device_resolved = _choose_device(args.device, cfg.device)
    OmegaConf.set_struct(cfg, False)
    cfg.device = args.device_resolved
    OmegaConf.set_struct(cfg, True)
    if args.device_resolved == "cpu" and str(cfg.get("sampler", "ddpm")).lower() == "ddpm":
        print(
            "[warn] Running the configured DDPM sampler on CPU is extremely slow "
            "(1000 reverse steps per sample). Use a CUDA node for real analysis; "
            "CPU is suitable only for smoke tests or low-step DDIM overrides."
        )
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ddpo.train_loop import _build_policy_and_pool

    print("[build] policy, train pool, reward")
    model_type, policy, train_pool, eval_dataset_cfg = _build_policy_and_pool(
        cfg_root, cfg, args.device_resolved
    )
    if model_type != "ldm_adv":
        print(f"[warn] config model_type is {model_type!r}; script was designed for ldm_adv.")
    if args.policy_ckpt:
        print(f"[build] loading policy checkpoint: {args.policy_ckpt}")
        _load_policy_checkpoint(policy, args.policy_ckpt)
    reward = _build_reward(cfg)

    pools = {}
    if "train" in args.splits:
        pools["train"] = train_pool
    for split in args.splits:
        if split == "train":
            continue
        print(f"[build] {split} pool")
        pools[split] = _build_split_pool(model_type, eval_dataset_cfg, cfg, split, args.device_resolved)

    all_scene_rows: list[dict[str, Any]] = []
    all_sample_rows: list[dict[str, Any]] = []
    all_condition_rows: list[dict[str, Any]] = []
    selected_counts = {}
    selected_indices = {}
    for split in args.splits:
        pool = pools[split]
        indices = _select_pool_indices(len(pool), args.num_scenes, args.seed, args.selection)
        selected_counts[split] = len(indices)
        selected_indices[split] = indices
        print(f"[run] split={split} pool={len(pool)} selected={len(indices)} samples={args.samples_per_scene}")
        scene_rows, sample_rows, condition_rows = _run_split(policy, pool, reward, split, indices, args)
        all_scene_rows.extend(scene_rows)
        all_sample_rows.extend(sample_rows)
        all_condition_rows.extend(condition_rows)

    scene_path = out_dir / "scene_summary.csv"
    samples_path = out_dir / "samples.csv"
    condition_path = out_dir / "conditioning_summary.csv"
    top_path = out_dir / "top_samples.csv"
    selected_path = out_dir / "selected_indices.json"
    aggregate_path = out_dir / "aggregate.json"
    report_path = out_dir / "report.md"

    _write_csv(scene_path, all_scene_rows)
    _write_csv(samples_path, all_sample_rows)
    _write_csv(condition_path, all_condition_rows)

    top_samples = sorted(
        all_sample_rows,
        key=lambda r: (
            float(r.get("reward", float("-inf"))),
            float(r.get("r_risk", float("-inf"))),
        ),
        reverse=True,
    )[: args.top_k_samples]
    _write_csv(top_path, top_samples)
    _json_dump(selected_path, selected_indices)

    aggregate_scene_input = all_scene_rows + all_condition_rows
    summary = {
        "args": vars(args),
        "selected_counts": selected_counts,
        "scene_summary": _aggregate_scene_rows(aggregate_scene_input, args.high_reward_threshold),
        "sample_summary": _aggregate_samples(all_sample_rows, args.high_reward_threshold),
    }
    _json_dump(aggregate_path, summary)
    _write_report(report_path, cfg_root, args, summary, selected_counts)

    print(f"[done] wrote {scene_path}")
    print(f"[done] wrote {samples_path}")
    print(f"[done] wrote {condition_path}")
    print(f"[done] wrote {top_path}")
    print(f"[done] wrote {aggregate_path}")
    print(f"[done] wrote {report_path}")


if __name__ == "__main__":
    main()
