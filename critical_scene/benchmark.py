from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cfgs.config import CONFIG_PATH
from ddpo.reward import PufferDriveReward

from .schema import assert_same_map

METRIC_KEYS = (
    "reward",
    "ego_collision",
    "ego_offroad",
    "init_invalid",
    "reached_goal",
    "ego_min_ttc",
    "goal_offlane_frac",
    "parking_mismatch_frac",
    "ego_adv_min_dist",
    "gen_agent_is_parked",
)


def _load_artifact(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _to_float(value) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def _mean_finite(values) -> float:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    return float(arr[finite].mean()) if finite.any() else float("nan")


def _rate(values) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float((arr > 0).mean()) if arr.size else float("nan")


def _compose_cfg(config_name: str, planner: str, overrides: list[str] | None = None):
    hydra_overrides = list(overrides or [])
    hydra_overrides.append(f"planner@ddpo.planner={planner}")
    with initialize_config_dir(config_dir=str(Path(CONFIG_PATH).resolve()), version_base=None):
        return compose(config_name=config_name, overrides=hydra_overrides)


def _build_reward(cfg_root):
    cfg = cfg_root.ddpo
    return PufferDriveReward(
        planner_cfg=cfg.get("planner", None),
        sim_steps=cfg.get("sim_steps", 91),
        deterministic=cfg.get("planner_deterministic", None),
        ttc_tau=cfg.get("ttc_tau", 3.0),
        init_overlap_margin=cfg.get("init_overlap_margin", 0.0),
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
        seed=cfg.get("seed", 0),
        backend=cfg.get("reward_backend", "numpy"),
        pufferdrive_root=cfg.get("pufferdrive_root", None),
    )


def _summary(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "reward": _mean_finite(metrics["reward"]),
        "reached_goal_rate": _rate(metrics["reached_goal"]),
        "ego_collision_rate": _rate(metrics["ego_collision"]),
        "ego_offroad_rate": _rate(metrics["ego_offroad"]),
        "init_invalid_rate": _rate(metrics["init_invalid"]),
        "ego_min_ttc": _mean_finite(metrics["ego_min_ttc"]),
        "goal_offlane_frac": _mean_finite(metrics["goal_offlane_frac"]),
        "parking_mismatch_frac": _mean_finite(metrics["parking_mismatch_frac"]),
        "ego_adv_min_dist": _mean_finite(metrics.get("ego_adv_min_dist", [])),
        "gen_agent_is_parked": _mean_finite(metrics.get("gen_agent_is_parked", [])),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _save_media(media_dir: Path, source: str, scenes, metrics: dict, metadata: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    from ddpo.viz import render_rollout

    lanes = scenes.lane_polylines.detach().cpu().numpy() if isinstance(scenes.lane_polylines, torch.Tensor) else scenes.lane_polylines
    lane_scene_idx = scenes.meta["lane_scene_idx"].detach().cpu().numpy()
    states = scenes.agent_states.detach().cpu().numpy()
    types = scenes.agent_types.detach().cpu().numpy()
    agent_scene_idx = scenes.agent_scene_idx.detach().cpu().numpy()
    source_dir = media_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)

    for scene_offset, scene_id in enumerate(metadata["scene_ids"]):
        a_sel = agent_scene_idx == scene_offset
        fig = render_rollout(
            metrics["trajectories"][scene_offset],
            lanes[lane_scene_idx == scene_offset],
            agent_states=states[a_sel],
            agent_types=types[a_sel],
            reward=metrics["reward"][scene_offset],
            ego_collision=metrics["ego_collision"][scene_offset] > 0,
            ego_offroad=metrics["ego_offroad"][scene_offset] > 0,
            init_invalid=metrics["init_invalid"][scene_offset] > 0,
            ego_min_ttc=metrics["ego_min_ttc"][scene_offset],
            goal_offlane_frac=metrics["goal_offlane_frac"][scene_offset],
            parking_mismatch_frac=metrics["parking_mismatch_frac"][scene_offset],
            title=f"{source} scene {scene_id}",
        )
        fig.savefig(source_dir / f"scene_{int(scene_id)}.png", dpi=160)
        plt.close(fig)


def benchmark_artifacts(options: dict[str, Any]) -> dict[str, Any]:
    input_paths = options.get("inputs", [])
    if isinstance(input_paths, str):
        input_paths = [p for p in input_paths.split(",") if p]
    if not input_paths:
        raise ValueError("eval_planner.py requires inputs=<artifact1.pt,artifact2.pt,...>")

    planner = options.get("planner", "selfplay_drive")
    config_name = options.get("config_name", "config_critical_scene_dm_goal_ddpm")
    cfg_root = _compose_cfg(config_name, planner, options.get("overrides", []))
    reward = _build_reward(cfg_root)
    paired = str(options.get("paired", "true")).lower() in {"1", "true", "yes", "y", "on"}
    save_media = str(options.get("save_media", "false")).lower() in {"1", "true", "yes", "y", "on"}
    record_trajectories = save_media or str(options.get("record_trajectories", "false")).lower() in {"1", "true", "yes", "y", "on"}

    artifacts = [_load_artifact(path) for path in input_paths]
    if paired and len(artifacts) > 1:
        reference = artifacts[0]["metadata"]
        for artifact in artifacts[1:]:
            assert_same_map(reference, artifact["metadata"])

    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for artifact in artifacts:
        scenes = artifact["scenes"]
        metadata = artifact["metadata"]
        source = metadata["source"]
        metrics = reward.evaluate(scenes, record_trajectories=record_trajectories)
        summaries[source] = _summary(metrics)
        scene_ids = metadata["scene_ids"]
        map_ids = metadata["map_ids"]
        lane_hash = metadata["lane_hashes"]
        for i, scene_id in enumerate(scene_ids):
            row = {
                "source": source,
                "scene_id": int(scene_id),
                "map_id": int(map_ids[i]),
                "lane_hash": lane_hash[i],
            }
            for key in METRIC_KEYS:
                if key in metrics:
                    row[key] = _to_float(metrics[key][i])
            rows.append(row)

    run_name = options.get("run_name") or f"benchmark_{planner}_same_map"
    out_dir = Path(options.get("out_dir") or Path("outputs") / "critical_scene" / run_name)
    media_dir = Path(
        options.get("media_dir")
        or Path(os.environ.get("SCRATCH_ROOT", ".")) / "critical_scene" / run_name / "media"
    )
    _write_csv(out_dir / "per_scene.csv", rows)
    payload = {
        "planner": planner,
        "paired": paired,
        "inputs": input_paths,
        "summaries": summaries,
    }
    _write_json(out_dir / "summary.json", payload)
    OmegaConf.save(config=OmegaConf.create(payload), f=str(out_dir / "manifest.yaml"))

    if save_media:
        for artifact in artifacts:
            scenes = artifact["scenes"]
            source = artifact["metadata"]["source"]
            metrics = reward.evaluate(scenes, record_trajectories=True)
            _save_media(media_dir, source, scenes, metrics, artifact["metadata"])

    return {"out_dir": str(out_dir), "summary_path": str(out_dir / "summary.json"), **payload}
