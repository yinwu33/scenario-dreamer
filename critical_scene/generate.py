from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from cfgs.config import CONFIG_PATH
from ddpo.conditioning import ConditioningPool, LDMGoalConditioningPool, prune_agents
from ddpo.policy import DMFixedMapAgentGoalDDPOPolicy, DMGoalDDPOPolicy
from ddpo.policy_ldm import LDMGoalDDPOPolicy
from datasets.waymo.dataset_dm_fixed_map_agent_goal_waymo import WaymoDatasetDMFixedMapAgentGoal
from utils.train_helpers import cache_latent_stats, set_latent_stats

from .schema import (
    SceneArtifactMetadata,
    artifact_payload,
    assert_same_map,
    batch_map_ids,
    batch_to_generated_scenes,
    lane_hashes,
)


def _cfg_get(cfg, key: str, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _set_dataset_name(cfg_node, dataset_name: str) -> None:
    OmegaConf.set_struct(cfg_node, False)
    cfg_node.dataset_name = dataset_name
    OmegaConf.set_struct(cfg_node, True)


def compose_cfg(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=str(Path(CONFIG_PATH).resolve()), version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


def default_scene_ids(num_scenes: int) -> list[int]:
    return list(range(int(num_scenes)))


def parse_scene_ids(value: str | None, num_scenes: int) -> list[int]:
    if value is None or value == "":
        return default_scene_ids(num_scenes)
    path = Path(value)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        return [int(v) for v in text.replace(",", "\n").split() if v.strip()]
    return [int(v) for v in value.split(",") if v.strip()]


def scratch_root(cfg_root) -> Path:
    value = os.environ.get("SCRATCH_ROOT")
    if value:
        return Path(value)
    try:
        resolved = OmegaConf.to_container(cfg_root, resolve=True)
        value = resolved.get("scratch_root")
    except Exception:
        value = None
    if not value or str(value).startswith("${"):
        raise RuntimeError("SCRATCH_ROOT is required for generated scene artifacts")
    return Path(str(value))


def _force_pool_indices(pool, scene_ids: list[int]) -> None:
    pool.pool_indices = np.asarray(scene_ids, dtype=np.int64)
    pool._cache.clear()


def _dm_goal_pool(cfg_root, cfg, split: str, scene_ids: list[int], device: str, seed: int) -> ConditioningPool:
    _set_dataset_name(cfg_root.dm_goal, cfg_root.dataset_name.name)
    pool = ConditioningPool(
        cfg_root.dm_goal.dataset,
        split_name=split,
        pool_size=max(max(scene_ids) + 1, len(scene_ids)),
        device=device,
        seed=seed,
        control_agent_num=cfg.get("control_agent_num", -1),
        ego_goal_override=cfg.get("ego_goal_override", None),
    )
    _force_pool_indices(pool, scene_ids)
    return pool


def _ldm_goal_pool(cfg_root, cfg, split: str, scene_ids: list[int], device: str, seed: int) -> LDMGoalConditioningPool:
    _set_dataset_name(cfg_root.ldm_goal, cfg_root.dataset_name.name)
    if not Path(cfg_root.ldm_goal.dataset.latent_stats_path).exists():
        cache_latent_stats(cfg_root.ldm_goal)
    ldm_cfg = set_latent_stats(cfg_root.ldm_goal)
    pool = LDMGoalConditioningPool(
        ldm_cfg.dataset,
        split_name=split,
        pool_size=max(max(scene_ids) + 1, len(scene_ids)),
        device=device,
        seed=seed,
    )
    _force_pool_indices(pool, scene_ids)
    return pool


def _batch_from_pool(pool, scene_ids: list[int]):
    return pool.batch_from_indices(list(range(len(scene_ids))))


def _dm_fixed_map_batch(cfg_root, split: str, scene_ids: list[int], *, control_agent_num: int | None = None):
    _set_dataset_name(cfg_root.dm_fixed_map_agent_goal, cfg_root.dataset_name.name)
    dataset = WaymoDatasetDMFixedMapAgentGoal(
        cfg_root.dm_fixed_map_agent_goal.dataset,
        split_name=split,
        mode="eval",
    )
    data_list = []
    valid_scene_ids = []
    for scene_id in scene_ids:
        data = dataset.get(int(scene_id))
        if data is None:
            raise RuntimeError(f"invalid fixed-map scene id {scene_id}: dataset returned None")
        if control_agent_num is not None:
            data = prune_agents(data, int(control_agent_num))
        data_list.append(data)
        valid_scene_ids.append(int(scene_id))
    return Batch.from_data_list(data_list), valid_scene_ids


def _dm_fixed_map_policy(
    cfg_root,
    cfg,
    ckpt: str,
    device: str,
    *,
    control_agent_num: int,
    control_ego: bool = False,
):
    return DMFixedMapAgentGoalDDPOPolicy(
        cfg_root.dm_fixed_map_agent_goal,
        ckpt_path=ckpt,
        mode="agent_only",
        device=device,
        use_ema_weights=cfg.get("use_ema_weights", True),
        inpaint_noised=cfg.get("inpaint_noised", True),
        control_ego=control_ego,
        control_agent_num=control_agent_num,
        sampler="ddpm",
        ddim_steps=None,
        ddim_eta=cfg.get("ddim_eta", 1.0),
        force_driving=cfg.get("force_driving", True),
    )


@torch.no_grad()
def _generate_dm_fixed_map(
    cfg_root,
    source: str,
    split: str,
    scene_ids: list[int],
    ckpt: str | None,
    device: str,
    seed: int,
    options: dict[str, Any],
):
    cfg = cfg_root.ddpo if "ddpo" in cfg_root else None
    raw_batch, scene_ids = _dm_fixed_map_batch(cfg_root, split, scene_ids)
    batch = raw_batch
    if source != "original" and cfg is None:
        raise ValueError(
            "generator=dm_fixed_map_agent_goal with diffusion sources "
            "requires a ddpo config, e.g. config_name=config_critical_scene_map_conditioned_dm_goal"
        )
    if source == "base_diffusion_one":
        control_agent_num = int(options.get("base_one_control_agent_num", options.get("base_control_agent_num", 1)))
        batch, scene_ids = _dm_fixed_map_batch(
            cfg_root, split, scene_ids, control_agent_num=control_agent_num
        )
    elif source == "ddpo_diffusion":
        control_agent_num = int(options.get("ddpo_control_agent_num", cfg.get("control_agent_num", 1)))
        batch, scene_ids = _dm_fixed_map_batch(
            cfg_root, split, scene_ids, control_agent_num=control_agent_num
        )
    elif source in {"base_diffusion", "base_diffusion_full"}:
        control_agent_num = int(options.get("base_control_agent_num", -1))
    else:
        control_agent_num = -1

    reference = batch_to_generated_scenes(batch, cfg_root.dm_fixed_map_agent_goal.dataset)
    reference_meta = {
        "scene_ids": scene_ids,
        "map_ids": batch_map_ids(batch),
        "lane_hashes": lane_hashes(reference),
    }
    if source == "original":
        return reference, reference_meta, None

    if source in {"base_diffusion", "base_diffusion_full", "base_diffusion_one"}:
        resolved_ckpt = ckpt or cfg.model_ckpt
    elif ckpt is None:
        raise ValueError(
            "source=ddpo_diffusion for generator=dm_fixed_map_agent_goal requires ckpt=<ddpo checkpoint>"
        )
    else:
        resolved_ckpt = ckpt

    policy = _dm_fixed_map_policy(
        cfg_root,
        cfg,
        str(resolved_ckpt),
        device,
        control_agent_num=control_agent_num,
        control_ego=False,
    )
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    scenes, _ = policy.sample(batch)
    candidate_meta = {
        "scene_ids": scene_ids,
        "map_ids": batch_map_ids(batch),
        "lane_hashes": lane_hashes(scenes),
    }
    assert_same_map(reference_meta, candidate_meta)
    return scenes, candidate_meta, resolved_ckpt


def _dm_goal_policy(cfg_root, cfg, ckpt: str | None, device: str, sampler: str | None, ddim_steps: int | None, eta: float):
    return DMGoalDDPOPolicy(
        cfg_root.dm_goal,
        ckpt_path=ckpt or cfg.model_ckpt,
        mode=cfg.mode,
        device=device,
        use_ema_weights=cfg.get("use_ema_weights", True),
        inpaint_noised=cfg.get("inpaint_noised", True),
        control_ego=cfg.get("control_ego", True),
        control_agent_num=cfg.get("control_agent_num", -1),
        sampler=sampler or cfg.get("sampler", "ddpm"),
        ddim_steps=ddim_steps if (sampler or cfg.get("sampler", "ddpm")) == "ddim" else None,
        ddim_eta=eta,
    )


def _ldm_goal_policy(cfg_root, cfg, ckpt: str | None, device: str):
    _set_dataset_name(cfg_root.ae_goal, cfg_root.dataset_name.name)
    ldm_cfg = set_latent_stats(cfg_root.ldm_goal)
    return LDMGoalDDPOPolicy(
        ldm_cfg,
        cfg_root.ae_goal,
        ldm_ckpt=ckpt or cfg.ldm_ckpt,
        ae_ckpt=cfg.ae_ckpt,
        device=device,
        use_ema_weights=cfg.get("use_ema_weights", True),
    )


@torch.no_grad()
def _generate_policy_scenes(
    cfg_root,
    *,
    source: str,
    split: str,
    scene_ids: list[int],
    ckpt: str | None,
    device: str,
    seed: int,
    sampler: str | None,
    ddim_steps: int | None,
    eta: float,
):
    cfg = cfg_root.ddpo
    model_type = cfg.get("model_type", "dm_goal")
    if model_type == "ldm_goal":
        pool = _ldm_goal_pool(cfg_root, cfg, split, scene_ids, device, seed)
        cond = _batch_from_pool(pool, scene_ids)
        policy = _ldm_goal_policy(cfg_root, cfg, ckpt, device)
    else:
        pool = _dm_goal_pool(cfg_root, cfg, split, scene_ids, device, seed)
        cond = _batch_from_pool(pool, scene_ids)
        policy = _dm_goal_policy(cfg_root, cfg, ckpt, device, sampler, ddim_steps, eta)

    original = policy.conditioning_scenes(cond)
    reference_meta = {
        "scene_ids": scene_ids,
        "map_ids": batch_map_ids(cond),
        "lane_hashes": lane_hashes(original),
    }
    if source == "original":
        return original, reference_meta, None

    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    scenes, _ = policy.sample(cond)
    candidate_meta = {
        "scene_ids": scene_ids,
        "map_ids": batch_map_ids(cond),
        "lane_hashes": lane_hashes(scenes),
    }
    assert_same_map(reference_meta, candidate_meta)
    resolved_ckpt = ckpt or (cfg.ldm_ckpt if model_type == "ldm_goal" else cfg.model_ckpt)
    return scenes, candidate_meta, resolved_ckpt


def save_manifest(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(json.loads(json.dumps(metadata))), f=str(path))


def generate_artifact(options: dict[str, Any]) -> dict[str, Any]:
    source = options.get("source", "original")
    valid_sources = {"original", "base_diffusion", "base_diffusion_full", "base_diffusion_one", "ddpo_diffusion"}
    if source not in valid_sources:
        raise ValueError(f"source must be one of {sorted(valid_sources)}; got {source!r}")

    config_name = options.get("config_name", "config_critical_scene_dm_goal_ddpm")
    cfg_root = compose_cfg(config_name, options.get("overrides", []))
    device = options.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    split = options.get("split", "val")
    seed = int(options.get("seed", 0))
    num_scenes = int(options.get("num_scenes", 4))
    scene_ids = parse_scene_ids(options.get("scene_ids"), num_scenes)
    generator = options.get("generator")
    ckpt = options.get("ckpt")
    sampler = options.get("sampler")
    ddim_steps = options.get("ddim_steps")
    ddim_steps = int(ddim_steps) if ddim_steps not in (None, "", "none") else None
    eta = float(options.get("eta", options.get("ddim_eta", 1.0)))

    if generator is None:
        if "ddpo" in cfg_root:
            generator = cfg_root.ddpo.get("model_type", "dm_goal")
        else:
            generator = "dm_fixed_map_agent_goal" if config_name.endswith("fixed_map_agent_goal_train") else "dm_goal"

    if generator == "dm_fixed_map_agent_goal":
        scenes, scene_meta, resolved_ckpt = _generate_dm_fixed_map(
            cfg_root, source, split, scene_ids, ckpt, device, seed, options
        )
    else:
        model_type = cfg_root.ddpo.get("model_type", "dm_goal")
        if generator != model_type:
            raise ValueError(
                f"generator={generator!r} does not match ddpo.model_type={model_type!r} "
                f"in config_name={config_name!r}"
            )
        scenes, scene_meta, resolved_ckpt = _generate_policy_scenes(
            cfg_root,
            source=source,
            split=split,
            scene_ids=scene_ids,
            ckpt=ckpt,
            device=device,
            seed=seed,
            sampler=sampler,
            ddim_steps=ddim_steps,
            eta=eta,
        )

    run_name = options.get("run_name") or f"critical_scene_{source}_{generator}_{split}"
    artifact_dir = Path(options.get("artifact_dir") or scratch_root(cfg_root) / "critical_scene" / run_name / "generated")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{source}.pt"
    metadata = SceneArtifactMetadata(
        source=source,
        scene_ids=scene_meta["scene_ids"],
        split=split,
        generator_ckpt=str(resolved_ckpt) if resolved_ckpt is not None else None,
        sampler=sampler or (cfg_root.ddpo.get("sampler", None) if "ddpo" in cfg_root else None),
        planner_target=options.get("planner_target"),
        seed=seed,
        same_map=True,
        map_ids=scene_meta["map_ids"],
        lane_hashes=scene_meta["lane_hashes"],
        config_name=config_name,
    ).to_dict()
    torch.save(artifact_payload(scenes, SceneArtifactMetadata(**metadata)), artifact_path)

    manifest_dir = Path(options.get("manifest_dir") or Path("outputs") / "critical_scene" / run_name)
    manifest = {**metadata, "artifact_path": str(artifact_path), "run_name": run_name}
    save_manifest(manifest_dir / "manifest.yaml", manifest)
    return {"artifact_path": str(artifact_path), "manifest_path": str(manifest_dir / "manifest.yaml"), **manifest}
