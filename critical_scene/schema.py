from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from ddpo.goal_schema import GOAL_SLICE, MIN_DISTANCE_TO_GOAL, fov_unnormalize
from ddpo.interfaces import GeneratedScenes
from models.scenario_dreamer_dm_goal import unnormalize_scene_with_goal


@dataclass
class SceneArtifactMetadata:
    source: str
    scene_ids: list[int]
    split: str
    generator_ckpt: str | None
    sampler: str | None
    planner_target: str | None
    seed: int
    same_map: bool
    map_ids: list[int]
    lane_hashes: list[str]
    config_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_cpu_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value).detach().cpu()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def batch_map_ids(batch) -> list[int]:
    value = batch["map_id"]
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def unnormalize_goal_scene(agent_states: torch.Tensor, lane_states: torch.Tensor, dataset_cfg):
    # Delegate to the authoritative dm_goal unnormalisation (goal cols [7:9] are
    # handled there); do not re-implement the transform here.
    return unnormalize_scene_with_goal(agent_states, lane_states, dataset_cfg)


def normalized_gt_parking_mask(agent_states: torch.Tensor, dataset_cfg) -> torch.Tensor:
    fov = float(dataset_cfg.fov)
    init_xy = fov_unnormalize(agent_states[:, 0:2].float(), fov)
    goal_xy = fov_unnormalize(agent_states[:, GOAL_SLICE].float(), fov)
    return torch.linalg.norm(goal_xy - init_xy, dim=-1) < MIN_DISTANCE_TO_GOAL


def batch_to_generated_scenes(batch, dataset_cfg, *, already_unnormalized: bool = False) -> GeneratedScenes:
    agent_states = batch["agent"].x.float().clone()
    lane_states = batch["lane"].x.float().clone()
    if not already_unnormalized:
        agent_states, lane_states = unnormalize_goal_scene(agent_states, lane_states, dataset_cfg)

    agent_type = batch["agent"].type
    agent_types = torch.argmax(agent_type, dim=-1) if agent_type.ndim > 1 else agent_type.long()
    meta = {"lane_scene_idx": batch["lane"].batch.detach().clone()}
    lane_edge_store = batch["lane", "to", "lane"]
    if "edge_index" in lane_edge_store:
        meta["lane_edge_index"] = lane_edge_store.edge_index.detach().clone()
    if "type" in lane_edge_store:
        meta["lane_edge_type"] = lane_edge_store.type.detach().clone()
    if not already_unnormalized and "x" in batch["agent"]:
        meta["gt_parking_mask"] = normalized_gt_parking_mask(batch["agent"].x, dataset_cfg)

    return GeneratedScenes(
        agent_states=agent_states.detach().cpu(),
        agent_types=agent_types.detach().cpu(),
        agent_scene_idx=batch["agent"].batch.detach().cpu(),
        lane_polylines=lane_states.detach().cpu(),
        num_scenes=int(batch.batch_size),
        meta={k: _to_cpu_tensor(v) if isinstance(v, torch.Tensor) else v for k, v in meta.items()},
    )


def lane_hashes(scenes: GeneratedScenes) -> list[str]:
    lanes = _to_numpy(scenes.lane_polylines).astype(np.float32, copy=False)
    lane_scene_idx = _to_numpy(scenes.meta["lane_scene_idx"]).astype(np.int64, copy=False)
    hashes: list[str] = []
    for scene_idx in range(int(scenes.num_scenes)):
        scene_lanes = np.ascontiguousarray(lanes[lane_scene_idx == scene_idx])
        hashes.append(hashlib.sha1(scene_lanes.tobytes()).hexdigest())
    return hashes


def assert_same_map(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    keys = ("scene_ids", "map_ids", "lane_hashes")
    for key in keys:
        if list(reference.get(key, [])) != list(candidate.get(key, [])):
            raise ValueError(
                f"same-map assertion failed for {key}: "
                f"reference={reference.get(key)} candidate={candidate.get(key)}"
            )


def artifact_payload(scenes: GeneratedScenes, metadata: SceneArtifactMetadata) -> dict[str, Any]:
    return {"scenes": scenes, "metadata": metadata.to_dict()}

