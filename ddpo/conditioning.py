"""Conditioning pools backed by native scenario-dreamer datasets.

For dm_goal, this replaces the offline dump step of the PufferDrive-hosted DDPO:
graphs are built on demand by ``WaymoDatasetDMGoal`` (mode="eval": no index
randomisation, so the SDC stays at local agent index 0 - the reward scores that
slot as the ego) and batched with ``Batch.from_data_list``.

The conditioning graph carries everything the policy needs per mode:
  * map (lane chain target) + node counts + edges  -> all modes
  * real agent states data['agent'].x / .type      -> inpainted in mode "goal"
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Batch

from cfgs.config import NON_PARTITIONED, PARTITIONED
from datasets.waymo.dataset_dm_goal_waymo import WaymoDatasetDMGoal
from datasets.waymo.dataset_ldm_waymo import WaymoDatasetLDM
from utils.data_helpers import reorder_indices
from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph


def prune_agents(d, control_agent_num: int):
    """Shrink a dm_goal conditioning graph to ``ego + control_agent_num`` agents.

    Ego is local agent index 0 and is always kept; the remaining nodes are the
    first ``control_agent_num`` non-ego agents in the dataset's deterministic
    (spatially sorted) order. This is the node-level analogue of the policy's
    ``control_agent_num`` mask: instead of generating ``k`` agents and inpainting
    the rest to GT, the rest are removed from the graph entirely, so the
    diffusion denoiser, the decoded scene, and the reward sim all see only
    ``1 + k`` agents.

    Agent-agent (complete) and lane-agent (bipartite) edges are rebuilt for the
    new count and ``num_agents`` is updated. dm_goal does not consume
    ``partition_mask`` / ``lg_type`` (the LDM/AE encoder does), so those are only
    sliced/recomputed to keep the Data object internally consistent.

    ``control_agent_num < 0`` (or a scene already at/below the target) returns the
    graph unchanged. Mutates ``d`` in place and returns it.
    """
    if control_agent_num < 0:
        return d
    num_agents = int(d["num_agents"])
    keep = control_agent_num + 1  # ego + k non-ego
    if keep >= num_agents:
        return d
    num_lanes = int(d["num_lanes"])

    d["agent"].x = d["agent"].x[:keep]
    d["agent"].type = d["agent"].type[:keep]
    if "partition_mask" in d["agent"]:
        d["agent"].partition_mask = d["agent"].partition_mask[:keep]
    d["num_agents"] = keep
    d["agent", "to", "agent"].edge_index = get_edge_index_complete_graph(keep)
    d["lane", "to", "agent"].edge_index = get_edge_index_bipartite(num_lanes, keep)
    if int(d.get("lg_type", NON_PARTITIONED)) == PARTITIONED:
        d["num_agents_after_origin"] = int((~d["agent"].partition_mask).sum().item())
    return d


@dataclass
class EgoGoalOverride:
    """Parameters for the rule-based ego-goal replacement (see _override_ego_goal)."""

    min_distance: float = 15.0   # metres; drop lane endpoints closer than this
    top_k: int = 8               # sample the goal among the top_k farthest endpoints
    noise: float = 1.0           # metres; uniform +/- jitter added to the picked point
    seed: int = 0                # base seed; combined with the scene index per scene

    @classmethod
    def from_cfg(cls, cfg):
        """Build from an OmegaConf/dict node; returns None when disabled or absent."""
        if cfg is None:
            return None
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
        if not bool(get("enabled", False)):
            return None
        return cls(
            min_distance=float(get("min_distance", 15.0)),
            top_k=int(get("top_k", 8)),
            noise=float(get("noise", 1.0)),
            seed=int(get("seed", 0)),
        )


def _override_ego_goal(d, dataset_cfg, params: EgoGoalOverride, rng) -> None:
    """Replace the ego (local index 0) goal with a far, forward, on-road point.

    Many real SDC trajectories barely move (goal ~= spawn), so the ego is never
    "controlled" by the planner and the criticality reward gets no signal. This
    gives the ego a driving task: collect all lane-polyline endpoints, drop those
    behind the ego or closer than ``min_distance``, randomly pick one of the
    ``top_k`` farthest, jitter it by +/- ``noise`` metres, and clip to the FOV.
    Falls back to the original (untouched) goal when no endpoint qualifies.

    Geometry is computed in physical SDC-local metres (agents use the FOV frame,
    lanes use the lane-x/y frame - the two normalisations differ, so both are
    unnormalised before comparing). The result is written back in the agent FOV
    frame. Mutates ``d['agent'].x[0, 7:9]`` in place.
    """
    lane = d["lane"].x
    if lane.numel() == 0:
        return
    fov = float(dataset_cfg.fov)
    ax = d["agent"].x
    ego_x = (float(ax[0, 0]) + 1.0) / 2.0 * fov - fov / 2.0
    ego_y = (float(ax[0, 1]) + 1.0) / 2.0 * fov - fov / 2.0
    cos_h, sin_h = float(ax[0, 3]), float(ax[0, 4])

    pts = lane.reshape(-1, 2).detach().cpu().numpy().astype(np.float64)
    min_lx, max_lx = float(dataset_cfg.min_lane_x), float(dataset_cfg.max_lane_x)
    min_ly, max_ly = float(dataset_cfg.min_lane_y), float(dataset_cfg.max_lane_y)
    px = (pts[:, 0] + 1.0) / 2.0 * (max_lx - min_lx) + min_lx
    py = (pts[:, 1] + 1.0) / 2.0 * (max_ly - min_ly) + min_ly
    finite = np.isfinite(px) & np.isfinite(py)
    px, py = px[finite], py[finite]
    if px.size == 0:
        return

    vx, vy = px - ego_x, py - ego_y
    dist = np.hypot(vx, vy)
    keep = ((vx * cos_h + vy * sin_h) > 0.0) & (dist >= params.min_distance)
    if not keep.any():
        return  # fall back to the original goal

    cand_x, cand_y, cand_d = px[keep], py[keep], dist[keep]
    order = np.argsort(cand_d)[::-1][: params.top_k]  # farthest endpoints first
    j = int(order[rng.integers(0, len(order))])
    half = fov / 2.0
    gx = float(np.clip(cand_x[j] + rng.uniform(-params.noise, params.noise), -half, half))
    gy = float(np.clip(cand_y[j] + rng.uniform(-params.noise, params.noise), -half, half))

    ax[0, 7] = 2.0 * ((gx + half) / fov) - 1.0
    ax[0, 8] = 2.0 * ((gy + half) / fov) - 1.0


class ConditioningPool:
    def __init__(
        self,
        dataset_cfg,
        *,
        split_name: str = "train",
        pool_size: int = 2048,
        device: str = "cuda",
        seed: int = 0,
        control_agent_num: int = -1,
        ego_goal_override=None,
    ):
        self.dataset = WaymoDatasetDMGoal(dataset_cfg, split_name=split_name, mode="eval")
        if len(self.dataset) == 0:
            raise RuntimeError(f"empty dm_goal dataset for split '{split_name}' "
                               f"({dataset_cfg.preprocess_dir})")
        self.dataset_cfg = dataset_cfg
        self.ego_goal_override = EgoGoalOverride.from_cfg(ego_goal_override)
        self.control_agent_num = int(control_agent_num)
        self.device = device
        self.rng = np.random.default_rng(seed)
        # fixed random subset of the split = the pool (graphs load lazily, cached)
        n = min(int(pool_size), len(self.dataset))
        self.pool_indices = self.rng.permutation(len(self.dataset))[:n]
        self._cache: dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.pool_indices)

    def _get(self, pool_idx: int):
        """Graph for pool slot ``pool_idx``; scenes without valid goals are
        skipped by probing subsequent dataset indices (deterministic per slot)."""
        if pool_idx in self._cache:
            return self._cache[pool_idx]
        ds_idx = int(self.pool_indices[pool_idx])
        for probe in range(len(self.dataset)):
            scene_idx = (ds_idx + probe) % len(self.dataset)
            d = self.dataset.get(scene_idx)
            if d is not None:
                if self.ego_goal_override is not None:
                    # deterministic per scene (cached): same goal each epoch
                    rng = np.random.default_rng((self.ego_goal_override.seed, scene_idx))
                    _override_ego_goal(d, self.dataset_cfg, self.ego_goal_override, rng)
                d = prune_agents(d, self.control_agent_num)
                self._cache[pool_idx] = d
                return d
        raise RuntimeError("no valid conditioning graphs in dataset")

    def batch_from_indices(self, indices) -> Batch:
        return Batch.from_data_list([self._get(int(i)) for i in indices]).to(self.device)

    def sample_batch(self, batch_size: int) -> Batch:
        idx = self.rng.integers(0, len(self.pool_indices), size=batch_size)
        return self.batch_from_indices(idx)


class LDMGoalConditioningPool:
    """Conditioning pool for ldm_goal DDPO.

    ``WaymoDatasetLDM`` provides normalised AE latents and graph topology. The
    reward also needs map polylines in physical units, so we attach sorted real
    lane points from the latent-cache pickle to every graph.
    """

    def __init__(
        self,
        dataset_cfg,
        *,
        split_name: str = "train",
        pool_size: int = 2048,
        device: str = "cuda",
        seed: int = 0,
    ):
        self.dataset = WaymoDatasetLDM(dataset_cfg, split_name=split_name)
        if len(self.dataset) == 0:
            raise RuntimeError(f"empty ldm_goal dataset for split '{split_name}' "
                               f"({dataset_cfg.dataset_path})")
        self.dataset_cfg = dataset_cfg
        self.device = device
        self.rng = np.random.default_rng(seed)
        n = min(int(pool_size), len(self.dataset))
        self.pool_indices = self.rng.permutation(len(self.dataset))[:n]
        self._cache: dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.pool_indices)

    def _unnormalize_lane_polylines(self, road_points):
        rp = torch.as_tensor(road_points, dtype=torch.float32).clone()
        rp[:, :, 0] = ((torch.clip(rp[:, :, 0], -1, 1) + 1) / 2) * (
            self.dataset_cfg.max_lane_x - self.dataset_cfg.min_lane_x
        ) + self.dataset_cfg.min_lane_x
        rp[:, :, 1] = ((torch.clip(rp[:, :, 1], -1, 1) + 1) / 2) * (
            self.dataset_cfg.max_lane_y - self.dataset_cfg.min_lane_y
        ) + self.dataset_cfg.min_lane_y
        return rp

    def _sorted_road_points(self, raw):
        _, _, road_points, _, _, _, _ = reorder_indices(
            raw["agent_mu"],
            raw["agent_log_var"],
            raw["road_points"],
            raw["road_points"],
            raw["edge_index_lane_to_lane"],
            raw["agent_states"],
            raw["road_points"],
            raw.get("scene_type", raw.get("lg_type", 0)),
            dataset="waymo",
        )
        return self._unnormalize_lane_polylines(road_points)

    def _get(self, pool_idx: int):
        if pool_idx in self._cache:
            return self._cache[pool_idx]
        ds_idx = int(self.pool_indices[pool_idx])
        d = self.dataset.get(ds_idx)
        with open(self.dataset.files[ds_idx], "rb") as f:
            raw = pickle.load(f)
        d["lane"].road_points = self._sorted_road_points(raw)
        self._cache[pool_idx] = d
        return d

    def batch_from_indices(self, indices) -> Batch:
        return Batch.from_data_list([self._get(int(i)) for i in indices]).to(self.device)

    def sample_batch(self, batch_size: int) -> Batch:
        idx = self.rng.integers(0, len(self.pool_indices), size=batch_size)
        return self.batch_from_indices(idx)
