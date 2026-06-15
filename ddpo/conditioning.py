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

import numpy as np
import torch
from torch_geometric.data import Batch

from datasets.waymo.dataset_dm_goal_waymo import WaymoDatasetDMGoal
from datasets.waymo.dataset_ldm_waymo import WaymoDatasetLDM
from utils.data_helpers import reorder_indices


class ConditioningPool:
    def __init__(
        self,
        dataset_cfg,
        *,
        split_name: str = "train",
        pool_size: int = 2048,
        device: str = "cuda",
        seed: int = 0,
    ):
        self.dataset = WaymoDatasetDMGoal(dataset_cfg, split_name=split_name, mode="eval")
        if len(self.dataset) == 0:
            raise RuntimeError(f"empty dm_goal dataset for split '{split_name}' "
                               f"({dataset_cfg.preprocess_dir})")
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
            d = self.dataset.get((ds_idx + probe) % len(self.dataset))
            if d is not None:
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
