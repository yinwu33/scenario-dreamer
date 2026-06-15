"""Conditioning pools backed by the native scenario-dreamer dm_goal dataset.

Replaces the offline dump step of the PufferDrive-hosted DDPO: graphs are built
on demand by ``WaymoDatasetDMGoal`` (mode="eval": no index randomisation, so the
SDC stays at local agent index 0 - the reward scores that slot as the ego) and
batched with ``Batch.from_data_list``.

The conditioning graph carries everything the policy needs per mode:
  * map (lane chain target) + node counts + edges  -> all modes
  * real agent states data['agent'].x / .type      -> inpainted in mode "goal"
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch

from datasets.waymo.dataset_dm_goal_waymo import WaymoDatasetDMGoal


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
