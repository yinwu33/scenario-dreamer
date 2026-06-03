import glob
import os
import pickle
from typing import Any

import hydra
import numpy as np
import torch
from torch_geometric.data import Dataset

from cfgs.config import CONFIG_PATH, NON_PARTITIONED, PARTITIONED
from utils.data_container import ScenarioDreamerData
from utils.data_helpers import normalize_scene, randomize_indices, reorder_indices
from utils.torch_helpers import from_numpy


class WaymoDatasetDM(Dataset):
    """Waymo vectorized scene dataset for direct diffusion training."""

    def __init__(self, cfg: Any, split_name: str = "train", mode: str = "train") -> None:
        super(WaymoDatasetDM, self).__init__()
        self.cfg = cfg
        self.split_name = split_name
        self.mode = mode
        self.dataset_dir = os.path.join(self.cfg.preprocess_dir, self.split_name)
        self.files = sorted(glob.glob(os.path.join(self.dataset_dir, "*.pkl")))
        self.dset_len = len(self.files)

        self.nocturne_compatible_filenames = set()
        if hasattr(self.cfg, "nocturne_train_filenames_path") and os.path.exists(self.cfg.nocturne_train_filenames_path):
            with open(self.cfg.nocturne_train_filenames_path, "rb") as f:
                self.nocturne_compatible_filenames.update(pickle.load(f))
        if hasattr(self.cfg, "nocturne_val_filenames_path") and os.path.exists(self.cfg.nocturne_val_filenames_path):
            with open(self.cfg.nocturne_val_filenames_path, "rb") as f:
                self.nocturne_compatible_filenames.update(pickle.load(f))

    def _get_map_id(self, path):
        filename = os.path.basename(path)
        parts = filename.split(".")
        if len(parts) > 1:
            scene_key = "_".join(parts[1].split("_")[:2])
        else:
            scene_key = "_".join(os.path.splitext(filename)[0].split("_")[:2])
        return int(scene_key in self.nocturne_compatible_filenames)

    def get_data(self, data, idx, path=None):
        agent_states = data["agent_states"]
        agent_types = data["agent_types"]
        road_points = data["road_points"]
        edge_index_lane_to_lane = data["edge_index_lane_to_lane"]
        edge_index_lane_to_agent = data["edge_index_lane_to_agent"]
        edge_index_agent_to_agent = data["edge_index_agent_to_agent"]
        road_connection_types = data["road_connection_types"]
        lg_type = data["lg_type"]
        num_lanes = data["num_lanes"]
        num_agents = data["num_agents"]

        agent_states, road_points = normalize_scene(
            agent_states,
            road_points,
            fov=self.cfg.fov,
            min_speed=self.cfg.min_speed,
            max_speed=self.cfg.max_speed,
            min_length=self.cfg.min_length,
            max_length=self.cfg.max_length,
            min_width=self.cfg.min_width,
            max_width=self.cfg.max_width,
            min_lane_x=self.cfg.min_lane_x,
            min_lane_y=self.cfg.min_lane_y,
            max_lane_x=self.cfg.max_lane_x,
            max_lane_y=self.cfg.max_lane_y,
        )

        if self.mode == "train":
            agent_states, agent_types, road_points, edge_index_lane_to_lane = randomize_indices(
                agent_states,
                agent_types,
                road_points,
                edge_index_lane_to_lane,
            )

        # Match the recursive ordering used by latent diffusion so partitioned
        # scenes condition on the before-partition half consistently.
        agent_states, agent_types, road_points, _, edge_index_lane_to_lane, agent_partition_mask, lane_partition_mask = reorder_indices(
            agent_states,
            agent_types,
            road_points,
            road_points,
            edge_index_lane_to_lane,
            agent_states,
            road_points,
            lg_type,
            dataset="waymo",
        )

        if lg_type == NON_PARTITIONED:
            agent_partition_mask = np.zeros(num_agents).astype(bool)
            lane_partition_mask = np.zeros(num_lanes).astype(bool)

        d = ScenarioDreamerData()
        d["idx"] = data.get("idx", idx)
        d["num_lanes"] = int(num_lanes)
        d["num_agents"] = int(num_agents)
        d["lg_type"] = int(lg_type)
        d["map_id"] = self._get_map_id(path) if path is not None else 0
        d["agent"].x = from_numpy(agent_states)
        d["agent"].type = from_numpy(agent_types)
        d["lane"].x = from_numpy(road_points)
        d["agent"].partition_mask = from_numpy(agent_partition_mask.astype(bool))
        d["lane"].partition_mask = from_numpy(lane_partition_mask.astype(bool))
        d["lane", "to", "lane"].edge_index = from_numpy(edge_index_lane_to_lane)
        d["lane", "to", "lane"].type = from_numpy(road_connection_types)
        d["agent", "to", "agent"].edge_index = from_numpy(edge_index_agent_to_agent)
        d["lane", "to", "agent"].edge_index = from_numpy(edge_index_lane_to_agent)

        if lg_type == PARTITIONED:
            d["num_agents_after_origin"] = int((~d["agent"].partition_mask).sum().item())
            d["num_lanes_after_origin"] = int((~d["lane"].partition_mask).sum().item())
        else:
            d["num_agents_after_origin"] = 0
            d["num_lanes_after_origin"] = 0

        return d

    def get(self, idx: int):
        path = self.files[idx]
        with open(path, "rb") as f:
            data = pickle.load(f)
        return self.get_data(data, idx, path)

    def len(self):
        return self.dset_len


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    dset = WaymoDatasetDM(cfg.dm.dataset, split_name="train")
    print(cfg.dm.dataset.preprocess_dir)
    print(len(dset))
    if len(dset) > 0:
        print(dset.get(0))


if __name__ == "__main__":
    main()
