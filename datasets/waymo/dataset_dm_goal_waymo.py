import glob
import os
import pickle
from typing import Any

import hydra
import numpy as np
from torch_geometric.data import Dataset

from cfgs.config import CONFIG_PATH, NON_PARTITIONED, PARTITIONED
from utils.data_container import ScenarioDreamerData
from utils.data_helpers import randomize_indices, reorder_indices
from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph
from utils.torch_helpers import from_numpy


class WaymoDatasetDMGoal(Dataset):
    """Waymo direct diffusion dataset that generates current scene and per-agent goals."""

    def __init__(self, cfg: Any, split_name: str = "train", mode: str = "train") -> None:
        super(WaymoDatasetDMGoal, self).__init__()
        self.cfg = cfg
        self.split_name = split_name
        self.mode = mode
        self.dataset_dir = os.path.join(self.cfg.preprocess_dir, self.split_name)
        self.files = sorted(glob.glob(os.path.join(self.dataset_dir, "*.pkl")))
        self.dset_len = len(self.files)

    def _normalize_agent_states(self, agent_states):
        agent_states = np.array(agent_states, copy=True)
        # Current position.
        agent_states[:, 0] = 2 * ((agent_states[:, 0] + self.cfg.fov / 2) / self.cfg.fov) - 1
        agent_states[:, 1] = 2 * ((agent_states[:, 1] + self.cfg.fov / 2) / self.cfg.fov) - 1
        # Speed.
        agent_states[:, 2] = 2 * ((agent_states[:, 2] - self.cfg.min_speed) / (self.cfg.max_speed - self.cfg.min_speed)) - 1
        # Length and width.
        agent_states[:, 5] = 2 * ((agent_states[:, 5] - self.cfg.min_length) / (self.cfg.max_length - self.cfg.min_length)) - 1
        agent_states[:, 6] = 2 * ((agent_states[:, 6] - self.cfg.min_width) / (self.cfg.max_width - self.cfg.min_width)) - 1
        # Clipped goal position in the same SDC-local 64m frame.
        agent_states[:, 7] = 2 * ((agent_states[:, 7] + self.cfg.fov / 2) / self.cfg.fov) - 1
        agent_states[:, 8] = 2 * ((agent_states[:, 8] + self.cfg.fov / 2) / self.cfg.fov) - 1
        return agent_states

    def _normalize_road_points(self, road_points):
        road_points = np.array(road_points, copy=True)
        road_points[:, :, 0] = 2 * ((road_points[:, :, 0] - self.cfg.min_lane_x) / (self.cfg.max_lane_x - self.cfg.min_lane_x)) - 1
        road_points[:, :, 1] = 2 * ((road_points[:, :, 1] - self.cfg.min_lane_y) / (self.cfg.max_lane_y - self.cfg.min_lane_y)) - 1
        return road_points

    def get_data(self, data, idx, path=None):
        valid_goal_mask = data["clipped_final_valid"].astype(bool)
        if valid_goal_mask.sum() == 0:
            return None

        agent_states = data["agent_states"][valid_goal_mask]
        agent_goal_xy = data["clipped_final_states"][valid_goal_mask, :2]
        agent_states = np.concatenate([agent_states, agent_goal_xy], axis=-1)
        agent_types = data["agent_types"][valid_goal_mask]
        num_agents = int(len(agent_states))

        road_points = data["road_points"]
        num_lanes = int(data["num_lanes"])
        edge_index_lane_to_lane = data["edge_index_lane_to_lane"]
        road_connection_types = data["road_connection_types"]
        edge_index_lane_to_agent = get_edge_index_bipartite(num_lanes, num_agents).numpy()
        edge_index_agent_to_agent = get_edge_index_complete_graph(num_agents).numpy()
        lg_type = int(data.get("lg_type", NON_PARTITIONED))

        agent_states = self._normalize_agent_states(agent_states)
        road_points = self._normalize_road_points(road_points)

        if self.mode == "train":
            agent_states, agent_types, road_points, edge_index_lane_to_lane = randomize_indices(
                agent_states,
                agent_types,
                road_points,
                edge_index_lane_to_lane,
            )

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
        d["num_lanes"] = num_lanes
        d["num_agents"] = num_agents
        d["lg_type"] = lg_type
        d["map_id"] = int(data.get("map_id", 0))
        d["agent"].x = from_numpy(agent_states.astype(np.float32))
        d["agent"].type = from_numpy(agent_types.astype(np.float32))
        d["lane"].x = from_numpy(road_points.astype(np.float32))
        d["agent"].partition_mask = from_numpy(agent_partition_mask.astype(bool))
        d["lane"].partition_mask = from_numpy(lane_partition_mask.astype(bool))
        d["lane", "to", "lane"].edge_index = from_numpy(edge_index_lane_to_lane)
        d["lane", "to", "lane"].type = from_numpy(road_connection_types.astype(np.float32))
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
    dset = WaymoDatasetDMGoal(cfg.dm_goal.dataset, split_name="train")
    print(cfg.dm_goal.dataset.preprocess_dir)
    print(len(dset))
    if len(dset) > 0:
        print(dset.get(0))


if __name__ == "__main__":
    main()
