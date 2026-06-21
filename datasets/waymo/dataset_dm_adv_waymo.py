import glob
import os
import pickle
from typing import Any

import hydra
import numpy as np
import torch
from torch_geometric.data import Dataset

from cfgs.config import CONFIG_PATH, NON_PARTITIONED
from utils.data_container import ScenarioDreamerData
from utils.data_helpers import normalize_scene, randomize_indices, reorder_indices
from utils.torch_helpers import from_numpy


class WaymoDatasetDMAdv(Dataset):
    """Waymo vectorized scene dataset for direct diffusion training with one
    adversarial agent per scene.

    Reuses the exact preprocessed pkls of the plain ``dm`` dataset and, on the
    fly, splits off a single non-ego agent into a separate ``adv`` node type
    (removing it from the normal ``agent`` set). The remaining agents are the
    context the adversary is conditioned on. Partitioning is ignored.

    Scenes with fewer than two agents (i.e. no non-ego agent to act as the
    adversary) are dropped by returning ``None`` -- the same convention used by
    the goal dataset -- so the adversary distribution stays pure (only real
    non-ego agents) and every retained scene has exactly one ``adv`` node.
    """

    def __init__(self, cfg: Any, split_name: str = "train", mode: str = "train") -> None:
        super(WaymoDatasetDMAdv, self).__init__()
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

    def _select_adv_index(self, num_agents):
        """Pick the index of the single adversarial agent: a random non-ego
        agent (ego is index 0 and never selected). Deterministic outside of
        training for reproducible evaluation. Assumes ``num_agents >= 2``."""
        non_ego = np.arange(1, num_agents)
        if self.mode == "train":
            return int(np.random.choice(non_ego))
        return int(non_ego[0])

    def _remove_agent(self, remove_idx, agent_states, agent_types, a2a_edge_index, l2a_edge_index):
        """Drop ``remove_idx`` from the agent set and reindex the (complete)
        a2a / l2a graphs so the remaining agents are contiguous 0..N-2."""
        num_agents = agent_states.shape[0]
        keep = np.ones(num_agents, dtype=bool)
        keep[remove_idx] = False
        # old agent index -> new agent index (removed node maps to -1, never used).
        old_to_new = np.cumsum(keep) - 1
        old_to_new[remove_idx] = -1

        agent_states = agent_states[keep]
        agent_types = agent_types[keep]

        a2a = np.asarray(a2a_edge_index)
        if a2a.shape[1] > 0:
            a2a_keep = (a2a[0] != remove_idx) & (a2a[1] != remove_idx)
            a2a = old_to_new[a2a[:, a2a_keep]]

        l2a = np.asarray(l2a_edge_index)
        if l2a.shape[1] > 0:
            l2a_keep = l2a[1] != remove_idx
            l2a = l2a[:, l2a_keep]
            l2a = np.stack([l2a[0], old_to_new[l2a[1]]], axis=0)

        return agent_states, agent_types, a2a, l2a

    def get_data(self, data, idx, path=None):
        num_agents = int(data["num_agents"])
        # Need the ego plus at least one non-ego agent to act as the adversary.
        if num_agents < 2:
            return None

        agent_states = data["agent_states"]
        agent_types = data["agent_types"]
        road_points = data["road_points"]
        edge_index_lane_to_lane = data["edge_index_lane_to_lane"]
        edge_index_lane_to_agent = data["edge_index_lane_to_agent"]
        edge_index_agent_to_agent = data["edge_index_agent_to_agent"]
        road_connection_types = data["road_connection_types"]
        lg_type = int(data.get("lg_type", NON_PARTITIONED))
        num_lanes = int(data["num_lanes"])

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

        # Deterministic, ego-first ordering so the positional encodings are
        # meaningful (partition outputs are ignored).
        agent_states, agent_types, road_points, _, edge_index_lane_to_lane, _, _ = reorder_indices(
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

        # Split off the single adversarial agent from the normal agent set.
        adv_idx = self._select_adv_index(agent_states.shape[0])
        adv_states = agent_states[adv_idx:adv_idx + 1]
        adv_types = agent_types[adv_idx:adv_idx + 1]
        (
            agent_states,
            agent_types,
            edge_index_agent_to_agent,
            edge_index_lane_to_agent,
        ) = self._remove_agent(
            adv_idx,
            agent_states,
            agent_types,
            edge_index_agent_to_agent,
            edge_index_lane_to_agent,
        )

        num_agents = agent_states.shape[0]

        d = ScenarioDreamerData()
        d["idx"] = data.get("idx", idx)
        d["num_lanes"] = int(num_lanes)
        d["num_agents"] = int(num_agents)
        d["lg_type"] = int(lg_type)
        d["map_id"] = self._get_map_id(path) if path is not None else 0
        d["agent"].x = from_numpy(agent_states)
        d["agent"].type = from_numpy(agent_types)
        d["lane"].x = from_numpy(road_points)
        d["adv"].x = from_numpy(adv_states)
        d["adv"].type = from_numpy(adv_types)
        d["lane", "to", "lane"].edge_index = from_numpy(edge_index_lane_to_lane)
        d["lane", "to", "lane"].type = from_numpy(road_connection_types)
        d["agent", "to", "agent"].edge_index = from_numpy(edge_index_agent_to_agent)
        d["lane", "to", "agent"].edge_index = from_numpy(edge_index_lane_to_agent)

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
    dset = WaymoDatasetDMAdv(cfg.dm.dataset, split_name="train")
    print(cfg.dm.dataset.preprocess_dir)
    print(len(dset))
    if len(dset) > 0:
        print(dset.get(0))


if __name__ == "__main__":
    main()
