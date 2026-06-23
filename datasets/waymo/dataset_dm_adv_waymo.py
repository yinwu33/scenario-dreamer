import glob
import os
import pickle
from typing import Any

import hydra
import numpy as np
from torch_geometric.data import Dataset

from cfgs.config import CONFIG_PATH, NON_PARTITIONED
from utils.data_container import ScenarioDreamerData
from utils.data_helpers import modify_agent_states, normalize_scene, randomize_indices, reorder_indices
from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph
from utils.torch_helpers import from_numpy


class WaymoDatasetDMAdv(Dataset):
    """Waymo direct-diffusion dataset that generates the current scene **and**
    per-agent goals, with one adversarial agent split off per scene.

    Mirrors ``WaymoDatasetDMGoal``: it reads the goal-augmented preprocessed pkls
    (``scene_goal_preprocess_waymo``), keeps only agents with a valid clipped
    goal, and builds a ``state_dim == 9`` agent layout
    ``[x, y, speed, cosθ, sinθ, length, width, goal_x, goal_y]``. After the
    deterministic ego-first ordering it splits a single non-ego agent off into a
    separate ``adv`` node type (which therefore also carries a goal); the
    remaining agents are the context the adversary is conditioned on. The
    agent-to-agent and lane-to-agent graphs are rebuilt fresh over the reduced
    agent set (exactly like the goal dataset rebuilds them from scratch).

    Scenes with fewer than two valid-goal agents (i.e. no non-ego agent to act as
    the adversary) are dropped by returning ``None`` so the adversary
    distribution stays pure (only real non-ego agents) and every retained scene
    has exactly one ``adv`` node.
    """

    def __init__(self, cfg: Any, split_name: str = "train", mode: str = "train") -> None:
        super(WaymoDatasetDMAdv, self).__init__()
        self.cfg = cfg
        self.split_name = split_name
        self.mode = mode
        self.dataset_dir = os.path.join(self.cfg.preprocess_dir, self.split_name)
        self.files = sorted(glob.glob(os.path.join(self.dataset_dir, "*.pkl")))
        self.dset_len = len(self.files)

    def _select_adv_index(self, num_agents):
        """Pick the index of the single adversarial agent: a random non-ego
        agent (ego is index 0 and never selected). Deterministic outside of
        training for reproducible evaluation. Assumes ``num_agents >= 2``."""
        non_ego = np.arange(1, num_agents)
        if self.mode == "train":
            return int(np.random.choice(non_ego))
        return int(non_ego[0])

    def get_data(self, data, idx, path=None):
        valid_goal_mask = data["clipped_final_valid"].astype(bool)
        # Need the ego plus at least one non-ego agent to act as the adversary,
        # and every retained agent must have a valid goal.
        if valid_goal_mask.sum() < 2:
            return None

        # Preprocessed states are raw [x, y, vx, vy, yaw, length, width]; convert
        # to the [x, y, speed, cosθ, sinθ, length, width] convention and append
        # the clipped goal position -> [..., goal_x, goal_y] (state_dim == 9).
        agent_states = modify_agent_states(data["agent_states"][valid_goal_mask])
        agent_goal_xy = data["clipped_final_states"][valid_goal_mask, :2]
        agent_states = np.concatenate([agent_states, agent_goal_xy], axis=-1)
        agent_types = data["agent_types"][valid_goal_mask]

        # Cap the agent count (closest-to-origin agents); the ego sits at the
        # origin so it always survives and stays at index 0.
        if len(agent_states) > self.cfg.max_num_agents:
            dist_to_origin = np.linalg.norm(agent_states[:, :2], axis=-1)
            closest_agent_ids = np.argsort(dist_to_origin)[: self.cfg.max_num_agents]
            agent_states = agent_states[closest_agent_ids]
            agent_types = agent_types[closest_agent_ids]

        road_points = data["road_points"]
        num_lanes = int(data["num_lanes"])
        edge_index_lane_to_lane = data["edge_index_lane_to_lane"]
        road_connection_types = data["road_connection_types"]
        lg_type = int(data.get("lg_type", NON_PARTITIONED))

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

        # Split off the single adversarial agent (with its goal) from the normal
        # agent set, then rebuild the agent graphs over the reduced agent set.
        adv_idx = self._select_adv_index(agent_states.shape[0])
        adv_states = agent_states[adv_idx:adv_idx + 1]
        adv_types = agent_types[adv_idx:adv_idx + 1]
        keep = np.ones(agent_states.shape[0], dtype=bool)
        keep[adv_idx] = False
        agent_states = agent_states[keep]
        agent_types = agent_types[keep]

        num_agents = int(agent_states.shape[0])
        edge_index_lane_to_agent = get_edge_index_bipartite(num_lanes, num_agents).numpy()
        edge_index_agent_to_agent = get_edge_index_complete_graph(num_agents).numpy()

        d = ScenarioDreamerData()
        d["idx"] = data.get("idx", idx)
        d["num_lanes"] = int(num_lanes)
        d["num_agents"] = int(num_agents)
        d["lg_type"] = int(lg_type)
        d["map_id"] = int(data.get("map_id", 0))
        d["agent"].x = from_numpy(agent_states.astype(np.float32))
        d["agent"].type = from_numpy(agent_types.astype(np.float32))
        d["lane"].x = from_numpy(road_points.astype(np.float32))
        d["adv"].x = from_numpy(adv_states.astype(np.float32))
        d["adv"].type = from_numpy(adv_types.astype(np.float32))
        d["lane", "to", "lane"].edge_index = from_numpy(edge_index_lane_to_lane)
        d["lane", "to", "lane"].type = from_numpy(road_connection_types.astype(np.float32))
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
    dset = WaymoDatasetDMAdv(cfg.dm_adv.dataset, split_name="train")
    print(cfg.dm_adv.dataset.preprocess_dir)
    print(len(dset))
    if len(dset) > 0:
        print(dset.get(0))


if __name__ == "__main__":
    main()
