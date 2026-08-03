import glob
import os
import pickle
import sys
from typing import Any

import hydra
import numpy as np
import torch
from torch_geometric.data import Dataset

np.set_printoptions(suppress=True, threshold=sys.maxsize)
torch.set_printoptions(threshold=100000)

from cfgs.config import CONFIG_PATH, NON_PARTITIONED
from utils.data_container import ScenarioDreamerData
from utils.data_helpers import normalize_scene, randomize_indices
from utils.goal_runtime import prepare_scene
from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph
from utils.torch_helpers import from_numpy


class WaymoDatasetAEGoal(Dataset):
    """Waymo autoencoder dataset that augments each agent state with its goal (final
    in-FOV position).

    Reads the **v2** SDC-centered goal records (``utils/goal_preprocess.py``), whose
    agent set is filtered exactly as the original Scenario Dreamer preprocessing filters
    it -- including the off-road vehicle removal the v1 pickles were missing. Each agent
    state becomes 9-dimensional ``[x, y, speed, cosθ, sinθ, length, width, goal_x,
    goal_y]`` so the autoencoder encodes/decodes the goal alongside the initial state.
    Only non-partitioned scenes are produced (no inpainting).

    The goal columns and any goal-driven filtering are applied at load time by
    ``utils.goal_runtime.prepare_scene``; everything else comes off disk, mirroring the
    baseline autoencoder's fast path.

    Setting ``cfg.include_goal = False`` keeps the goal-driven *filtering* but drops the
    two goal columns, yielding the baseline 7-D states on exactly the same agent set --
    the apples-to-apples setup for evaluating a 7-D checkpoint against this dataset.
    """

    def __init__(self, cfg: Any, split_name: str = "train", mode: str = "train") -> None:
        super(WaymoDatasetAEGoal, self).__init__()
        self.cfg = cfg
        self.split_name = split_name
        self.mode = mode
        self.preprocess = self.cfg.preprocess
        # When False the goal columns are dropped again after prepare_scene, so the states
        # are the baseline 7-D ones while the *agent set* stays exactly the goal-filtered
        # one. That is what makes a 7-D and a 9-D autoencoder comparable on the same data.
        self.include_goal = getattr(self.cfg, "include_goal", True)
        # mirrors the autoencoder dataset interface; preprocess_dir holds the cached pickles
        self.preprocessed_dir = os.path.join(self.cfg.preprocess_dir, f"{self.split_name}")
        self.files = sorted(glob.glob(os.path.join(self.preprocessed_dir, "*.pkl")))
        self.dset_len = len(self.files)

    def get_data(self, data: dict, idx: int):
        # Everything the original preprocessing did (FOV crop, closest-N cap, off-road
        # vehicle removal, modify_agent_states) is already baked into the v2 record.
        # prepare_scene only adds the goal-side work, which is deliberately kept at
        # runtime so goal definitions and goal filters stay adjustable.
        
        if not self.preprocess:
            # TODO: preprocess code
            raise NotImplementedError("preprocess=False is not implemented for WaymoDatasetAEGoal")
            return
        
        scene = prepare_scene(data, self.cfg)
        agent_states = scene["agent_states"] # 【N, 9】, includes goal columns
        if not self.include_goal:
            agent_states = agent_states[:, :-2] # [N, 7], baseline state layout
        agent_types = scene["agent_types"]  # [N, ?]
        goal_xy = scene["goal_xy"]  # [N, 2]
        goal_valid = scene["goal_valid"]  # [N,]
        goal_timestep = scene["goal_timestep"]  # [N,]
        goal_dist = scene["goal_dist"]  # [N,]
        
        num_agents = int(len(agent_states))
        assert num_agents != 0

        # read from data
        road_points = np.array(data["road_points"], copy=True)
        num_lanes = int(data["num_lanes"])
        edge_index_lane_to_lane = np.array(data["edge_index_lane_to_lane"])
        road_connection_types = np.array(data["road_connection_types"])

        # min-max normalize agent states (incl. goal cols) and lanes into [-1, 1]
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

        # training-only randomization of non-ego agent and lane ordering
        if self.mode == "train":
            agent_states, agent_types, road_points, edge_index_lane_to_lane = randomize_indices(
                agent_states, agent_types, road_points, edge_index_lane_to_lane
            )

        edge_index_lane_to_lane = torch.from_numpy(np.ascontiguousarray(edge_index_lane_to_lane))
        edge_index_agent_to_agent = get_edge_index_complete_graph(num_agents)
        edge_index_lane_to_agent = get_edge_index_bipartite(num_lanes, num_agents)

        if self.cfg.remove_left_right_connections:
            # keep only none/pred/succ/self
            road_connection_types = road_connection_types[:, [0, 1, 2, 5]]

        # non-partitioned scene: no inpainting machinery
        lg_type = NON_PARTITIONED
        a2a_mask = torch.ones(edge_index_agent_to_agent.shape[1]).bool()
        l2l_mask = torch.ones(edge_index_lane_to_lane.shape[1]).bool()
        l2a_mask = torch.ones(edge_index_lane_to_agent.shape[1]).bool()
        lane_partition_mask = np.zeros(num_lanes).astype(bool)

        d = ScenarioDreamerData()
        d["idx"] = idx  # dataset index (used by latent caching to locate the source file)
        d["num_lanes"] = num_lanes
        d["num_agents"] = num_agents
        d["lg_type"] = lg_type
        d["agent"].x = from_numpy(agent_states.astype(np.float32))
        d["agent"].type = from_numpy(agent_types.astype(np.float32))
        d["lane"].x = from_numpy(road_points.astype(np.float32))
        d["lane"].partition_mask = from_numpy(lane_partition_mask)
        d["num_agents_after_origin"] = 0
        d["num_lanes_after_origin"] = 0

        d["lane", "to", "lane"].edge_index = edge_index_lane_to_lane
        d["lane", "to", "lane"].type = from_numpy(road_connection_types.astype(np.float32))
        d["agent", "to", "agent"].edge_index = edge_index_agent_to_agent
        d["lane", "to", "agent"].edge_index = edge_index_lane_to_agent
        d["lane", "to", "lane"].encoder_mask = l2l_mask
        d["lane", "to", "agent"].encoder_mask = l2a_mask
        d["agent", "to", "agent"].encoder_mask = a2a_mask

        return d

    def get(self, idx: int):
        with open(self.files[idx], "rb") as f:
            data = pickle.load(f)
        return self.get_data(data, idx)

    def len(self):
        return self.dset_len


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    dset = WaymoDatasetAEGoal(cfg.ae_goal.dataset, split_name="train")
    print(cfg.ae_goal.dataset.preprocess_dir)
    print(len(dset))
    if len(dset) > 0:
        d = dset.get(0)
        print(d)
        print("agent.x shape:", d["agent"].x.shape)


if __name__ == "__main__":
    main()
