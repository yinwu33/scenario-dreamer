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
from utils.data_helpers import normalize_scene, randomize_indices, reorder_indices
from utils.goal_runtime import prepare_scene
from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph
from utils.torch_helpers import from_numpy


class WaymoDatasetDMGoal(Dataset):
    """Waymo **data-space** diffusion dataset: the current scene plus per-agent goals.

    This is the dataset of the SceneControl-style baseline. It reads the **v2**
    SDC-centered goal records and produces exactly the agent set the goal
    autoencoder (``WaymoDatasetAEGoal``) produces -- same ``prepare_scene`` call,
    same off-road / valid-goal filtering, same ``max_num_agents`` cap -- so the
    data-space model and the latent chain (ae_goal -> ldm_adv) are trained on an
    identical distribution and their metrics are comparable. The only difference
    is what gets diffused: raw 9-D geometry here, autoencoder latents there.

    Each agent state is ``[x, y, speed, cosθ, sinθ, length, width, goal_x, goal_y]``
    (``state_dim == 9``), min-max normalized into ``[-1, 1]`` by ``normalize_scene``
    (which handles the two goal columns in the same FOV frame as the position).

    Unlike the adversary datasets there is **no** ``adv`` node type: every agent is
    symmetric. The adversary of the guidance baseline is chosen at sampling time by
    applying the guidance cost to one agent's rows of ``x̂₀`` -- the model itself
    stays a plain unconditional scene generator, as in SceneControl.

    Agents are reordered ego-first with the deterministic hierarchical sort
    (``reorder_indices``) so the DiT positional encodings are meaningful; the ego
    is index 0 and is never moved.
    """

    def __init__(self, cfg: Any, split_name: str = "train", mode: str = "train") -> None:
        super(WaymoDatasetDMGoal, self).__init__()
        self.cfg = cfg
        self.split_name = split_name
        self.mode = mode
        self.dataset_dir = os.path.join(self.cfg.preprocess_dir, self.split_name)
        self.files = sorted(glob.glob(os.path.join(self.dataset_dir, "*.pkl")))
        self.dset_len = len(self.files)

    def get_data(self, data, idx, path=None):
        # Everything the original preprocessing did (FOV crop, closest-N cap, off-road
        # vehicle removal, modify_agent_states) is already baked into the v2 record;
        # prepare_scene adds the goal columns and the goal-driven filtering at runtime.
        # NOTE: do NOT call modify_agent_states here -- v2 stores agent_states already
        # converted to [x, y, speed, cosθ, sinθ, l, w]. Applying it a second time reads
        # (speed, cos, sin) as (vx, vy, yaw) and silently corrupts every heading.
        scene = prepare_scene(data, self.cfg)
        agent_states = scene["agent_states"]  # [N, 9], goal columns included
        agent_types = scene["agent_types"]  # [N, num_agent_types]

        num_agents = int(len(agent_states))
        assert num_agents != 0

        # normalize_scene mutates in place, so work on copies of the stored tensors.
        road_points = np.array(data["road_points"], copy=True)
        num_lanes = int(data["num_lanes"])
        edge_index_lane_to_lane = np.array(data["edge_index_lane_to_lane"], copy=True)
        road_connection_types = np.array(data["road_connection_types"], copy=True)
        lg_type = int(data.get("lg_type", NON_PARTITIONED))

        # min-max normalize agent states (incl. the two goal columns) and lanes into [-1, 1]
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

        # training-only randomization of non-ego agent and lane ordering; it only
        # affects how ties (within reorder_indices' tolerance) are broken below.
        if self.mode == "train":
            agent_states, agent_types, road_points, edge_index_lane_to_lane = randomize_indices(
                agent_states,
                agent_types,
                road_points,
                edge_index_lane_to_lane,
            )

        # Deterministic ego-first ordering so the positional encodings are meaningful.
        # Reuses the generic permutation machinery by passing states/types in the
        # agent slots and road_points in both lane slots (same idiom as the adv
        # datasets); the duplicated lane output is dropped.
        (
            agent_states,
            agent_types,
            road_points,
            _,
            edge_index_lane_to_lane,
            agent_partition_mask,
            lane_partition_mask,
        ) = reorder_indices(
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

        if self.cfg.remove_left_right_connections:
            # keep only none/pred/succ/self
            road_connection_types = road_connection_types[:, [0, 1, 2, 5]]

        # v2 records are always non-partitioned (no inpainting machinery).
        agent_partition_mask = np.zeros(num_agents).astype(bool)
        lane_partition_mask = np.zeros(num_lanes).astype(bool)

        edge_index_lane_to_agent = get_edge_index_bipartite(num_lanes, num_agents).numpy()
        edge_index_agent_to_agent = get_edge_index_complete_graph(num_agents).numpy()

        d = ScenarioDreamerData()
        d["idx"] = int(data.get("idx", idx))
        d["num_lanes"] = num_lanes
        d["num_agents"] = num_agents
        d["lg_type"] = lg_type
        # v2 records carry no nocturne metadata; the DiT scene-type label is
        # therefore constant across the dataset (see cfgs/dm_goal/train.yaml).
        d["map_id"] = int(data.get("map_id", 0))
        d["agent"].x = from_numpy(agent_states.astype(np.float32))
        d["agent"].type = from_numpy(agent_types.astype(np.float32))
        d["lane"].x = from_numpy(road_points.astype(np.float32))
        d["agent"].partition_mask = from_numpy(agent_partition_mask)
        d["lane"].partition_mask = from_numpy(lane_partition_mask)
        d["lane", "to", "lane"].edge_index = from_numpy(edge_index_lane_to_lane)
        d["lane", "to", "lane"].type = from_numpy(road_connection_types.astype(np.float32))
        d["agent", "to", "agent"].edge_index = from_numpy(edge_index_agent_to_agent)
        d["lane", "to", "agent"].edge_index = from_numpy(edge_index_lane_to_agent)
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


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config_dm_goal")
def main(cfg):
    dset = WaymoDatasetDMGoal(cfg.dm_goal.dataset, split_name="val", mode="eval")
    print(cfg.dm_goal.dataset.preprocess_dir)
    print(len(dset))
    if len(dset) > 0:
        d = dset.get(0)
        print(d)
        print("agent.x shape:", d["agent"].x.shape)


if __name__ == "__main__":
    main()
