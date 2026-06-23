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

from cfgs.config import CONFIG_PATH
from ddpo.goal_schema import MIN_DISTANCE_TO_GOAL
from utils.data_container import ScenarioDreamerData
from utils.data_helpers import reorder_indices, reparameterize, sample_latents
from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph
from utils.torch_helpers import from_numpy


class WaymoDatasetLDMAdv(Dataset):
    """Latent-diffusion dataset that mirrors :class:`WaymoDatasetLDM` but splits a
    single adversarial agent off into a separate ``adv`` node.

    Reuses the **exact** goal-autoencoder latent cache
    (``scenario_dreamer_ae_goal_latents_waymo``): the autoencoder is completely
    unaware of the adversary, so an adversary is simply one agent's latent. After
    the deterministic ego-first reordering (same as the base LDM dataset) one
    non-ego agent's ``mu/log_var/latent`` is split into the ``adv`` node and the
    agent-to-agent / lane-to-agent graphs are rebuilt over the reduced agent set
    (exactly like ``WaymoDatasetDMAdv`` does in raw space).

    Scenes with fewer than two agents (no non-ego agent to act as the adversary)
    are dropped by returning ``None`` -- the datamodule's collater filters those
    out -- so every retained scene has exactly one ``adv`` node.
    """

    def __init__(self, cfg: Any, split_name: str = "train", mode: str = "train") -> None:
        super(WaymoDatasetLDMAdv, self).__init__()
        self.cfg = cfg
        self.split_name = split_name
        self.mode = mode
        self.dataset_dir = os.path.join(self.cfg.dataset_path, f"{self.split_name}")
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir, exist_ok=True)
        self.files = sorted(glob.glob(os.path.join(self.dataset_dir, "*.pkl")))
        self.dset_len = len(self.files)

    def _select_adv_index(self, num_agents):
        """Pick the index of the single adversarial agent: a random non-ego agent
        (ego is index 0 after the ego-first reorder and is never selected).
        Deterministic outside of training for reproducible evaluation. Assumes
        ``num_agents >= 2``."""
        non_ego = np.arange(1, num_agents)
        if self.mode == "train":
            return int(np.random.choice(non_ego))
        return int(non_ego[0])

    def _adv_condition(self, agent_states, agent_types, adv_idx):
        """Discretize the adversary's own (normalized) state into the
        ``[type, motion, dist]`` label triple consumed by the conditioned DiT adv
        stream. ``agent_states``/``agent_types`` must already be reordered with the
        same permutation as the latents (ego at index 0). Returns a ``LongTensor``
        of shape ``(1, 3)`` so it batches to ``(batch_size, 3)``."""
        # agent type: one-hot [N, num_agent_types] -> class id (0=veh, 1=ped, 2=cyc)
        adv_type = int(np.argmax(agent_types[adv_idx]))

        adv_xy = agent_states[adv_idx, :2]
        # parked / moving: match visualization/DDPO semantics. Normalized xy and
        # goal xy are scaled by fov/2 per unit -> metres.
        adv_goal_xy = agent_states[adv_idx, 7:9]
        phys_goal_dist = float(np.linalg.norm(adv_goal_xy - adv_xy)) * (self.cfg.fov / 2.0)
        adv_motion = 0 if phys_goal_dist < MIN_DISTANCE_TO_GOAL else 1

        # distance bucket: adversary distance from ego in metres
        ego_xy = agent_states[0, :2]
        phys_dist = float(np.linalg.norm(adv_xy - ego_xy)) * (self.cfg.fov / 2.0)
        if phys_dist < self.cfg.adv_cond_dist_near_threshold:
            adv_dist = 0
        elif phys_dist < self.cfg.adv_cond_dist_far_threshold:
            adv_dist = 1
        else:
            adv_dist = 2

        return torch.tensor([[adv_type, adv_motion, adv_dist]], dtype=torch.long)

    def get_data(self, data, idx):
        agent_mu = data["agent_mu"]
        agent_log_var = data["agent_log_var"]
        lane_mu = data["lane_mu"]
        lane_log_var = data["lane_log_var"]
        agent_states = data["agent_states"]
        agent_types = data["agent_types"]
        road_points = data["road_points"]
        edge_index_lane_to_lane = data["edge_index_lane_to_lane"]
        scene_type = data["scene_type"]
        map_id = data["nocturne_compatible"]

        # Need the ego plus at least one non-ego agent to act as the adversary.
        if agent_mu.shape[0] < 2:
            return None

        idx = data["idx"]
        num_lanes = lane_mu.shape[0]

        # Reorder the raw adv-conditioning state + type with the SAME deterministic
        # permutation the latents get below (reorder_indices sorts on agent_states),
        # so the condition labels stay aligned with the reordered latent rows. We
        # reuse the generic permutation machinery by passing states/types in the
        # agent slots -- identical idiom to dataset_dm_adv_waymo.py /
        # verify_adv_overlap.py. Done before the latent reorder so the original
        # (un-reordered) lane inputs are still available. Lane outputs are dropped.
        agent_states_r, agent_types_r = reorder_indices(
            agent_states,
            agent_types,
            lane_mu,
            lane_log_var,
            edge_index_lane_to_lane,
            agent_states,
            road_points,
            scene_type,
            dataset="waymo",
        )[:2]

        # Deterministic ego-first ordering (identical to the base LDM dataset) so
        # the positional encodings are meaningful and ego stays at index 0.
        (
            agent_mu,
            agent_log_var,
            lane_mu,
            lane_log_var,
            edge_index_lane_to_lane,
            agent_partition_mask,
            lane_partition_mask,
        ) = reorder_indices(
            agent_mu,
            agent_log_var,
            lane_mu,
            lane_log_var,
            edge_index_lane_to_lane,
            agent_states,
            road_points,
            scene_type,
            dataset="waymo",
        )
        edge_index_lane_to_lane = torch.from_numpy(edge_index_lane_to_lane)

        # Split off the single adversarial agent latent from the normal agent set.
        adv_idx = self._select_adv_index(agent_mu.shape[0])
        adv_mu = agent_mu[adv_idx:adv_idx + 1]
        adv_log_var = agent_log_var[adv_idx:adv_idx + 1]
        keep = np.ones(agent_mu.shape[0], dtype=bool)
        keep[adv_idx] = False
        agent_mu = agent_mu[keep]
        agent_log_var = agent_log_var[keep]
        agent_partition_mask = agent_partition_mask[keep]

        num_agents = int(agent_mu.shape[0])
        # Rebuild the agent graphs over the reduced agent set (one fewer agent).
        edge_index_lane_to_agent = get_edge_index_bipartite(num_lanes, num_agents)
        edge_index_agent_to_agent = get_edge_index_complete_graph(num_agents)

        d = ScenarioDreamerData()
        d["idx"] = idx
        d["num_lanes"] = num_lanes
        d["num_agents"] = num_agents
        d["lg_type"] = scene_type
        d["map_id"] = map_id
        d["agent"].x = from_numpy(agent_mu)
        d["lane"].x = from_numpy(lane_mu)
        d["agent"].log_var = from_numpy(agent_log_var)
        d["lane"].log_var = from_numpy(lane_log_var)
        d["agent"].partition_mask = from_numpy(agent_partition_mask)
        d["lane"].partition_mask = from_numpy(lane_partition_mask)
        d["adv"].x = from_numpy(adv_mu)
        d["adv"].log_var = from_numpy(adv_log_var)
        d["adv"].partition_mask = torch.zeros(1).bool()
        # Discretized adv-only conditioning labels [type, motion, dist] (shape (1, 3))
        # aligned with the reordered adv row, batches to (batch_size, 3).
        d["adv"].cond = self._adv_condition(agent_states_r, agent_types_r, adv_idx)

        # Sample (and normalize) the normal-agent and lane latents for diffusion.
        d["agent"].latents, d["lane"].latents = sample_latents(
            d,
            self.cfg.agent_latents_mean,
            self.cfg.agent_latents_std,
            self.cfg.lane_latents_mean,
            self.cfg.lane_latents_std,
            normalize=True,
        )
        # The adversary shares the agent latent statistics (it is an agent latent).
        adv_latents = reparameterize(d["adv"].x, d["adv"].log_var)
        adv_latents = (adv_latents - self.cfg.agent_latents_mean) / self.cfg.agent_latents_std
        d["adv"].latents = adv_latents

        d["lane", "to", "lane"].edge_index = edge_index_lane_to_lane
        d["agent", "to", "agent"].edge_index = edge_index_agent_to_agent
        d["lane", "to", "agent"].edge_index = edge_index_lane_to_agent

        return d

    def get(self, idx: int):
        with open(self.files[idx], "rb") as f:
            data = pickle.load(f)
        return self.get_data(data, idx)

    def len(self):
        return self.dset_len


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    cfg = cfg.ldm_adv
    cfg.dataset.agent_latents_mean = 0.0
    cfg.dataset.agent_latents_std = 1.0
    cfg.dataset.lane_latents_mean = 0.0
    cfg.dataset.lane_latents_std = 1.0
    dset = WaymoDatasetLDMAdv(cfg.dataset, split_name="train")
    print(cfg.dataset.dataset_path)
    print(len(dset))
    if len(dset) > 0:
        d = dset.get(0)
        print(d)


if __name__ == "__main__":
    main()
