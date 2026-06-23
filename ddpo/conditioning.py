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
from datasets.waymo.dataset_ldm_adv_waymo import WaymoDatasetLDMAdv
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
    if "parking" in d["agent"]:
        d["agent"].parking = d["agent"].parking[:keep]
    if "parking_label" in d["agent"]:
        d["agent"].parking_label = d["agent"].parking_label[:keep]
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
    # Override can be undone in decode if ego remains parked (parking head->spawn goal),
    # so force ego to not-parked in the conditioning graph when fields exist.
    if "parking_label" in d["agent"]:
        d["agent"].parking_label[0] = 0
    if "parking" in d["agent"]:
        d["agent"].parking[0] = 0.0
        d["agent"].parking[0, 0] = 1.0


class ConditioningPool:
    def __init__(
        self,
        dataset_cfg,
        *,
        split_name: str = "train",
        pool_size: int = 2048,
        dataset_cls=WaymoDatasetDMGoal,
        device: str = "cuda",
        seed: int = 0,
        control_agent_num: int = -1,
        ego_goal_override=None,
    ):
        self.dataset = dataset_cls(dataset_cfg, split_name=split_name, mode="eval")
        if len(self.dataset) == 0:
            raise RuntimeError(f"empty conditioning dataset for split '{split_name}' "
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

    def sample_group_batch(self, num_groups: int, group_size: int):
        """Sample ``num_groups`` distinct contexts, each replicated ``group_size``
        times, for per-context (GRPO-style) advantage normalisation.

        Returns ``(batch, group_ids)`` where ``group_ids`` is a CPU LongTensor of
        shape ``[num_groups * group_size]`` mapping each scene to its context
        group. The same conditioning graph is repeated ``group_size`` times; the
        policy draws independent per-node noise, so the repeats yield different
        samples that share map / ego / ego-goal. Per-group whitening of the
        resulting rewards then isolates "which generation is more critical in
        THIS context" from "which contexts are intrinsically easy".
        """
        pool_n = len(self.pool_indices)
        replace = int(num_groups) > pool_n
        groups = self.rng.choice(pool_n, size=int(num_groups), replace=replace)
        idx = np.repeat(groups, int(group_size))
        group_ids = torch.repeat_interleave(
            torch.arange(int(num_groups)), int(group_size)
        )
        return self.batch_from_indices(idx), group_ids


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

    def sample_group_batch(self, num_groups: int, group_size: int):
        """See ``ConditioningPool.sample_group_batch``."""
        pool_n = len(self.pool_indices)
        replace = int(num_groups) > pool_n
        groups = self.rng.choice(pool_n, size=int(num_groups), replace=replace)
        idx = np.repeat(groups, int(group_size))
        group_ids = torch.repeat_interleave(
            torch.arange(int(num_groups)), int(group_size)
        )
        return self.batch_from_indices(idx), group_ids


class LDMAdvConditioningPool:
    """Conditioning pool for ldm_adv DDPO (init_adv flow).

    Each graph carries the real ego + the real normal agents + lane latents (all
    held fixed by the policy as conditioning) plus one ``adv`` node that the
    policy regenerates from noise. Two knobs adapt it for the criticality reward
    (copied from the map-conditioned dm flow):

      * **prune_base_to_ego** (default ``False``) -- when ``True`` only the ego is
        kept among the real base agents (the rest dropped, graphs rebuilt), so the
        decoded scene is ``ego + adv`` and the reward's ego-vs-all TTC / collision
        unambiguously measures the adversary. When ``False`` (the default) the
        full real normal scene is kept: ldm_adv's intended setting, where the
        adversary is generated in the context of all real neighbours and the
        criticality credit is de-biased by GRPO per-context whitening (the normal
        scene is identical across a group, so its constant contribution is
        baselined out). ``controlled_mask`` still flags only the adv, so the
        approach / lane / parking terms and the green viz highlight stay
        adv-specific either way.
      * **near-stationary egos filtered out** -- a scene whose real ego barely
        drives (GT goal within ``min_ego_drive`` metres of spawn) gives the
        criticality reward no signal, so it is skipped at pool-build time (the
        data-side analogue of the dm flow's ego-goal override, which we do not
        do here: the ego keeps its real, on-road goal).

    Like ``LDMGoalConditioningPool`` it attaches sorted physical lane polylines
    (needed by the reward) from the latent-cache pickle.
    """

    def __init__(
        self,
        dataset_cfg,
        *,
        split_name: str = "train",
        pool_size: int = 2048,
        device: str = "cuda",
        seed: int = 0,
        min_ego_drive: float = 10.0,
        prune_base_to_ego: bool = False,
        adv_cond_target=None,
    ):
        self.dataset = WaymoDatasetLDMAdv(dataset_cfg, split_name=split_name, mode="eval")
        if len(self.dataset) == 0:
            raise RuntimeError(f"empty ldm_adv dataset for split '{split_name}' "
                               f"({dataset_cfg.dataset_path})")
        self.dataset_cfg = dataset_cfg
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.min_ego_drive = float(min_ego_drive)
        self.prune_base_to_ego = bool(prune_base_to_ego)
        self.fov = float(dataset_cfg.fov)
        # Fixed adversary conditioning target (type/motion/dist label triple). The
        # adv is generated from noise, so the real adv's labels are irrelevant --
        # override them with the desired target so the conditioned base model
        # samples the requested adversary category (e.g. car / moving / near).
        # None (or enabled=false) keeps each scene's real adv labels.
        self.adv_cond_target = self._parse_adv_cond_target(adv_cond_target)
        n = min(int(pool_size), len(self.dataset))
        self.pool_indices = self.rng.permutation(len(self.dataset))[:n]
        self._cache: dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.pool_indices)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _parse_adv_cond_target(spec):
        """Parse the fixed adv conditioning target into a LongTensor ``[1, 3]``
        ([type, motion, dist]), or ``None`` to keep each scene's real adv labels.
        Accepts an OmegaConf/dict node with ``enabled`` plus ``type/motion/dist``."""
        if spec is None:
            return None
        get = spec.get if hasattr(spec, "get") else (lambda k, d=None: getattr(spec, k, d))
        if not bool(get("enabled", False)):
            return None
        return torch.tensor(
            [[int(get("type", 0)), int(get("motion", 1)), int(get("dist", 0))]],
            dtype=torch.long,
        )

    def _apply_target_cond(self, d):
        """Override the adv node's conditioning labels with the fixed target."""
        if self.adv_cond_target is not None:
            d["adv"].cond = self.adv_cond_target.clone()
        return d

    def _ego_drives_enough(self, raw) -> bool:
        """True if the real ego's GT goal is >= ``min_ego_drive`` metres from its
        spawn. ``agent_states`` is stored min-max normalised to [-1, 1] over the
        FOV frame, so a difference of two positions scales to metres by fov/2; the
        ego is always raw row 0 (reorder_indices never moves it)."""
        a = np.asarray(raw["agent_states"], dtype=np.float64)
        if a.shape[0] < 2 or a.shape[1] < 9:
            return False
        norm_dist = float(np.linalg.norm(a[0, 7:9] - a[0, 0:2]))
        return norm_dist * (self.fov / 2.0) >= self.min_ego_drive

    def _prune_base_to_ego(self, d):
        """Keep only the ego (local index 0) among the real base agents; the adv
        node is untouched. Agent-agent / lane-agent graphs are rebuilt for a
        single base agent, matching ``prune_agents`` in the dm flow."""
        num_lanes = int(d["num_lanes"])
        d["agent"].x = d["agent"].x[:1]
        d["agent"].latents = d["agent"].latents[:1]
        if "log_var" in d["agent"]:
            d["agent"].log_var = d["agent"].log_var[:1]
        if "partition_mask" in d["agent"]:
            d["agent"].partition_mask = d["agent"].partition_mask[:1]
        d["num_agents"] = 1
        d["agent", "to", "agent"].edge_index = get_edge_index_complete_graph(1)
        d["lane", "to", "agent"].edge_index = get_edge_index_bipartite(num_lanes, 1)
        return d

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
        """Graph for pool slot ``pool_idx``. Scenes with no non-ego agent or a
        near-stationary ego are skipped by probing subsequent dataset indices
        (deterministic per slot, cached)."""
        if pool_idx in self._cache:
            return self._cache[pool_idx]
        ds_idx = int(self.pool_indices[pool_idx])
        for probe in range(len(self.dataset)):
            scene_idx = (ds_idx + probe) % len(self.dataset)
            with open(self.dataset.files[scene_idx], "rb") as f:
                raw = pickle.load(f)
            # Cheap filters on the raw pickle before building the graph.
            if not self._ego_drives_enough(raw):
                continue
            d = self.dataset.get(scene_idx)
            if d is None:
                continue
            d["lane"].road_points = self._sorted_road_points(raw)
            if self.prune_base_to_ego:
                d = self._prune_base_to_ego(d)
            d = self._apply_target_cond(d)
            self._cache[pool_idx] = d
            return d
        raise RuntimeError("no valid (driving-ego) conditioning graphs in dataset")

    def batch_from_indices(self, indices) -> Batch:
        return Batch.from_data_list([self._get(int(i)) for i in indices]).to(self.device)

    def sample_batch(self, batch_size: int) -> Batch:
        idx = self.rng.integers(0, len(self.pool_indices), size=batch_size)
        return self.batch_from_indices(idx)

    def sample_group_batch(self, num_groups: int, group_size: int):
        """See ``ConditioningPool.sample_group_batch``."""
        pool_n = len(self.pool_indices)
        replace = int(num_groups) > pool_n
        groups = self.rng.choice(pool_n, size=int(num_groups), replace=replace)
        idx = np.repeat(groups, int(group_size))
        group_ids = torch.repeat_interleave(
            torch.arange(int(num_groups)), int(group_size)
        )
        return self.batch_from_indices(idx), group_ids
