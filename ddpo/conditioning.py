"""Conditioning pool for ldm_adv DDPO, backed by the native dataset.

Graphs are built on demand by ``WaymoDatasetLDMAdv`` (mode="eval": no index
randomisation, so the SDC stays at local agent index 0 - the reward scores that
slot as the ego) and batched with ``Batch.from_data_list``.
"""

from __future__ import annotations

import pickle

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from datasets.waymo.dataset_ldm_adv_waymo import WaymoDatasetLDMAdv
from utils.data_helpers import reorder_indices
from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph


# --- adv-conditioning target vocabulary ------------------------------------
# Symbolic name -> bucket id, and the number of buckets per field. These match
# the discretization in WaymoDatasetLDMAdv._adv_condition (type one-hot argmax;
# motion binary parked/moving; goal_dist / ego_dist near/middle/far buckets).
_ADV_COND_FIELDS = ("type", "motion", "goal_dist", "ego_dist")
_ADV_COND_VOCAB = {
    "type": {"vehicle": 0, "car": 0, "veh": 0, "pedestrian": 1, "ped": 1, "cyclist": 2, "cyc": 2},
    "motion": {"parked": 0, "static": 0, "stationary": 0, "moving": 1},
    "goal_dist": {"near": 0, "middle": 1, "mid": 1, "far": 2},
    "ego_dist": {"near": 0, "middle": 1, "mid": 1, "far": 2},
}
_ADV_COND_NUM = {"type": 3, "motion": 2, "goal_dist": 3, "ego_dist": 3}
# Per-field null-token id = that field's bucket count. The conditioned DiT's
# LabelEmbedder adds an unconditional row at index ``num_classes`` (trained via
# cond_dropout_prob > 0); a field set to null in adv_cond_target draws this id, so
# the frozen embedder looks it up as the trained null embedding (genuinely
# unconstrained) and the reward invalid-check skips the field. MUST match
# ``_ADV_COND_NULL`` in ddpo/reward_hooks.py.
_ADV_COND_NULL = {f: n for f, n in _ADV_COND_NUM.items()}


def _resolve_adv_label(field: str, value) -> int:
    """Map one adv-cond value (int bucket id or symbolic name) to its bucket id."""
    if isinstance(value, str):
        key = value.strip().lower()
        vocab = _ADV_COND_VOCAB[field]
        if key not in vocab:
            raise ValueError(
                f"unknown {field} label {value!r}; expected one of "
                f"{sorted(set(vocab))} or an int in [0, {_ADV_COND_NUM[field] - 1}]"
            )
        return vocab[key]
    v = int(value)
    if not 0 <= v < _ADV_COND_NUM[field]:
        raise ValueError(
            f"{field} bucket {v} out of range [0, {_ADV_COND_NUM[field] - 1}]"
        )
    return v


def _parse_adv_field(field: str, spec) -> list:
    """One field's config value -> list of candidate label ids that the per-scene
    draw samples from.

    ``None`` (or an empty list) -> the null token (``_ADV_COND_NULL[field]``): the
    field is fed the model's trained unconditional embedding and is skipped by the
    reward invalid-check, i.e. genuinely unconstrained. A non-empty list/tuple ->
    that explicit set of concrete buckets, sampled uniformly per scene. A scalar ->
    a single fixed concrete bucket."""
    if spec is None:
        return [_ADV_COND_NULL[field]]
    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            return [_ADV_COND_NULL[field]]
        return [_resolve_adv_label(field, v) for v in spec]
    return [_resolve_adv_label(field, spec)]


class LDMAdvConditioningPool:
    """Conditioning pool for ldm_adv DDPO (init_adv flow).

    Each graph carries the real ego + the real normal agents + lane latents (all
    held fixed by the policy as conditioning) plus one ``adv`` node that the
    policy regenerates from noise. Two knobs adapt it for the criticality
    reward:

      * **insert_adv_as_extra** (default ``False``) -- when ``True`` the selected
        real adversary remains in the normal-agent context and the generated
        adversary is inserted as an extra agent during decode. This preserves the
        full real scene for DDPO while keeping the base ldm_adv training path as
        replacement-style missing-agent modelling.
      * **prune_base_to_ego** (default ``False``) -- when ``True`` only the ego is
        kept among the real base agents (the rest dropped, graphs rebuilt), so the
        decoded scene is ``ego + adv`` and the reward's ego-vs-all TTC / collision
        unambiguously measures the adversary. When ``False`` (the default) the
        full real normal scene is kept: ldm_adv's intended setting, where the
        adversary is generated in the context of all real neighbours and the
        criticality credit is de-biased by GRPO per-context whitening (the normal
        scene is identical across a group, so its constant contribution is
        baselined out). ``gen_agent_mask`` still flags only the adv, so the
        approach / lane / parking terms and the green viz highlight stay
        adv-specific either way.
      * **near-stationary egos filtered out** -- a scene whose real ego barely
        drives (GT goal within ``min_ego_drive`` metres of spawn) gives the
        criticality reward no signal, so it is skipped at pool-build time (the
        ego keeps its real, on-road goal).

    Sorted physical lane polylines (needed by the reward) are attached to every
    graph from the latent-cache pickle.
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
        insert_adv_as_extra: bool = False,
        adv_cond_target=None,
    ):
        self.dataset = WaymoDatasetLDMAdv(
            dataset_cfg,
            split_name=split_name,
            mode="eval",
            keep_adv_in_agents=insert_adv_as_extra,
        )
        if len(self.dataset) == 0:
            raise RuntimeError(f"empty ldm_adv dataset for split '{split_name}' "
                               f"({dataset_cfg.dataset_path})")
        self.dataset_cfg = dataset_cfg
        self.device = device
        self.rng = np.random.default_rng(seed)
        # Base seed for the per-scene (deterministic) adv-cond draw; kept separate
        # from self.rng so the sampled target depends only on (seed, scene_idx),
        # not on pool draw order -> constant within a GRPO group and across epochs.
        self.adv_cond_seed = int(seed)
        self.min_ego_drive = float(min_ego_drive)
        self.prune_base_to_ego = bool(prune_base_to_ego)
        self.insert_adv_as_extra = bool(insert_adv_as_extra)
        self.fov = float(dataset_cfg.fov)
        # Adversary conditioning target as per-field candidate bucket lists
        # ({type, motion, goal_dist, ego_dist} -> [int, ...]). The adv is generated
        # from noise, so the real adv's labels are irrelevant -- a per-scene draw
        # from these candidates overrides them so the conditioned base model samples
        # the requested adversary category. Each field can be fixed (a single
        # value), randomized over a set (a list), or unconstrained (null -> the
        # model's null token, skipped by the reward check); see
        # _parse_adv_cond_target. None (or enabled=false) keeps each scene's real
        # adv labels.
        self.adv_cond_target = self._parse_adv_cond_target(adv_cond_target)
        n = min(int(pool_size), len(self.dataset))
        self.pool_indices = self.rng.permutation(len(self.dataset))[:n]
        self._cache: dict[int, object] = {}
        # pool slot -> dataset index the slot actually resolved to after the
        # driving-ego probing in _get (for reproducibility manifests).
        self.resolved_scene_idx: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.pool_indices)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _parse_adv_cond_target(spec):
        """Parse the adv conditioning target into per-field candidate bucket lists
        ``{type, motion, goal_dist, ego_dist} -> [int, ...]``, or ``None`` to keep
        each scene's real adv labels.

        Accepts an OmegaConf/dict node with ``enabled`` plus the four fields. Each
        field may be a single value (fixed), a list (sampled uniformly per scene),
        or null / omitted (the null token: unconditional and skipped by the reward
        invalid-check). Values may be int bucket ids or symbolic names -- type:
        vehicle/pedestrian/cyclist; motion:
        parked/moving; goal_dist & ego_dist: near/middle/far. The per-scene draw
        itself happens in ``_apply_target_cond`` so it stays constant within a GRPO
        group. ``enabled=false`` keeps each scene's real adv labels (returns None).
        """
        if spec is None:
            return None
        if OmegaConf.is_config(spec):
            spec = OmegaConf.to_container(spec, resolve=True)
        get = spec.get if hasattr(spec, "get") else (lambda k, d=None: getattr(spec, k, d))
        if not bool(get("enabled", False)):
            return None
        return {f: _parse_adv_field(f, get(f, None)) for f in _ADV_COND_FIELDS}

    def _apply_target_cond(self, d, scene_idx):
        """Override the adv node's conditioning labels with a per-scene sample from
        the target candidates. The draw is deterministic in ``(seed, scene_idx)``,
        so a scene's adv target is constant within a GRPO group (per-context
        whitening compares samples that share the SAME conditioning) and stable
        across epochs."""
        if self.adv_cond_target is None:
            return d
        rng = np.random.default_rng((self.adv_cond_seed, int(scene_idx)))
        labels = [int(rng.choice(self.adv_cond_target[f])) for f in _ADV_COND_FIELDS]
        d["adv"].cond = torch.tensor([labels], dtype=torch.long)
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
        single base agent."""
        num_lanes = int(d["num_lanes"])
        d["agent"].x = d["agent"].x[:1]
        d["agent"].latents = d["agent"].latents[:1]
        if "log_var" in d["agent"]:
            d["agent"].log_var = d["agent"].log_var[:1]
        if "partition_mask" in d["agent"]:
            d["agent"].partition_mask = d["agent"].partition_mask[:1]
        if "gt_x" in d["agent"]:
            d["agent"].gt_x = d["agent"].gt_x[:1]
        if "gt_type" in d["agent"]:
            d["agent"].gt_type = d["agent"].gt_type[:1]
        # Per-agent conditioning labels must shrink with the agent set, otherwise
        # the conditioned DiT adds an (N, 3) embedding onto a single ego token.
        if "cond" in d["agent"]:
            d["agent"].cond = d["agent"].cond[:1]
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

    def build_scene(self, scene_idx: int, *, require_driving_ego: bool = True):
        """Build the full conditioning graph for dataset index ``scene_idx`` with
        every config-driven transform applied: sorted physical lane polylines, the
        optional base->ego prune, and the per-scene adversary-conditioning target
        override. Returns ``None`` when the scene has no non-ego agent (or, when
        ``require_driving_ego``, a near-stationary ego).

        Shared by the pool's own random sampling (``_get``) and external callers
        (the ldm_adv viz test scripts) so they condition the model exactly the way
        DDPO training does."""
        with open(self.dataset.files[scene_idx], "rb") as f:
            raw = pickle.load(f)
        # Cheap filter on the raw pickle before building the graph.
        if require_driving_ego and not self._ego_drives_enough(raw):
            return None
        d = self.dataset.get(scene_idx)
        if d is None:
            return None
        d["lane"].road_points = self._sorted_road_points(raw)
        if self.prune_base_to_ego:
            d = self._prune_base_to_ego(d)
        d = self._apply_target_cond(d, scene_idx)
        return d

    def _get(self, pool_idx: int):
        """Graph for pool slot ``pool_idx``. Scenes with no non-ego agent or a
        near-stationary ego are skipped by probing subsequent dataset indices
        (deterministic per slot, cached)."""
        if pool_idx in self._cache:
            return self._cache[pool_idx]
        ds_idx = int(self.pool_indices[pool_idx])
        for probe in range(len(self.dataset)):
            scene_idx = (ds_idx + probe) % len(self.dataset)
            d = self.build_scene(scene_idx, require_driving_ego=True)
            if d is not None:
                self._cache[pool_idx] = d
                self.resolved_scene_idx[pool_idx] = scene_idx
                return d
        raise RuntimeError("no valid (driving-ego) conditioning graphs in dataset")

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
