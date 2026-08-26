"""The scene contract: what any scene *source* hands to the rollout.

``GeneratedScenes`` is the single input type of ``sim.runner.RolloutRunner``, so
it is also the seam that makes scene sources interchangeable. Today's producers
are the DDPO diffusion policy (``ddpo.policy_ldm_adv``), the real-log loader
(``critical_scene.log_scenes``) and the paired-source evaluation
(``critical_scene.ldm_adv_eval``); none of them is privileged, and nothing below
this module knows which one it is looking at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from cfgs.config import LANE_CONNECTION_TYPES_WAYMO


@dataclass
class GeneratedScenes:
    """Decoded, simulator-ready output of one ``sample`` call (a batch of scenes).

    All agent tensors are flattened across the batch (PyG style); ``agent_scene_idx``
    maps every agent row to its scene in ``[0, num_scenes)``. By convention the
    ego/SDC is local index 0 within each scene (the dm_goal dataset is SDC-centric).

    ``agent_states`` layout (physical units, from dm_goal decode):
        [x, y, speed, cos_theta, sin_theta, length, width, goal_x, goal_y]
    """

    agent_states: torch.Tensor      # [N_agents, 9]
    agent_types: torch.Tensor       # [N_agents] int class ids (0 veh, 1 ped, 2 cyc)
    agent_scene_idx: torch.Tensor   # [N_agents] -> scene id in [0, num_scenes)
    lane_polylines: Any             # [N_lanes, P, 2] map geometry (fixed or generated)
    num_scenes: int
    # Canonical single-adversary handle: per-scene sim-local index of THE generated
    # adversary (the sole generated non-ego agent), or -1 if the scene has none
    # (e.g. no-adv conditioning scenes). ``meta['gen_agent_mask']`` is the same
    # information as a per-node bool, kept only for viz/analysis scripts.
    adv_local_idx: torch.Tensor | None = None   # [num_scenes] long, -1 == no adversary
    # Free-form side channel. Conventional keys:
    #   "lane_scene_idx"  -- [N_lanes] -> scene id, the lane counterpart of agent_scene_idx
    #   "gen_agent_mask"  -- [N_agents] bool, per-node form of adv_local_idx (viz/analysis)
    #   "lane_graph"      -- list[dict], one per scene: {"succ": [2, E], "lateral": [2, E]}
    #                        arrays of SCENE-LOCAL lane indices. ``succ`` edges run in the
    #                        driving direction; ``lateral`` pairs left/right neighbours.
    #                        Optional: the lane GRAPH, as opposed to the lane geometry in
    #                        lane_polylines. Route-planning planners (sim.routes) need it;
    #                        the frozen neural planners do not, so most producers leave it
    #                        out and SimScene.lane_graph stays None.
    meta: dict = field(default_factory=dict)


def _as_numpy(x, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def lane_graph_edges(edge_index, connection_types) -> dict[str, np.ndarray]:
    """``{"succ": [2, E], "lateral": [2, E]}`` lane connectivity over lane rows.

    Takes the two arrays rather than a scene record so every producer shares this
    decoding: the log loader passes the stored pair, generated caches rebuild the
    edge index they were flattened against (``critical_scene.gen_scenes``), and
    the DDPO decode passes the autoencoder's PREDICTED connection types
    (``batched_lane_graphs``).

    ``succ`` runs in the driving direction; ``lateral`` pairs left/right
    neighbours (used to widen the route search's candidate lanes, not as
    traversable edges -- see ``sim.routes``).

    Edge-direction note, verified against the geometry on 60 val scenes: the
    stored graph is in PyG message-passing form, where the type labels the SOURCE
    node's relation to the destination. ``build_road_connection_types``
    (``utils/goal_preprocess.py:189``) tags edge ``(src, dst)`` as ``succ`` when
    ``suc_adj[dst, src]`` -- i.e. when *src is the successor of dst*. So the
    driving direction of a ``succ`` edge is ``dst -> src`` and the column order
    must be flipped here. (Checked: with the flip, the end of the upstream lane
    coincides with the start of the downstream one to 0.000 m; without it the gap
    is ~40 m.) Lateral edges are symmetric, so their orientation does not matter.
    """
    edge_index = _as_numpy(edge_index, dtype=np.int64)
    types = _as_numpy(connection_types).argmax(axis=-1)
    if edge_index.shape[1] != types.shape[0]:
        raise ValueError(
            "lane_graph_edges: edge index and connection types disagree "
            f"({edge_index.shape[1]} edges vs {types.shape[0]} types)"
        )
    succ = edge_index[:, types == LANE_CONNECTION_TYPES_WAYMO["succ"]]
    lateral = edge_index[
        :,
        (types == LANE_CONNECTION_TYPES_WAYMO["left"])
        | (types == LANE_CONNECTION_TYPES_WAYMO["right"]),
    ]
    return {"succ": succ[::-1].copy(), "lateral": lateral.copy()}


def batched_lane_graphs(
    edge_index, connection_types, lane_scene_idx, num_scenes: int
) -> list[dict[str, np.ndarray]]:
    """Split one batch's lane-to-lane edges into per-scene, scene-local graphs.

    The batched (PyG-style) inputs are what a decode pass has on hand:
    ``edge_index`` over GLOBAL lane rows, one ``connection_types`` row per edge in
    the same order, and ``lane_scene_idx`` mapping each lane row to its scene.
    ``meta['lane_graph']`` is per-scene and scene-LOCAL (that is what
    ``RolloutRunner._build_scenes`` hands to each ``SimScene``), so the global
    lane offset is subtracted here.

    Each edge belongs to the scene of its endpoints -- both are in the same scene,
    so the source row decides, the same way ``utils.data_container.get_batches``
    derives its lane-connection batch vector.
    """
    edge_index = _as_numpy(edge_index, dtype=np.int64).reshape(2, -1)
    conn = _as_numpy(connection_types)
    lane_scene_idx = _as_numpy(lane_scene_idx, dtype=np.int64)

    # The waymo 6-class layout is what lane_graph_edges indexes into. A 4-class
    # array (dataset.remove_left_right_connections, or nuplan) would silently
    # decode 'self' edges as 'left', so refuse it rather than guess.
    num_types = len(LANE_CONNECTION_TYPES_WAYMO)
    if conn.ndim != 2 or conn.shape[1] != num_types:
        raise ValueError(
            "batched_lane_graphs: connection_types must be one-hot over the "
            f"{num_types} waymo lane-connection classes, got shape {conn.shape}"
        )
    # Scene-local indices are built by subtracting each scene's first lane row,
    # which is only the right offset while the batch is grouped by scene. PyG
    # batching guarantees that; assert it rather than emit wrong indices.
    if lane_scene_idx.size and np.any(np.diff(lane_scene_idx) < 0):
        raise ValueError(
            "batched_lane_graphs: lane_scene_idx must be non-decreasing "
            "(lane rows grouped by scene, as PyG batching produces)"
        )
    offsets = np.searchsorted(lane_scene_idx, np.arange(num_scenes))

    empty = np.zeros((2, 0), dtype=np.int64)
    edge_scene = (
        lane_scene_idx[edge_index[0]] if edge_index.shape[1] else np.zeros(0, np.int64)
    )
    graphs = []
    for s in range(num_scenes):
        mask = edge_scene == s
        if not mask.any():
            graphs.append({"succ": empty.copy(), "lateral": empty.copy()})
            continue
        graphs.append(lane_graph_edges(edge_index[:, mask] - offsets[s], conn[mask]))
    return graphs


def single_adv_local_idx(gen_agent_mask, agent_scene_idx, num_scenes):
    """Per-scene sim-local index of the single generated non-ego adversary.

    Returns a ``[num_scenes]`` long tensor; ``-1`` marks a scene with no generated
    adversary. The local index is the agent's *order of appearance* within its
    scene (ego is local 0), which is exactly how ``Planner._build_scenes``
    slices each ``SimScene`` -- so it doubles as the sim-local index downstream.

    Enforces the single-adversary contract: at most one generated non-ego agent
    per scene (raises otherwise). ``agent_scene_idx`` need not be contiguous /
    sorted (ldm_adv appends the adv after the whole base set).
    """
    device = agent_scene_idx.device
    out = torch.full((num_scenes,), -1, dtype=torch.long, device=device)
    if gen_agent_mask is None or agent_scene_idx.numel() == 0:
        return out

    n = agent_scene_idx.shape[0]
    order = torch.arange(n, device=device)
    scene = agent_scene_idx.to(torch.long)
    # Order-preserving within-scene rank: stable-sort by (scene, appearance).
    perm = torch.argsort(scene * n + order)
    counts = torch.bincount(scene, minlength=num_scenes)
    offsets = torch.cumsum(counts, 0) - counts
    local = torch.empty(n, dtype=torch.long, device=device)
    local[perm] = order - offsets[scene[perm]]

    sel = gen_agent_mask.to(torch.bool) & (local > 0)   # generated, non-ego
    sel_scene = scene[sel]
    if sel_scene.numel():
        if int(torch.bincount(sel_scene, minlength=num_scenes).max()) > 1:
            raise ValueError(
                "single_adv_local_idx: more than one generated non-ego agent in a "
                "scene -- the DDPO reward assumes a single adversary "
                "(check control_agent_num)."
            )
        out[sel_scene] = local[sel]
    return out


def slice_scenes(scenes: "GeneratedScenes", lo: int, hi: int) -> "GeneratedScenes":
    """The scenes ``[lo, hi)`` as a standalone, self-consistent batch.

    Used by ``sim.parallel`` to hand each worker its shard: every scene index is
    renumbered to ``[0, hi - lo)`` so the shard's hooks index their own metric
    arrays exactly as a single-process rollout over those scenes would. Agent
    ids stay SCENE-LOCAL and so are untouched, which is why the sliced rollout
    reproduces the full one bitwise.

    Tensors are returned on the CPU: the shard is pickled to a worker process
    that has no CUDA context, and ``RolloutRunner._build_scenes`` would move
    them to the host anyway.

    Every payload is classified by its leading dimension -- per agent, per lane,
    or per scene. Anything that matches none of the three is passed through
    unchanged, and anything AMBIGUOUS raises rather than being sliced on a
    guess (a silently mis-sliced conditioning field would corrupt the rollout
    without ever failing).
    """
    if not (0 <= lo < hi <= scenes.num_scenes):
        raise ValueError(f"slice [{lo}, {hi}) out of range for {scenes.num_scenes} scenes")

    n_scenes = int(scenes.num_scenes)
    a_idx = _as_numpy(scenes.agent_scene_idx, dtype=np.int64)
    l_idx = _as_numpy(scenes.meta["lane_scene_idx"], dtype=np.int64)
    n_agents, n_lanes = a_idx.shape[0], l_idx.shape[0]
    a_keep = (a_idx >= lo) & (a_idx < hi)
    l_keep = (l_idx >= lo) & (l_idx < hi)

    def _cpu(x):
        return x.detach().cpu() if isinstance(x, torch.Tensor) else x

    def _take(x, keep):
        return _cpu(x)[torch.as_tensor(keep)] if isinstance(x, torch.Tensor) else np.asarray(x)[keep]

    def _by_leading_dim(key, value):
        """Slice one payload by whichever axis its length identifies."""
        if isinstance(value, (torch.Tensor, np.ndarray)):
            n = value.shape[0]
        elif isinstance(value, list):
            n = len(value)
        else:
            return value  # scalar / mapping / anything not indexed by row
        matches = [n == n_agents, n == n_lanes, n == n_scenes]
        if sum(matches) > 1:
            raise ValueError(
                f"meta[{key!r}] has leading dim {n}, which is ambiguous between "
                f"agents({n_agents}) / lanes({n_lanes}) / scenes({n_scenes}); "
                "sim.parallel cannot shard this batch -- give the field a "
                "distinguishable layout or special-case it in slice_scenes"
            )
        if matches[0]:
            return _take(value, a_keep)
        if matches[1]:
            return _take(value, l_keep)
        if matches[2]:
            return value[lo:hi] if isinstance(value, list) else _cpu(value)[lo:hi]
        return value

    meta = {}
    for key, value in scenes.meta.items():
        if key == "lane_scene_idx":
            meta[key] = torch.as_tensor(l_idx[l_keep] - lo)
        else:
            meta[key] = _by_leading_dim(key, value)

    adv = scenes.adv_local_idx
    return GeneratedScenes(
        agent_states=_take(scenes.agent_states, a_keep),
        agent_types=_take(scenes.agent_types, a_keep),
        agent_scene_idx=torch.as_tensor(a_idx[a_keep] - lo),
        lane_polylines=_take(scenes.lane_polylines, l_keep),
        num_scenes=hi - lo,
        adv_local_idx=None if adv is None else _cpu(adv)[lo:hi],
        meta=meta,
    )
