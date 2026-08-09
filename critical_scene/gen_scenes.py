"""Scene sources fed by a generative model's own sample cache.

The "Scene Initialization = generated" rows of the SUT x scene-initialization
table. A cache is whatever ``utils.data_helpers.convert_batch_to_scenarios``
wrote with ``cache_samples=True`` -- e.g. the 10k unconditional ldm_adv-base
samples of ``eval.py --config-name config_ldm_adv_base ldm_adv.eval.mode=init_scene``
-- so this loader is per *format*, not per model, and a new checkpoint's cache
is a new ``benchmark.gen_dirs`` entry rather than new code.

The cache is already sim-ready, which is why nothing here resembles the log
loader's preparation step:

  * ``agent_states`` is the 9-column ``GeneratedScenes`` layout INCLUDING the
    goal columns -- the goal autoencoder decodes goals as part of the state, so
    there is no ``utils.goal_runtime.prepare_scene`` call (and no way to make
    one: the cache carries no trajectories to derive goals from). The agent-set
    filters that call applies were likewise already applied to the data the
    model was trained on, so its samples inherit them distributionally.
  * the lane graph is reconstructed rather than read: see
    ``dense_lane_edge_index``.

Two properties of these scenes that the table must not silently absorb, both
measured on the ldm_adv-base 10k cache against 400 val log scenes:

  * the maps are GENERATED, so lane connectivity is only approximately
    consistent with lane geometry -- the gap between an upstream lane's end and
    its ``succ`` lane's start is a median 0.25 m (p90 1.0 m) where the log's is
    exactly 0. The topological route search absorbs that, but expect the
    ``route_unavailable_rate`` diagnostic to sit above the log row's for reasons
    that have nothing to do with the planner being benchmarked.
  * generated egos have nearer goals (median spawn->goal 15.3 m vs the log's
    29.5 m), so success rates are NOT comparable across scene sources at face
    value; read them next to ``ego_goal_dist_mean``.

Agent-set statistics are, by contrast, close enough to compare directly:
10.0 vs 9.2 agents/scene, and 0.5% vs 0.7% of egos spawning farther than the
2.75 m off-road proxy from every centerline.
"""

from __future__ import annotations

import pickle
from glob import glob
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from critical_scene.log_scenes import lane_graph_edges
from sim.scenes import GeneratedScenes

# ``GeneratedScenes.agent_states`` width. A cache written by a model without the
# goal decoder (the baseline LDM) stores 7 columns and is rejected on sight.
STATE_DIM = 9


def list_gen_scene_files(sample_dir: str | Path) -> list[str]:
    """Every cached sample, in name order.

    A cache is FLAT -- one directory of ``<i>_<batch_idx>.pkl``, with no
    train/val split, because the samples are drawn from the layout prior rather
    than from a split of real scenes. So an index here indexes the cache and
    means nothing outside it, unlike a log scene index.
    """
    return sorted(glob(str(Path(sample_dir) / "*.pkl")))


def dense_lane_edge_index(num_lanes: int) -> np.ndarray:
    """The ``[2, L*L]`` edge index ``road_connection_types`` rows are aligned to.

    ``convert_batch_to_scenarios`` caches the connection types but not the edge
    index, so it has to be rebuilt. The generator emits one type (including
    'none') for every ORDERED lane pair, laid out row-major with the source
    index slowest -- verified identical to the ``edge_index_lane_to_lane`` the
    preprocessing stores alongside the same array on real scenes, and the row
    count is checked against it here.
    """
    idx = np.arange(int(num_lanes), dtype=np.int64)
    return np.stack([np.repeat(idx, len(idx)), np.tile(idx, len(idx))])


def _scene_lane_graph(record: dict[str, Any], num_lanes: int) -> dict[str, np.ndarray]:
    types = np.asarray(record["road_connection_types"])
    if types.shape[0] != num_lanes * num_lanes:
        raise ValueError(
            "gen_scenes: road_connection_types must hold one row per ordered lane "
            f"pair ({num_lanes}^2 = {num_lanes * num_lanes}), got {types.shape[0]} -- "
            "the cache was not written by convert_batch_to_scenarios"
        )
    return lane_graph_edges(dense_lane_edge_index(num_lanes), types)


def load_gen_scenes(
    sample_dir: str | Path,
    indices: Sequence[int],
    *,
    files: Sequence[str] | None = None,
) -> tuple[GeneratedScenes, list[int]]:
    """Batch the cached samples at ``indices`` into one ``GeneratedScenes``.

    Returns ``(scenes, kept_indices)``: a sample with no lanes or no agents
    cannot be rolled out and is skipped, so the caller must label per-scene
    results with ``kept_indices`` rather than ``indices``.

    The adversary is always the LAST agent of each scene (``convert_batch_to_scenarios``
    appends the ``adv`` node after the base set); the cache records no marker, so
    that ordering is the only handle and it holds only for caches written by a
    model with an adversary branch.
    """
    paths = list(files) if files is not None else list_gen_scene_files(sample_dir)

    states, types, agent_scene, lanes, lane_scene, lane_graph = [], [], [], [], [], []
    kept: list[int] = []
    for idx in indices:
        with open(paths[int(idx)], "rb") as f:
            record = pickle.load(f)

        agent_states = np.asarray(record["agent_states"], dtype=np.float32)
        road_points = np.asarray(record["road_points"], dtype=np.float32)
        if len(road_points) == 0 or len(agent_states) == 0:
            continue
        if agent_states.shape[1] != STATE_DIM:
            raise ValueError(
                f"gen_scenes: agent_states must have {STATE_DIM} columns "
                "[x, y, speed, cos, sin, length, width, goal_x, goal_y], got "
                f"{agent_states.shape[1]} in {paths[int(idx)]} -- a cache without "
                "goal columns cannot be rolled out"
            )

        s = len(kept)
        states.append(agent_states)
        # Cached types are one-hot; GeneratedScenes uses model-side class ids.
        types.append(np.asarray(record["agent_types"]).argmax(axis=-1).astype(np.int64))
        agent_scene.append(np.full(len(agent_states), s, dtype=np.int64))
        lanes.append(road_points)
        lane_scene.append(np.full(len(road_points), s, dtype=np.int64))
        lane_graph.append(_scene_lane_graph(record, len(road_points)))
        kept.append(int(idx))

    if not kept:
        raise ValueError("load_gen_scenes: every requested sample was empty or lane-less")

    num_scenes = len(kept)
    adv_local_idx = torch.tensor([len(st) - 1 for st in states], dtype=torch.long)
    # A single-agent scene is ego-only: local index 0 is the ego, never an
    # adversary (RolloutRunner reads > 0 as "has adversary" anyway).
    adv_local_idx[adv_local_idx == 0] = -1

    scenes = GeneratedScenes(
        agent_states=torch.from_numpy(np.concatenate(states, axis=0)),
        agent_types=torch.from_numpy(np.concatenate(types, axis=0)),
        agent_scene_idx=torch.from_numpy(np.concatenate(agent_scene, axis=0)),
        lane_polylines=np.concatenate(lanes, axis=0),
        num_scenes=num_scenes,
        adv_local_idx=adv_local_idx,
        meta={
            "lane_scene_idx": torch.from_numpy(np.concatenate(lane_scene, axis=0)),
            "lane_graph": lane_graph,
        },
    )
    return scenes, kept
