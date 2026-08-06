"""Scene source ``log``: real Waymo scenes straight out of the preprocessed pickles.

The "Scene Initialization = Log" row of the SUT x scene-initialization table.
Unlike ``critical_scene.ldm_adv_eval._gt_scenes`` -- which reaches the same
ground-truth scenes through an ``LDMAdvConditioningPool``, and therefore drags in
the latent cache, the LDM dataset config and the AE checkpoint -- this loader
reads ``data/advscene_preprocess_waymo/<split>/*.pkl`` directly and emits
physical-unit ``GeneratedScenes``. Nothing here needs a model, and the
normalize/unnormalize round trip is skipped entirely.

Two things the DDPO path does not carry are added:

  * goals, via ``utils.goal_runtime.prepare_scene`` -- goals are NOT stored on
    disk, they are derived at load time as the last in-FOV valid trajectory
    point, and the same call applies the off-road / valid-goal / max-agent
    filters the training datasets use, so this row sees the same agent set;
  * the lane GRAPH (``meta['lane_graph']``), which rule-based planners need to
    route from spawn to goal. The DDPO conditioning path deliberately drops it
    (``ddpo/conditioning.py``), keeping geometry only.
"""

from __future__ import annotations

import pickle
from glob import glob
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from cfgs.config import LANE_CONNECTION_TYPES_WAYMO
from sim.scenes import GeneratedScenes
from utils.goal_runtime import prepare_scene


def list_scene_files(preprocess_dir: str | Path, split: str) -> list[str]:
    """Preprocessed scenario files for one split, in the dataset's own order.

    Same discovery + ordering as ``datasets/waymo/dataset_ae_goal_waymo.py``, so
    an index here means the same scene it means there.
    """
    return sorted(glob(str(Path(preprocess_dir) / split / "*.pkl")))


def lane_graph_edges(record: dict[str, Any]) -> dict[str, np.ndarray]:
    """``{"succ": [2, E], "lateral": [2, E]}`` lane connectivity over lane rows.

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
    edge_index = np.asarray(record["edge_index_lane_to_lane"], dtype=np.int64)
    types = np.asarray(record["road_connection_types"]).argmax(axis=-1)
    succ = edge_index[:, types == LANE_CONNECTION_TYPES_WAYMO["succ"]]
    lateral = edge_index[
        :,
        (types == LANE_CONNECTION_TYPES_WAYMO["left"])
        | (types == LANE_CONNECTION_TYPES_WAYMO["right"]),
    ]
    return {"succ": succ[::-1].copy(), "lateral": lateral.copy()}


def load_log_scenes(
    preprocess_dir: str | Path,
    split: str,
    indices: Sequence[int],
    dataset_cfg: Any,
    *,
    files: Sequence[str] | None = None,
) -> tuple[GeneratedScenes, list[int]]:
    """Batch the scenes at ``indices`` into one ``GeneratedScenes``.

    Returns ``(scenes, kept_indices)``: a scene with no lanes or no agents cannot
    be rolled out and is skipped, so the caller must use ``kept_indices`` (not
    ``indices``) when labelling per-scene results.
    """
    paths = list(files) if files is not None else list_scene_files(preprocess_dir, split)

    states, types, agent_scene, lanes, lane_scene, lane_graph = [], [], [], [], [], []
    kept: list[int] = []
    for idx in indices:
        with open(paths[int(idx)], "rb") as f:
            record = pickle.load(f)

        road_points = np.asarray(record["road_points"], dtype=np.float32)
        scene = prepare_scene(record, dataset_cfg)
        agent_states = np.asarray(scene["agent_states"], dtype=np.float32)
        if len(road_points) == 0 or len(agent_states) == 0:
            continue

        s = len(kept)
        states.append(agent_states)
        # GeneratedScenes.agent_types uses model-side class ids (0 veh / 1 ped /
        # 2 cyc); the stored form is a one-hot.
        types.append(np.asarray(scene["agent_types"]).argmax(axis=-1).astype(np.int64))
        agent_scene.append(np.full(len(agent_states), s, dtype=np.int64))
        lanes.append(road_points)
        lane_scene.append(np.full(len(road_points), s, dtype=np.int64))
        lane_graph.append(lane_graph_edges(record))
        kept.append(int(idx))

    if not kept:
        raise ValueError("load_log_scenes: every requested scene was empty or lane-less")

    num_scenes = len(kept)
    scenes = GeneratedScenes(
        agent_states=torch.from_numpy(np.concatenate(states, axis=0)),
        agent_types=torch.from_numpy(np.concatenate(types, axis=0)),
        agent_scene_idx=torch.from_numpy(np.concatenate(agent_scene, axis=0)),
        lane_polylines=np.concatenate(lanes, axis=0),
        num_scenes=num_scenes,
        # A log scene has no generated adversary: the adv role drives nobody and
        # no agent gets the adversary conditioning override.
        adv_local_idx=torch.full((num_scenes,), -1, dtype=torch.long),
        meta={
            "lane_scene_idx": torch.from_numpy(np.concatenate(lane_scene, axis=0)),
            "lane_graph": lane_graph,
        },
    )
    return scenes, kept
