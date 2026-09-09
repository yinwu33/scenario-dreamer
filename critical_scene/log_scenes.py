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
    route from spawn to goal. Here it is decoded from the stored ground-truth
    connectivity; the DDPO path builds the same structure from the
    autoencoder's predicted connection types (``sim.scenes.batched_lane_graphs``).
"""

from __future__ import annotations

import pickle
from glob import glob
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from sim.scenes import GeneratedScenes, lane_graph_edges
from sim.schema import VEHICLE_TYPE_ID
from utils.goal_runtime import prepare_scene


def list_scene_files(preprocess_dir: str | Path, split: str) -> list[str]:
    """Preprocessed scenario files for one split, in the dataset's own order.

    Same discovery + ordering as ``datasets/waymo/dataset_ae_goal_waymo.py``, so
    an index here means the same scene it means there.
    """
    return sorted(glob(str(Path(preprocess_dir) / split / "*.pkl")))


def closest_agent_adv_idx(agent_states, agent_types, agent_scene_idx, num_scenes):
    """Per-scene local index of the agent designated the adversary in a LOG scene.

    The "Log (closest agent as adversary)" row's designation. A logged scene has no
    generated adversary, so every adversary-scoped metric (ego-vs-adversary
    collision, ego fault, ego min TTC) is undefined until one agent is named; this
    names the ego's nearest neighbour, VEHICLES FIRST.

    Vehicles first because the adversary is what the adv planner drives and what the
    generated rows put there (``adv_cond_target.type: vehicle``); a pedestrian is not
    a comparable object. But a scene whose only neighbours are pedestrians or
    cyclists still gets one -- falling back to the nearest agent of any type keeps
    the row's denominator the whole scene set rather than silently dropping the
    scenes with no car near the ego.

    Distance is spawn-to-spawn (columns 0,1 of the [N, 9] state), a property of the
    initialisation and not of any rollout, so the designation is planner-independent
    and one artifact serves every SUT.

    Returns ``[num_scenes]`` int64, ``-1`` where the scene has no non-ego agent at
    all. Local index 0 is the ego: ``_build_scenes`` slices with
    ``agent_scene_idx == s``, so a scene's local order is its payload order.

    Shared on purpose. ``eval_rollout.log_payload`` and
    ``scripts/make_closest_adv.py`` each had their own copy and they disagreed:
    one took the nearest agent of any type, the other the nearest vehicle and -1
    when there was none, which differed on 91 of 1000 val scenes (66 a different
    agent, 25 no vehicle at all). Two definitions of one table row is one too many.
    """
    states = np.asarray(agent_states, dtype=np.float32)
    types = np.asarray(agent_types, dtype=np.int64)
    scene_idx = np.asarray(agent_scene_idx, dtype=np.int64)

    adv = np.full(int(num_scenes), -1, dtype=np.int64)
    for s in range(int(num_scenes)):
        rows = np.flatnonzero(scene_idx == s)
        if len(rows) < 2:
            continue
        dist = np.linalg.norm(states[rows[1:], :2] - states[rows[0], :2], axis=-1)
        is_vehicle = types[rows[1:]] == VEHICLE_TYPE_ID
        if is_vehicle.any():
            dist = np.where(is_vehicle, dist, np.inf)
        adv[s] = int(np.argmin(dist)) + 1
    return adv


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
        lane_graph.append(
            lane_graph_edges(
                record["edge_index_lane_to_lane"], record["road_connection_types"]
            )
        )
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
