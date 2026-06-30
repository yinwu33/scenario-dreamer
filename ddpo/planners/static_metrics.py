"""Init-state (non-rollout) validity metrics for planners that don't use hooks.

The numpy ``SimScene`` planners compute goal-off-lane / parking / generated-agent
parking fractions through rollout hooks. Backends that own their own loop (e.g.
``puffer_drive``) cannot run those hooks, so they call ``add_static_metrics``
to fill the same metric keys directly from the generated init states + lanes.

Mirrors ``RewardHookGoalOfflane`` / ``RewardHookParkingMismatch`` /
``RewardHookGenAgentParking``.
"""

from __future__ import annotations

import numpy as np
import torch

from ..interfaces import GeneratedScenes
from ..pufferdrive_sim import MIN_DISTANCE_TO_GOAL, TYPE_VEHICLE, SimConfig
from .type_utils import to_puffer_agent_types


def _to_numpy(value, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def _lane_distance(points: np.ndarray, lane_polylines: np.ndarray) -> np.ndarray:
    """Min distance from points [M, 2] to lane centerline segments."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float32))
    starts, ends = [], []
    for poly in np.asarray(lane_polylines, dtype=np.float32):
        valid = np.isfinite(poly).all(axis=1)
        p = poly[valid]
        if p.shape[0] >= 2:
            starts.append(p[:-1])
            ends.append(p[1:])
    if not starts:
        return np.full(points.shape[0], np.inf, dtype=np.float32)
    a = np.concatenate(starts, axis=0)
    b = np.concatenate(ends, axis=0)
    ab = b - a
    denom = np.maximum((ab * ab).sum(-1), 1e-9)
    ap = points[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(-1) / denom[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    d = points[:, None, :] - proj
    return np.sqrt((d * d).sum(-1)).min(axis=1).astype(np.float32)


def add_static_metrics(
    scenes: GeneratedScenes,
    metrics: dict,
    *,
    sim_cfg: SimConfig,
    goal_offlane_threshold: float,
    goal_onroad_threshold: float,
) -> dict:
    """Fill generated-agent lane / parking-mismatch / parking fractions.

    Goal-off-lane and lane-distance metrics are computed only over DDPO-generated
    non-ego vehicles, mirroring ``RewardHookGoalOfflane``. Adds the keys in place and
    returns ``metrics``. ``ego_adv_min_dist`` needs per-step rollout positions
    this static path does not have; it is left at +inf (zero shaping bonus)
    unless the caller already populated it.
    """
    states = _to_numpy(scenes.agent_states, np.float32)
    types = _to_numpy(scenes.agent_types, np.int64)
    agent_scene_idx = _to_numpy(scenes.agent_scene_idx, np.int64)
    lanes = _to_numpy(scenes.lane_polylines, np.float32)
    lane_scene_idx = _to_numpy(scenes.meta["lane_scene_idx"], np.int64)

    goal_offlane = np.zeros(scenes.num_scenes, dtype=np.float32)
    goal_lane_dist = np.zeros(scenes.num_scenes, dtype=np.float32)
    spawn_lane_dist = np.zeros(scenes.num_scenes, dtype=np.float32)
    parking_mismatch = np.zeros(scenes.num_scenes, dtype=np.float32)
    gen_agent_parking = np.zeros(scenes.num_scenes, dtype=np.float32)
    gt_parking = scenes.meta.get("gt_parking_mask")
    if gt_parking is not None:
        gt_parking = _to_numpy(gt_parking, bool)
    gen_agent_mask = scenes.meta.get("gen_agent_mask")
    if gen_agent_mask is not None:
        gen_agent_mask = _to_numpy(gen_agent_mask, bool)

    ptype = to_puffer_agent_types(types)
    for s in range(scenes.num_scenes):
        a_sel = agent_scene_idx == s
        s_states = states[a_sel]
        if s_states.shape[0] == 0:
            continue

        spawn = s_states[:, :2]
        goal = s_states[:, 7:9]
        gen_dist = np.hypot(goal[:, 0] - spawn[:, 0], goal[:, 1] - spawn[:, 1])

        if gt_parking is not None:
            gt_p = gt_parking[a_sel]
            if len(gt_p):
                parking_mismatch[s] = float(((gen_dist < MIN_DISTANCE_TO_GOAL) != gt_p).mean())

        gen_agent_local = np.zeros(0, dtype=np.int64)
        if gen_agent_mask is not None:
            gen_agent_s = gen_agent_mask[a_sel]
            gen_agent_local = np.nonzero(gen_agent_s)[0]
            gen_agent_local = gen_agent_local[gen_agent_local > 0]  # drop ego
            if len(gen_agent_local):
                gen_agent_parking[s] = float(
                    (gen_dist[gen_agent_local] < MIN_DISTANCE_TO_GOAL).mean()
                )

        gen_agent_local = gen_agent_local[: int(sim_cfg.max_controlled_agents)]
        if len(gen_agent_local) == 0:
            continue

        global_idx = np.nonzero(a_sel)[0][gen_agent_local]
        eligible = ptype[global_idx] == TYPE_VEHICLE
        if not eligible.any():
            continue

        # Mirror RewardHookGoalOfflane: only DDPO-generated non-ego vehicles are
        # scored; ego/fixed GT agents are excluded from the lane constraint.
        s_lanes = lanes[lane_scene_idx == s]
        spawn_d = _lane_distance(spawn[gen_agent_local], s_lanes)
        goal_d = _lane_distance(goal[gen_agent_local], s_lanes)
        offlane = eligible & (
            (np.isfinite(spawn_d) & (spawn_d > goal_onroad_threshold))
            | (np.isfinite(goal_d) & (goal_d > goal_offlane_threshold))
        )
        goal_offlane[s] = float(offlane.sum() / eligible.sum())
        gd = goal_d[eligible & np.isfinite(goal_d)]
        sd = spawn_d[eligible & np.isfinite(spawn_d)]
        if gd.size:
            goal_lane_dist[s] = float(gd.max())
        if sd.size:
            spawn_lane_dist[s] = float(sd.max())

    metrics["goal_offlane_frac"] = goal_offlane
    metrics["goal_lane_dist"] = goal_lane_dist
    metrics["spawn_lane_dist"] = spawn_lane_dist
    metrics["parking_mismatch_frac"] = parking_mismatch
    metrics["gen_agent_is_parked"] = gen_agent_parking
    metrics.setdefault(
        "ego_adv_min_dist", np.full(scenes.num_scenes, np.inf, dtype=np.float32)
    )
    return metrics
