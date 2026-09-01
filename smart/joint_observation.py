"""Per-scene tensors for the joint net, shared by training and rollout.

Same contract as ``smart.observation``: written once so a scene looks identical
whether it came off a record or out of a live ``SimScene``. What changes is the
shape. The per-agent design produced one flat row per agent; this produces one
SCENE -- agents, the map, and the pairwise geometry between them -- because the
model attends across agents and cannot be fed them one at a time.

Every relative quantity is expressed in the QUERY agent's frame, which is what
makes the network equivariant: rotating a scene rotates nothing the model sees.
"""

from __future__ import annotations

import numpy as np

from sim.world import MAX_ROAD_SEGMENT_LENGTH, MAX_SPEED, MAX_VEH_LEN, MAX_VEH_WIDTH

from .joint_net import (
    HISTORY_FEATURES,
    HISTORY_STEPS,
    MAP_FEATURES,
    MAX_AGENTS,
    MAX_MAP_TOKENS,
    REL_AGENT_FEATURES,
    REL_MAP_FEATURES,
    SELF_FEATURES,
)

POS_SCALE = 0.02
POSE_SLOTS = HISTORY_STEPS + 1


def select_map(num_segments: int) -> np.ndarray:
    """Which segments to keep. Uniform stride over the LANE-MAJOR segment list,
    so the whole map is represented rather than a disc around anybody -- possible
    only because the map is encoded once per scene. Striding an intrinsic
    ordering keeps the choice independent of the world frame."""
    if num_segments <= MAX_MAP_TOKENS:
        return np.arange(num_segments)
    return np.linspace(0, num_segments - 1, MAX_MAP_TOKENS).astype(np.int64)


def build_scene(pose, self_feat, poses, first_valid_delta, seg_mid, seg_dir,
                seg_half_len) -> dict:
    """One scene as padded arrays.

    ``pose`` [A, 3] global (x, y, heading); ``self_feat`` [A, SELF_FEATURES];
    ``poses`` [A, POSE_SLOTS, 3] the rolling history; ``seg_*`` the scene's road
    segments. Agents beyond ``MAX_AGENTS`` and segments beyond ``MAX_MAP_TOKENS``
    are dropped; the rest is zero padded and masked.
    """
    a = min(len(pose), MAX_AGENTS)
    pose, self_feat, poses = pose[:a], self_feat[:a], poses[:a]
    ch, sh = np.cos(pose[:, 2]), np.sin(pose[:, 2])

    out = {
        "agent_self": np.zeros((MAX_AGENTS, SELF_FEATURES), np.float32),
        "agent_hist": np.zeros((MAX_AGENTS, HISTORY_STEPS, HISTORY_FEATURES), np.float32),
        "agent_valid": np.zeros(MAX_AGENTS, bool),
        "map_feat": np.zeros((MAX_MAP_TOKENS, MAP_FEATURES), np.float32),
        "rel_agent": np.zeros((MAX_AGENTS, MAX_AGENTS, REL_AGENT_FEATURES), np.float32),
        "rel_map": np.zeros((MAX_AGENTS, MAX_MAP_TOKENS, REL_MAP_FEATURES), np.float32),
        "mask_agent": np.zeros((MAX_AGENTS, MAX_AGENTS), bool),
        "mask_map": np.zeros((MAX_AGENTS, MAX_MAP_TOKENS), bool),
    }
    out["agent_self"][:a] = self_feat
    out["agent_valid"][:a] = True

    # ---- history, in each agent's own frame -------------------------------
    if first_valid_delta < HISTORY_STEPS:
        d = np.diff(poses[:, :, :2], axis=1)
        dyaw = np.diff(poses[:, :, 2], axis=1)
        hist = np.zeros((a, HISTORY_STEPS, HISTORY_FEATURES), np.float32)
        hist[:, :, 0] = (d[:, :, 0] * ch[:, None] + d[:, :, 1] * sh[:, None]) * POS_SCALE
        hist[:, :, 1] = (-d[:, :, 0] * sh[:, None] + d[:, :, 1] * ch[:, None]) * POS_SCALE
        hist[:, :, 2] = np.cos(dyaw)
        hist[:, :, 3] = np.sin(dyaw)
        hist[:, :first_valid_delta] = 0.0
        hist[:, first_valid_delta:, 4] = 1.0
        out["agent_hist"][:a] = hist

    # ---- agent -> agent, in the query's frame -----------------------------
    dx = pose[None, :, 0] - pose[:, None, 0]
    dy = pose[None, :, 1] - pose[:, None, 1]
    rel = np.zeros((a, a, REL_AGENT_FEATURES), np.float32)
    rel[:, :, 0] = (dx * ch[:, None] + dy * sh[:, None]) * POS_SCALE
    rel[:, :, 1] = (-dx * sh[:, None] + dy * ch[:, None]) * POS_SCALE
    dyaw = pose[None, :, 2] - pose[:, None, 2]
    rel[:, :, 2], rel[:, :, 3] = np.cos(dyaw), np.sin(dyaw)
    rel[:, :, 4] = np.hypot(dx, dy) * POS_SCALE
    rel[:, :, 5] = 1.0
    out["rel_agent"][:a, :a] = rel
    m = np.ones((a, a), bool)
    np.fill_diagonal(m, False)          # an agent is not its own neighbour
    out["mask_agent"][:a, :a] = m

    # ---- agent -> map, in the query's frame -------------------------------
    keep = select_map(len(seg_mid))
    n = len(keep)
    if n:
        mid, dirn, half = seg_mid[keep], seg_dir[keep], seg_half_len[keep]
        out["map_feat"][:n, 0] = half / MAX_ROAD_SEGMENT_LENGTH
        out["map_feat"][:n, 1] = 1.0
        rx = mid[None, :, 0] - pose[:, None, 0]
        ry = mid[None, :, 1] - pose[:, None, 1]
        rm = np.zeros((a, n, REL_MAP_FEATURES), np.float32)
        rm[:, :, 0] = (rx * ch[:, None] + ry * sh[:, None]) * POS_SCALE
        rm[:, :, 1] = (-rx * sh[:, None] + ry * ch[:, None]) * POS_SCALE
        rm[:, :, 2] = dirn[None, :, 0] * ch[:, None] + dirn[None, :, 1] * sh[:, None]
        rm[:, :, 3] = -dirn[None, :, 0] * sh[:, None] + dirn[None, :, 1] * ch[:, None]
        rm[:, :, 4] = half[None, :] / MAX_ROAD_SEGMENT_LENGTH
        rm[:, :, 5] = 1.0
        out["rel_map"][:a, :n] = rm
        out["mask_map"][:a, :n] = True
    return out


def self_features(speed, width, length, ptype, collision, removed) -> np.ndarray:
    """[A, SELF_FEATURES] -- the same layout ``smart.observation`` uses."""
    a = len(speed)
    f = np.zeros((a, SELF_FEATURES), np.float32)
    f[:, 0] = speed / MAX_SPEED
    f[:, 1] = width / MAX_VEH_WIDTH
    f[:, 2] = length / MAX_VEH_LEN
    f[np.arange(a), 3 + ptype.astype(np.int64) - 1] = 1.0
    f[:, 6] = collision
    f[:, 7] = removed
    return f
