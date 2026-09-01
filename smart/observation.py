"""The observation spec, written ONCE and shared by training and rollout.

An agent-centric row is built from exactly the same code whether it comes from a
logged record (``smart.dataset``) or from a live ``SimScene``
(``smart.planner``). That is deliberate: an observation builder implemented
twice is the standard way a behavior model ends up trained on something subtly
different from what it is served, and the failure is silent -- the model simply
drives worse than it should and nothing reports why.

Layout and sizes live in ``smart.net``; this module only fills the row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.world import (
    GRID_CELL_SIZE,
    MAX_ROAD_SEGMENT_LENGTH,
    MAX_SPEED,
    MAX_VEH_LEN,
    MAX_VEH_WIDTH,
)

from .net import (
    HISTORY_FEATURES,
    HISTORY_OFF,
    HISTORY_STEPS,
    MAX_NEIGHBORS,
    MAX_ROAD_SEGMENTS,
    NEIGHBOR_FEATURES,
    NEIGHBOR_OFF,
    OBS_DIM,
    ROAD_FAR_LANES,
    ROAD_FAR_POINTS,
    ROAD_FEATURES,
    ROAD_NEAR_SEGMENTS,
    ROAD_OFF,
)


def _lane_tokens(lanes, x, y, ch, sh):
    """Points walked along the nearest lanes, end to end within the FOV.

    Range, as opposed to the resolution the nearest-segment half supplies. Lanes
    are ranked by their closest point and then sampled at a fixed stride along
    their own polyline, which is a property of the lane rather than of the world
    grid, so this is rotation and translation invariant.
    """
    feat = np.zeros((ROAD_FAR_LANES * ROAD_FAR_POINTS, ROAD_FEATURES), dtype=np.float32)
    if not len(lanes):
        return feat
    valid = np.isfinite(lanes).all(axis=2)
    d2 = np.where(valid, (lanes[:, :, 0] - x) ** 2 + (lanes[:, :, 1] - y) ** 2, np.inf)
    order = np.argsort(d2.min(axis=1), kind="stable")[:ROAD_FAR_LANES]
    row = 0
    for li in order:
        pts = lanes[li][valid[li]]
        if len(pts) < 2:
            continue
        idx = np.linspace(0, len(pts) - 2, ROAD_FAR_POINTS).astype(np.int64)
        seg_a, seg_b = pts[idx], pts[idx + 1]
        mid = 0.5 * (seg_a + seg_b)
        d = seg_b - seg_a
        n = np.hypot(d[:, 0], d[:, 1])
        dn = d / np.maximum(n, 1e-12)[:, None]
        rx, ry = mid[:, 0] - x, mid[:, 1] - y
        k = len(idx)
        feat[row:row + k, 0] = (rx * ch + ry * sh) * POS_SCALE
        feat[row:row + k, 1] = (-rx * sh + ry * ch) * POS_SCALE
        feat[row:row + k, 2] = 0.5 * n / MAX_ROAD_SEGMENT_LENGTH
        feat[row:row + k, 3] = dn[:, 0] * ch + dn[:, 1] * sh
        feat[row:row + k, 4] = -dn[:, 0] * sh + dn[:, 1] * ch
        feat[row:row + k, 5] = 1.0
        row += k
    return feat


# Same position scaling the PufferDrive observation uses, so the two planners'
# inputs live on comparable numeric ranges.
POS_SCALE = 0.02
# Poses kept per agent: HISTORY_STEPS deltas need one more pose than deltas.
POSE_SLOTS = HISTORY_STEPS + 1


@dataclass
class Frame:
    """Every agent's CURRENT state, plus who is present. All arrays are [A]."""

    x: np.ndarray
    y: np.ndarray
    heading_x: np.ndarray
    heading_y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    length: np.ndarray
    width: np.ndarray
    ptype: np.ndarray          # PufferDrive ids: 1 vehicle, 2 pedestrian, 3 cyclist
    active: np.ndarray         # [A] bool, agents present in the scene this step
    collision: np.ndarray      # [A] bool
    removed: np.ndarray        # [A] bool

    @property
    def slot_order(self) -> np.ndarray:
        return np.nonzero(self.active)[0]


def agent_cell(grid, x: float, y: float) -> int:
    """Grid cell of a position, matching ``SimScene._agent_cell`` exactly."""
    if not grid["grid_ok"]:
        return -1
    # C-style (int) cast truncates toward zero, matching getGridIndex
    gx = int((x - grid["min_x"]) / GRID_CELL_SIZE)
    gy = int((y - grid["min_y"]) / GRID_CELL_SIZE)
    if gx < 0 or gx >= grid["grid_cols"] or gy < 0 or gy >= grid["grid_rows"]:
        return -1
    return gy * grid["grid_cols"] + gx


def build(frame: Frame, grid, ids: np.ndarray, poses: np.ndarray,
          first_valid_delta: int) -> np.ndarray:
    """[len(ids), OBS_DIM] in ``ids`` order, each row in that agent's frame.

    ``poses`` is [A, POSE_SLOTS, 3] of (x, y, heading); ``first_valid_delta`` is
    the first history slot whose two bracketing poses are both real, so an empty
    or partial history is expressed rather than faked.
    """
    obs = np.zeros((len(ids), OBS_DIM), dtype=np.float32)
    slot_order = frame.slot_order

    for k, i in enumerate(ids):
        o = obs[k]
        ch, sh = frame.heading_x[i], frame.heading_y[i]
        speed = float(np.hypot(frame.vx[i], frame.vy[i]))
        v_dot_h = frame.vx[i] * ch + frame.vy[i] * sh

        # ---- self ---------------------------------------------------------
        o[0] = np.copysign(speed, v_dot_h) / MAX_SPEED
        o[1] = frame.width[i] / MAX_VEH_WIDTH
        o[2] = frame.length[i] / MAX_VEH_LEN
        o[3 + int(frame.ptype[i]) - 1] = 1.0
        o[6] = 1.0 if frame.collision[i] else 0.0
        o[7] = 1.0 if frame.removed[i] else 0.0

        # ---- history: per-step motion, rotated into the current frame ------
        if first_valid_delta < HISTORY_STEPS:
            p = poses[i]
            dx = p[1:, 0] - p[:-1, 0]
            dy = p[1:, 1] - p[:-1, 1]
            dyaw = p[1:, 2] - p[:-1, 2]
            hist = np.zeros((HISTORY_STEPS, HISTORY_FEATURES), dtype=np.float32)
            hist[:, 0] = (dx * ch + dy * sh) * POS_SCALE
            hist[:, 1] = (-dx * sh + dy * ch) * POS_SCALE
            hist[:, 2] = np.cos(dyaw)
            hist[:, 3] = np.sin(dyaw)
            hist[:first_valid_delta] = 0.0
            hist[first_valid_delta:, 4] = 1.0
            o[HISTORY_OFF:NEIGHBOR_OFF] = hist.reshape(-1)

        # ---- neighbours: the MAX_NEIGHBORS closest active agents -----------
        cand = slot_order[slot_order != i]
        if len(cand):
            dx = frame.x[cand] - frame.x[i]
            dy = frame.y[cand] - frame.y[i]
            # stable so ties resolve by slot order, which sharding preserves
            order = np.argsort(dx * dx + dy * dy, kind="stable")[:MAX_NEIGHBORS]
            cand, dx, dy = cand[order], dx[order], dy[order]
            m = len(cand)
            rel = np.zeros((m, NEIGHBOR_FEATURES), dtype=np.float32)
            rel[:, 0] = (dx * ch + dy * sh) * POS_SCALE
            rel[:, 1] = (-dx * sh + dy * ch) * POS_SCALE
            rel[:, 2] = frame.heading_x[cand] * ch + frame.heading_y[cand] * sh
            rel[:, 3] = frame.heading_y[cand] * ch - frame.heading_x[cand] * sh
            sp = np.hypot(frame.vx[cand], frame.vy[cand])
            vdh = frame.vx[cand] * frame.heading_x[cand] + frame.vy[cand] * frame.heading_y[cand]
            rel[:, 4] = np.copysign(sp, vdh) / MAX_SPEED
            rel[:, 5] = frame.width[cand] / MAX_VEH_WIDTH
            rel[:, 6] = frame.length[cand] / MAX_VEH_LEN
            rel[:, 7] = 1.0
            o[NEIGHBOR_OFF : NEIGHBOR_OFF + m * NEIGHBOR_FEATURES] = rel.reshape(-1)

        # ---- road: the MAX_ROAD_SEGMENTS closest centreline segments -------
        cell = agent_cell(grid, frame.x[i], frame.y[i])
        segs = grid["cell_cache"].get(cell) if cell >= 0 else None
        if segs is not None and len(segs):
            mid = grid["seg_mid"][segs]
            rx = mid[:, 0] - frame.x[i]
            ry = mid[:, 1] - frame.y[i]
            order = np.argsort(rx * rx + ry * ry, kind="stable")[:ROAD_NEAR_SEGMENTS]
            segs, rx, ry = segs[order], rx[order], ry[order]
            dn = grid["seg_dir"][segs]
            m = len(segs)
            feat = np.zeros((m, ROAD_FEATURES), dtype=np.float32)
            feat[:, 0] = (rx * ch + ry * sh) * POS_SCALE
            feat[:, 1] = (-rx * sh + ry * ch) * POS_SCALE
            feat[:, 2] = grid["seg_half_len"][segs] / MAX_ROAD_SEGMENT_LENGTH
            feat[:, 3] = dn[:, 0] * ch + dn[:, 1] * sh
            feat[:, 4] = -dn[:, 0] * sh + dn[:, 1] * ch
            feat[:, 5] = 1.0
            o[ROAD_OFF : ROAD_OFF + m * ROAD_FEATURES] = feat.reshape(-1)

        # ---- road, far half: the nearest lanes walked end to end ------------
        far_off = ROAD_OFF + ROAD_NEAR_SEGMENTS * ROAD_FEATURES
        o[far_off:] = _lane_tokens(grid["lanes"], frame.x[i], frame.y[i], ch, sh).reshape(-1)
    return obs


def frame_from_sim(sim) -> Frame:
    """A ``Frame`` view of a live ``SimScene`` (no copies of the state arrays)."""
    active = np.zeros(sim.n, dtype=bool)
    active[sim.slot_order] = True
    return Frame(
        x=sim.x, y=sim.y, heading_x=sim.heading_x, heading_y=sim.heading_y,
        vx=sim.vx, vy=sim.vy, length=sim.length, width=sim.width, ptype=sim.ptype,
        active=active, collision=sim.collision_state > 0, removed=sim.removed,
    )


def grid_from_sim(sim) -> dict:
    """The lane grid a ``SimScene`` already built, in the dict shape ``build`` wants."""
    return {
        "seg_mid": sim.seg_mid, "seg_half_len": sim.seg_half_len, "seg_dir": sim.seg_dir,
        "lanes": sim.lane_polylines,
        "cell_cache": sim._cell_cache, "grid_ok": sim._grid_ok,
        "grid_cols": sim.grid_cols, "grid_rows": sim.grid_rows,
        "min_x": sim.min_x, "min_y": sim.min_y,
    }
