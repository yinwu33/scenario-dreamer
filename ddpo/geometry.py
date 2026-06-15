"""Vectorised 2D collision / off-road geometry.

Faithful numpy port of the oriented-box separating-axis test
(`check_aabb_collision`) and the road-edge crossing test from ``drive.h``, used to
read per-ego metrics from collected rollout trajectories without modifying the C
binding. Kept identical in spirit to the simulator so the reward matches what the
planner actually experiences.
"""

from __future__ import annotations

import numpy as np


def _corners(x, y, heading, length, width):
    """Return [M, 4, 2] oriented-box corners (matches drive.h corner order).

    Inputs are 1-D arrays of length M (one box each).
    """
    x, y, heading, length, width = (np.atleast_1d(np.asarray(v, dtype=np.float64))
                                    for v in (x, y, heading, length, width))
    c, s = np.cos(heading), np.sin(heading)
    hl, hw = length / 2.0, width / 2.0
    # corner sign pattern: (+l,+w), (+l,-w), (-l,-w), (-l,+w)
    sl = np.array([1.0, 1.0, -1.0, -1.0])
    sw = np.array([1.0, -1.0, -1.0, 1.0])
    ox = hl[:, None] * sl          # [M,4]
    oy = hw[:, None] * sw          # [M,4]
    cx = x[:, None] + ox * c[:, None] - oy * s[:, None]
    cy = y[:, None] + ox * s[:, None] + oy * c[:, None]
    return np.stack([cx, cy], axis=-1)  # [M,4,2]


def _sat_overlap(box_a, boxes_b):
    """SAT test of one box [4,2] against many boxes [M,4,2] -> bool [M]."""
    M = boxes_b.shape[0]
    if M == 0:
        return np.zeros(0, dtype=bool)
    overlap = np.ones(M, dtype=bool)
    a = box_a[None]  # [1,4,2]
    for box, idx in ((a, 0), (boxes_b, 1)):
        # two perpendicular axes from this box's first edge pair
        edge_x = box[:, 1, 0] - box[:, 0, 0]
        edge_y = box[:, 1, 1] - box[:, 0, 1]
        for ax, ay in ((edge_x, edge_y), (-edge_y, edge_x)):
            norm = np.sqrt(ax * ax + ay * ay) + 1e-9
            ax_n, ay_n = ax / norm, ay / norm
            pa = a[..., 0] * ax_n[:, None] + a[..., 1] * ay_n[:, None]   # [1,4]
            pb = boxes_b[..., 0] * ax_n[:, None] + boxes_b[..., 1] * ay_n[:, None]  # [M,4]
            a_min, a_max = pa.min(1), pa.max(1)
            b_min, b_max = pb.min(1), pb.max(1)
            sep = (a_max < b_min) | (b_max < a_min)
            overlap &= ~sep
    return overlap


def ego_collides(ego, others) -> bool:
    """ego/others: dicts of arrays (x,y,heading,length,width). Returns any-overlap."""
    if np.atleast_1d(others["x"]).shape[0] == 0:
        return False
    ego_box = _corners(ego["x"], ego["y"], ego["heading"], ego["length"], ego["width"])[0]
    other_boxes = _corners(others["x"], others["y"], others["heading"], others["length"], others["width"])
    return bool(_sat_overlap(ego_box, other_boxes).any())


def ego_offroad(ego, road_edges, dist_threshold: float = 0.0) -> bool:
    """Approximate: ego off-road if its box centre crosses any road-edge segment.

    road_edges: list of [P,2] polylines. Best-effort; tune against drive.h's
    OFFROAD_ROAD_EDGE semantics once running.
    """
    ego_box = _corners(np.array([ego["x"]]), np.array([ego["y"]]), np.array([ego["heading"]]),
                       np.array([ego["length"]]), np.array([ego["width"]]))[0]
    for poly in road_edges:
        poly = np.asarray(poly)
        if poly.shape[0] < 2:
            continue
        for k in range(4):
            p1, p2 = ego_box[k], ego_box[(k + 1) % 4]
            for j in range(poly.shape[0] - 1):
                if _segments_cross(p1, p2, poly[j], poly[j + 1]):
                    return True
    return False


def _segments_cross(p1, p2, q1, q2) -> bool:
    d1 = p2 - p1
    d2 = q2 - q1
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return False
    d3 = p1 - q1
    s = (d1[0] * d3[1] - d1[1] * d3[0]) / cross
    t = (d2[0] * d3[1] - d2[1] * d3[0]) / cross
    return (0 <= s <= 1) and (0 <= t <= 1)
