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


def _poly_signed_area(poly) -> float:
    """Shoelace signed area of an ordered polygon (CCW positive)."""
    n = len(poly)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def _clip_polygon(subject, clip):
    """Sutherland-Hodgman clip of ``subject`` by the convex ``clip`` polygon.

    Both are sequences of (x, y) vertices in any consistent winding. Returns the
    intersection polygon as a list of (x, y) tuples (empty if disjoint).
    """
    clip = [tuple(p) for p in clip]
    sign = 1.0 if _poly_signed_area(clip) >= 0.0 else -1.0

    def inside(p, a, b):
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        return cr * sign >= 0.0

    def intersect(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output = [tuple(p) for p in subject]
    K = len(clip)
    for i in range(K):
        if not output:
            break
        a, b = clip[i], clip[(i + 1) % K]
        inp, output = output, []
        s = inp[-1]
        for e in inp:
            if inside(e, a, b):
                if not inside(s, a, b):
                    output.append(intersect(s, e, a, b))
                output.append(e)
            elif inside(s, a, b):
                output.append(intersect(s, e, a, b))
            s = e
    return output


def _obb_iou(box_a, boxes_b):
    """IoU of one oriented box [4,2] against many [M,4,2] -> float [M] in [0,1].

    Continuous companion to ``_sat_overlap``: exact convex-polygon intersection
    area over (area_a + area_b - inter). 0 when separated or merely touching;
    ramps toward 1 with deeper interpenetration. Used for the soft init-overlap
    penalty (a graded signal the flat SAT boolean cannot give).
    """
    M = boxes_b.shape[0]
    out = np.zeros(M, dtype=np.float64)
    if M == 0:
        return out
    area_a = abs(_poly_signed_area(box_a))
    for i in range(M):
        inter = _clip_polygon(box_a, boxes_b[i])
        if len(inter) < 3:
            continue
        inter_area = abs(_poly_signed_area(inter))
        if inter_area <= 0.0:
            continue
        area_b = abs(_poly_signed_area(boxes_b[i]))
        denom = area_a + area_b - inter_area
        if denom > 1e-9:
            out[i] = inter_area / denom
    return out


def _obb_overlap_frac(box_a, boxes_b):
    """Fraction of box_a's area overlapped by each of many [M,4,2] -> float [M] in [0,1].

    Like ``_obb_iou`` but normalised by box_a's OWN area instead of the union, so it
    measures how much of box_a (the adversary) is interpenetrating, independent of
    boxes_b's size: a half-buried adversary reads 0.5 whether the neighbour is a car
    or a bus (IoU would dilute that toward 0 for a large neighbour). 0 when separated
    or merely touching; 1 when box_a is fully contained.
    """
    M = boxes_b.shape[0]
    out = np.zeros(M, dtype=np.float64)
    if M == 0:
        return out
    area_a = abs(_poly_signed_area(box_a))
    if area_a <= 1e-9:
        return out
    for i in range(M):
        inter = _clip_polygon(box_a, boxes_b[i])
        if len(inter) < 3:
            continue
        inter_area = abs(_poly_signed_area(inter))
        if inter_area > 0.0:
            out[i] = inter_area / area_a
    return out


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
