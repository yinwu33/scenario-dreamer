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


def sat_first_contact_time(box_a, boxes_b, rel_vel, dt, horizon):
    """Closed-form first-contact time of a static box vs boxes translating at
    constant relative velocity -- analytic replacement for the per-step SAT sweep.

    ``box_a`` [4,2] is fixed; each of ``boxes_b`` [M,4,2] translates by
    ``rel_vel[m] * t`` (``rel_vel`` [M,2]). Returns [M] float: the smallest grid
    time ``k*dt`` (k = 0, 1, ... up to ``ceil(horizon/dt)``) at which box_a and
    the translated box_b overlap under the same 4-axis SAT convention as
    ``_sat_overlap``, or ``inf`` if they never overlap within the horizon.

    Same axes/normalisation/inclusive-touch semantics as the sweep, so the result
    matches ``for step: _sat_overlap(box_a, boxes_b + rel_vel*step*dt)`` to within
    grid quantisation. On each SAT axis n the two projection intervals are static
    (box_a) and linearly shifting (box_b, by ``(rel_vel.n) t``), so "projections
    overlap" holds on a time interval; the boxes overlap on the intersection of
    the 4 per-axis intervals, whose clamped left endpoint is first contact.
    """
    boxes_b = np.asarray(boxes_b, dtype=np.float64)
    M = boxes_b.shape[0]
    if M == 0:
        return np.zeros(0, dtype=np.float64)
    box_a = np.asarray(box_a, dtype=np.float64)
    rel_vel = np.asarray(rel_vel, dtype=np.float64).reshape(M, 2)

    def _first_edge_axes(box):
        # two perpendicular axes from the box's first edge (corner0 -> corner1),
        # matching _sat_overlap's axis choice + (+1e-9) normalisation.
        ex = box[..., 1, 0] - box[..., 0, 0]
        ey = box[..., 1, 1] - box[..., 0, 1]
        a1 = np.stack([ex, ey], axis=-1)
        a2 = np.stack([-ey, ex], axis=-1)
        out = np.stack([a1, a2], axis=-2)  # [...,2,2]
        norm = np.sqrt((out * out).sum(-1)) + 1e-9
        return out / norm[..., None]

    ea = np.broadcast_to(_first_edge_axes(box_a), (M, 2, 2))  # ego axes [M,2,2]
    eb = _first_edge_axes(boxes_b)                            # per-b axes [M,2,2]
    axes = np.concatenate([ea, eb], axis=1)                   # [M,4,2]

    pa = np.einsum("mad,cd->mac", axes, box_a)                # [M,4,4] ego corners
    a_min, a_max = pa.min(-1), pa.max(-1)                     # [M,4]
    pb0 = np.einsum("mad,mcd->mac", axes, boxes_b)            # [M,4,4] at t=0
    b_min0, b_max0 = pb0.min(-1), pb0.max(-1)                 # [M,4]
    s = np.einsum("mad,md->ma", axes, rel_vel)                # [M,4] shift rate

    # projections overlap (inclusive) iff  s*t in [lo, hi]  (hi >= lo always).
    lo = a_min - b_max0
    hi = a_max - b_min0
    enter = np.full((M, 4), -np.inf)
    leave = np.full((M, 4), np.inf)
    pos, neg = s > 0.0, s < 0.0
    enter[pos], leave[pos] = lo[pos] / s[pos], hi[pos] / s[pos]
    enter[neg], leave[neg] = hi[neg] / s[neg], lo[neg] / s[neg]
    zero = ~(pos | neg)
    # s == 0: axis is time-independent -> overlaps for all t iff already overlapping.
    zero_never = zero & ~((lo <= 0.0) & (hi >= 0.0))
    # (zero & overlapping) keeps the default [-inf, inf], i.e. no time constraint.

    t_enter = np.maximum(enter.max(1), 0.0)   # first time all axes could overlap
    t_leave = leave.min(1)
    never = zero_never.any(1) | (t_enter > t_leave)

    k_max = int(np.ceil(horizon / dt))
    k0 = np.ceil(t_enter / dt - 1e-9)          # first grid index >= t_enter
    t_grid = k0 * dt
    ok = (~never) & (t_grid <= t_leave + 1e-9) & (k0 <= k_max) & (k0 >= 0)
    return np.where(ok, t_grid, np.inf)


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


def point_line_offset(pt, a, b) -> float:
    """Perpendicular distance from ``pt`` to the INFINITE line through a -> b.

    The lateral half of a lane-corridor test: a car following 15 m behind the
    ego is 15 m from the ego's spawn->goal *segment* but ~0 m from its line, and
    it is the latter that says "same lane". Falls back to the point distance
    when the line is degenerate (a == b).
    """
    pt, a, b = (np.asarray(v, dtype=np.float64) for v in (pt, a, b))
    ab = b - a
    n = float(np.hypot(*ab))
    if n < 1e-6:
        return float(np.hypot(*(pt - a)))
    return float(abs(ab[0] * (pt[1] - a[1]) - ab[1] * (pt[0] - a[0])) / n)


def segment_closest_approach(p0, p1, q0, q1):
    """Closest approach of two 2D segments [p0,p1] and [q0,q1].

    Returns ``(dist, s_p, s_q)``: the minimum distance between the segments and
    the arc length travelled along each one to reach its closest point. Distance
    0 means the segments intersect, so this is a graded generalisation of
    ``_segments_cross`` -- which is what the path-conflict screen needs: a strict
    crossing test answers "do these chords intersect" and misses the geometry a
    rear-end or a same-lane head-on produces (near-collinear chords that never
    formally cross), while the distance degrades smoothly through those cases.

    Standard clamped segment-segment solve (Ericson, Real-Time Collision
    Detection 5.1.9): solve the unconstrained quadratic, clamp each parameter to
    its segment, and re-solve the other against the clamp.
    """
    p0, p1, q0, q1 = (np.asarray(v, dtype=np.float64) for v in (p0, p1, q0, q1))
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a = float(u @ u)          # |u|^2
    c = float(v @ v)          # |v|^2
    b = float(u @ v)
    d = float(u @ w)
    e = float(v @ w)
    den = a * c - b * b
    # Degenerate segments (a parked agent's spawn == goal) collapse to a point.
    if a <= 1e-12 and c <= 1e-12:
        return float(np.hypot(*w)), 0.0, 0.0
    if a <= 1e-12:
        t = min(max(e / c, 0.0), 1.0)
        s = 0.0
    elif c <= 1e-12:
        s = min(max(-d / a, 0.0), 1.0)
        t = 0.0
    elif den <= 1e-12:                      # parallel: pin s, solve t
        s = 0.0
        t = min(max(e / c, 0.0), 1.0)
    else:
        s = min(max((b * e - c * d) / den, 0.0), 1.0)
        t = (b * s + e) / c
        if t < 0.0:
            t, s = 0.0, min(max(-d / a, 0.0), 1.0)
        elif t > 1.0:
            t, s = 1.0, min(max((b - d) / a, 0.0), 1.0)
    diff = w + s * u - t * v
    return float(np.hypot(*diff)), float(s * np.sqrt(a)), float(t * np.sqrt(c))


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
