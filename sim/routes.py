"""Lane-graph route search for rule-based planners.

The DDPO rollout carries lane *geometry* everywhere (``SimScene.lane_polylines``,
plus the flattened segment grid used for observations) but the lane *graph* only
when the scene source supplies it (``GeneratedScenes.meta['lane_graph']``,
attached to ``SimScene.lane_graph`` by ``RolloutRunner._build_scenes``). A neural
planner does not need the graph -- it reads raw road segments out of its
observation -- but a rule-based planner does: to drive from its spawn to its goal
an IDM agent needs an explicit reference path, and at an intersection the only
thing that distinguishes "go straight" from "turn left" is which successor lane
the route takes.

``build_route`` returns a path that ALWAYS follows lane centerlines. There is no
straight-line fallback: a route that cuts from spawn to goal through open space
is not a route, it is the absence of one, and quietly substituting it makes a
planner look like it is driving badly when really it was never given a path. When
no lane path exists ``build_route`` returns ``None`` and the caller is expected to
report that separately.

Three things matter for coverage, in descending order of how much they cost:

  * **Lateral neighbours are candidates, not edges.** ~12% of agents end up in a
    lane their spawn lane has no successor path to, because they change lanes. A
    lane change cannot be a graph edge here: two parallel lanes are traversed in
    the same direction, so appending one after the other would drive to the end
    of lane A and then restart at the *beginning* of lane B, i.e. backwards.
    Instead the left/right neighbours of the nearest lanes JOIN the candidate
    sets. Since spawn and goal are pinned as the route's true endpoints, a route
    that starts in the adjacent lane is exactly a lane change at the start, and
    one that ends there is a lane change at the end -- which pure pursuit then
    executes as a merge. (A lane change in the *middle* of a long route is not
    representable; it needs per-lane enter/exit arc positions.)
  * **Start and goal lanes are chosen jointly.** Picking each independently
    yields pairs with no forward progress ~3% of the time (the goal projects
    behind the spawn). Every candidate pair is searched and scored.
  * **Trimming is by arc length, not vertex index.** Lane centerlines are only 20
    points, so trimming to the nearest *vertex* collapses ~4% of same-lane routes
    to a bare two-point line. Projection interpolates along the segments.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from utils.lane_graph_helpers import resample_polyline_every

# Radius (m) around a point inside which lanes compete to be its lane. Beyond it
# only the single nearest lane is considered.
SEARCH_RADIUS = 6.0
# Reject a start-lane candidate whose tangent disagrees with the agent heading by
# more than this (radians). 90 deg keeps merges/turn-ins while ruling out the
# oncoming lane of the same road, which is often the geometrically nearest one.
MAX_START_HEADING_DIFF = np.pi / 2
# Metres of lateral distance that one radian of heading disagreement is worth
# when ranking start-lane candidates.
HEADING_TIEBREAK_WEIGHT = 4.0
# How many nearest lanes seed each candidate set (their lateral neighbours are
# added on top).
NUM_LANE_CANDIDATES = 3
# A route must advance at least this far (m) along the centerline, or the lane
# pair is not actually taking the agent from its spawn to its goal.
MIN_ROUTE_PROGRESS = 1.0
# A candidate route may be longer than the straight line -- that is what turning a
# corner costs -- but not by this much. Without the cap the joint candidate
# search happily accepts a pair that connects the long way round the block
# (observed at 4x), which is not the path the agent covered in 9 seconds. The
# additive slack keeps short goals from being over-constrained: a 90-degree turn
# 5 m away is legitimately much longer than 5 m.
MAX_DETOUR_RATIO = 2.0
MAX_DETOUR_SLACK = 10.0


@dataclass
class Route:
    """A reference path for one agent, in scene coordinates.

    The segment decomposition is precomputed once here rather than per
    projection: a planner projects itself and every neighbour onto its route on
    every one of the 91 steps, so this is the hot path.
    """

    points: np.ndarray  # [P, 2], ordered spawn -> goal, roughly evenly spaced
    source: str         # "graph" (multi-lane path) | "lane" (single lane)

    def __post_init__(self) -> None:
        self.seg_a = self.points[:-1]                          # [S, 2]
        self.ab = self.points[1:] - self.points[:-1]           # [S, 2]
        self.seg_len = np.linalg.norm(self.ab, axis=-1)        # [S]
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)]).astype(np.float32)
        self.total = float(self.cum[-1])
        self._denom = np.maximum((self.ab * self.ab).sum(-1), 1e-9)

    def project(self, points: np.ndarray):
        """Frenet ``(s, d)`` of ``points`` [N, 2] against this route."""
        rel = points[:, None, :] - self.seg_a[None, :, :]                    # [N, S, 2]
        t = np.clip((rel * self.ab[None, :, :]).sum(-1) / self._denom[None, :], 0.0, 1.0)
        proj = self.seg_a[None, :, :] + t[..., None] * self.ab[None, :, :]
        dist = np.linalg.norm(points[:, None, :] - proj, axis=-1)            # [N, S]
        k = dist.argmin(axis=1)
        rows = np.arange(len(points))
        return self.cum[k] + t[rows, k] * self.seg_len[k], dist[rows, k]

    def point_at(self, s: float) -> np.ndarray:
        """The route point at arc length ``s``.

        Past the end the path is EXTRAPOLATED along its final tangent rather than
        clamped to the last point. Clamping makes an agent that has driven past
        its goal steer at a target behind itself, so it loiters in a circle --
        which reads as erratic traffic and causes collisions that have nothing to
        do with the planner being benchmarked.
        """
        if s > self.total and self.seg_len[-1] > 1e-6:
            return (self.points[-1] + self.ab[-1] / self.seg_len[-1] * (s - self.total)).astype(
                np.float32
            )
        return np.array(
            [np.interp(s, self.cum, self.points[:, 0]),
             np.interp(s, self.cum, self.points[:, 1])],
            dtype=np.float32,
        )


# --------------------------------------------------------------- geometry
def project_point_to_segments(seg_a: np.ndarray, seg_b: np.ndarray, pos: np.ndarray):
    """Distance from ``pos`` to each segment ``[seg_a, seg_b]``.

    ``seg_a`` / ``seg_b`` broadcast to any leading shape ``[..., 2]``; ``pos`` is
    ``[2]``. Returns ``(dist[...], t[...])`` where ``t`` is the clamped
    parametric position of the foot of the perpendicular along each segment.
    """
    ab = seg_b - seg_a
    denom = np.maximum((ab * ab).sum(-1), 1e-9)
    t = np.clip(((pos - seg_a) * ab).sum(-1) / denom, 0.0, 1.0)
    proj = seg_a + t[..., None] * ab
    return np.linalg.norm(proj - pos, axis=-1), t


def polyline_length(polyline: np.ndarray) -> float:
    if len(polyline) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(polyline, axis=0), axis=-1).sum())


def _arclength(polyline: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=-1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _project_arclength(polyline: np.ndarray, pos: np.ndarray) -> float:
    """Arc length of the foot of the perpendicular from ``pos`` onto ``polyline``.

    Interpolates within the segment rather than snapping to a vertex -- lane
    centerlines carry only 20 points, so vertex snapping is metres-coarse.
    """
    seg_a, seg_b = polyline[:-1], polyline[1:]
    dist, t = project_point_to_segments(seg_a, seg_b, pos)
    k = int(dist.argmin())
    cum = _arclength(polyline)
    return float(cum[k] + t[k] * (cum[k + 1] - cum[k]))


# ------------------------------------------------------------ lane lookup
def _lane_distances(lanes: np.ndarray, pos: np.ndarray):
    """Per-lane distance from ``pos`` to the lane polyline, plus segment index."""
    if lanes.shape[1] < 2:
        d = np.linalg.norm(lanes[:, 0, :] - pos, axis=-1)
        return d, np.zeros(len(lanes), dtype=np.int64)
    dist, _ = project_point_to_segments(lanes[:, :-1, :], lanes[:, 1:, :], pos)  # [L, S]
    return dist.min(axis=1), dist.argmin(axis=1)


def _segment_heading(lane: np.ndarray, seg: int) -> float:
    a, b = lane[seg], lane[min(seg + 1, len(lane) - 1)]
    return float(np.arctan2(b[1] - a[1], b[0] - a[0]))


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _lateral_neighbours(lane_graph, lane_ids) -> set[int]:
    """Left/right neighbours of ``lane_ids`` (both directions of the relation)."""
    if not lane_graph:
        return set()
    edges = lane_graph.get("lateral")
    if edges is None or np.asarray(edges).size == 0:
        return set()
    e = np.asarray(edges, dtype=np.int64).reshape(2, -1)
    wanted = set(int(i) for i in lane_ids)
    out: set[int] = set()
    for src, dst in zip(e[0], e[1]):
        if int(src) in wanted:
            out.add(int(dst))
        if int(dst) in wanted:
            out.add(int(src))
    return out - wanted


def lane_candidates(
    lanes: np.ndarray,
    pos: np.ndarray,
    lane_graph,
    *,
    heading: float | None = None,
    k: int = NUM_LANE_CANDIDATES,
) -> list[tuple[int, float]]:
    """Plausible lanes for ``pos``, as ``(lane_id, score)`` sorted by score.

    Seeded by the ``k`` nearest lanes within ``SEARCH_RADIUS`` and widened with
    their lateral neighbours, so a route may begin or end one lane over -- which
    is how a lane change is expressed (see the module docstring).

    When ``heading`` is given, candidates pointing the wrong way are dropped and
    the remainder are ranked by distance plus a heading-disagreement penalty:
    without it the oncoming lane of a two-way road wins roughly half the time
    (it can be the geometrically nearest centerline) and routes the agent into
    head-on traffic. A goal carries no heading, so it ranks on distance alone.
    """
    if len(lanes) == 0:
        return []
    dist, seg = _lane_distances(lanes, pos)
    near = np.flatnonzero(dist <= SEARCH_RADIUS)
    if near.size == 0:
        near = np.array([int(dist.argmin())])
    seeds = [int(i) for i in near[np.argsort(dist[near])][:k]]
    ids = seeds + sorted(_lateral_neighbours(lane_graph, seeds))

    scored: list[tuple[int, float]] = []
    for i in ids:
        score = float(dist[i])
        if heading is not None:
            diff = abs(_wrap(_segment_heading(lanes[i], int(seg[i])) - heading))
            if diff > MAX_START_HEADING_DIFF:
                continue
            score += HEADING_TIEBREAK_WEIGHT * diff
        scored.append((i, score))
    if not scored:
        # Every nearby lane points the wrong way (a reversing agent, or a spawn
        # heading that disagrees with the map): fall back to pure distance.
        return [(int(dist.argmin()), float(dist.min()))]
    return sorted(scored, key=lambda p: p[1])


# ------------------------------------------------------------ graph search
def _successors(lane_graph, num_lanes: int) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in range(num_lanes)]
    edges = (lane_graph or {}).get("succ")
    if edges is None or np.asarray(edges).size == 0:
        return out
    e = np.asarray(edges, dtype=np.int64).reshape(2, -1)
    for src, dst in zip(e[0], e[1]):
        if 0 <= src < num_lanes and 0 <= dst < num_lanes and src != dst:
            out[int(src)].append(int(dst))
    return out


def shortest_lane_path(
    lanes: np.ndarray,
    lane_graph,
    start: int,
    goal: int,
    *,
    max_depth: int = 12,
    successors: list[list[int]] | None = None,
) -> list[int] | None:
    """Cheapest successor path ``start -> goal``, weighted by lane arc length.

    Arc-length weights (rather than hop count) matter at intersections, where a
    turn is modelled as a short connector lane: hop count would happily route
    through three connectors to avoid one long straight lane.

    ``successors`` may be passed in to avoid rebuilding the adjacency for every
    candidate pair of the same scene.
    """
    if start == goal:
        return [start]
    succ = successors if successors is not None else _successors(lane_graph, len(lanes))
    cost = [polyline_length(lane) for lane in lanes]

    best = {start: 0.0}
    prev: dict[int, int] = {}
    pq = [(0.0, 0, start)]  # (cost, depth, lane)
    while pq:
        c, depth, u = heapq.heappop(pq)
        if u == goal:
            path = [u]
            while path[-1] != start:
                path.append(prev[path[-1]])
            return path[::-1]
        if c > best.get(u, np.inf) or depth >= max_depth:
            continue
        for v in succ[u]:
            nc = c + cost[v]
            if nc < best.get(v, np.inf):
                best[v] = nc
                prev[v] = u
                heapq.heappush(pq, (nc, depth + 1, v))
    return None


# ------------------------------------------------------------------ route
def _concat_lanes(lanes: np.ndarray, path: list[int]) -> np.ndarray:
    """Stitch the path's lane polylines, dropping duplicated shared endpoints."""
    pts = [lanes[path[0]]]
    for lane_id in path[1:]:
        lane = lanes[lane_id]
        if np.linalg.norm(lane[0] - pts[-1][-1]) < 1e-3:
            lane = lane[1:]
        if len(lane):
            pts.append(lane)
    return np.concatenate(pts, axis=0)


def _trim(polyline: np.ndarray, spawn: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
    """Cut the stitched centerline down to the spawn->goal stretch.

    Both ends are located by projecting onto the polyline (interpolating within
    the segment), and the spawn and goal are then pinned as the true endpoints:
    the agent starts wherever it starts -- usually a little off-centre, and up to
    a lane width off when the route begins in the adjacent lane -- and
    ``reached_goal`` is measured against the goal point itself, not the
    centerline near it.

    Returns ``None`` when the goal does not lie meaningfully ahead of the spawn
    along this path, i.e. this lane pair does not actually take the agent where
    it is going.
    """
    cum = _arclength(polyline)
    s0 = _project_arclength(polyline, spawn)
    s1 = _project_arclength(polyline, goal)
    if s1 - s0 < MIN_ROUTE_PROGRESS:
        return None
    mid = polyline[(cum > s0) & (cum < s1)]
    return np.concatenate([spawn[None, :], mid, goal[None, :]], axis=0)


def _resample(polyline: np.ndarray, spacing: float) -> np.ndarray | None:
    """Even spacing along the path, with the final point kept.

    ``resample_polyline_every`` samples at ``arange(0, total, spacing)`` and so
    drops the endpoint; the endpoint here IS the goal, which the planner needs.
    """
    if len(polyline) < 2 or polyline_length(polyline) < 1e-3:
        return None
    out = resample_polyline_every(polyline, every=spacing)
    if len(out) == 0:
        out = polyline[:1]
    if np.linalg.norm(out[-1] - polyline[-1]) > 1e-3:
        out = np.concatenate([out, polyline[-1:]], axis=0)
    return out.astype(np.float32) if len(out) >= 2 else None


def build_route(
    lane_polylines: np.ndarray,
    lane_graph,
    spawn_xy: np.ndarray,
    goal_xy: np.ndarray,
    heading: float,
    *,
    spacing: float = 1.0,
    max_depth: int = 12,
    num_candidates: int = NUM_LANE_CANDIDATES,
) -> Route | None:
    """Lane-following reference path from ``spawn_xy`` to ``goal_xy``.

    Every candidate (start lane, goal lane) pair is searched and the cheapest one
    that yields a forward-progressing path wins; pairs are scored by how well
    each lane explains its endpoint, so the route prefers the lane the agent is
    actually in and the lane the goal actually sits on.

    Returns ``None`` when no lane path connects the two -- deliberately, rather
    than substituting a straight line (see the module docstring).
    """
    spawn = np.asarray(spawn_xy, dtype=np.float32).reshape(2)
    goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)

    lanes = np.asarray(lane_polylines, dtype=np.float32)
    if lanes.ndim != 3 or lanes.shape[0] == 0 or lanes.shape[1] < 2:
        return None

    starts = lane_candidates(lanes, spawn, lane_graph, heading=heading, k=num_candidates)
    goals = lane_candidates(lanes, goal, lane_graph, heading=None, k=num_candidates)
    if not starts or not goals:
        return None

    straight_dist = float(np.linalg.norm(goal - spawn))
    max_length = max(MAX_DETOUR_RATIO * straight_dist, straight_dist + MAX_DETOUR_SLACK)

    succ = _successors(lane_graph, len(lanes))
    best: tuple[tuple[float, float], np.ndarray, int] | None = None
    for start, start_score in starts:
        for goal_lane, goal_score in goals:
            path = shortest_lane_path(
                lanes, lane_graph, start, goal_lane, max_depth=max_depth, successors=succ
            )
            if path is None:
                continue
            trimmed = _trim(_concat_lanes(lanes, path), spawn, goal)
            if trimmed is None:
                continue
            length = polyline_length(trimmed)
            if length > max_length:
                continue
            # Primary: how well the two lanes explain the endpoints. Tie-break on
            # the shorter path, so equally plausible lane pairs prefer the direct
            # one.
            score = (start_score + goal_score, length)
            if best is None or score < best[0]:
                best = (score, trimmed, len(path))
    if best is None:
        return None

    points = _resample(best[1], spacing)
    if points is None:
        return None
    return Route(points, "lane" if best[2] == 1 else "graph")
