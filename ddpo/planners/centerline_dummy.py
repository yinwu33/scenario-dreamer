"""Centerline-following deterministic rollout planner.

This planner is a rule-based alternative to ``dummy``: controlled agents first
snap to a nearby lane centerline, follow a successor-connected lane route at a
constant speed, then leave the centerline to reach the exact generated goal. It
does not reason about collisions or right-of-way.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any

import numpy as np
import torch

from ..pufferdrive_sim import MAX_SPEED, SimScene
from .base import NumpyPlanner, SimulatorConfig, register_planner

PRED_CONN = 1
SUCC_CONN = 2
LEFT_CONN = 3
RIGHT_CONN = 4
POINT_EPS = 1e-4


def _to_numpy(value: Any, dtype=None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


@dataclass
class LaneInfo:
    points: np.ndarray
    cumlen: np.ndarray
    length: float


@dataclass
class Projection:
    lane: int
    s: float
    point: np.ndarray
    dist: float


@register_planner("centerline_dummy")
class CenterlineDummyPlanner(NumpyPlanner):
    """Rule planner that moves each controlled agent along lane centerlines."""

    def __init__(self, planner_cfg, params: SimulatorConfig, *, device: str | None = None):
        super().__init__(planner_cfg, params, device=device)
        self.candidate_lanes = int(planner_cfg.get("candidate_lanes", 5))
        self.connect_radius = float(planner_cfg.get("connect_radius", 2.5))
        self.allow_lane_changes = bool(planner_cfg.get("allow_lane_changes", False))
        self.allow_backward_same_lane = bool(
            planner_cfg.get("allow_backward_same_lane", False)
        )
        self.fallback_to_straight = bool(planner_cfg.get("fallback_to_straight", True))
        self.min_speed = float(planner_cfg.get("min_speed", 0.0))
        max_speed = planner_cfg.get("max_speed", MAX_SPEED)
        self.max_speed = None if max_speed is None else float(max_speed)

    # ------------------------------------------------------------------ build
    def _build_scenes(self, scenes):
        sims = super()._build_scenes(scenes)

        lanes = _to_numpy(scenes.lane_polylines, np.float32)
        lane_scene_idx = _to_numpy(scenes.meta["lane_scene_idx"], np.int64)
        edge_index = _to_numpy(scenes.meta.get("lane_edge_index"), np.int64)
        edge_type = _to_numpy(scenes.meta.get("lane_edge_type"), np.float32)

        for s, sim in enumerate(sims):
            global_lanes = np.nonzero(lane_scene_idx == s)[0]
            sim._centerline_lanes = lanes[global_lanes]
            sim._centerline_edge_index = None
            sim._centerline_edge_type = None
            if edge_index is None or edge_type is None or len(global_lanes) == 0:
                continue
            in_scene = np.zeros(lane_scene_idx.shape[0], dtype=bool)
            in_scene[global_lanes] = True
            edge_mask = in_scene[edge_index[0]] & in_scene[edge_index[1]]
            if not edge_mask.any():
                continue
            global_to_local = np.full(lane_scene_idx.shape[0], -1, dtype=np.int64)
            global_to_local[global_lanes] = np.arange(len(global_lanes), dtype=np.int64)
            local_edges = global_to_local[edge_index[:, edge_mask]]
            valid = (local_edges[0] >= 0) & (local_edges[1] >= 0)
            sim._centerline_edge_index = local_edges[:, valid]
            sim._centerline_edge_type = edge_type[edge_mask][valid]
        return sims

    # ---------------------------------------------------------------- advance
    def _advance(self, sims: list[SimScene], active: list[int]) -> None:
        for s in active:
            sim = sims[s]
            if not hasattr(sim, "_centerline_routes"):
                self._prepare_routes(sim)
            self._step_routes(sim)

    def _prepare_routes(self, sim: SimScene) -> None:
        lane_infos = self._lane_infos(getattr(sim, "_centerline_lanes", None))
        adjacency = self._lane_adjacency(sim, lane_infos)
        routes: dict[int, np.ndarray] = {}
        cursors: dict[int, int] = {}
        for i in sim.controlled:
            route = self._build_agent_route(sim, int(i), lane_infos, adjacency)
            if route is None and self.fallback_to_straight:
                route = np.asarray(
                    [[sim.x[i], sim.y[i]], [sim.goal[i, 0], sim.goal[i, 1]]],
                    dtype=np.float32,
                )
            if route is not None and len(route) >= 2:
                routes[int(i)] = route
                cursors[int(i)] = 0
        sim._centerline_lane_infos = lane_infos
        sim._centerline_adjacency = adjacency
        sim._centerline_routes = routes
        sim._centerline_cursors = cursors

    def _step_routes(self, sim: SimScene) -> None:
        stopped = sim.controlled[sim.stopped[sim.controlled]]
        if len(stopped):
            sim.vx[stopped] = 0.0
            sim.vy[stopped] = 0.0

        idx = sim.controlled[~sim.stopped[sim.controlled]]
        idx = idx[~sim.removed[idx]]
        if len(idx) == 0:
            return
        for i in idx:
            speed = float(sim.speed0[i])
            if self.max_speed is not None:
                speed = min(speed, self.max_speed)
            speed = max(speed, self.min_speed)
            if speed <= 1e-6:
                sim.vx[i] = 0.0
                sim.vy[i] = 0.0
                continue
            route = sim._centerline_routes.get(int(i))
            if route is None or len(route) < 2:
                if self.fallback_to_straight:
                    self._step_direct(sim, int(i), speed)
                else:
                    sim.vx[i] = 0.0
                    sim.vy[i] = 0.0
                continue
            self._step_along_route(sim, int(i), route, speed)

    def _step_direct(self, sim: SimScene, i: int, speed: float) -> None:
        vec = sim.goal[i] - np.asarray([sim.x[i], sim.y[i]], dtype=np.float32)
        dist = float(np.hypot(vec[0], vec[1]))
        if dist <= 1e-6:
            sim.vx[i] = 0.0
            sim.vy[i] = 0.0
            return
        direction = vec / dist
        step = min(speed * sim.dt, dist)
        sim.x[i] += float(direction[0] * step)
        sim.y[i] += float(direction[1] * step)
        sim.vx[i] = float(direction[0] * step / sim.dt)
        sim.vy[i] = float(direction[1] * step / sim.dt)
        self._set_heading(sim, i, direction)

    def _step_along_route(
        self, sim: SimScene, i: int, route: np.ndarray, speed: float
    ) -> None:
        cur = np.asarray([sim.x[i], sim.y[i]], dtype=np.float32)
        cursor = int(sim._centerline_cursors.get(i, 0))
        remaining = speed * sim.dt
        moved = 0.0
        last_dir = None

        while remaining > 1e-6 and cursor < len(route) - 1:
            target = route[cursor + 1]
            vec = target - cur
            dist = float(np.hypot(vec[0], vec[1]))
            if dist <= POINT_EPS:
                cursor += 1
                cur = target.astype(np.float32, copy=False)
                continue
            direction = vec / dist
            step = min(remaining, dist)
            cur = cur + direction * step
            remaining -= step
            moved += step
            last_dir = direction
            if step >= dist - POINT_EPS:
                cur = target.astype(np.float32, copy=False)
                cursor += 1

        sim._centerline_cursors[i] = cursor
        sim.x[i] = float(cur[0])
        sim.y[i] = float(cur[1])
        if moved <= 1e-6 or last_dir is None:
            sim.vx[i] = 0.0
            sim.vy[i] = 0.0
            return
        speed_eff = moved / sim.dt
        sim.vx[i] = float(last_dir[0] * speed_eff)
        sim.vy[i] = float(last_dir[1] * speed_eff)
        self._set_heading(sim, i, last_dir)

    @staticmethod
    def _set_heading(sim: SimScene, i: int, direction: np.ndarray) -> None:
        norm = float(np.hypot(direction[0], direction[1]))
        if norm <= 1e-6:
            return
        sim.heading[i] = float(np.arctan2(direction[1], direction[0]))
        sim.heading_x[i] = float(direction[0] / norm)
        sim.heading_y[i] = float(direction[1] / norm)

    # --------------------------------------------------------------- routing
    def _lane_infos(self, lanes: np.ndarray | None) -> list[LaneInfo | None]:
        if lanes is None:
            return []
        out: list[LaneInfo | None] = []
        for poly in np.asarray(lanes, dtype=np.float32):
            valid = np.isfinite(poly).all(axis=1)
            pts = poly[valid]
            if len(pts) >= 2:
                keep = np.ones(len(pts), dtype=bool)
                keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > POINT_EPS
                pts = pts[keep]
            if len(pts) < 2:
                out.append(None)
                continue
            seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cumlen = np.concatenate([[0.0], np.cumsum(seg_len)]).astype(np.float32)
            length = float(cumlen[-1])
            out.append(LaneInfo(points=pts, cumlen=cumlen, length=length))
        return out

    def _lane_adjacency(
        self, sim: SimScene, lane_infos: list[LaneInfo | None]
    ) -> list[list[int]]:
        adj = [set() for _ in lane_infos]
        edge_index = getattr(sim, "_centerline_edge_index", None)
        edge_type = getattr(sim, "_centerline_edge_type", None)
        if edge_index is not None and edge_type is not None and edge_index.size:
            conn = edge_type.argmax(axis=1) if edge_type.ndim > 1 else edge_type
            for (src, dst), typ in zip(edge_index.T, conn):
                src, dst, typ = int(src), int(dst), int(typ)
                if (
                    src == dst
                    or not self._valid_lane(lane_infos, src)
                    or not self._valid_lane(lane_infos, dst)
                ):
                    continue
                if typ == PRED_CONN:
                    adj[src].add(dst)
                elif typ == SUCC_CONN:
                    adj[dst].add(src)
                elif self.allow_lane_changes and typ in (LEFT_CONN, RIGHT_CONN):
                    adj[src].add(dst)
        else:
            self._add_geometric_adjacency(adj, lane_infos)
        return [sorted(v) for v in adj]

    def _add_geometric_adjacency(
        self, adj: list[set[int]], lane_infos: list[LaneInfo | None]
    ) -> None:
        valid = [i for i, info in enumerate(lane_infos) if info is not None]
        for i in valid:
            info_i = lane_infos[i]
            end_i = info_i.points[-1]
            tan_i = info_i.points[-1] - info_i.points[-2]
            tan_i = tan_i / max(float(np.linalg.norm(tan_i)), 1e-6)
            for j in valid:
                if i == j:
                    continue
                info_j = lane_infos[j]
                start_j = info_j.points[0]
                if float(np.linalg.norm(end_i - start_j)) > self.connect_radius:
                    continue
                tan_j = info_j.points[1] - info_j.points[0]
                tan_j = tan_j / max(float(np.linalg.norm(tan_j)), 1e-6)
                if float(np.dot(tan_i, tan_j)) > 0.0:
                    adj[i].add(j)

    @staticmethod
    def _valid_lane(lane_infos: list[LaneInfo | None], lane: int) -> bool:
        return 0 <= lane < len(lane_infos) and lane_infos[lane] is not None

    def _build_agent_route(
        self,
        sim: SimScene,
        i: int,
        lane_infos: list[LaneInfo | None],
        adjacency: list[list[int]],
    ) -> np.ndarray | None:
        if not lane_infos:
            return None
        start = np.asarray([sim.x[i], sim.y[i]], dtype=np.float32)
        goal = np.asarray(sim.goal[i], dtype=np.float32)
        start_cands = self._project_candidates(start, lane_infos)
        goal_cands = self._project_candidates(goal, lane_infos)
        if not start_cands or not goal_cands:
            return None

        best = None
        for sp in start_cands:
            for gp in goal_cands:
                lane_path = self._lane_path(sp, gp, lane_infos, adjacency)
                if lane_path is None:
                    continue
                cost = self._route_cost(sp, gp, lane_path, lane_infos)
                if best is None or cost < best[0]:
                    best = (cost, sp, gp, lane_path)
        if best is None:
            return None
        _, sp, gp, lane_path = best
        return self._route_polyline(start, goal, sp, gp, lane_path, lane_infos)

    def _project_candidates(
        self, point: np.ndarray, lane_infos: list[LaneInfo | None]
    ) -> list[Projection]:
        candidates: list[Projection] = []
        for lane, info in enumerate(lane_infos):
            if info is None:
                continue
            pts = info.points
            a = pts[:-1]
            b = pts[1:]
            ab = b - a
            denom = np.maximum((ab * ab).sum(axis=1), 1e-9)
            t = np.clip(((point[None] - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
            proj = a + t[:, None] * ab
            d2 = ((proj - point[None]) ** 2).sum(axis=1)
            j = int(np.argmin(d2))
            s = float(info.cumlen[j] + t[j] * (info.cumlen[j + 1] - info.cumlen[j]))
            candidates.append(
                Projection(
                    lane=lane,
                    s=s,
                    point=proj[j].astype(np.float32),
                    dist=float(np.sqrt(d2[j])),
                )
            )
        candidates.sort(key=lambda p: p.dist)
        return candidates[: max(self.candidate_lanes, 1)]

    def _lane_path(
        self,
        sp: Projection,
        gp: Projection,
        lane_infos: list[LaneInfo | None],
        adjacency: list[list[int]],
    ) -> list[int] | None:
        if sp.lane == gp.lane:
            if gp.s >= sp.s or self.allow_backward_same_lane:
                return [sp.lane]
            return None
        return self._shortest_path(sp.lane, gp.lane, lane_infos, adjacency)

    def _shortest_path(
        self,
        start: int,
        goal: int,
        lane_infos: list[LaneInfo | None],
        adjacency: list[list[int]],
    ) -> list[int] | None:
        heap = [(0.0, start)]
        dist = {start: 0.0}
        prev: dict[int, int | None] = {start: None}
        while heap:
            cost, node = heapq.heappop(heap)
            if cost != dist[node]:
                continue
            if node == goal:
                break
            for nxt in adjacency[node]:
                info = lane_infos[nxt]
                if info is None:
                    continue
                ncost = cost + max(info.length, 1e-3)
                if ncost < dist.get(nxt, float("inf")):
                    dist[nxt] = ncost
                    prev[nxt] = node
                    heapq.heappush(heap, (ncost, nxt))
        if goal not in prev:
            return None
        out = []
        node: int | None = goal
        while node is not None:
            out.append(node)
            node = prev[node]
        return out[::-1]

    def _route_cost(
        self,
        sp: Projection,
        gp: Projection,
        lane_path: list[int],
        lane_infos: list[LaneInfo | None],
    ) -> float:
        cost = sp.dist + gp.dist
        if len(lane_path) == 1:
            return cost + abs(gp.s - sp.s)
        cost += lane_infos[sp.lane].length - sp.s
        for lane in lane_path[1:-1]:
            cost += lane_infos[lane].length
        cost += gp.s
        return float(cost)

    def _route_polyline(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        sp: Projection,
        gp: Projection,
        lane_path: list[int],
        lane_infos: list[LaneInfo | None],
    ) -> np.ndarray:
        points: list[np.ndarray] = []
        self._append_point(points, start)
        self._append_point(points, sp.point)

        if len(lane_path) == 1:
            self._append_section(points, lane_infos[sp.lane], sp.s, gp.s)
        else:
            self._append_section(
                points, lane_infos[sp.lane], sp.s, lane_infos[sp.lane].length
            )
            for lane in lane_path[1:-1]:
                self._append_section(
                    points, lane_infos[lane], 0.0, lane_infos[lane].length
                )
            self._append_section(points, lane_infos[gp.lane], 0.0, gp.s)
        self._append_point(points, gp.point)
        self._append_point(points, goal)
        return np.asarray(points, dtype=np.float32)

    def _append_section(
        self,
        points: list[np.ndarray],
        info: LaneInfo,
        s0: float,
        s1: float,
    ) -> None:
        self._append_point(points, self._interp_lane(info, s0))
        if s1 >= s0:
            mask = (info.cumlen > s0 + POINT_EPS) & (info.cumlen < s1 - POINT_EPS)
            for p in info.points[mask]:
                self._append_point(points, p)
        else:
            mask = (info.cumlen < s0 - POINT_EPS) & (info.cumlen > s1 + POINT_EPS)
            for p in info.points[mask][::-1]:
                self._append_point(points, p)
        self._append_point(points, self._interp_lane(info, s1))

    @staticmethod
    def _interp_lane(info: LaneInfo, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, info.length))
        j = int(np.searchsorted(info.cumlen, s, side="right") - 1)
        j = min(max(j, 0), len(info.points) - 2)
        seg_len = max(float(info.cumlen[j + 1] - info.cumlen[j]), 1e-9)
        t = (s - float(info.cumlen[j])) / seg_len
        return (info.points[j] + t * (info.points[j + 1] - info.points[j])).astype(
            np.float32
        )

    @staticmethod
    def _append_point(points: list[np.ndarray], point: np.ndarray) -> None:
        point = np.asarray(point, dtype=np.float32)
        if points and float(np.linalg.norm(points[-1] - point)) <= POINT_EPS:
            return
        points.append(point)
