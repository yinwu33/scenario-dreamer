"""idm planner: rule-based Intelligent-Driver-Model agent on a lane-graph route.

The reference planner for the SUT x scene-initialization benchmark. Unlike
the ``ppo_*`` family (frozen neural policies that read raw road segments out of their
observation) this planner is explicit about where it is going: it searches a
route from spawn to goal through the lane graph (``sim.routes``) and then

  * longitudinal -- classic IDM against the nearest vehicle ahead ON THAT ROUTE
    (Frenet projection, not a Euclidean cone), plus the end of the route as a
    virtual stationary obstacle so the agent decelerates into its goal;
  * lateral -- pure pursuit to a speed-dependent lookahead point, resolved by
    forward-simulating each of the 13 discrete steering values and keeping the
    one that lands closest to the target.

Both are then quantised onto PufferDrive's 7x13 discrete action table, so IDM
and the neural planners drive through the *identical* ``step_dynamics``
integrator -- which is what makes their metrics comparable in one table.

The steering search deliberately avoids inverting the dynamics analytically:
``step_dynamics`` applies a slip angle ``beta = tanh(0.5*tan(steer))`` and turns
on ``heading + beta``, so a closed-form inverse is easy to get subtly wrong and
would silently drift out of sync if the sim's model ever changed. Rolling the
real update forward for a few steps cannot.

Routes are built lazily on the first ``plan`` call and cached on the sim
(``sim._idm_routes``), mirroring how ``PPOPlanner`` stashes its LSTM carry
-- planner-owned state on a per-scene object, never world state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..routes import LaneIndex, Route, RoutePack, build_route
from ..world import (
    ACCELERATION_VALUES,
    NUM_STEER,
    STEERING_VALUES,
    TYPE_PEDESTRIAN,
    SimScene,
)
from .base import Planner, PlanItem, require

# Coast: acceleration -0.0, steering 0.0. Used when an agent has no usable route.
_IDLE_ACCEL_IDX = int(np.argmin(np.abs(ACCELERATION_VALUES)))
_IDLE_STEER_IDX = int(np.argmin(np.abs(STEERING_VALUES)))
IDLE_ACTION = _IDLE_ACCEL_IDX * NUM_STEER + _IDLE_STEER_IDX

# Hardest brake the discrete action table can command. The lead-vehicle gate is
# sized against it so the gate can never hide an obstacle the agent could still
# stop for (see IDMPlanner._search_radius).
MAX_TABLE_DECEL = float(-ACCELERATION_VALUES.min())


def steering_indices(
    route: Route,
    s_ego: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    length: float,
    signed_speed: np.ndarray,
    accel: np.ndarray,
    dt: float,
    *,
    lookahead_time: float,
    lookahead_min: float,
    lookahead_max: float,
    preview_steps: int,
    lateral_offset: np.ndarray,
) -> np.ndarray:
    """Vectorized discrete pure-pursuit action used by IDM-style planners.

    ``lateral_offset`` shifts each row's lookahead point that many metres to the
    LEFT of the route (negative = right), i.e. it tracks a path parallel to the
    centerline rather than the centerline itself. IDM passes zeros; PDM uses it
    to give its proposals the lateral spread that is the whole reason a proposal
    set can beat a single IDM policy.
    """
    s_ego = np.asarray(s_ego, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    heading = np.asarray(heading, dtype=np.float64)
    signed_speed = np.asarray(signed_speed, dtype=np.float64)
    accel = np.asarray(accel, dtype=np.float64)

    lookahead = np.clip(
        lookahead_time * np.abs(signed_speed), lookahead_min, lookahead_max
    )
    target_s = s_ego + lookahead
    target = np.stack(
        [
            np.interp(target_s, route.cum, route.points[:, 0]),
            np.interp(target_s, route.cum, route.points[:, 1]),
        ],
        axis=-1,
    )
    past_end = target_s > route.total
    target[past_end] = (
        route.points[-1]
        + (target_s[past_end] - route.total)[:, None] * route.end_tangent
    )

    # Shift the target onto the parallel path. The tangent comes from the
    # segment the target lands in (exact, and already precomputed by Route);
    # past the end the route's own extrapolation tangent is the right one.
    lateral_offset = np.asarray(lateral_offset, dtype=np.float64)
    if np.any(lateral_offset):
        seg = np.clip(
            np.searchsorted(route.cum, target_s, side="right") - 1, 0, len(route.ab) - 1
        )
        tangent = route.ab[seg] / route.seg_len[seg][:, None]
        tangent[past_end] = route.end_tangent
        target = target + lateral_offset[:, None] * np.stack(
            [-tangent[:, 1], tangent[:, 0]], axis=-1
        )

    return steering_from_target(
        target, x, y, heading, length, signed_speed, accel, dt,
        preview_steps=preview_steps,
    )


def steering_from_target(
    target: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    length,
    signed_speed: np.ndarray,
    accel: np.ndarray,
    dt: float,
    *,
    preview_steps: int,
) -> np.ndarray:
    """Discrete steering choice, given the lookahead point each row is tracking.

    The route-dependent half (where the target is) is the caller's job, so the
    same forward simulation serves a single ``Route`` and a batched
    ``RoutePack``. ``length`` is a scalar or a per-row array.
    """
    length = np.asarray(length, dtype=np.float64)
    if length.ndim:
        length = length[:, None]
    steer = STEERING_VALUES[None, :]
    beta = np.tanh(0.5 * np.tan(steer))
    speed = np.clip(signed_speed + accel * dt, -100.0, 100.0)[:, None]
    px = np.broadcast_to(x[:, None], (len(x), NUM_STEER)).copy()
    py = np.broadcast_to(y[:, None], (len(y), NUM_STEER)).copy()
    ph = np.broadcast_to(heading[:, None], (len(heading), NUM_STEER)).copy()
    yaw_rate = speed * np.cos(beta) * np.tan(steer) / length
    for _ in range(max(preview_steps, 1)):
        px += speed * np.cos(ph + beta) * dt
        py += speed * np.sin(ph + beta) * dt
        ph += yaw_rate * dt
    return np.argmin(
        (px - target[:, 0, None]) ** 2 + (py - target[:, 1, None]) ** 2,
        axis=1,
    )


def steering_targets_packed(
    pack, rows: np.ndarray, s_ego: np.ndarray, signed_speed: np.ndarray, *,
    lookahead_time: float, lookahead_min: float, lookahead_max: float,
    lateral_offset: np.ndarray,
) -> np.ndarray:
    """``steering_indices``' lookahead point, for one route per row.

    Same lookahead law, same past-the-end extrapolation and same lateral shift
    as the single-route version; ``RoutePack`` supplies the interpolation and
    the segment tangent for every row at once.
    """
    lookahead = np.clip(
        lookahead_time * np.abs(signed_speed), lookahead_min, lookahead_max
    )
    target_s = s_ego + lookahead
    target = pack.point_at(rows, target_s)
    lateral_offset = np.asarray(lateral_offset, dtype=np.float64)
    if np.any(lateral_offset):
        tangent = pack.tangent_at(rows, target_s)
        target = target + np.atleast_1d(lateral_offset)[:, None] * np.stack(
            [-tangent[:, 1], tangent[:, 0]], axis=-1
        )
    return target


@dataclass(frozen=True)
class SceneState:
    """Every active scene's agent arrays, padded to ``[num_items, Nmax]``.

    One of these is built per ``plan`` call; every row of the batch reads its own
    scene's neighbours out of it by ``[item_index, agent_index]``.
    """

    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    width: np.ndarray
    length: np.ndarray
    heading: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    cand: np.ndarray


class IDMPlanner(Planner):
    def __init__(self, planner_cfg, *, role: str, device: str | None = None):
        super().__init__(planner_cfg, role=role, device=device)
        # Every value is required: an IDM agent's behaviour IS these numbers, so
        # a missing key must fail rather than quietly drive with a hidden default.
        # --- IDM longitudinal ---
        self.target_speed = float(self._require("target_speed"))
        self.min_gap = float(self._require("min_gap"))
        self.headway_time = float(self._require("headway_time"))
        self.max_accel = float(self._require("max_accel"))
        self.comfort_decel = float(self._require("comfort_decel"))
        self.accel_exponent = float(self._require("accel_exponent"))
        # --- lead-vehicle gating ---
        self.lateral_margin = float(self._require("lateral_margin"))
        self.lead_search_radius = float(self._require("lead_search_radius"))
        # --- pure pursuit ---
        self.lookahead_time = float(self._require("lookahead_time"))
        self.lookahead_min = float(self._require("lookahead_min"))
        self.lookahead_max = float(self._require("lookahead_max"))
        self.steer_preview_steps = int(self._require("steer_preview_steps"))
        # --- route search ---
        route_cfg = self._require("route")
        self.route_spacing = float(require(route_cfg, self.name, "spacing", "route.spacing"))
        self.route_max_depth = int(require(route_cfg, self.name, "max_depth", "route.max_depth"))

    # ------------------------------------------------------------- routes
    def _routes_for(self, sim: SimScene) -> dict[int, Route | None]:
        routes = getattr(sim, "_idm_routes", None)
        if routes is None:
            # No lane graph means the route search cannot chain lanes: only an
            # agent whose spawn and goal sit on the SAME polyline would get a
            # route, everyone else would coast. That degradation is silent and
            # is always a scene-source plumbing bug, so refuse to run instead.
            if sim.lane_graph is None:
                raise ValueError(
                    f"planner {self.name!r} needs the lane graph, but this scene "
                    "source did not provide one: set GeneratedScenes.meta"
                    "['lane_graph'] (see sim.scenes.lane_graph_edges / "
                    "batched_lane_graphs)"
                )
            routes = sim._idm_routes = {}
            # Adjacency + per-lane arc lengths, built once for the whole scene.
            sim._idm_lane_index = LaneIndex(sim.lane_polylines, sim.lane_graph)
            # Per-agent provenance of the route ("graph"/"lane"/"straight"/"none"),
            # aggregated after the rollout by the benchmark's diagnostics hook.
            sim._idm_route_sources = {}
        return routes

    def _route(self, sim: SimScene, i: int) -> Route | None:
        routes = self._routes_for(sim)
        if i not in routes:
            route = build_route(
                sim._idm_lane_index,
                np.array([sim.x[i], sim.y[i]], dtype=np.float32),
                sim.goal[i].astype(np.float32),
                float(sim.heading[i]),
                spacing=self.route_spacing,
                max_depth=self.route_max_depth,
            )
            routes[i] = route
            # "none": the lane graph has no path from this agent's spawn to its
            # goal. It coasts rather than being handed a straight line through
            # open space, and the benchmark reports how often this happened --
            # a scene with no route measures route coverage, not driving.
            sim._idm_route_sources[i] = route.source if route is not None else "none"
        return routes[i]

    # -------------------------------------------------------- longitudinal
    def _search_radius(self, sim: SimScene, i: int, speed: float) -> float:
        """Broad-phase radius, never shorter than the current braking distance.

        ``lead_search_radius`` on its own is a FIXED distance, so past
        ``sqrt(2 * MAX_TABLE_DECEL * radius)`` (~17.9 m/s at the shipped 40 m) a
        lead the agent could still stop for is discarded before IDM ever sees
        it, and the agent drives into it at undiminished speed. Growing the gate
        with ``v^2 / 2a`` keeps what the gate can see in step with what the
        action table can physically avoid; the configured radius stays the
        free-flow following range, which is the larger of the two at low speed.
        """
        return self._search_radius_for(float(sim.length[i]), speed)

    def _search_radius_for(self, length, speed):
        """``_search_radius`` over arrays; the scalar form above delegates here."""
        braking = speed * speed / (2.0 * MAX_TABLE_DECEL)
        return np.maximum(self.lead_search_radius, braking + self.min_gap + length)

    # ------------------------------------------------------- batched plan
    # ``plan`` drives EVERY agent of this role, across every active scene, in one
    # set of numpy operations. The per-agent loop it replaces was the single most
    # expensive phase of a rule-based DDPO iteration (measured single-process,
    # batch 128: idm traffic 16.1 s of a 28.0 s iteration = 57.5%; pdm SUT
    # 73.2 s of 81.9 s = 89.3%) and almost all of that was numpy call overhead --
    # roughly 30 array operations per agent per step on arrays of one row.
    #
    # Everything below is the same computation as before, in the same order,
    # with the agent axis promoted to a real dimension: the lead-vehicle gate
    # still runs before the Frenet projection, the lead is still the smallest
    # arc length ahead inside the corridor with ties going to the lowest agent
    # index, and the accel/steer tables are still resolved by the same argmin.
    # The one deliberate change is dtype: ``RoutePack`` projects in float64
    # where ``Route.project`` inherited float32 from the caller's ``pos``.

    def _route_pack(self, items: Sequence[PlanItem]):
        """``(RoutePack, [rows per item])`` for this role, built once per rollout.

        Routes are immutable for the life of a ``SimScene`` and agents only ever
        retire, so the first ``plan`` call of a rollout sees the maximal agent
        set and the pack built from it stays valid. The token is an object
        identity rather than a flag, so a later rollout's fresh SimScenes can
        never be mistaken for these.
        """
        token = getattr(self, "_pack_token", None)
        if token is not None and all(
            getattr(sim, "_idm_pack_token", {}).get(self.role) is token
            for sim, _ in items
        ):
            return self._pack, [sim._idm_pack_row[self.role][ids] for sim, ids in items]

        token = object()
        routes, rows_per_item = [], []
        for sim, ids in items:
            row = np.full(sim.n, -1, dtype=np.int64)
            for i in ids:
                i = int(i)
                if self._route(sim, i) is not None:
                    row[i] = len(routes)
                    routes.append(sim._idm_routes[i])
            if not hasattr(sim, "_idm_pack_row"):
                sim._idm_pack_row, sim._idm_pack_token = {}, {}
            sim._idm_pack_row[self.role] = row
            sim._idm_pack_token[self.role] = token
            rows_per_item.append(row[ids])
        self._pack = RoutePack(routes)
        self._pack_token = token
        return self._pack, rows_per_item

    @staticmethod
    def _scene_state(items: Sequence[PlanItem]):
        """Per-scene agent arrays padded to ``[len(items), Nmax]``.

        The lead search only ever looks inside one scene, so the neighbour axis
        is the scene's own agent slots; padding lets every driven agent's search
        share one array expression.
        """
        nmax = max(sim.n for sim, _ in items)
        m = len(items)
        x = np.zeros((m, nmax)); y = np.zeros((m, nmax))
        vx = np.zeros((m, nmax)); vy = np.zeros((m, nmax))
        w = np.zeros((m, nmax)); ln = np.zeros((m, nmax))
        hd = np.zeros((m, nmax)); hx = np.zeros((m, nmax)); hy = np.zeros((m, nmax))
        cand = np.zeros((m, nmax), dtype=bool)
        for t, (sim, _) in enumerate(items):
            n = sim.n
            x[t, :n] = sim.x; y[t, :n] = sim.y
            vx[t, :n] = sim.vx; vy[t, :n] = sim.vy
            w[t, :n] = sim.width; ln[t, :n] = sim.length
            hd[t, :n] = sim.heading; hx[t, :n] = sim.heading_x; hy[t, :n] = sim.heading_y
            cand[t, :n] = (~sim.removed) & (sim.ptype != TYPE_PEDESTRIAN)
        return SceneState(x, y, vx, vy, w, ln, hd, hx, hy, cand)

    def _lead(self, pack, rows, item_of, agent_of, s_ego, speed, st):
        """``(gap, lead_speed)`` per row: the nearest vehicle ahead on each route.

        Batched form of the old ``_closest_obstacle``. Same two-stage gate --
        the braking-distance radius first (so the Frenet projection only runs on
        candidates that could matter), then ahead-on-route inside the combined
        half-width corridor.
        """
        X, Y, VX, VY, W, L, CAND = st.x, st.y, st.vx, st.vy, st.width, st.length, st.cand
        k = len(rows)
        nmax = X.shape[1]
        gap = np.full(k, np.inf)
        lead_speed = np.zeros(k)
        if k == 0:
            return gap, lead_speed

        ex, ey = X[item_of, agent_of], Y[item_of, agent_of]
        e_len, e_w = L[item_of, agent_of], W[item_of, agent_of]
        radius = self._search_radius_for(e_len, speed)
        dx = X[item_of] - ex[:, None]
        dy = Y[item_of] - ey[:, None]
        near = (
            CAND[item_of]
            & (np.arange(nmax)[None, :] != agent_of[:, None])
            & ((dx * dx + dy * dy) <= (radius * radius)[:, None])
        )
        pr, pc = np.nonzero(near)
        if not len(pr):
            return gap, lead_speed

        pts = np.stack([X[item_of[pr], pc], Y[item_of[pr], pc]], axis=-1).astype(np.float32)
        s_o, d_o = pack.project(rows[pr], pts)
        corridor = (W[item_of[pr], pc] + e_w[pr]) / 2.0 + self.lateral_margin
        ahead = (s_o > s_ego[pr]) & (d_o <= corridor)
        # Dense arc lengths so argmin breaks ties on the LOWEST agent index,
        # which is the order the flatnonzero candidate list used to have.
        s_mat = np.full((k, nmax), np.inf)
        s_mat[pr[ahead], pc[ahead]] = s_o[ahead]
        j = s_mat.argmin(axis=1)
        q = np.arange(k)
        s_best = s_mat[q, j]
        found = np.isfinite(s_best)
        veh_gap = s_best - s_ego - e_len / 2.0 - L[item_of, j] / 2.0
        gap = np.where(found, np.maximum(veh_gap, 0.0), np.inf)
        lead_speed = np.where(found, np.hypot(VX[item_of, j], VY[item_of, j]), 0.0)
        return gap, lead_speed

    # --------------------------------------------------------------- plan
    def plan(self, items: Sequence[PlanItem]) -> list:
        counts = [len(ids) for _, ids in items]
        plans = [np.full(c, IDLE_ACTION, dtype=np.int64) for c in counts]
        if not items or not sum(counts):
            return plans
        dt = float(items[0][0].dt)
        pack, rows_per_item = self._route_pack(items)
        st = self._scene_state(items)

        rows = np.concatenate(rows_per_item)
        item_of = np.concatenate(
            [np.full(c, t, dtype=np.int64) for t, c in enumerate(counts)]
        )
        agent_of = np.concatenate([np.asarray(ids, dtype=np.int64) for _, ids in items])
        slot_of = np.concatenate([np.arange(c, dtype=np.int64) for c in counts])

        # Agents the route search could not serve keep the IDLE action.
        keep = rows >= 0
        rows, item_of, agent_of, slot_of = (
            rows[keep], item_of[keep], agent_of[keep], slot_of[keep]
        )
        if not len(rows):
            return plans

        ex, ey = st.x[item_of, agent_of], st.y[item_of, agent_of]
        vx, vy = st.vx[item_of, agent_of], st.vy[item_of, agent_of]
        hx, hy = st.hx[item_of, agent_of], st.hy[item_of, agent_of]
        heading = st.heading[item_of, agent_of]
        e_len = st.length[item_of, agent_of]

        # float32 positions, matching the scalar path's explicit float32 pos
        s_ego, _ = pack.project(rows, np.stack([ex, ey], axis=-1).astype(np.float32))
        signed_speed = np.copysign(np.hypot(vx, vy), vx * hx + vy * hy)
        speed = np.maximum(signed_speed, 0.0)

        gap, lead_speed = self._lead(pack, rows, item_of, agent_of, s_ego, speed, st)
        s_star = (
            self.min_gap
            + speed * self.headway_time
            + speed * (speed - lead_speed)
            / (2.0 * np.sqrt(self.max_accel * self.comfort_decel))
        )
        s_alpha = np.maximum(gap, self.min_gap)
        free = 1.0 - (speed / self.target_speed) ** self.accel_exponent
        a_des = self.max_accel * (free - (s_star / s_alpha) ** 2)

        # Nearest table acceleration that cannot reverse a stopped agent. The
        # allowed set is never empty (a_min <= 0 <= ACCELERATION_VALUES.max()),
        # so the scalar version's empty-set branch has no batched counterpart.
        a_min = -speed / dt
        allowed = ACCELERATION_VALUES[None, :] >= a_min[:, None] - 1e-6
        ia = np.argmin(
            np.where(allowed, np.abs(ACCELERATION_VALUES[None, :] - a_des[:, None]), np.inf),
            axis=1,
        )
        target = steering_targets_packed(
            pack, rows, s_ego, signed_speed,
            lookahead_time=self.lookahead_time,
            lookahead_min=self.lookahead_min,
            lookahead_max=self.lookahead_max,
            lateral_offset=np.zeros(len(rows)),
        )
        is_ = steering_from_target(
            target, ex, ey, heading, e_len, signed_speed,
            ACCELERATION_VALUES[ia].astype(np.float64), dt,
            preview_steps=self.steer_preview_steps,
        )

        actions = ia * NUM_STEER + is_
        for t in range(len(items)):
            sel = item_of == t
            if sel.any():
                plans[t][slot_of[sel]] = actions[sel]
        return plans

    def apply(self, items: Sequence[PlanItem], plans: list) -> None:
        for (sim, ids), actions in zip(items, plans):
            sim.step_dynamics(actions, ids)
