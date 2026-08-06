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

from typing import Sequence

import numpy as np

from ..routes import Route, build_route
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
            routes = sim._idm_routes = {}
            # Per-agent provenance of the route ("graph"/"lane"/"straight"/"none"),
            # aggregated after the rollout by the benchmark's diagnostics hook.
            sim._idm_route_sources = {}
        return routes

    def _route(self, sim: SimScene, i: int) -> Route | None:
        routes = self._routes_for(sim)
        if i not in routes:
            route = build_route(
                sim.lane_polylines,
                sim.lane_graph,
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
    def _idm_accel(self, sim: SimScene, i: int, route: Route, s_ego: float, speed: float) -> float:
        """IDM acceleration against the closest obstacle ahead on the route."""
        gap, lead_speed = self._closest_obstacle(sim, i, route, s_ego)

        # s_star: the desired dynamic gap. Not clamped at 0 -- when the lead is
        # faster than us the interaction term legitimately relaxes below the
        # static minimum, exactly as in policies/idm_policy.py.
        s_star = (
            self.min_gap
            + speed * self.headway_time
            + speed * (speed - lead_speed) / (2.0 * np.sqrt(self.max_accel * self.comfort_decel))
        )
        s_alpha = max(gap, self.min_gap)  # clamped to avoid division by zero
        free = 1.0 - (speed / self.target_speed) ** self.accel_exponent
        return float(self.max_accel * (free - (s_star / s_alpha) ** 2))

    def _closest_obstacle(self, sim: SimScene, i: int, route: Route, s_ego: float):
        """``(gap, lead_speed)`` for the nearest vehicle ahead on the route.

        Vehicles count when their centre projects ahead of us on the route AND
        lies within a combined-half-width corridor of it. The Frenet gate is what
        makes this a lane-follower rather than a Euclidean cone: a car in the
        neighbouring lane of a straight road is correctly ignored, one around a
        bend ahead is correctly seen.

        The end of the route is deliberately NOT an obstacle. A goal here is the
        last in-FOV point of a trajectory that carried on past it, not a stop
        line -- and IDM's whole design is to hold a standstill gap ``s0``, so
        treating it as a stationary lead parks the agent ``s0 + length/2`` ~ 3.6 m
        short of its own goal, outside the 2 m goal radius, and the agent never
        arrives. Agents drive through their goal instead; ``SimScene.goal_step``
        applies the configured goal behaviour when they get there.
        """
        half_len = float(sim.length[i]) / 2.0
        gap, lead_speed = np.inf, 0.0

        others = np.flatnonzero(
            (~sim.removed) & (sim.ptype != TYPE_PEDESTRIAN) & (np.arange(sim.n) != i)
        )
        if others.size:
            dx = sim.x[others] - sim.x[i]
            dy = sim.y[others] - sim.y[i]
            others = others[(dx * dx + dy * dy) <= self.lead_search_radius ** 2]
        if others.size:
            pos = np.stack([sim.x[others], sim.y[others]], axis=-1)
            s, d = route.project(pos)
            corridor = (sim.width[others] + float(sim.width[i])) / 2.0 + self.lateral_margin
            ahead = (s > s_ego) & (d <= corridor)
            if ahead.any():
                cand = np.flatnonzero(ahead)
                k = cand[int(np.argmin(s[cand]))]
                j = others[k]
                # Bumper-to-bumper gap: centre separation along the route minus
                # both half-lengths.
                veh_gap = float(s[k] - s_ego) - half_len - float(sim.length[j]) / 2.0
                if veh_gap < gap:
                    gap = max(veh_gap, 0.0)
                    lead_speed = float(np.hypot(sim.vx[j], sim.vy[j]))
        return gap, lead_speed

    def _accel_index(self, a_desired: float, signed_speed: float, dt: float) -> int:
        """Nearest table acceleration, never one that reverses a stopped agent.

        ``step_dynamics`` integrates a *signed* speed, so an IDM brake command
        applied to an already-stopped agent would drive it backwards down the
        road. Restrict the choice to accelerations that keep speed >= 0.
        """
        a_min = -max(signed_speed, 0.0) / dt
        allowed = np.flatnonzero(ACCELERATION_VALUES >= a_min - 1e-6)
        if allowed.size == 0:
            return int(np.argmax(ACCELERATION_VALUES))
        return int(allowed[np.argmin(np.abs(ACCELERATION_VALUES[allowed] - a_desired))])

    # ------------------------------------------------------------- lateral
    def _steer_index(self, sim: SimScene, i: int, route: Route, s_ego: float,
                     signed_speed: float, accel: float, dt: float) -> int:
        """Pick the discrete steering that best tracks the lookahead point.

        Each of the 13 candidates is forward-simulated with ``step_dynamics``'
        own update, held constant for ``steer_preview_steps``; the winner is the
        one whose final position is closest to the target. Previewing several
        steps (rather than one) accounts for the heading catching up to the slip
        angle, which is what stops the agent from crabbing and oscillating.
        """
        lookahead = float(
            np.clip(self.lookahead_time * abs(signed_speed), self.lookahead_min, self.lookahead_max)
        )
        target = route.point_at(s_ego + lookahead)

        steer = STEERING_VALUES                                  # [13]
        beta = np.tanh(0.5 * np.tan(steer))
        # Speed is shared by all candidates: the acceleration is already chosen.
        speed = np.clip(signed_speed + accel * dt, -100.0, 100.0)

        x = np.full(NUM_STEER, sim.x[i], dtype=np.float64)
        y = np.full(NUM_STEER, sim.y[i], dtype=np.float64)
        heading = np.full(NUM_STEER, sim.heading[i], dtype=np.float64)
        yaw_rate = speed * np.cos(beta) * np.tan(steer) / float(sim.length[i])
        for _ in range(max(self.steer_preview_steps, 1)):
            x += speed * np.cos(heading + beta) * dt
            y += speed * np.sin(heading + beta) * dt
            heading += yaw_rate * dt
        return int(np.argmin((x - target[0]) ** 2 + (y - target[1]) ** 2))

    # --------------------------------------------------------------- plan
    def plan(self, items: Sequence[PlanItem]) -> list:
        plans = []
        for sim, ids in items:
            dt = float(sim.dt)
            actions = np.full(len(ids), IDLE_ACTION, dtype=np.int64)
            for k, i in enumerate(ids):
                i = int(i)
                route = self._route(sim, i)
                if route is None:
                    continue
                pos = np.array([[sim.x[i], sim.y[i]]], dtype=np.float32)
                s_ego = float(route.project(pos)[0][0])
                speed_mag = float(np.hypot(sim.vx[i], sim.vy[i]))
                v_dot_h = sim.vx[i] * sim.heading_x[i] + sim.vy[i] * sim.heading_y[i]
                signed_speed = float(np.copysign(speed_mag, v_dot_h))

                a_des = self._idm_accel(sim, i, route, s_ego, max(signed_speed, 0.0))
                ia = self._accel_index(a_des, signed_speed, dt)
                is_ = self._steer_index(
                    sim, i, route, s_ego, signed_speed, float(ACCELERATION_VALUES[ia]), dt
                )
                actions[k] = ia * NUM_STEER + is_
            plans.append(actions)
        return plans

    def apply(self, items: Sequence[PlanItem], plans: list) -> None:
        for (sim, ids), actions in zip(items, plans):
            sim.step_dynamics(actions, ids)
