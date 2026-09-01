"""Centerline-only PDM-Closed adaptation for the local rollout simulator.

Each controlled agent gets one lane-graph route from ``IDMPlanner``.  The
planner unrolls a Cartesian product of IDM target speeds, standstill gaps and
headways against constant-velocity traffic, scores the proposals by collision,
short-horizon TTC and route progress, and applies the best proposal's first
action through ``SimScene.step_dynamics``.

This is deliberately the closed-loop, rule-based half of PDM: there are no
learned weights.  nuPlan-only score terms that need lane polygons, roadblocks,
traffic lights or speed limits are absent because generated scenes carry lane
centerlines and connectivity only.  Comfort is also intentionally unscored.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..geometry import _corners
from ..routes import Route
from ..world import (
    ACCELERATION_VALUES,
    NUM_STEER,
    STEERING_VALUES,
    TYPE_PEDESTRIAN,
    SimScene,
)
from .base import PlanItem, require
from .idm import IDLE_ACTION, IDMPlanner, steering_indices


def _pairwise_overlap(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Oriented-box SAT for every pair in ``boxes_a[P] x boxes_b[M]``."""
    p, m = len(boxes_a), len(boxes_b)
    if m == 0:
        return np.zeros((p, 0), dtype=bool)

    def axes(boxes):
        edge = boxes[:, 1] - boxes[:, 0]
        out = np.stack([edge, np.stack([-edge[:, 1], edge[:, 0]], axis=-1)], axis=1)
        return out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-9)

    axes_a = axes(boxes_a)  # [P, 2, 2]
    axes_b = axes(boxes_b)  # [M, 2, 2]

    a_on_a = np.einsum("pka,pca->pkc", axes_a, boxes_a)
    b_on_a = np.einsum("pka,mca->pmkc", axes_a, boxes_b)
    sep_a = (
        (a_on_a.min(-1)[:, None] > b_on_a.max(-1))
        | (b_on_a.min(-1) > a_on_a.max(-1)[:, None])
    ).any(-1)

    a_on_b = np.einsum("mka,pca->pmkc", axes_b, boxes_a)
    b_on_b = np.einsum("mka,mca->mkc", axes_b, boxes_b)
    sep_b = (
        (a_on_b.min(-1) > b_on_b.max(-1)[None])
        | (b_on_b.min(-1)[None] > a_on_b.max(-1))
    ).any(-1)
    return ~(sep_a | sep_b)


class PDMPlanner(IDMPlanner):
    """Select the best finite-horizon proposal from several IDM policies."""

    def __init__(self, planner_cfg, *, role: str, device: str | None = None):
        super().__init__(planner_cfg, role=role, device=device)
        cfg = self._require("proposal")
        self.horizon_steps = int(require(cfg, self.name, "horizon_steps", "proposal.horizon_steps"))
        self.ttc_steps = int(require(cfg, self.name, "ttc_steps", "proposal.ttc_steps"))
        self.progress_weight = float(
            require(cfg, self.name, "progress_weight", "proposal.progress_weight")
        )
        self.ttc_weight = float(require(cfg, self.name, "ttc_weight", "proposal.ttc_weight"))

        def values(key: str) -> np.ndarray:
            out = np.asarray(require(cfg, self.name, key, f"proposal.{key}"), dtype=np.float64)
            if out.ndim != 1 or out.size == 0:
                raise ValueError(f"cfgs/planner/{self.name}.yaml: proposal.{key} must be non-empty")
            if np.any(out <= 0.0):
                raise ValueError(f"cfgs/planner/{self.name}.yaml: proposal.{key} must be positive")
            return out

        target, gap, headway = np.meshgrid(
            values("target_speeds"),
            values("min_gaps"),
            values("headway_times"),
            indexing="ij",
        )
        self.policy_target_speed = target.ravel()
        self.policy_min_gap = gap.ravel()
        self.policy_headway = headway.ravel()

        if self.horizon_steps < 1:
            raise ValueError("proposal.horizon_steps must be >= 1")
        if self.ttc_steps < 0:
            raise ValueError("proposal.ttc_steps must be >= 0")
        if self.progress_weight < 0.0 or self.ttc_weight < 0.0:
            raise ValueError("proposal score weights must be non-negative")
        if self.progress_weight + self.ttc_weight == 0.0:
            raise ValueError("at least one proposal score weight must be positive")

    def _proposal_actions(
        self, sim: SimScene, i: int, route: Route
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return first actions, scores, first-contact times and progress."""
        count = len(self.policy_target_speed)
        dt = float(sim.dt)
        x = np.full(count, sim.x[i], dtype=np.float64)
        y = np.full(count, sim.y[i], dtype=np.float64)
        heading = np.full(count, sim.heading[i], dtype=np.float64)
        speed_mag = float(np.hypot(sim.vx[i], sim.vy[i]))
        v_dot_h = sim.vx[i] * sim.heading_x[i] + sim.vy[i] * sim.heading_y[i]
        speed = np.full(count, np.copysign(speed_mag, v_dot_h), dtype=np.float64)
        length = float(sim.length[i])
        width = float(sim.width[i])
        s0 = float(route.project(np.array([[sim.x[i], sim.y[i]]]))[0][0])

        others = np.flatnonzero(
            (~sim.removed) & (sim.ptype != TYPE_PEDESTRIAN) & (np.arange(sim.n) != i)
        )
        ox0 = sim.x[others].astype(np.float64)
        oy0 = sim.y[others].astype(np.float64)
        ovx = sim.vx[others].astype(np.float64)
        ovy = sim.vy[others].astype(np.float64)
        oheading = sim.heading[others].astype(np.float64)
        olength = sim.length[others].astype(np.float64)
        owidth = sim.width[others].astype(np.float64)
        ospeed = np.hypot(ovx, ovy)
        corridor = (owidth + width) / 2.0 + self.lateral_margin

        first_actions = np.full(count, IDLE_ACTION, dtype=np.int64)
        first_contact = np.full(count, np.inf, dtype=np.float64)
        progress = np.zeros(count, dtype=np.float64)
        total_steps = self.horizon_steps + self.ttc_steps

        for step in range(total_steps):
            t = step * dt
            ox, oy = ox0 + ovx * t, oy0 + ovy * t
            s = route.project(np.stack([x, y], axis=-1))[0]

            if len(others):
                os, od = route.project(np.stack([ox, oy], axis=-1))
                dist2 = (ox[None] - x[:, None]) ** 2 + (oy[None] - y[:, None]) ** 2
                valid = (
                    (os[None] > s[:, None])
                    & (od[None] <= corridor[None])
                    & (dist2 <= self.lead_search_radius ** 2)
                )
                gaps = os[None] - s[:, None] - length / 2.0 - olength[None] / 2.0
                gaps = np.where(valid, np.maximum(gaps, 0.0), np.inf)
                lead_idx = np.argmin(gaps, axis=1)
                gap = gaps[np.arange(count), lead_idx]
                lead_speed = np.where(np.isfinite(gap), ospeed[lead_idx], 0.0)
            else:
                gap = np.full(count, np.inf)
                lead_speed = np.zeros(count)

            forward_speed = np.maximum(speed, 0.0)
            desired_gap = (
                self.policy_min_gap
                + forward_speed * self.policy_headway
                + forward_speed * (forward_speed - lead_speed)
                / (2.0 * np.sqrt(self.max_accel * self.comfort_decel))
            )
            accel = self.max_accel * (
                1.0
                - (forward_speed / self.policy_target_speed) ** self.accel_exponent
                - (desired_gap / np.maximum(gap, self.policy_min_gap)) ** 2
            )

            min_accel = -forward_speed / dt
            allowed = ACCELERATION_VALUES[None] >= min_accel[:, None] - 1e-6
            accel_idx = np.argmin(
                np.where(allowed, np.abs(ACCELERATION_VALUES[None] - accel[:, None]), np.inf),
                axis=1,
            )
            discrete_accel = ACCELERATION_VALUES[accel_idx]
            steer_idx = steering_indices(
                route,
                s,
                x,
                y,
                heading,
                length,
                speed,
                discrete_accel,
                dt,
                lookahead_time=self.lookahead_time,
                lookahead_min=self.lookahead_min,
                lookahead_max=self.lookahead_max,
                preview_steps=self.steer_preview_steps,
            )
            actions = accel_idx * NUM_STEER + steer_idx
            if step == 0:
                first_actions[:] = actions

            steer = STEERING_VALUES[steer_idx]
            speed = np.clip(speed + discrete_accel * dt, -100.0, 100.0)
            beta = np.tanh(0.5 * np.tan(steer))
            yaw_rate = speed * np.cos(beta) * np.tan(steer) / length
            vx = speed * np.cos(heading + beta)
            vy = speed * np.sin(heading + beta)
            x += vx * dt
            y += vy * dt
            heading += yaw_rate * dt

            if len(others):
                collision_t = (step + 1) * dt
                obstacle_boxes = _corners(
                    ox0 + ovx * collision_t,
                    oy0 + ovy * collision_t,
                    oheading,
                    olength,
                    owidth,
                )
                proposal_boxes = _corners(x, y, heading, length, width)
                hit = _pairwise_overlap(proposal_boxes, obstacle_boxes).any(axis=1)
                first_contact[np.isinf(first_contact) & hit] = collision_t

            if step + 1 == self.horizon_steps:
                progress[:] = route.project(np.stack([x, y], axis=-1))[0] - s0

        horizon_time = self.horizon_steps * dt
        total_time = total_steps * dt
        collision_free = first_contact > horizon_time + 1e-9
        ttc_score = np.where(
            np.isinf(first_contact), 1.0, np.clip(first_contact / total_time, 0.0, 1.0)
        )
        progress_scale = float(self.policy_target_speed.max()) * horizon_time
        progress_score = np.clip(progress / progress_scale, 0.0, 1.0)
        scores = collision_free * (
            self.progress_weight * progress_score + self.ttc_weight * ttc_score
        )
        return first_actions, scores, first_contact, progress

    def plan(self, items: Sequence[PlanItem]) -> list:
        plans = []
        for sim, ids in items:
            actions = np.full(len(ids), IDLE_ACTION, dtype=np.int64)
            for k, raw_i in enumerate(ids):
                i = int(raw_i)
                route = self._route(sim, i)
                if route is None:
                    continue
                first, scores, contact, _ = self._proposal_actions(sim, i, route)
                if np.any(scores > 0.0):
                    best = int(np.argmax(scores))
                else:
                    best = int(np.argmax(contact))
                actions[k] = first[best]
            plans.append(actions)
        return plans
