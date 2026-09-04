"""Centerline-only PDM-Closed adaptation for the local rollout simulator.

Each controlled agent gets one lane-graph route from ``IDMPlanner``.  The
planner unrolls PDM-Closed's proposal set -- IDM target speeds crossed with
lateral centerline offsets -- against constant-velocity traffic, scores the
proposals by collision, short-horizon TTC and route progress, and applies the
best proposal's first action through ``SimScene.step_dynamics``.

The lateral axis is load-bearing.  With longitudinal proposals only, every
proposal tracks the identical path and the selection can do no more than choose
a speed, which makes the planner a strictly weaker IDM (measured: it lost to
``idm`` on every column of the benchmark).  The offsets are what let a proposal
steer around an obstacle rather than only brake for it.

This is deliberately the closed-loop, rule-based half of PDM: there are no
learned weights.  nuPlan-only score terms that need lane polygons, roadblocks,
traffic lights or speed limits are absent because generated scenes carry lane
centerlines and connectivity only.  Comfort is also intentionally unscored.

The TTC subscore is currently inert.  PDM computes it from a SEPARATE
constant-velocity extrapolation of the ego, independent of the proposal's own
rollout; here both come from the same simulation, so once the collision gate
covers the whole simulated window every surviving proposal scores ``ttc = 1.0``.
``ttc_steps`` still matters -- it is how far past ``horizon_steps`` the gate
looks -- but ``ttc_weight`` no longer changes any ranking.  Restoring a real TTC
term means adding that second extrapolation, not widening this one.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..geometry import _corners, sat_pairs
from ..world import (
    ACCELERATION_VALUES,
    NUM_STEER,
    STEERING_VALUES,
    TYPE_PEDESTRIAN,
    SimScene,
)
from .base import PlanItem, require
from .idm import (
    IDLE_ACTION,
    MAX_TABLE_DECEL,
    IDMPlanner,
    steering_from_target,
    steering_targets_packed,
)


def _pairwise_overlap_batched(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Oriented-box SAT for ``boxes_a[..., P, 4, 2]`` x ``boxes_b[..., M, 4, 2]``.

    The leading axes broadcast together, so one call covers every agent's whole
    proposal-vs-obstacle grid. Axes, normalisation and the inclusive-touch
    convention are the same four-axis test used everywhere else in ``sim``.
    """
    if boxes_b.shape[-3] == 0:
        return np.zeros(boxes_a.shape[:-2] + (0,), dtype=bool)

    def axes(boxes):
        edge = boxes[..., 1, :] - boxes[..., 0, :]
        out = np.stack(
            [edge, np.stack([-edge[..., 1], edge[..., 0]], axis=-1)], axis=-2
        )
        return out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-9)

    axes_a = axes(boxes_a)  # [..., P, 2, 2]
    axes_b = axes(boxes_b)  # [..., M, 2, 2]

    a_on_a = np.einsum("...pka,...pca->...pkc", axes_a, boxes_a)
    b_on_a = np.einsum("...pka,...mca->...pmkc", axes_a, boxes_b)
    sep_a = (
        (a_on_a.min(-1)[..., None, :] > b_on_a.max(-1))
        | (b_on_a.min(-1) > a_on_a.max(-1)[..., None, :])
    ).any(-1)

    a_on_b = np.einsum("...mka,...pca->...pmkc", axes_b, boxes_a)
    b_on_b = np.einsum("...mka,...mca->...mkc", axes_b, boxes_b)
    sep_b = (
        (a_on_b.min(-1) > np.expand_dims(b_on_b.max(-1), -3))
        | (np.expand_dims(b_on_b.min(-1), -3) > a_on_b.max(-1))
    ).any(-1)
    return ~(sep_a | sep_b)


def _pairwise_overlap(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Oriented-box SAT for every pair in ``boxes_a[P] x boxes_b[M]``."""
    return _pairwise_overlap_batched(boxes_a, boxes_b)


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
        self.offset_weight = float(
            require(cfg, self.name, "offset_weight", "proposal.offset_weight")
        )

        def values(key: str) -> np.ndarray:
            out = np.asarray(require(cfg, self.name, key, f"proposal.{key}"), dtype=np.float64)
            if out.ndim != 1 or out.size == 0:
                raise ValueError(f"cfgs/planner/{self.name}.yaml: proposal.{key} must be non-empty")
            if np.any(out <= 0.0):
                raise ValueError(f"cfgs/planner/{self.name}.yaml: proposal.{key} must be positive")
            return out

        # Lateral offsets may be zero or negative, so they get their own
        # validator rather than the positive-only one above.
        offsets = np.asarray(
            require(cfg, self.name, "lateral_offsets", "proposal.lateral_offsets"),
            dtype=np.float64,
        )
        if offsets.ndim != 1 or offsets.size == 0:
            raise ValueError(
                f"cfgs/planner/{self.name}.yaml: proposal.lateral_offsets must be non-empty"
            )

        target, gap, headway, offset = np.meshgrid(
            values("target_speeds"),
            values("min_gaps"),
            values("headway_times"),
            offsets,
            indexing="ij",
        )
        self.policy_target_speed = target.ravel()
        self.policy_min_gap = gap.ravel()
        self.policy_headway = headway.ravel()
        self.policy_offset = offset.ravel()

        if self.horizon_steps < 1:
            raise ValueError("proposal.horizon_steps must be >= 1")
        if self.ttc_steps < 0:
            raise ValueError("proposal.ttc_steps must be >= 0")
        if min(self.progress_weight, self.ttc_weight, self.offset_weight) < 0.0:
            raise ValueError("proposal score weights must be non-negative")
        if self.progress_weight + self.ttc_weight == 0.0:
            raise ValueError("at least one proposal score weight must be positive")

    # The proposal rollout, for EVERY driven agent at once. Previously this was a
    # Python loop over agents around a 40-step loop over the horizon, and it cost
    # 73.2 s of an 81.9 s single-process DDPO iteration (89.3%, batch 128, pdm
    # SUT). The horizon loop stays -- it is genuinely sequential -- but each of
    # its 40 iterations now works on an [agents, proposals] grid instead of one
    # agent's 15 proposals, so the numpy call count drops by the agent count.
    #
    # Every rule is unchanged: the per-proposal braking-distance gate, the
    # offset-following corridor, the lead being the smallest arc length ahead,
    # the collision gate covering horizon + ttc, and the scoring.

    def _proposal_actions(self, pack, rows, st, item_of, agent_of, dt):
        """[A, P] first actions, scores, first-contact times, progress, safe mask."""
        count = len(self.policy_target_speed)
        a_n = len(rows)
        q = np.arange(a_n)
        n_slots = st.x.shape[1]

        ex = st.x[item_of, agent_of].astype(np.float64)
        ey = st.y[item_of, agent_of].astype(np.float64)
        length = st.length[item_of, agent_of].astype(np.float64)
        width = st.width[item_of, agent_of].astype(np.float64)
        speed_mag = np.hypot(st.vx[item_of, agent_of], st.vy[item_of, agent_of])
        v_dot_h = (
            st.vx[item_of, agent_of] * st.hx[item_of, agent_of]
            + st.vy[item_of, agent_of] * st.hy[item_of, agent_of]
        )
        x = np.repeat(ex[:, None], count, axis=1)
        y = np.repeat(ey[:, None], count, axis=1)
        heading = np.repeat(st.heading[item_of, agent_of].astype(np.float64)[:, None], count, axis=1)
        speed = np.repeat(np.copysign(speed_mag, v_dot_h).astype(np.float64)[:, None], count, axis=1)

        s0, _ = pack.project(rows, np.stack([ex, ey], axis=-1))

        # Other agents, as a padded [A, n_slots] block with a validity mask. PDM
        # deliberately has NO distance gate here: the collision check below must
        # see every box, so the mask is only "is this a real, non-pedestrian,
        # not-me agent".
        omask = st.cand[item_of] & (np.arange(n_slots)[None, :] != agent_of[:, None])
        ox0 = st.x[item_of].astype(np.float64)
        oy0 = st.y[item_of].astype(np.float64)
        ovx = st.vx[item_of].astype(np.float64)
        ovy = st.vy[item_of].astype(np.float64)
        oheading = st.heading[item_of].astype(np.float64)
        olength = st.length[item_of].astype(np.float64)
        owidth = st.width[item_of].astype(np.float64)
        ospeed = np.hypot(ovx, ovy)
        corridor = (owidth + width[:, None]) / 2.0 + self.lateral_margin
        any_other = omask.any()
        pr, pc = np.nonzero(omask)

        first_actions = np.full((a_n, count), IDLE_ACTION, dtype=np.int64)
        first_contact = np.full((a_n, count), np.inf)
        progress = np.zeros((a_n, count))
        total_steps = self.horizon_steps + self.ttc_steps

        tgt_speed = self.policy_target_speed[None, :]
        min_gap_p = self.policy_min_gap[None, :]
        headway_p = self.policy_headway[None, :]
        offset_p = self.policy_offset[None, :]
        offset_flat = np.tile(self.policy_offset, a_n)
        rows_prop = np.repeat(rows, count)

        hd_prop = np.hypot(length / 2.0, width / 2.0)
        hd_obs = np.hypot(olength / 2.0, owidth / 2.0)
        if any_other:
            # Obstacle boxes are a pure translation over the horizon: heading and
            # extents are constant, so the corners are rotated once here and each
            # step only adds the displacement.
            base_box_of = np.zeros((a_n, n_slots, 4, 2))
            base_box_of[pr, pc] = _corners(
                ox0[pr, pc], oy0[pr, pc], oheading[pr, pc],
                olength[pr, pc], owidth[pr, pc],
            )

        for step in range(total_steps):
            t = step * dt
            ox, oy = ox0 + ovx * t, oy0 + ovy * t
            s, _ = pack.project(rows_prop, np.stack([x.ravel(), y.ravel()], axis=-1))
            s = s.reshape(a_n, count)

            forward_speed = np.maximum(speed, 0.0)

            if any_other:
                dist2 = (ox[:, None, :] - x[:, :, None]) ** 2 + (
                    oy[:, None, :] - y[:, :, None]
                ) ** 2
                radius = np.maximum(
                    self.lead_search_radius,
                    forward_speed ** 2 / (2.0 * MAX_TABLE_DECEL)
                    + self.min_gap + length[:, None],
                )
                in_radius = dist2 <= (radius ** 2)[:, :, None]
                # Only project what could still qualify as a lead. An agent
                # outside EVERY proposal's radius fails ``valid`` regardless of
                # its arc length, so skipping its projection cannot change the
                # chosen lead -- the same gate-before-project order the scalar
                # IDM lead search has always used.
                reach = omask & in_radius.any(axis=1)
                qr, qc = np.nonzero(reach)
                os = np.zeros((a_n, n_slots)); od = np.zeros((a_n, n_slots))
                if len(qr):
                    os[qr, qc], od[qr, qc] = pack.project(
                        rows[qr], np.stack([ox[qr, qc], oy[qr, qc]], axis=-1)
                    )
                valid = (
                    reach[:, None, :]
                    & (os[:, None, :] > s[:, :, None])
                    & (np.abs(od[:, None, :] - offset_p[..., None]) <= corridor[:, None, :])
                    & in_radius
                )
                gaps = (
                    os[:, None, :] - s[:, :, None]
                    - length[:, None, None] / 2.0 - olength[:, None, :] / 2.0
                )
                gaps = np.where(valid, np.maximum(gaps, 0.0), np.inf)
                lead_idx = np.argmin(gaps, axis=2)
                gap = np.take_along_axis(gaps, lead_idx[:, :, None], axis=2)[:, :, 0]
                lead_speed = np.where(
                    np.isfinite(gap), np.take_along_axis(ospeed, lead_idx, axis=1), 0.0
                )
            else:
                gap = np.full((a_n, count), np.inf)
                lead_speed = np.zeros((a_n, count))

            desired_gap = (
                min_gap_p
                + forward_speed * headway_p
                + forward_speed * (forward_speed - lead_speed)
                / (2.0 * np.sqrt(self.max_accel * self.comfort_decel))
            )
            accel = self.max_accel * (
                1.0
                - (forward_speed / tgt_speed) ** self.accel_exponent
                - (desired_gap / np.maximum(gap, min_gap_p)) ** 2
            )

            min_accel = -forward_speed / dt
            allowed = ACCELERATION_VALUES[None, None, :] >= min_accel[:, :, None] - 1e-6
            accel_idx = np.argmin(
                np.where(
                    allowed,
                    np.abs(ACCELERATION_VALUES[None, None, :] - accel[:, :, None]),
                    np.inf,
                ),
                axis=2,
            )
            discrete_accel = ACCELERATION_VALUES[accel_idx]
            target = steering_targets_packed(
                pack, rows_prop, s.ravel(), speed.ravel(),
                lookahead_time=self.lookahead_time,
                lookahead_min=self.lookahead_min,
                lookahead_max=self.lookahead_max,
                lateral_offset=offset_flat,
            )
            steer_idx = steering_from_target(
                target, x.ravel(), y.ravel(), heading.ravel(),
                np.repeat(length, count), speed.ravel(),
                discrete_accel.ravel().astype(np.float64), dt,
                preview_steps=self.steer_preview_steps,
            ).reshape(a_n, count)

            if step == 0:
                first_actions[:] = accel_idx * NUM_STEER + steer_idx

            steer = STEERING_VALUES[steer_idx]
            speed = np.clip(speed + discrete_accel * dt, -100.0, 100.0)
            beta = np.tanh(0.5 * np.tan(steer))
            yaw_rate = speed * np.cos(beta) * np.tan(steer) / length[:, None]
            vx = speed * np.cos(heading + beta)
            vy = speed * np.sin(heading + beta)
            x = x + vx * dt
            y = y + vy * dt
            heading = heading + yaw_rate * dt

            if any_other:
                collision_t = (step + 1) * dt
                onx = ox0 + ovx * collision_t
                ony = oy0 + ovy * collision_t
                # Broad phase: two oriented boxes cannot overlap while their
                # centres are farther apart than the sum of their half-diagonals,
                # so the SAT only runs on the handful of pairs that survive this.
                # Exact -- the gate is a bound on the SAT, not an approximation.
                cdx = x[:, :, None] - onx[:, None, :]
                cdy = y[:, :, None] - ony[:, None, :]
                reach2 = (hd_prop[:, None, None] + hd_obs[:, None, :]) ** 2
                touching = omask[:, None, :] & ((cdx * cdx + cdy * cdy) <= reach2)
                hit = np.zeros((a_n, count), dtype=bool)
                br, bp, bm = np.nonzero(touching)
                if len(br):
                    prop_boxes = _corners(
                        x[br, bp], y[br, bp], heading[br, bp],
                        length[br], width[br],
                    )
                    obs_boxes = base_box_of[br, bm] + np.stack(
                        [ovx[br, bm], ovy[br, bm]], axis=-1
                    )[:, None, :] * collision_t
                    k = len(br)
                    pair_hit = sat_pairs(
                        np.concatenate([prop_boxes, obs_boxes]),
                        np.arange(k), np.arange(k) + k,
                    )
                    if pair_hit.any():
                        hit[br[pair_hit], bp[pair_hit]] = True
                first_contact = np.where(
                    np.isinf(first_contact) & hit, collision_t, first_contact
                )

            if step + 1 == self.horizon_steps:
                s_now, _ = pack.project(
                    rows_prop, np.stack([x.ravel(), y.ravel()], axis=-1)
                )
                progress = s_now.reshape(a_n, count) - s0[:, None]

        horizon_time = self.horizon_steps * dt
        total_time = total_steps * dt
        collision_free = np.isinf(first_contact)
        ttc_score = np.where(
            collision_free, 1.0, np.clip(first_contact / total_time, 0.0, 1.0)
        )
        progress_scale = float(self.policy_target_speed.max()) * horizon_time
        progress_score = np.clip(progress / progress_scale, 0.0, 1.0)
        scores = collision_free * (
            self.progress_weight * progress_score
            + self.ttc_weight * ttc_score
            - self.offset_weight * np.abs(offset_p)
        )
        return first_actions, scores, first_contact, progress, collision_free

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

        keep = rows >= 0
        rows, item_of, agent_of, slot_of = (
            rows[keep], item_of[keep], agent_of[keep], slot_of[keep]
        )
        if not len(rows):
            return plans

        first, scores, contact, _, safe = self._proposal_actions(
            pack, rows, st, item_of, agent_of, dt
        )
        # Rank only the proposals that clear the horizon: the offset penalty can
        # push a safe-but-slow one below zero, so a score sign test would drop
        # into the trapped branch by accident. Trapped agents (nothing avoids
        # contact) buy the most time instead.
        has_safe = safe.any(axis=1)
        best_safe = np.argmax(np.where(safe, scores, -np.inf), axis=1)
        best_trapped = np.argmax(contact, axis=1)
        best = np.where(has_safe, best_safe, best_trapped)
        actions = np.take_along_axis(first, best[:, None], axis=1)[:, 0]

        for t in range(len(items)):
            sel = item_of == t
            if sel.any():
                plans[t][slot_of[sel]] = actions[sel]
        return plans
