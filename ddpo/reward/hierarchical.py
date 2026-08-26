"""Hierarchical reward: strictly ordered outcome bands with a fallback grade.

``path_conflict`` is only an admission bit (the rollout ran); the measured
outcome decides the band:

  invalid < fallback < parked-close < closing-close < TTC < collision

With ``h_split_close`` off the two close bands merge into one (levels 0..4).
A rare event therefore always wins its GRPO group when one exists, and a group
with no event still ranks its samples by what the rollout did measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ddpo.reward import terms
from ddpo.reward.base import BaseRewardConfig, RewardAssembler, component_dict


@dataclass
class HierarchicalRewardConfig(BaseRewardConfig):
    tier0_lo: float
    tier0_hi: float
    h_fallback_lo: float
    h_fallback_hi: float
    h_park_lo: float
    h_park_hi: float
    h_close_lo: float
    h_close_hi: float
    h_ttc_lo: float
    h_ttc_hi: float
    h_collision_lo: float
    h_collision_hi: float
    h_ego_fault_extra: float
    # Close band: dmin ramp, optional split/scaling by how much the adversary
    # closed in, optional floor keeping pre-warmup contacts out of the fallback.
    h_close_dist: float
    h_close_floor: float
    h_split_close: bool
    h_close_delta: float
    h_close_gate: bool
    h_close_static_floor: float
    h_trivial_collision_floor: bool
    # Fallback grade: "spawn" ranks by spawn distance, "measured" by the dmin
    # the rollout reported (chord clearance when the rollout was skipped).
    h_fallback_mode: str
    h_init_near: float
    h_init_far: float
    h_fb_dmin_near: float
    h_fb_dmin_far: float
    h_fb_clr_near: float
    h_fb_clr_far: float

    name = "hierarchical"

    def __post_init__(self):
        super().__post_init__()
        if self.h_fallback_mode not in ("spawn", "measured"):
            raise ValueError(
                f"h_fallback_mode must be 'spawn' or 'measured', got {self.h_fallback_mode!r}"
            )
        bands = [
            self.tier0_lo, self.tier0_hi,
            self.h_fallback_lo, self.h_fallback_hi,
            self.h_park_lo, self.h_park_hi,
            self.h_close_lo, self.h_close_hi,
            self.h_ttc_lo, self.h_ttc_hi,
            self.h_collision_lo, self.h_collision_hi,
        ]
        if any(a >= b for a, b in zip(bands, bands[1:])):
            raise ValueError(
                "hierarchical reward bands must be strictly ordered: "
                "tier0 < fallback < parked < close < TTC < collision"
            )


class HierarchicalReward(RewardAssembler):
    name = "hierarchical"
    config_cls = HierarchicalRewardConfig
    requires_path_conflict = True

    def assemble(self, m: dict) -> tuple[np.ndarray, dict]:
        cfg = self.cfg

        c_overlap = np.asarray(m["init_overlap_frac"], dtype=np.float32)
        c_parking = np.asarray(m["gen_agent_is_parked"], dtype=np.float32)
        c_invalid, reason, sev_reject = terms.reject_terms(
            m,
            gen_invalid_enabled=self.gen_invalid_enabled,
            grade_scale=cfg.invalid_grade_scale,
            parked_when_disabled=True,
        )
        invalid = (c_overlap > 0.0) | (c_invalid > 0.0)
        admitted = np.asarray(m["path_conflict"], dtype=np.float32) > 0.0
        valid = ~invalid

        r_ttc = terms.ttc_grade(m, cfg.ttc_tau)
        dmin = np.asarray(m["ego_adv_min_dist_warmup"], dtype=np.float32)
        r_close = (1.0 - terms.smoothstep(dmin, cfg.h_close_floor, cfg.h_close_dist)).astype(
            np.float32
        )

        collision = np.asarray(m["ego_collision"], dtype=np.float32)
        ctime = np.asarray(m["ego_collision_time"], dtype=np.float32)
        r_collision = terms.collision_ramp(m, cfg.collision_warmup, cfg.collision_window)
        c_trivial = terms.trivial_collision(m, cfg.trivial_collision_t)
        ego_fault = np.asarray(m["ego_fault_collision"], dtype=np.float32)

        tier4 = valid & admitted & (r_collision > 0.0)
        tier3 = valid & admitted & ~tier4 & (r_ttc > 0.0)
        close_ok = np.isfinite(dmin) & (dmin < cfg.h_close_dist)
        if cfg.h_trivial_collision_floor:
            # A pre-warmup contact earns no collision credit, but its large
            # post-warmup dmin must not drop it below a quiet sample.
            close_ok = close_ok | ((collision > 0.0) & (ctime < cfg.collision_warmup))
        tier2 = valid & admitted & ~tier4 & ~tier3 & close_ok

        closed_in = terms.closed_in(m)
        if cfg.h_split_close:
            parked = tier2 & (closed_in <= cfg.h_close_delta)
            tier2 = tier2 & ~parked
        else:
            parked = np.zeros_like(tier2)
        tier1 = valid & ~(parked | tier2 | tier3 | tier4)

        sev0 = np.maximum(c_overlap, sev_reject).astype(np.float32)
        r0 = cfg.tier0_hi + (cfg.tier0_lo - cfg.tier0_hi) * np.clip(sev0, 0.0, 1.0)

        # Lane quality only orders within a band, so it can never demote a real
        # event below a weaker one.
        c_spawn_lane, c_goal_lane = terms.lane_costs(m, cfg.lane_soft, cfg.lane_hard)
        c_lane = (c_spawn_lane + c_goal_lane).astype(np.float32)
        lane_grade_cost = (0.5 * cfg.lane_penalty * c_lane).astype(np.float32)

        clr = np.asarray(m["path_conflict_dist"], dtype=np.float32)
        if cfg.h_fallback_mode == "measured":
            init_grade = np.where(
                np.isfinite(dmin),
                1.0 - terms.smoothstep(dmin, cfg.h_fb_dmin_near, cfg.h_fb_dmin_far),
                1.0 - terms.smoothstep(clr, cfg.h_fb_clr_near, cfg.h_fb_clr_far),
            )
            init_grade = np.where(np.isfinite(clr), init_grade, 0.0)
        else:
            init_grade = 1.0 - terms.smoothstep(
                m["ego_adv_spawn_dist"], cfg.h_init_near, cfg.h_init_far
            )
            init_grade = np.where(np.isfinite(m["ego_adv_spawn_dist"]), init_grade, 0.0)
        if cfg.h_close_gate:
            gate = np.clip(closed_in / max(cfg.h_close_delta, 1e-6), 0.0, 1.0)
            r_close = r_close * (
                cfg.h_close_static_floor + (1.0 - cfg.h_close_static_floor) * gate
            )

        g1 = np.clip(init_grade - lane_grade_cost, 0.0, 1.0)
        g2 = np.clip(r_close - lane_grade_cost, 0.0, 1.0)
        g3 = np.clip(r_ttc - lane_grade_cost, 0.0, 1.0)
        g4 = np.clip(r_collision - lane_grade_cost, 0.0, 1.0)

        r1 = cfg.h_fallback_lo + (cfg.h_fallback_hi - cfg.h_fallback_lo) * g1
        r_park = cfg.h_park_lo + (cfg.h_park_hi - cfg.h_park_lo) * g2
        r2 = cfg.h_close_lo + (cfg.h_close_hi - cfg.h_close_lo) * g2
        r3 = cfg.h_ttc_lo + (cfg.h_ttc_hi - cfg.h_ttc_lo) * g3
        r4 = (
            cfg.h_collision_lo
            + (cfg.h_collision_hi - cfg.h_collision_lo) * g4
            + cfg.h_ego_fault_extra * ego_fault
        )

        if cfg.h_split_close:
            conds = [invalid, tier1, parked, tier2, tier3]
            vals = [r0, r1, r_park, r2, r3]
            levels = [0.0, 1.0, 2.0, 3.0, 4.0]
            top = 5.0
        else:
            conds = [invalid, tier1, tier2, tier3]
            vals = [r0, r1, r2, r3]
            levels = [0.0, 1.0, 2.0, 3.0]
            top = 4.0
        total = np.select(conds, vals, default=r4).astype(np.float32)
        tier = np.select(conds, levels, default=top).astype(np.float32)

        components = component_dict(
            r_ttc=np.where(admitted, r_ttc, 0.0).astype(np.float32),
            r_approach=np.where(tier2 | parked, r_close, 0.0).astype(np.float32),
            # Kept TTC-based, so r_risk > .5 still means TTC < tau/2.
            r_risk=np.where(admitted, r_ttc, 0.0).astype(np.float32),
            r_collision=r_collision,
            r_bonus=np.where(tier4, (r4 - cfg.h_collision_lo).astype(np.float32), 0.0).astype(
                np.float32
            ),
            criticality=np.where(tier3 | tier4, r_ttc, 0.0).astype(np.float32),
            c_spawn_lane=c_spawn_lane,
            c_goal_lane=c_goal_lane,
            c_parking=c_parking,
            c_invalid=c_invalid,
            c_invalid_sev=sev_reject,
            c_invalid_reason=reason,
            c_trivial=c_trivial,
            c_overlap=c_overlap,
            constraint=(cfg.lane_penalty * c_lane).astype(np.float32),
        )
        components.update(
            tier=tier,
            c_path_conflict=admitted.astype(np.float32),
            c_path_dist=clr,
        )
        return total, components
