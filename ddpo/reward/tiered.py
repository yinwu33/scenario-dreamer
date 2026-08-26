"""Tiered reward: three disjoint bands keyed on the pre-rollout path geometry.

  tier 2  ego/adversary spawn->goal chords conflict (the scene was rolled out):
          [tier2_floor, 1] = clip(criticality - lane cost, 0, 1) + collision bonus
  tier 1  no conflict: [tier1_lo, tier1_hi], graded by chord clearance blended
          with the adversary's spawn distance to the ego
  tier 0  spawn interpenetration or condition violation: [tier0_lo, tier0_hi],
          graded by severity
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ddpo.reward import terms
from ddpo.reward.base import (
    AnnealedApproachAssembler,
    ApproachRewardConfig,
    component_dict,
)


@dataclass
class TieredRewardConfig(ApproachRewardConfig):
    collision_bonus: float
    ego_fault_bonus: float
    tier2_floor: float
    tier1_lo: float
    tier1_hi: float
    tier0_lo: float
    tier0_hi: float
    tier1_path_near: float
    tier1_path_far: float
    tier1_ego_near: float
    tier1_ego_far: float
    tier1_path_weight: float
    tier1_ego_weight: float

    name = "tiered"


class TieredReward(AnnealedApproachAssembler):
    name = "tiered"
    config_cls = TieredRewardConfig
    requires_path_conflict = True

    def assemble(self, m: dict) -> tuple[np.ndarray, dict]:
        cfg = self.cfg

        # --- membership (priority order 0 -> 1 -> 2) ------------------------
        c_overlap = np.asarray(m["init_overlap_frac"], dtype=np.float32)
        c_parking = np.asarray(m["gen_agent_is_parked"], dtype=np.float32)
        c_invalid, reason, sev_reject = terms.reject_terms(
            m,
            gen_invalid_enabled=self.gen_invalid_enabled,
            grade_scale=cfg.invalid_grade_scale,
            parked_when_disabled=False,
        )
        degenerate = (c_overlap > 0.0) | (c_invalid > 0.0)
        conflict = np.asarray(m["path_conflict"], dtype=np.float32) > 0
        tier2 = conflict & ~degenerate
        tier1 = ~conflict & ~degenerate

        # --- tier 0 --------------------------------------------------------
        sev0 = np.maximum(c_overlap, sev_reject).astype(np.float32)
        r_tier0 = cfg.tier0_hi + (cfg.tier0_lo - cfg.tier0_hi) * np.clip(sev0, 0.0, 1.0)

        # --- tier 1 --------------------------------------------------------
        g_path = 1.0 - terms.smoothstep(
            m["path_conflict_dist"], cfg.tier1_path_near, cfg.tier1_path_far
        )
        # Car-following was denied tier 2 and its chord clearance is ~0 by
        # construction: crediting it here would re-open that shortcut.
        g_path = np.where(np.asarray(m["path_following"]) > 0, 0.0, g_path)
        g_ego = 1.0 - terms.smoothstep(
            m["ego_adv_spawn_dist"], cfg.tier1_ego_near, cfg.tier1_ego_far
        )
        w_sum = max(cfg.tier1_path_weight + cfg.tier1_ego_weight, 1e-6)
        t1_grade = (
            (cfg.tier1_path_weight * g_path + cfg.tier1_ego_weight * g_ego) / w_sum
        ).astype(np.float32)
        # No adversary in the scene -> infinite chord distance.
        t1_grade = np.where(np.isfinite(m["path_conflict_dist"]), t1_grade, 0.0)
        r_tier1 = cfg.tier1_lo + (cfg.tier1_hi - cfg.tier1_lo) * t1_grade

        # --- tier 2 --------------------------------------------------------
        r_ttc = terms.ttc_grade(m, cfg.ttc_tau)
        # The approach term is fenced inside tier 2: reaching it already
        # requires conflicting geometry, so it cannot be farmed by parking.
        r_approach = terms.approach_grade(
            m,
            cfg.approach_d_safe,
            cfg.approach_d_scale,
            cfg.approach_close_delta,
            cfg.approach_close_scale,
        )
        r_risk = (
            1.0 - (1.0 - r_ttc) * (1.0 - self._approach_coef * r_approach)
        ).astype(np.float32)
        criticality_raw = (cfg.risk_coef * r_risk).astype(np.float32)

        r_collision = terms.collision_ramp(m, cfg.collision_warmup, cfg.collision_window)
        c_trivial = terms.trivial_collision(m, cfg.trivial_collision_t)
        ego_fault = np.asarray(m["ego_fault_collision"], dtype=np.float32)
        r_bonus = (
            r_collision * (cfg.collision_bonus + cfg.ego_fault_bonus * ego_fault)
        ).astype(np.float32)

        c_spawn_lane, c_goal_lane = terms.lane_costs(m, cfg.lane_soft, cfg.lane_hard)
        constraint = (cfg.lane_penalty * (c_spawn_lane + c_goal_lane)).astype(np.float32)
        # Floored into its own band: the lane cost orders within tier 2 instead
        # of leaking across the boundary.
        core = np.clip(criticality_raw - constraint, 0.0, 1.0).astype(np.float32)
        r_tier2 = cfg.tier2_floor + (1.0 - cfg.tier2_floor) * core + r_bonus

        total = np.select([degenerate, tier1], [r_tier0, r_tier1], default=r_tier2)
        total = total.astype(np.float32)

        components = component_dict(
            r_ttc=np.where(tier2, r_ttc, 0.0).astype(np.float32),
            r_approach=np.where(tier2, r_approach, 0.0).astype(np.float32),
            r_risk=np.where(tier2, r_risk, 0.0).astype(np.float32),
            r_collision=r_collision,
            r_bonus=np.where(tier2, r_bonus, 0.0).astype(np.float32),
            criticality=np.where(tier2, criticality_raw, 0.0).astype(np.float32),
            c_spawn_lane=c_spawn_lane,
            c_goal_lane=c_goal_lane,
            c_parking=c_parking,
            c_invalid=c_invalid,
            c_invalid_sev=sev_reject,
            c_invalid_reason=reason,
            c_trivial=c_trivial,
            c_overlap=c_overlap,
            constraint=constraint,
        )
        components.update(
            tier=np.select([degenerate, tier1], [0.0, 1.0], default=2.0).astype(np.float32),
            t1_grade=np.where(tier1, t1_grade, 0.0).astype(np.float32),
            c_path_conflict=conflict.astype(np.float32),
            c_path_dist=np.asarray(m["path_conflict_dist"], dtype=np.float32),
        )
        return total, components
