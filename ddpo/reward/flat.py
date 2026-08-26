"""Flat reward: dense TTC + approach criticality, soft constraints, bonus.

Three branches, checked in order:
  reject        condition violation (or parked adversary when that gate is off):
                -(base + (1-base) * severity)
  init_invalid  adversary interpenetrates a neighbour at spawn: -constraint
  valid         clip(criticality - constraint, -1, 1) + collision bonus
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
class FlatRewardConfig(ApproachRewardConfig):
    init_overlap_penalty: float
    invalid_penalty_base: float
    collision_bonus: float
    ego_fault_bonus: float

    name = "flat"


class FlatReward(AnnealedApproachAssembler):
    name = "flat"
    config_cls = FlatRewardConfig

    def assemble(self, m: dict) -> tuple[np.ndarray, dict]:
        cfg = self.cfg

        r_ttc = terms.ttc_grade(m, cfg.ttc_tau)
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

        c_overlap = np.asarray(m["init_overlap_frac"], dtype=np.float32)
        c_spawn_lane, c_goal_lane = terms.lane_costs(m, cfg.lane_soft, cfg.lane_hard)
        constraint = (
            cfg.lane_penalty * (c_spawn_lane + c_goal_lane)
            + cfg.init_overlap_penalty * c_overlap
        ).astype(np.float32)

        c_parking = np.asarray(m["gen_agent_is_parked"], dtype=np.float32)
        c_invalid, reason, sev = terms.reject_terms(
            m,
            gen_invalid_enabled=self.gen_invalid_enabled,
            grade_scale=cfg.invalid_grade_scale,
            parked_when_disabled=True,
        )
        reject = c_invalid > 0.0
        init_invalid = c_overlap > 0.0
        valid = ~(reject | init_invalid)

        base_pen = float(np.clip(cfg.invalid_penalty_base, 0.0, 1.0))
        r_reject = -(base_pen + (1.0 - base_pen) * sev)

        # Only true spawn interpenetration gates criticality; lane costs are
        # soft-subtracted so aggressive off-lane geometries keep their gradient.
        total = np.select(
            [reject, init_invalid],
            [
                r_reject.astype(np.float32),
                np.clip(-constraint, -1.0, 0.0),
            ],
            # The bonus sits outside the clip so a saturated dense term cannot
            # absorb it.
            default=np.clip(criticality_raw - constraint, -1.0, 1.0) + r_bonus,
        ).astype(np.float32)

        components = component_dict(
            r_ttc=r_ttc,
            r_approach=r_approach,
            r_risk=r_risk,
            r_collision=r_collision,
            r_bonus=np.where(valid, r_bonus, 0.0).astype(np.float32),
            criticality=np.where(valid, criticality_raw, 0.0).astype(np.float32),
            c_spawn_lane=c_spawn_lane,
            c_goal_lane=c_goal_lane,
            c_parking=c_parking,
            c_invalid=c_invalid,
            c_invalid_sev=sev,
            c_invalid_reason=reason,
            c_trivial=c_trivial,
            c_overlap=c_overlap,
            constraint=constraint,
        )
        return total, components
