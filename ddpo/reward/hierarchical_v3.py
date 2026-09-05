"""Four-level ego-fault reward.

    invalid -> collision (+ ego-fault bonus) -> min TTC_ego -> d_min

Every level above ``invalid`` measures ONE phenomenon at a different severity:
the ego running into the adversary. ``ego_min_ttc`` is already gated by
``SimScene._ego_approaching_mask``, the cone counterpart of the front-face
contact test that decides ``ego_fault_collision``, so "almost ran into it" and "ran into it" are the same
event observed earlier or later -- unlike ``hierarchical``, whose collision level
was fault-agnostic while its TTC level was ego-gated.

Three deliberate differences from ``hierarchical``:

  * A contact before ``hard_collision_t`` is INVALID, not merely uncredited. An
    adversary placed close enough to be hit within a second is a spawn defect,
    and grading it as a near-miss (its post-contact d_min is large, because the
    cars separated) is what let those samples outrank quiet ones before.
  * The proximity level is graded on ABSOLUTE d_min, but only for adversaries
    that actually closed in: distance alone is farmable by spawning alongside
    the ego and driving parallel, which is the hack EgoAdvMinDistHook warns
    about and which the relative form (1 - d_min/d0) would hide.
  * Any valid collision enters the collision level; ego fault is a BONUS on top,
    not a separate level. Making fault its own level inverts the expected
    ordering: fault|collision is 44% (measured on ppo-ppo_norm base_gen), and
    whether the ego ends up the aggressor is mostly decided by the frozen ego,
    so a fault-only top level makes a reliable near miss worth more in
    expectation than actually causing a crash. With the collision base at 0.9
    and the TTC ceiling at 0.85, every collision outranks every near miss
    (0.444*1.0 + 0.556*0.9 = 0.944 > 0.85) while fault still pays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ddpo.reward import terms
from ddpo.reward.base import BaseRewardConfig, RewardAssembler, component_dict


@dataclass
class HierarchicalV3RewardConfig(BaseRewardConfig):
    name: ClassVar[str] = "hierarchical_v3"

    # Contact earlier than this is a spawn defect -> invalid.
    hard_collision_t: float

    # Proximity level: linear ramp on absolute d_min, gated on closing in.
    d_near: float
    d_far: float
    close_delta: float

    # Level values.
    r_invalid: float
    r_collision: float
    r_fault_bonus: float
    ttc_lo: float
    ttc_hi: float
    prox_hi: float

    def __post_init__(self):
        super().__post_init__()
        if not self.d_far > self.d_near:
            raise ValueError(f"d_far ({self.d_far}) must exceed d_near ({self.d_near})")
        if not self.ttc_lo < self.ttc_hi < self.r_collision:
            raise ValueError("levels must be ordered: ttc_lo < ttc_hi < r_collision")
        if self.r_fault_bonus < 0.0:
            raise ValueError("r_fault_bonus must be >= 0")
        if not self.prox_hi <= self.ttc_lo:
            raise ValueError(f"prox_hi ({self.prox_hi}) must not exceed ttc_lo ({self.ttc_lo})")


class HierarchicalV3Reward(RewardAssembler):
    name = "hierarchical_v3"
    config_cls = HierarchicalV3RewardConfig
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

        collision = np.asarray(m["ego_collision"], dtype=np.float32) > 0.0
        ctime = np.asarray(m["ego_collision_time"], dtype=np.float32)
        ego_fault = np.asarray(m["ego_fault_collision"], dtype=np.float32) > 0.0
        # Hard gate: a contact inside hard_collision_t is a placement artifact.
        too_early = collision & (ctime < cfg.hard_collision_t)

        invalid = (c_overlap > 0.0) | (c_invalid > 0.0) | too_early
        admitted = np.asarray(m["path_conflict"], dtype=np.float32) > 0.0
        valid = ~invalid

        g_ttc = terms.ttc_grade(m, cfg.ttc_tau)
        dmin = np.asarray(m["ego_adv_min_dist_warmup"], dtype=np.float32)
        closed_in = terms.closed_in(m)
        g_d = np.clip(
            (cfg.d_far - dmin) / (cfg.d_far - cfg.d_near), 0.0, 1.0
        ).astype(np.float32)
        # Distance alone is farmable by spawning alongside and driving parallel.
        g_d = np.where(
            np.isfinite(dmin) & (closed_in > cfg.close_delta), g_d, 0.0
        ).astype(np.float32)

        tier3 = valid & admitted & collision            # it crashed
        tier2 = valid & admitted & ~tier3 & (g_ttc > 0.0)   # it nearly did
        tier1 = valid & ~tier3 & ~tier2                 # it got close, or nothing

        r_prox = (cfg.prox_hi * g_d).astype(np.float32)
        r_ttc_band = (cfg.ttc_lo + (cfg.ttc_hi - cfg.ttc_lo) * g_ttc).astype(np.float32)
        r_coll = (cfg.r_collision + cfg.r_fault_bonus * ego_fault).astype(np.float32)

        total = np.select(
            [invalid, tier2, tier3],
            [np.full_like(r_prox, cfg.r_invalid), r_ttc_band, r_coll],
            default=r_prox,
        ).astype(np.float32)
        tier = np.select([invalid, tier1, tier2], [0.0, 1.0, 2.0], default=3.0).astype(
            np.float32
        )

        c_spawn_lane, c_goal_lane = terms.lane_costs(m, cfg.lane_soft, cfg.lane_hard)
        components = component_dict(
            r_ttc=np.where(admitted, g_ttc, 0.0).astype(np.float32),
            r_approach=np.where(tier1, g_d, 0.0).astype(np.float32),
            r_risk=np.where(admitted, g_ttc, 0.0).astype(np.float32),
            r_collision=collision.astype(np.float32),
            r_bonus=(cfg.r_fault_bonus * ego_fault).astype(np.float32),
            criticality=np.where(tier2 | tier3, g_ttc, 0.0).astype(np.float32),
            c_spawn_lane=c_spawn_lane,
            c_goal_lane=c_goal_lane,
            c_parking=c_parking,
            c_invalid=c_invalid,
            c_invalid_sev=sev_reject,
            c_invalid_reason=reason,
            c_trivial=too_early.astype(np.float32),
            c_overlap=c_overlap,
            constraint=np.zeros_like(r_prox),
        )
        components.update(
            tier=tier,
            c_path_conflict=admitted.astype(np.float32),
            c_path_dist=np.asarray(m["path_conflict_dist"], dtype=np.float32),
        )
        return total, components
