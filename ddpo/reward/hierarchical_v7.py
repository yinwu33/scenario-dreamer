"""Five-band reward: a ram earns nothing UNLESS the ego genuinely bore down on
the adversary first, in which case the near miss it caused still counts.

    invalid  ->  uncredited ram  ->  d_min  ->  min TTC_ego  ->  ego-fault collision
      -1              0            0..0.3      0.4..0.85            1.0

Identical to ``hierarchical_v7`` except for which rams reach the uncredited
band. v6 sent every one there, including scenes where the ego had been closing
on the adversary inside its own fault cone before the contact landed elsewhere.
Two ppo-idm rollouts make the case: ego_min_ttc 1.30 s and 1.40 s, so g_ttc 0.57
and 0.53 -- a real ego-caused near miss by this reward's own definition -- both
scored 0 because the scene happened to end with the adversary hitting the ego.
Here they keep the TTC band; only a ram with no approach at all (g_ttc == 0)
falls to 0.

WHAT THIS IS WORTH, measured over the eight finished table_main_v6 cells
(ddpo_gen, driving subset): 222 rams, of which 14 (6.3%) have g_ttc > 0. The
share is wildly uneven -- ppo-idm 52.9% (9 of the 14), idm-idm 22.2%, pdm-idm
10.0%, and three cells at exactly 0.0%. Rams are about 2% of scenes, so v7
rescores roughly 0.2% of them. That is far below the resolution of the 1000-scene
protocol, where Coll._f alone is 2-11 events; this reward cannot be validated by
that eval, and a run should be judged on its training-side tier mix instead.

Note the asymmetry in that 6.3%: it was measured on policies trained under v6,
which forbade this path. It is a floor, not a ceiling, on what v7's policies may
do.

v6's docstring claimed that crediting these "would leave a ram-tolerant path
open". That overstates it. The adversary is a placement and a goal, not a
controller -- its motion comes from the frozen `adv` planner -- so the policy
cannot execute an approach-then-ram strategy deliberately, only pick placements
that tend to produce one. The exploit risk is correspondingly smaller than that
sentence implied.

Relation to the earlier versions: v7 is v6's zero for the approach-less ram,
v4's TTC credit for the ram that follows a real approach, and v5's off-lane
rejection. It is deliberately not v4 for the approach-less case -- there a ram
lands on d_min with d_min ~ 0, collecting that band's full 0.3, which is what
made ramming worth about four times a quiet scene across v4's twelve cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ddpo.reward import terms
from ddpo.reward.base import BaseRewardConfig, RewardAssembler, component_dict


@dataclass
class HierarchicalV7RewardConfig(BaseRewardConfig):
    name: ClassVar[str] = "hierarchical_v7"

    # Contact earlier than this is a spawn defect -> invalid.
    hard_collision_t: float

    # Proximity level: linear ramp on absolute d_min, gated on closing in.
    d_near: float
    d_far: float
    close_delta: float

    # Level values. There is no fault BONUS here: fault is the level's
    # admission test, so r_collision is what an ego-fault collision is worth.
    r_invalid: float
    r_collision: float
    ttc_lo: float
    ttc_hi: float
    prox_hi: float

    def __post_init__(self):
        super().__post_init__()
        if not self.d_far > self.d_near:
            raise ValueError(f"d_far ({self.d_far}) must exceed d_near ({self.d_near})")
        if not self.ttc_lo < self.ttc_hi < self.r_collision:
            raise ValueError("levels must be ordered: ttc_lo < ttc_hi < r_collision")
        if not self.prox_hi <= self.ttc_lo:
            raise ValueError(f"prox_hi ({self.prox_hi}) must not exceed ttc_lo ({self.ttc_lo})")


class HierarchicalV7Reward(RewardAssembler):
    name = "hierarchical_v7"
    config_cls = HierarchicalV7RewardConfig
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
        # Applied to ANY contact, fault or not -- an early ram is still a spawn
        # defect, and letting it fall through to d_min would pay for it.
        too_early = collision & (ctime < cfg.hard_collision_t)

        # Off-lane is the adversary's own placement and stays a rejection. The
        # ram is not: it is scored 0 below, outside the invalid band.
        rammed = collision & ~ego_fault
        offlane = np.asarray(m["goal_offlane_frac"], dtype=np.float32) > 0.0
        invalid = (c_overlap > 0.0) | (c_invalid > 0.0) | too_early | offlane
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

        # A ram is uncredited only when there was no ego approach to credit.
        # With g_ttc > 0 the ego was closing on the adversary inside the fault
        # cone, which is a near miss it caused whatever the contact did next, so
        # the scene keeps the TTC band.
        ram_band = valid & rammed & (g_ttc <= 0.0)
        scoring = valid & ~ram_band
        # `scoring` still admits the credited rams, so the fault test is needed
        # here explicitly -- it is no longer implied by excluding every ram.
        tier3 = scoring & admitted & collision & ego_fault
        tier2 = scoring & admitted & ~tier3 & (g_ttc > 0.0)
        tier1 = scoring & ~tier3 & ~tier2

        r_prox = (cfg.prox_hi * g_d).astype(np.float32)
        r_ttc_band = (cfg.ttc_lo + (cfg.ttc_hi - cfg.ttc_lo) * g_ttc).astype(np.float32)
        r_coll = np.full_like(r_prox, cfg.r_collision)

        total = np.select(
            [invalid, ram_band, tier2, tier3],
            [np.full_like(r_prox, cfg.r_invalid), np.zeros_like(r_prox), r_ttc_band, r_coll],
            default=r_prox,
        ).astype(np.float32)
        # 4 is appended, not inserted in value order, so 0..3 keep the meaning
        # they have in v3/v4/v5 and the logged tier mix stays comparable.
        tier = np.select(
            [invalid, ram_band, tier1, tier2], [0.0, 4.0, 1.0, 2.0], default=3.0
        ).astype(np.float32)

        c_spawn_lane, c_goal_lane = terms.lane_costs(m, cfg.lane_soft, cfg.lane_hard)
        components = component_dict(
            r_ttc=np.where(admitted, g_ttc, 0.0).astype(np.float32),
            r_approach=np.where(tier1, g_d, 0.0).astype(np.float32),
            r_risk=np.where(admitted, g_ttc, 0.0).astype(np.float32),
            # The rewarded event, so the training log's coll_rate tracks what the
            # reward actually pays for; ego_collision stays available unmodified
            # in the metric dict for the fault-agnostic rate.
            r_collision=(collision & ego_fault).astype(np.float32),
            r_bonus=np.zeros_like(r_prox),
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
        # component_dict has a fixed schema and rejects unknown keys, so the two
        # new reject reasons are attached here alongside tier.
        components.update(
            c_rammed=rammed.astype(np.float32),
            c_offlane=offlane.astype(np.float32),
            tier=tier,
            c_path_conflict=admitted.astype(np.float32),
            c_path_dist=np.asarray(m["path_conflict_dist"], dtype=np.float32),
        )
        return total, components
