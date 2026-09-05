"""Four-level reward: only an ego-fault collision counts, everything else the
adversary can do to force contact is INVALID.

    invalid  ->  ego-fault collision  ->  min TTC_ego  ->  d_min

Levels and grading are ``hierarchical_v5``'s. Two things join the invalid level,
both because v4 measurably paid for them:

  * **A collision the ego did not cause.** ``ego_collision`` is already scoped to
    the adversary (sim/hooks.py), so this is exactly "the adversary drove into
    the ego". v4 excluded it from the top level but let it fall to ``d_min``,
    where a contact scores that level's maximum -- a ram IS zero distance.
    Measured over the 12 table_main_v4 cells, that paid: mean v4 reward 0.138 /
    0.162 / 0.126 for a ram in idm-, pdm- and ppo-ppo_aggressive against 0.035 /
    0.033 / 0.034 for a quiet scene, i.e. roughly four times what doing nothing
    is worth, in exactly the three cells with the highest ram rate (9.5% / 9.4%
    / 3.2%). Rams are 2.9% of scenes pooled.
  * **An adversary spawned or aiming off the lane graph.** ``goal_offlane_frac``
    already flags either endpoint past ``simulator.goal_onroad_threshold`` /
    ``goal_offlane_threshold`` (2.75 m), so no new threshold is introduced. v4
    scored these positively: one idm-ppo_caution scene spawned the adversary
    25.45 m from the nearest centerline and still collected +0.224. Pooled,
    1.5% of adversaries spawn more than 5 m off, 1.0% more than 10 m, and 3.6%
    trip the off-lane flag at either end.

Both were the same hole: ``d_min``'s ``0.3 * g_d`` pays for being close to the
ego without asking how that happened. Excluding them means the only way up is
the one the reward is named for.

The cost, stated plainly: the invalid band grows from ~8% of samples to roughly
14%, and every one of those is a -1. If that swamps the within-group contrast
GRPO needs, it will show up as ``grp_std`` falling and the reward sitting at
-1; screen it before spending a run.

``ego_fault_collision`` itself changed underneath this: it now tests whether the
ego's FRONT FACE overlaps the other's box rather than where that agent's centre
sits in a cone (``SimScene._ego_front_contact_mask``). On idm-idm's v4
``ddpo_gen`` that lifts the fault share of collisions from 33.3% to 47.1%. Every
``Coll._f`` and ``ego_min_ttc`` number measured before that change is stale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ddpo.reward import terms
from ddpo.reward.base import BaseRewardConfig, RewardAssembler, component_dict


@dataclass
class HierarchicalV5RewardConfig(BaseRewardConfig):
    name: ClassVar[str] = "hierarchical_v5"

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


class HierarchicalV5Reward(RewardAssembler):
    name = "hierarchical_v5"
    config_cls = HierarchicalV5RewardConfig
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

        # A contact the ego did not cause, and an adversary that is not on the
        # lane graph at either end, are rejected outright rather than allowed to
        # collect the d_min level -- see the module docstring for what each was
        # worth under v4.
        rammed = collision & ~ego_fault
        offlane = np.asarray(m["goal_offlane_frac"], dtype=np.float32) > 0.0
        invalid = (c_overlap > 0.0) | (c_invalid > 0.0) | too_early | rammed | offlane
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

        # The collision level tests ego_fault; a collision the ego did not cause
        # is now invalid above, not merely excluded from here, so every
        # collision that survives `valid` is an ego-fault one.
        tier3 = valid & admitted & collision
        tier2 = valid & admitted & ~tier3 & (g_ttc > 0.0)
        tier1 = valid & ~tier3 & ~tier2

        r_prox = (cfg.prox_hi * g_d).astype(np.float32)
        r_ttc_band = (cfg.ttc_lo + (cfg.ttc_hi - cfg.ttc_lo) * g_ttc).astype(np.float32)
        r_coll = np.full_like(r_prox, cfg.r_collision)

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
