"""Five-band reward: a collision the ego did not cause earns NOTHING, rather
than costing -1.

    invalid  ->  uncredited ram  ->  d_min  ->  min TTC_ego  ->  ego-fault collision
      -1              0            0..0.3      0.4..0.85            1.0

Identical to ``hierarchical_v6`` except for where the ram goes. v5 put it in the
invalid band at -1; here it is its own band worth exactly 0, the same as a quiet
scene that never approached.

WHY, measured on v5's three scored table_main_v5 cells (driving subset,
ddpo_gen):

    cell                 invalid   ram  offlane overlap early   ram share
    idm-ppo_aggressive     10.7%  2.7%    3.2%    4.9%   0.4%      25.7%
    pdm-ppo_aggressive     12.5%  3.6%    4.6%    4.7%   0.5%      28.5%
    ppo-ppo_aggressive     10.9%  2.3%    4.2%    5.3%   0.7%      21.5%

A quarter of v5's -1 band is the ram, and whether a placement ends in one is
largely the frozen ego's decision, not the adversary's: the same spawn is a -1
or a positive score depending on whether the ego brakes. Charging -1 for it
injects variance the policy cannot reduce, which is the same structural error as
rewarding a fault collision the ego alone controls. Scoring it 0 keeps the
incentive gone -- ramming still buys nothing over doing nothing -- without
punishing the adversary for the ego's choice.

Off-lane stays at -1. That IS the adversary's own placement, fully under its
control, and it is the larger share of the band (30-40%).

What v5 established, and v6 does not change: removing the ram incentive works
and is measurable. Collisions fell 9.87% -> 2.95% (97 -> 29 scenes) for
idm-ppo_aggressive and 9.46% -> 3.76% for pdm-ppo_aggressive. What it did not
do is move the top level: tier3 stayed at 2-4% of samples across four cells and
its net change over 500 iterations was +0.001 to +0.006, and the eval Coll._f is
2-8 events per 1000 scenes, which cannot resolve a difference either way.

So the question v6 asks is narrow and worth asking alone: with the -1 removed
from the part the policy cannot control, does the invalid band finally come
down? Under v5 it never did -- 22.4% flat over 500 iterations, 22.6% after a
rebound from 16.8%, 18.3% rising, 14.9% rising, in four cells. If v6's band
drops to the 7-9% that overlap + offlane + early alone account for and tier3
still does not move, the reward is not what is holding this back.

Tier codes keep v3/v4/v5's meaning -- 0 invalid, 1 d_min, 2 near miss, 3
ego-fault collision -- so the training log and the tables stay comparable across
versions; the ram band is appended as 4 rather than inserted in value order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ddpo.reward import terms
from ddpo.reward.base import BaseRewardConfig, RewardAssembler, component_dict


@dataclass
class HierarchicalV6RewardConfig(BaseRewardConfig):
    name: ClassVar[str] = "hierarchical_v6"

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


class HierarchicalV6Reward(RewardAssembler):
    name = "hierarchical_v6"
    config_cls = HierarchicalV6RewardConfig
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

        # A ram short-circuits every credited band, including the TTC one: a
        # genuine near miss that ends in the adversary hitting the ego earns 0,
        # not 0.85. Crediting it would leave a ram-tolerant path open, which is
        # the whole thing this reward removes.
        ram_band = valid & rammed
        scoring = valid & ~rammed
        tier3 = scoring & admitted & collision   # ~rammed => ego_fault
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
