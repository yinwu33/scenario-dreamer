"""Four-level reward whose top level is the EGO-FAULT collision alone.

    invalid -> ego-fault collision -> min TTC_ego -> d_min

Same four levels as ``hierarchical_v3`` and the same grading inside each, with
one change: the collision level admits only collisions the ego caused. A
collision the ego did not cause is not penalised, it simply is not a collision
as far as this reward is concerned, and falls to whichever level its own
``ego_min_ttc`` and ``d_min`` put it in.

WHY, measured on the 12 table_main_v3 cells (1000 val scenes each, driving-ego
subset, per-scene arrays in ``scored_adv.npz``):

  * ``hierarchical_v3`` credits any valid collision, and the cheapest way to
    obtain one is to drive the adversary into the ego. Pooled over the 12
    ddpo_gen sources, 88.1% of collisions never put the ego in aggressor
    geometry at ANY step, and only 3.9% were ego-fault at contact. In the two
    highest-collision cells (pdm-ppo_aggressive 158 collisions,
    idm-ppo_aggressive 150) the ego was never the aggressor in 100% and 98.7%
    of them respectively.
  * DDPO drives the fault share DOWN in every cell it is trained on
    (ppo-ppo_norm 44.4% -> 8.3%, pdm-ppo_norm 7.7% -> 0.0%): the 0.1 fault
    bonus in v3 does not outweigh how much easier a ram is.

WHY THIS DOES NOT STARVE THE GRADIENT. ``ego_min_ttc`` carries the same
``_ego_approaching_mask`` cone that ``ego_fault_collision``'s front-face test
is the contact-time form of, so a ram has
``ego_min_ttc = inf``, ``g_ttc = 0``, and therefore cannot enter the TTC level
either: it lands on ``d_min``, worth at most ``prox_hi``. The levels a ram can
reach are strictly below the near-miss band, so the incentive to ram is removed
rather than merely reduced. Expected value of causing a collision, under the
measured 3.9 / 8.0 / 88.1 split:

    0.039*1.0 + 0.080*(<=0.85) + 0.881*(<=0.30)  <=  0.371   vs  0.85 near miss

And the TTC level is the fault collision's own precursor -- it pays for putting
the adversary where the ego will drive INTO it -- so the policy climbs
d_min -> TTC -> fault collision instead of having to hit a <1% event blind.
The TTC level holds ~9% of samples (~11 per 128-scene batch), which is what
GRPO needs for within-group contrast; the fault level being nearly empty at the
start is therefore survivable in a way it would not be if the ram still paid.

The open risk this trades into: the TTC ceiling (0.85) and the fault collision
(1.0) differ by 0.15, while crossing between them means driving a ~9% event down
to well under 1%. The policy may simply sit at the TTC ceiling. That is a
convergence question, not a gradient question, and it is what the first run has
to answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ddpo.reward import terms
from ddpo.reward.base import BaseRewardConfig, RewardAssembler, component_dict


@dataclass
class HierarchicalV4RewardConfig(BaseRewardConfig):
    name: ClassVar[str] = "hierarchical_v4"

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


class HierarchicalV4Reward(RewardAssembler):
    name = "hierarchical_v4"
    config_cls = HierarchicalV4RewardConfig
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

        # THE difference from v3: the collision level tests ego_fault, so a
        # collision the ego did not cause falls through to the TTC level (if the
        # ego was ever the aggressor) or to d_min (if it never was).
        tier3 = valid & admitted & collision & ego_fault
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
        components.update(
            tier=tier,
            c_path_conflict=admitted.astype(np.float32),
            c_path_dist=np.asarray(m["path_conflict_dist"], dtype=np.float32),
        )
        return total, components
