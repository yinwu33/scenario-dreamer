# DDPO Critical-Scene Training — Improvement Plan

Derived from ANALYSIS.md, re-prioritised against the actual code state
(collision currently disabled in `reward_hooks.py`, KL-to-ref / cos-sin heading /
inpaint log-prob mask already implemented). Ordered cheapest-highest-impact first.

## Phase 0 — Diagnostics / observability (no behaviour change)
- [ ] `reward.evaluate` returns per-component arrays (ttc_term, dist/approach,
      offlane_pen, parking_pen), not just the scalar total.
- [ ] `train_loop` logs per component: mean / std / non-zero fraction /
      correlation with total reward.
- [ ] Rename `critical_rate` (reward>0); add `near_miss_rate`,
      `meaningful_collision_rate`, `trivial_collision_rate`, `start_lane_distance`,
      `goal_lane_distance`, `adv_init_dist`, `adv_min_dist_warmup`.

## Phase 1 — Per-context advantage normalisation (highest leverage)  ✅ DONE
- [x] `ConditioningPool.sample_group_batch(num_groups, group_size)`: replicate
      each context K times (`[i]*K + [j]*K + ...`), return `group_ids`.
      (added to both ConditioningPool and LDMGoalConditioningPool)
- [x] `compute_advantages(rewards, cfg, group_ids=None)`: normalise within group;
      `adv_mode: zscore | rank` (rank advantage); skip groups with std < eps.
- [x] `train_loop`: group sampling + thread `group_ids`; config keys
      `group_size`, `ddpo.{adv_mode, per_context, group_skip_std}`;
      log `group_reward_std`.

## Phase 2 — Reward shaping: continuous + decoupled  ✅ DONE
- [x] `RewardHookEgoAdvMinDist`: record initial clearance `d0` and post-warmup `dmin`
      (`ego_adv_init_dist`, `ego_adv_min_dist_warmup`).
- [x] `RewardHookGoalOfflane` (+ static_metrics): continuous `goal_lane_dist` /
      `spawn_lane_dist` (max over controlled) alongside the binary fraction.
- [x] `RewardHookEgoCollision`: togglable (`collision_enabled`) + records
      `ego_collision_time` (first-collision step * dt) for trivial gating.
- [x] `reward.py`: distance_bonus -> gated approach bonus
      `sigma((d_safe-dmin)/s)*sigma((d0-dmin-delta)/s)`; binary offlane ->
      smoothstep continuous penalty; trivial-collision (<0.75s) penalty;
      time-gated collision *bonus*; returns `R_criticality` + `R_constraint`
      + every component (for logging).
- [x] New config weights + RolloutParams plumbing.

NOTE: collision stays disabled by default (`collision_enabled: false`,
`collision_coef: 0`) — decide whether to re-enable now that the trivial/late
gating exists (open question from the review).

## Phase 3 — Counterfactual ego reward (pairs with Phase 1 grouping)
- [ ] Cache one no-adversary baseline rollout per context (ego/map/goal fixed).
- [ ] `R_cf_progress`, `R_cf_collision` from with/without-adversary delta.

## Phase 4 — Route-based conflict potential (paper-level, high cost)
- [ ] `ddpo/routes.py`: lane-graph route search for ego + adversary, conflict
      point, time-to-arrival; `R_conflict = 1{route conflict} * exp(-dTTA^2/2tau^2)`.

## Phase 5 — Constrained DDPO + reference data mixing
- [ ] Separate criticality vs constraint advantage normalisation.
- [ ] Lagrangian dynamic lambda for validity/route constraints (replace fixed
      penalty coefs).
- [ ] Mix a batch of Waymo denoising loss every N updates; optionally train only
      LoRA / last layers.

## Phase 6 — Structured map-relative representation + curriculum (research)
- [ ] Generate (lane anchor, s, d, v); heading = lane tangent + residual; goal as
      reachable route branch. Curriculum: position-only -> +goal -> +heading.
- [ ] Gate on ablation evidence from Phases 1-3 that the joint Euclidean action
      space is actually the credit-assignment bottleneck.
