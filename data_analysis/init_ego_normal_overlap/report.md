# Frame-1 ego-crash: normal vs adversary at spawn (ldm_adv DDPO)

**Question.** In ldm_adv DDPO rollouts some scenes end on frame 1 because the ego
collides immediately. How often is the culprit a *real normal* agent that the AE
decodes overlapping the ego at spawn (NOT the generated adversary)?

**Setup.** `config_critical_scene_ldm_adv_ddpo`, train split, `prune_base_to_ego=false`
(full real normal scene kept, mean ~9 normals/scene), frozen/base policy = start of
DDPO. The general collision response `SimScene.latch_ego_crash` freezes + stops the
scene on ANY ego↔vehicle overlap regardless of fault, so a normal-on-ego init kills
the scene on frame 1. Neither `init_invalid` nor `ego_collision` records this (both
are adversary-only). Two independent measurements:

## 1. Spawn-state overlap (`init_ego_normal_overlap.py`)
Replays the sim's exact collision test (`_corners`/`_sat_overlap`, 15 m gate, ped
exclusion) on the decoded t=0 state.

| metric | cond (real adv, n=8000) | sampled (gen adv, n=1024) |
|---|---|---|
| P(ego overlaps a **NORMAL** @ spawn) | **0.34%** (27) | **0.29%** (3) |
| P(ego overlaps the **ADV** @ spawn) | 0.01% (1) | 2.54% (26) |
| P(ego overlaps anything @ spawn) | 0.35% | 2.83% |

The normal-overlap rate is ~0.3% and **sample-independent** (normals are fixed
conditioning; cond and sampled agree), confirming it is a base-scene AE
reconstruction artifact, not a policy effect.

## 2. Real-rollout frame-1 crash attribution (`init_frame1_crash_attribution.py`)
Runs the actual `bad_driver` planner and records the first ego-crash step + partner
(n=2048, base policy, max_steps=5).

| metric | value |
|---|---|
| P(ego crashes on frame 1) | **2.29%** (47/2048) |
| of frame-1 crashes: hit a NORMAL | 10.6% |
| of frame-1 crashes: hit the ADV | 89.4% |
| **P(frame-1 crash by NORMAL, all scenes)** | **0.24%** ← the user's case |
| P(frame-1 crash by ADV, all scenes) | 2.05% |

## Conclusion
- The user's case — a **real normal agent decoded overlapping the ego at spawn**,
  ending the scene on frame 1 — occurs in **~0.3% of scenes** (≈1 in 300). It is a
  fixed AE-reconstruction artifact, independent of the policy.
- It is the **minority** of frame-1 terminations. Of all scenes that die on frame 1
  (~2.3%), **~90% are the GENERATED ADVERSARY spawning on the ego** (~2.0% absolute),
  not a normal. That part IS the adversary's doing and is already shaped by the
  reward (`init_overlap_frac` gate / `init_invalid`); it should shrink as DDPO
  trains. The ~0.3% normal-on-ego floor will not.
