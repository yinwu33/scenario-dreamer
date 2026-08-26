# Hierarchical reward preflight

Frozen policy data: `data/reward_screen/idm-idm-idm_512x32.npz` (512 IDM-IDM-IDM
contexts, 32 base-model samples/context, 16,384 full rollouts).  Production GRPO
group size is 8.  Goal-distance condition rejection is disabled exactly as in
`config_ldm_adv_ddpo_idm_idm_hierarchical`; type and motion checks remain.

## Selected design

Strict order, with continuous grading inside every level:

1. invalid / spawn overlap: `[-1.00, -0.82]`;
2. no measured interaction: `[-0.75, -0.45]`, ranked by initial distance;
3. post-warmup distance `< 8 m`: `[-0.35, -0.05]`, ranked by minimum distance;
4. finite TTC `< 3 s`: `[0.05, 0.65]`, ranked by TTC;
5. non-trivial collision: `[0.75, 1.35]`, plus `0.40` for ego-at-fault.

The 14 m chord-distance predicate only decides whether to run the planner.  It
does not change a sample's reward tier.  The former car-following exclusion is
disabled because the dump shows that it removes real TTC/collision events.

## Prefilter selection

The selected 14 m threshold executes 82.7% of rollouts (17.3% skipped) and
retains 99.5% of valid non-trivial collisions, 98.1% of valid TTC<1.5 s events,
96.9% of valid TTC<3 s events, and 99.6% of actual <8 m approaches.  A PET arm
was swept as an OR condition; it only moved the chosen point toward more
rollouts without a useful recall gain.  Threshold 12 m was cheaper but reduced
conditional TTC winner accuracy to 94.2%; 16 m improved it to 96.5% while
skipping only 13.4%, so 14 m is the current knee.

## GRPO signal, group size 8

| metric | result |
|---|---:|
| degenerate groups | 0.0% |
| live within-group reward std | 0.257 |
| mean group headroom | 0.467 |
| collision winner, when available | 99.5% |
| TTC winner, when available and no collision | 95.2% |
| dynamic-close winner, when available and no TTC/collision | 99.9% |

Sample tier occupancy is T0/T1/T2/T3/T4 =
`8.0/64.4/20.2/5.8/1.5%`.  Winner occupancy is
`0.0/27.3/43.5/20.2/9.0%`: the many `quiet` winners are now intentional
fallback winners, not samples promoted merely because their chords intersect.

For comparison, TTC-only has 69.6% degenerate groups.  `full@it0` has dense
contrast but 78.8% quiet winners.  The old three-tier reward has 79.2% quiet
winners because path admission itself occupies its highest ordinary band.

## Stability checks

The first/second 256-context halves respectively produced:

- degenerate groups: `0.0% / 0.0%`;
- collision winner given available: `100.0% / 99.0%`;
- TTC winner given available: `95.8% / 94.5%`;
- close winner given available: `100.0% / 99.8%`.

At group sizes 4/8/16, conditional collision winner accuracy was
`99.5/99.5/99.3%`, TTC winner accuracy `96.5/95.2/95.9%`, and close winner
accuracy `99.9/99.9/100.0%`; no tested size produced degenerate groups.

## Decision and pilot gates

This reward passes the frozen-base preflight: every group has a fallback
ordering, and rare real events dominate when present.  It is suitable for a
short on-policy pilot, not yet evidence of long-run convergence.  Continue to a
full run only if a 200--500 iteration pilot shows: T1 winner/rate falling while
T2/T3/T4 rise, TTC/collision metrics improving rather than only reward, reject
and overlap not rising, KL remaining controlled, and per-context diversity not
collapsing.
