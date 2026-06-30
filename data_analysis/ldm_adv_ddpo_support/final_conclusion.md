# LDMAdv DDPO Feasibility Conclusion

This conclusion is based on the configured DDPM base sampler from
`cfgs/config_critical_scene_ldm_adv_ddpo.yaml`.

## Run Scope

- Policy: frozen base `ldm_adv_ckpt`, not a DDPO checkpoint.
- Reward/planner: same `bad_driver` reward configuration as DDPO.
- Splits: train and val.
- Contexts: 1000 random contexts per split from the configured 40000-scene pool.
- Samples: 8 independent DDPM samples per context.
- Total base samples: 16000.
- Conditioning baseline: real cached adversary rolled out once per context.

## Main Result

DDPO is feasible for this base model.

The frozen diffusion model already has a substantial high-reward tail:

- Train base: `reward > 0.3` in `11.39%` of samples.
- Val base: `reward > 0.3` in `11.34%` of samples.
- Train contexts with at least one high-reward sample among 8: `49.0%`.
- Val contexts with at least one high-reward sample among 8: `48.8%`.
- Train reward p95/p99/max: `0.735 / 0.984 / 1.0`.
- Val reward p95/p99/max: `0.707 / 0.981 / 1.0`.

This is not a weak-support case. DDPO does not need to invent critical scenes far
outside the base distribution; it can mostly reweight/amplify samples that the
base model already produces.

## Artifact Check

High reward is not primarily caused by artifacts:

- Train high-reward artifact rate: `9.33%`.
- Val high-reward artifact rate: `9.26%`.
- Train clean high-reward sample rate: `10.33%`.
- Val clean high-reward sample rate: `10.29%`.

Among high-reward samples:

- Train near-miss rate: `75.08%`; collision rate: `9.33%`.
- Val near-miss rate: `73.43%`; collision rate: `10.47%`.
- Median ego-adversary min distance in clean high-reward samples:
  - Train: `5.24 m`
  - Val: `5.52 m`

So most high reward comes from TTC/approach criticality rather than invalid,
offlane, parked, or small-vehicle artifacts.

## Important Caveats

The base sampler also has non-trivial invalid/artifact mass:

- Train init-invalid rate: `15.70%`.
- Val init-invalid rate: `14.34%`.
- Train goal-offlane rate: `11.49%`.
- Val goal-offlane rate: `12.66%`.
- Train small-vehicle rate: `6.05%`.
- Val small-vehicle rate: `6.54%`.

This does not block DDPO, but it explains why DDPO training can become noisy and
why KL/reward spikes appear: the reward has a real high-criticality tail, but the
sampler also has enough invalid/off-manifold mass that optimization must be
kept constrained.

## Conditioning Baseline

The real cached adversary is much less critical than base diffusion samples:

- Train conditioning `reward > 0.3`: `1.7%`.
- Val conditioning `reward > 0.3`: `2.3%`.
- Train conditioning median reward: `0.0`.
- Val conditioning median reward: `0.0`.
- Conditioning adversaries are often parked (`~40%`) and far from ego
  (median min distance about `27 m`).

This means the critical tail is coming from the diffusion sampler plus the
adversary-conditioning target, not simply from the original real adversary.

## Decision

Proceed with DDPO, but treat it as a constrained tail-amplification problem:

- DDPO is justified because base support is strong and appears on both train and
  val.
- The reward/KL spikes seen in training are not evidence that base support is
  missing; they are more likely from amplifying an already-present critical tail,
  random timestep KL estimation, and remaining artifact modes.
- Before trusting long DDPO runs, fix or verify the adversary vehicle-size
  projection issue, because small-vehicle artifacts still appear in about `6%`
  of base samples and about `9%` of high-reward samples.
- Continue monitoring clean high-reward rate separately from total reward.

