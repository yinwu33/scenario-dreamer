# LDMAdv DDPO Support Analysis

## Run
- config: `config_critical_scene_ldm_adv_ddpo`
- splits: `train, val`
- selected scenes: `{'train': 1000, 'val': 1000}`
- samples per scene: `8`
- device: `cuda`
- sampler: `ddpm`
- high reward threshold: `0.3`

## Conclusion
- ddpo_feasibility: `favorable`
- base high-reward sample rate=0.1139
- base scene coverage with any high sample=0.4900
- artifact among high reward=0.0933
- init-invalid mean=0.1570
- small-vehicle mean=0.0605

## Split Summary
### train/base
- n_samples: `8000`
- high_reward_sample_rate: `0.113875`
- positive_sample_rate: `0.639500`
- reward_p95 / p99 / max: `0.7353` / `0.9843` / `1.0000`
- scene_with_high_reward_sample_rate: `0.490000`
- near_miss_rate_mean: `0.127250`
- collision_rate_mean: `0.014375`
- init_invalid_rate_mean: `0.157000`
- goal_offlane_rate_mean: `0.114875`
- controlled_parking_rate_mean: `0.021250`
- small_vehicle_rate_mean: `0.060500`

### val/base
- n_samples: `8000`
- high_reward_sample_rate: `0.113375`
- positive_sample_rate: `0.625125`
- reward_p95 / p99 / max: `0.7073` / `0.9814` / `1.0000`
- scene_with_high_reward_sample_rate: `0.488000`
- near_miss_rate_mean: `0.128375`
- collision_rate_mean: `0.015625`
- init_invalid_rate_mean: `0.143375`
- goal_offlane_rate_mean: `0.126625`
- controlled_parking_rate_mean: `0.021375`
- small_vehicle_rate_mean: `0.065375`

## Interpretation Rules
- `favorable`: base diffusion already has a usable high-reward tail; DDPO should mainly increase its probability.
- `conditional`: base support exists but is sparse; DDPO may work only with careful KL/reward/artifact controls.
- `weak_base_support`: base almost never samples high reward; DDPO will likely leave the base manifold.
- `blocked_by_artifacts`: high reward is dominated by invalid/offlane/parked/small-vehicle artifacts.

## Config Snapshot
```yaml
model_type: ldm_adv
device: cuda
seed: 0
min_ego_drive: 10.0
prune_base_to_ego: false
force_adv_vehicle: true
adv_cond_target:
  enabled: true
  type: vehicle
  motion: moving
  goal_dist:
  - middle
  - far
  ego_dist: null
ldm_adv_ckpt: data/checkpoints/scenario_dreamer_ldm_adv_train/last.ckpt
ae_ckpt: data/checkpoints/scenario_dreamer_ae_goal_waymo/last.ckpt
use_ema_weights: true
train_split: train
eval_split: val
pool_size: 40000
reward_backend: numpy
pufferdrive_root: /home/tjhu78u/workspace/scenario-dreamer/PufferDrive
sim_steps: 91
planner_deterministic: true
ttc_tau: 3.0
init_overlap_margin: 0.0
goal_offlane_threshold: 2.75
goal_onroad_threshold: 2.75
risk_coef: 1.0
approach_d_safe: 6.0
approach_d_scale: 2.0
approach_close_delta: 2.0
approach_close_scale: 1.0
approach_warmup_time: 0.5
goal_offlane_penalty: 1.0
lane_soft: 1.75
parking_mismatch_penalty: 0.5
controlled_parking_penalty: 0.5
collision_enabled: true
collision_coef: 0.3
collision_warmup: 0.75
collision_window: 0.5
trivial_collision_t: 0.75
trivial_collision_penalty: 0.5
min_dist_coef: 0.0
min_dist_dmax: 10.0
sampler: ddpm
ddim_steps: 50
ddim_eta: 1.0
batch_size: 64
group_size: 8
num_iterations: 40000
inner_epochs: 1
k_steps: 16
min_diffusion_t: 1
bf16: false
logprob_bf16: false
lr: 1.0e-05
weight_decay: 0.0001
grad_clip: 1.0
save_every: 100
resume: true
output_dir: data/critical_scene/critical_scene_ddpo_ldm_adv_ddpm_bad_driver
ddpo:
  estimator: is
  clip_range: 0.0001
  kl_coef: 0.2
  adv_clip: 5.0
  adv_eps: 1.0e-06
  logratio_clip: 20.0
  adv_mode: zscore
  per_context: true
  group_skip_std: 0.0001
wandb:
  enabled: true
  project: critical_scene
  entity: null
  run_name: critical_scene_ddpo_ldm_adv_ddpm_bad_driver
eval_every: 50
eval_num_scenes: 8
eval_visualize_reference: true
eval_visualize_conditioning: true
eval_visualize_train: true
save_gif: true
gif_fps: 10
gif_max_frames: 90
planner:
  name: bad_driver
  checkpoint: /home/tjhu78u/workspace/scenario-dreamer/planner/bad_driver/bad_driver_178126787233.pt
  device: auto
  deterministic: true
  policy:
    input_size: 64
    hidden_size: 256
  rnn_name: Recurrent
  rnn:
    input_size: 256
    hidden_size: 256
  sim:
    dt: 0.1
    goal_radius: 2.0
    goal_speed: 100.0
    goal_behavior: continue
    max_controlled_agents: 64
    condition_sample_mode: fixed
    fixed_collision_factor: 0.5
    fixed_offroad_factor: 0.5
    fixed_lane_width: 3.5
    collision_factor_range:
    - 0.0
    - 2.0
    offroad_factor_range:
    - 0.0
    - 2.0
    lane_width_range:
    - 1.0
    - 5.0

```