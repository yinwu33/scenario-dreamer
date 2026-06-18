```
source scripts/define_env_variables.sh
# 1. 训 goal autoencoder
python train.py model_name=autoencoder_goal
# 2. 缓存 latent（train/val 各跑一次）
python eval.py model_name=autoencoder_goal ae_goal.eval.cache_latents.enable_caching=True ae_goal.eval.cache_latents.split_name=train
python eval.py model_name=autoencoder_goal ae_goal.eval.cache_latents.enable_caching=True ae_goal.eval.cache_latents.split_name=val
# 3. 训 goal latent diffusion
python train.py model_name=ldm_goal
# 4. 采样/可视化（viz 已自动画 goal）
python eval.py model_name=ldm_goal ldm_goal.eval.visualize=True
```


critical_scene Stage 2 — DDPO（scenario-dreamer 内置，单 venv，无需 dump / .bin）
```
# 0. 一次性：把 PufferDrive 训好的 frozen planner 拷到 planner/selfplay_drive/
#    然后在 planner/selfplay_drive/config.yaml 里设置 checkpoint

source scripts/define_env_variables.sh

# 训练（统一走 train.py；ddpo.mode 选 goal_only | agent_only | full）
python train.py --config-name config_critical_scene_dm_goal_ddpm ddpo.mode=goal_only
python train.py --config-name config_critical_scene_dm_goal_ddpm ddpo.mode=agent_only
python train.py --config-name config_critical_scene_dm_goal_ddim                 # 随机 DDIM 子采样
python train.py --config-name config_critical_scene_ldm_goal                     # latent-space 策略

# 冒烟测试（三种模式各跑一个 mini iteration）
.venv/bin/python scripts/smoke_test_ddpo.py

# 入口配置：cfgs/config_critical_scene_*.yaml（experiment.* 派生 run_name / output_dir / wandb）
# DDPO 组配置：cfgs/ddpo/{dm_goal,ldm_goal}.yaml（batch_size / kl_coef / 阈值等）
# planner 配置：cfgs/planner/*.yaml + planner/selfplay_drive/config.yaml（checkpoint / device 等）
# simulator 配置：ddpo/config_sim.yaml（goal 行为 / dt / conditioning ranges 等）
# 代码：ddpo/（policy.py 三模式封装；pufferdrive_sim.py 纯 numpy 复刻 PufferDrive rollout；
#       goal_schema.py 是 agent-state 9/12 维 layout 的单一真源）
# 产物：${SCRATCH_ROOT}/critical_scene/<run_name>/（checkpoints/generated/media）；
#       repo outputs/critical_scene/<run_name>/ 只留 summary.json / per_scene.csv / manifest
# checkpoint 存为 Lightning 兼容格式（diff_model.* 前缀），可被 eval.py/viz 直接加载

# 评估（两步：先生成 same-map artifact，再用 planner benchmark；不加载 diffusion model）
python generate_scene.py source=original       split=val num_scenes=4
python generate_scene.py source=base_diffusion split=val num_scenes=4
python generate_scene.py source=ddpo_diffusion split=val num_scenes=4 ckpt=<ddpo ckpt>
python eval_planner.py planner=selfplay_drive inputs=<original.pt>,<base.pt>,<ddpo.pt>
```

ddpo（旧版：PufferDrive 寄宿，已被上面替代）
```
# 1. dump conditioning（SD venv，一次性）——训练池 + 独立 val 可视化池
source scripts/define_env_variables.sh
.venv/bin/python PufferDrive/scene_init_ddpo/tools/dump_conditioning_ldm.py \
    --num-scenes 20000 --split train \
    --out PufferDrive/scene_init_ddpo/data/cond_pool_ldm
.venv/bin/python PufferDrive/scene_init_ddpo/tools/dump_conditioning_ldm.py \
    --num-scenes 2000 --split val \
    --out PufferDrive/scene_init_ddpo/data/cond_pool_ldm_val_eval

# 2. 确认 config/ddpo_ldm_goal.yaml 里的 planner_ckpt 指向你的 frozen planner
#    （ldm_ckpt / ae_ckpt 已指向已验证存在的 checkpoint）

# 3. 训练（PufferDrive venv）
cd PufferDrive
.venv/bin/python -m scene_init_ddpo.train --config scene_init_ddpo/config/ddpo_ldm_goal.yaml
```
