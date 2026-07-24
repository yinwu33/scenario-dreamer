```
source scripts/define_env_variables.sh
# 1. 训 goal autoencoder（专用入口，配置在 cfgs/ae_goal/）
python train.py --config-name config_ae_goal
# 2. 缓存 latent（train/val 各跑一次）
python eval.py --config-name config_ae_goal ae_goal.eval.cache_latents.enable_caching=True ae_goal.eval.cache_latents.split_name=train
python eval.py --config-name config_ae_goal ae_goal.eval.cache_latents.enable_caching=True ae_goal.eval.cache_latents.split_name=val
# 3. 训 ldm_adv（在 goal-AE latent 上扩散 normal agents + 1 个 adversary）
python train.py --config-name config_ldm_adv_train
```


critical_scene Stage 2 — DDPO（scenario-dreamer 内置，单 venv，无需 dump / .bin）
```
# 0. 一次性：把 PufferDrive 训好的 frozen bad_driver checkpoint 拷到 planner/bad_driver/
#    路径在 cfgs/planner/bad_driver.yaml 里配置

source scripts/define_env_variables.sh

# 训练（统一走 train.py；只微调 adversary 分支，基座场景冻结）
python train.py --config-name config_critical_scene_ldm_adv_ddpo                 # DDPM 采样
python train.py --config-name config_critical_scene_ldm_adv_ddim                 # 随机 DDIM 子采样

# 入口配置：cfgs/config_critical_scene_ldm_adv_ddpo.yaml（experiment.* 派生 run_name / output_dir / wandb；
#       ddim 入口继承它，只覆盖 sampler）。四个正交的配置组由入口 defaults 组装：
#   flow：cfgs/ddpo/ldm_adv.yaml（conditioning / ckpt / pool + simulator / reward + 优化与采样）
#   算法：cfgs/ddpo/algo/grpo.yaml（group_size / 白化 / clip / KL 信任域；换 PPO 时
#         加一个 sibling yaml，ddpo/algo@ddpo.algo=ppo 一键切换，代码读 cfg.algo 不变）
#   planner：cfgs/planner/bad_driver.yaml（checkpoint / 网络结构 + 该策略的
#         conditioning obs：collision/offroad factor、lane_width，标量或 [lo,hi] 采样），
#         按角色组装 planner@ddpo.planner.{sut,env,adv}；adv 用 bad_driver_reckless
#         变体（collision_factor 0，只有 adversary 不避让）
#   rollout 动力学：cfgs/rollout/base.yaml（dt / goal 行为 / map_extent；
#         组装到 ddpo.planner.sim，完整显式、无隐藏默认）
# 代码：ddpo/（policy_ldm_adv.py 采样封装；planners/ RolloutRunner + 每角色 Planner(plan/apply)；
#       hooks 由 reward.py 组装注入；pufferdrive_sim.py 纯 numpy 复刻 PufferDrive rollout；
#       goal_schema.py 是 agent-state 9/12 维 layout 的单一真源）
# 产物：${SCRATCH_ROOT}/critical_scene/<run_name>/（checkpoints/generated/media）
# checkpoint 存为 Lightning 兼容格式（diff_model.* 前缀），可被 eval.py/viz 直接加载

# 评估（四源对比 original / base_gen / ddpo_gen / original_ddpo_adv，bad_driver rollout）
.venv/bin/python scripts/run_ldm_adv_bad_driver_table.py \
    --num-scenes 1000 --chunk-size 32 --out-dir data/critical_scene/ldm_adv_bad_driver_eval
# 2x2 GIF 可视化（依赖上一步的 benchmark 输出）
.venv/bin/python scripts/render_ldm_adv_2x2_gifs.py --out-dir data/critical_scene/ldm_adv_bad_driver_eval
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
