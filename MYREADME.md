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


ddpo（新版：scenario-dreamer 内置，单 venv，无需 dump / .bin）
```
# 0. 一次性：把 PufferDrive 训好的 frozen planner 拷到 planner/selfplay_drive/
#    然后在 planner/selfplay_drive/config.yaml 里设置 checkpoint

source scripts/define_env_variables.sh

# 训练（三种模式，conditioning 直接来自原生 dm_goal 数据集）
python train_ddpo.py ddpo.mode=goal        # 固定 map + agent 初始状态，只训 goal point
python train_ddpo.py ddpo.mode=init_goal   # 固定 map，训 agent 初始状态 + goal
python train_ddpo.py ddpo.mode=all         # map（lane 链）+ agent + goal 全部训练

# 冒烟测试（三种模式各跑一个 mini iteration）
.venv/bin/python scripts/smoke_test_ddpo.py

# 配置：cfgs/ddpo/waymo_dm_goal.yaml（batch_size / kl_coef / wandb 等）
# planner 配置：planner/selfplay_drive/config.yaml（checkpoint / device / 网络宽度等）
# simulator 配置：ddpo/config_sim.yaml（goal 行为 / dt / conditioning ranges 等）
# 代码：ddpo/（policy.py 三模式封装；pufferdrive_sim.py 纯 numpy 复刻 PufferDrive
#       rollout——obs 构造 + classic 动力学 + planner 网络精确移植，reward 不再走 C env）
# checkpoint 存为 Lightning 兼容格式（diff_model.* 前缀），可被 eval.py/viz 直接加载
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
