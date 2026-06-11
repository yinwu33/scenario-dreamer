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


ddpo
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