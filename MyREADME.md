# Before

```bash
source .venv/bin/activate
source ./scripts/define_env_variables.sh
```

# Data (v2 goal scenes)

The goal dataset is aligned with the original Scenario Dreamer preprocessing: the agent
set is filtered **offline** (FOV crop -> closest `max_num_agents` -> off-road vehicle
removal at 1.5 m, ego and non-vehicles exempt), while everything goal-specific stays at
**runtime** (`utils/goal_runtime.py`) so goal definitions and goal filters can change
without regenerating. v1 lacked the off-road filter entirely, which kept parking-lot
clusters in the training distribution -- ~33% of agents, and a 10x higher ground-truth
collision rate.

```bash
# official path: raw TFRecords -> v2 (run where the raw Waymo data lives)
python data_processing/waymo/preprocess_waymo_selfplay_dataset.py \
    --split train --num-workers 64 \
    --waymo-dir /path/to/waymo/scenario/training \
    --output-dir $DATASET_ROOT/scene_goal_preprocess_waymo_v2
python data_processing/waymo/preprocess_waymo_selfplay_dataset.py \
    --split val --num-workers 64 \
    --waymo-dir /path/to/waymo/scenario/validation \
    --output-dir $DATASET_ROOT/scene_goal_preprocess_waymo_v2

# temporary path: reuse the v1 pickles already on disk (scripts/tmp_convert_goal_v1_to_v2.py,
# not committed). --verify recomputes what v1 stored and checks the two agree.
python scripts/tmp_convert_goal_v1_to_v2.py --split val --verify 200
python scripts/tmp_convert_goal_v1_to_v2.py --split val   --num-workers 64
python scripts/tmp_convert_goal_v1_to_v2.py --split train --num-workers 64
```

Everything downstream must then be rebuilt in this order:
`ae_goal` retrain -> latent cache (`advscene_ae_goal_latents_waymo_v2`) ->
`scripts/create_waymo_goal_val_eval_set.py` -> `scripts/create_goal_init_prob_matrix.py`
-> `ldm_adv` retrain. The baseline LDM only needs its metrics recomputed against the new
reference set.

# Train

### Training of Adv Scene

First generate latent vectors from ae

```bash
# generate train latent
python eval.py \
  --config-name config_ae_goal \
  ae_goal.eval.cache_latents.enable_caching=True \
  ae_goal.eval.cache_latents.split_name=train

# generate val latent
python eval.py \
  --config-name config_ae_goal \
  ae_goal.eval.cache_latents.enable_caching=True \
  ae_goal.eval.cache_latents.split_name=val
```

# Evaluation

### Evaluation of Autoencoder


Scenario Dreamer Baseline

```bash
# download data/checkpoints/scenario_dreamer_autoencoder_waymo/last.ckpt

# run on original dataset
python eval.py dataset_name=waymo model_name=autoencoder ae.eval。run_name=scenario_dreamer_autoencoder_waymo

# run on my dataset
python eval.py --config-name config_ae_my_dataset
```

AdvScene 

```bash
# run on my dataset
python eval.py --config-name config_ae_goal +ae_goal.eval.split_name=val ae_goal.eval.run_name=scenario_dreamer_ae_goal_waymo

```



### Evaluation of Generation

#### Fair LDM vs LDM-Adv comparison

Both models are scored under the same protocol:

* **same reference data** — `metadata/waymo_goal_val_eval_set.pkl` over
  `data/scene_goal_preprocess_waymo/val`, prepared with `gt_format=goal`
  (valid-goal filter + `max_num_agents` closest), for both models.
* **same sampling protocol** — both generate *unconditionally* from a
  `(num_lanes, num_agents)` prior, with no dependence on the eval scenes.
  `ldm_adv.eval.sample_source=prior` leaves every conditioning label at its trained
  null token; the old `sample_source=dataset` path (ground-truth layout + labels) is
  a reconstruction setting, kept for the conditioned modes and diagnostics.
* **same metrics** — the lane/agent metrics only read the first 7 unified-format
  columns, so they are goal-agnostic ("w/o goal"). LDM-Adv additionally reports the
  goal-specific metrics ("w goal") automatically when generated samples carry goals.

Every metric is a distribution-vs-distribution comparison (JSD / Frechet) over two
independently pooled sets — nothing is matched scene-to-scene. So the two counts are
separate knobs:

* `metrics.num_samples` — how many generated scenes to score (default: all cached).
  10k is plenty for stable JSDs.
* `metrics.num_gt_samples` — how many real scenes to pool (default: the whole
  43658-scene eval set). Keep it at the full set: a bigger reference is strictly
  better, and it costs only ~1 min of preprocessing.

⚠️ Finite-sample JSD is biased by pool size, so when comparing two models give them
the **same** generated count and the **same** reference set.

Each model keeps the layout prior of its own training preprocessing: the baseline
uses `metadata/initial_prob_matrix_waymo.pt`, LDM-Adv uses
`metadata/initial_prob_matrix_goal_waymo.pt` (the goal preprocessing drops agents
without a valid goal and caps the scene at `max_num_agents`, so its layout
distribution differs).

⚠️ **Caveat**: `scenario_dreamer_ldm_large_waymo` was trained on
`scenario_dreamer_ae_preprocess_waymo`, so scoring it against the goal-preprocessed
val split measures it *out of* its training preprocessing. The evaluation protocol
is matched; the training data is not. A goal-dataset LDM baseline would be needed to
remove that caveat.

Rebuild the shared artifacts (one-off):

```bash
# shared reference set (43658 scenes: non-partitioned, >= 2 agents with a valid goal)
python scripts/create_waymo_goal_val_eval_set.py --num-workers 48

# (num_lanes, num_agents) prior for prior-mode ldm_adv sampling
python scripts/create_goal_init_prob_matrix.py --num-workers 48
```

```bash

# TODO: delete, fast check
python eval.py \
    dataset_name=waymo \
    model_name=ldm \
    ldm.eval.mode=initial_scene \
    ldm.eval.run_name=scenario_dreamer_ldm_large_waymo \
    ldm.model.autoencoder_run_name=scenario_dreamer_autoencoder_waymo \
    ldm.model.num_l2l_blocks=3 \
    ldm.eval.num_samples=10 \
    ldm.eval.batch_size=256 \
    ldm.eval.cache_samples=False \
    ldm.eval.visualize=True \
    hydra.run.dir=slurm_logs/eval_ldm_50k_b256
```

```bash
# scenario dreamer baseline LDM Large
# note this is large model, not base model size
# generate samples 10k
python eval.py \
    dataset_name=waymo \
    model_name=ldm \
    ldm.eval.mode=initial_scene \
    ldm.eval.run_name=scenario_dreamer_ldm_large_waymo \
    ldm.model.autoencoder_run_name=scenario_dreamer_autoencoder_waymo \
    ldm.model.num_l2l_blocks=3 \
    ldm.eval.num_samples=10000 \
    ldm.eval.batch_size=256 \
    ldm.eval.cache_samples=True \
    ldm.eval.visualize=False \
    hydra.run.dir=slurm_logs/eval_ldm_10k_b256

# calculate metrics against the SHARED goal-val reference set (w/o goal metrics).
# eval_set / gt_test_dir / gt_format now default to the shared set in
# cfgs/eval/waymo_ldm.yaml, so no override is needed. num_samples caps the GENERATED
# pool; the real pool defaults to all 43658 reference scenes.
python eval.py \
    dataset_name=waymo \
    model_name=ldm \
    ldm.eval.mode=metrics \
    ldm.eval.run_name=scenario_dreamer_ldm_large_waymo \
    +ldm.eval.metrics.num_samples=10000 \
    hydra.run.dir=slurm_logs/metrics_ldm_goal_val_10k
```

LDM-Adv, same protocol (unconditional prior sampling, shared reference set). Samples
land in `init_scene_prior_samples` and metrics in `metrics_init_scene_prior.pkl`, so
they never collide with the `dataset`-mode outputs.

```bash
# generate 10k unconditional samples (no dependence on the eval scenes).
# Same count as the baseline above -- finite-sample JSD is biased by pool size.
python eval.py \
    --config-name config_ldm_adv_train \
    ldm_adv.eval.run_name=scenario_dreamer_ldm_adv_train \
    ldm_adv.eval.mode=init_scene \
    ldm_adv.eval.sample_source=prior \
    ldm_adv.eval.num_samples=10000 \
    ldm_adv.eval.batch_size=256 \
    ldm_adv.eval.cache_samples=True \
    ldm_adv.eval.visualize=False \
    hydra.run.dir=slurm_logs/eval_adv_prior_init_scene_10k

# metrics: prints both the [w/o goal] lane+agent tables (directly comparable to the
# baseline above) and the [w goal] goal table; all three are saved in one pickle
python eval.py \
    --config-name config_ldm_adv_train \
    ldm_adv.eval.run_name=scenario_dreamer_ldm_adv_train \
    ldm_adv.eval.mode=metrics \
    ldm_adv.eval.metrics.mode=init_scene \
    ldm_adv.eval.sample_source=prior \
    +ldm_adv.eval.metrics.num_samples=10000 \
    hydra.run.dir=slurm_logs/metrics_adv_prior_init_scene_10k
```

Reconstruction setting (ground-truth layout + conditioning labels), for diagnostics
and the conditioned modes — NOT comparable to the baseline:

```bash
python eval.py \
    --config-name config_ldm_adv_train \
    ldm_adv.eval.run_name=scenario_dreamer_ldm_adv_train \
    ldm_adv.eval.mode=init_scene \
    ldm_adv.eval.sample_source=dataset \
    ldm_adv.eval.num_samples=10000 \
    ldm_adv.eval.batch_size=256 \
    ldm_adv.eval.cache_samples=True \
    ldm_adv.eval.visualize=False \
    hydra.run.dir=slurm_logs/eval_adv_dataset_init_scene_10k
```
