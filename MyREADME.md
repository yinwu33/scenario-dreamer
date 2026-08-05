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
  `data/advscene_preprocess_waymo/val`, prepared with `gt_format=goal`
  (valid-goal filter + `max_num_agents` closest), for both models.
* **same sampling protocol** — both generate *unconditionally* from a
  `(num_lanes, num_agents)` prior, with no dependence on the eval scenes.
  LDM-Adv's prior-mode graph contains no conditioning labels, so every label uses
  its trained null token. Dataset/reconstruction sampling is no longer exposed by
  the base-evaluation entrypoint.
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

Both models use `metadata/initial_prob_matrix_goal_waymo.pt`, so they draw total
lane/agent counts from the same goal-data joint distribution. This prior is 2D and
has no `map_id` axis. The baseline stores `map_id=0` only as a schema placeholder
and runs with `ldm.train.guidance_scale=0.0`, so that placeholder does not condition
its generated scenes.

⚠️ **Caveat**: `scenario_dreamer_ldm_large_waymo` was trained on
`scenario_dreamer_ae_preprocess_waymo`, so scoring it against the goal-preprocessed
val split measures it *out of* its training preprocessing. The evaluation protocol
is matched; the training data is not. A goal-dataset LDM baseline would be needed to
remove that caveat.

The shared artifacts currently contain 43658 val scenes. Rebuild the LDM-Adv layout
prior after changing the goal latent cache:

```bash
# (num_lanes, num_agents) prior for prior-mode ldm_adv sampling
python scripts/create_goal_init_prob_matrix.py --num-workers 48
```

```bash
# scenario dreamer baseline LDM Large
# Its checkpoint is currently archived under data/checkpoints. Use a fresh
# sample directory rather than mixing these samples into the existing 50k cache.
python eval.py \
    dataset_name=waymo \
    model_name=ldm \
    ldm.eval.mode=initial_scene \
    ldm.eval.run_name=scenario_dreamer_ldm_large_waymo \
    ldm.eval.save_dir=$DATASET_ROOT/checkpoints \
    ldm.eval.init_prob_matrix_path=$PROJECT_ROOT/metadata/initial_prob_matrix_goal_waymo.pt \
    ldm.train.guidance_scale=0.0 \
    ldm.model.autoencoder_run_name=scenario_dreamer_autoencoder_waymo \
    ldm.model.num_l2l_blocks=3 \
    ldm.eval.num_samples=10000 \
    ldm.eval.batch_size=256 \
    ldm.eval.cache_samples=True \
    ldm.eval.visualize=False \
    +ldm.eval.cache_dir=$DATASET_ROOT/checkpoints/scenario_dreamer_ldm_large_waymo/initial_scene_advscene_fair10k_samples \
    hydra.run.dir=$PROJECT_ROOT/slurm_logs/eval_ldm_large_advscene_fair10k

# Score against all 43658 shared advscene val scenes. Only its [w/o goal]
# lane/agent tables are comparable to LDM-Adv.
python eval.py \
    dataset_name=waymo \
    model_name=ldm \
    ldm.eval.mode=metrics \
    ldm.eval.run_name=scenario_dreamer_ldm_large_waymo \
    ldm.eval.save_dir=$DATASET_ROOT/checkpoints \
    ldm.eval.metrics.samples_path=$DATASET_ROOT/checkpoints/scenario_dreamer_ldm_large_waymo/initial_scene_advscene_fair10k_samples \
    ldm.eval.metrics.metrics_save_path=$DATASET_ROOT/checkpoints/scenario_dreamer_ldm_large_waymo \
    ldm.eval.metrics.metrics_filename=metrics_advscene_fair10k.pkl \
    ldm.eval.metrics.eval_set=$PROJECT_ROOT/metadata/waymo_goal_val_eval_set.pkl \
    ldm.eval.metrics.gt_test_dir=$DATASET_ROOT/advscene_preprocess_waymo/val \
    ldm.eval.metrics.gt_format=goal \
    +ldm.eval.metrics.num_samples=10000 \
    +ldm.eval.metrics.num_gt_samples=43658 \
    hydra.run.dir=$PROJECT_ROOT/slurm_logs/metrics_ldm_large_advscene_fair10k
```

LDM-Adv Base, same layout prior, sample count, batch size, default seed (0), metric
code, and real reference pool.

```bash
# Generate 10k unconditional samples into a fresh cache.
python eval.py \
    --config-name config_ldm_adv_base \
    ldm_adv.eval.run_name=advscene_ldm_adv_base \
    ldm_adv.eval.mode=init_scene \
    ldm_adv.eval.num_samples=10000 \
    ldm_adv.eval.batch_size=256 \
    ldm_adv.eval.cache_samples=True \
    ldm_adv.eval.visualize=False \
    ldm_adv.eval.cache_dir=$DATASET_ROOT/checkpoints/advscene_ldm_adv_base/init_scene_advscene_fair10k_samples \
    hydra.run.dir=$PROJECT_ROOT/slurm_logs/eval_ldm_adv_base_advscene_fair10k

# metrics: prints both the [w/o goal] lane+agent tables (directly comparable to the
# baseline above) and the [w goal] goal table; all three are saved in one pickle
python eval.py \
    --config-name config_ldm_adv_base \
    ldm_adv.eval.run_name=advscene_ldm_adv_base \
    ldm_adv.eval.mode=metrics \
    ldm_adv.eval.metrics.mode=init_scene \
    ldm_adv.eval.metrics.samples_path=$DATASET_ROOT/checkpoints/advscene_ldm_adv_base/init_scene_advscene_fair10k_samples \
    ldm_adv.eval.metrics.metrics_save_path=$DATASET_ROOT/checkpoints/advscene_ldm_adv_base \
    ldm_adv.eval.metrics.metrics_filename=metrics_advscene_fair10k.pkl \
    ldm_adv.eval.metrics.eval_set=$PROJECT_ROOT/metadata/waymo_goal_val_eval_set.pkl \
    ldm_adv.eval.metrics.gt_test_dir=$DATASET_ROOT/advscene_preprocess_waymo/val \
    ldm_adv.eval.metrics.gt_format=goal \
    +ldm_adv.eval.metrics.num_samples=10000 \
    +ldm_adv.eval.metrics.num_gt_samples=43658 \
    hydra.run.dir=$PROJECT_ROOT/slurm_logs/metrics_ldm_adv_base_advscene_fair10k
```

# Planner benchmark (SUT x traffic planner x scene initialization)

Fills one cell of the evaluation table: an ego planner (the system under test) and a
traffic planner share a scene, and the ego's Succ. / Off. / Coll. rates are measured.
The table's two planner axes ARE the rollout's role axes (`planner.sut` = row,
`planner.env` = column), so a cell is selected purely by composing
`cfgs/planner/<name>.yaml` -- the harness never special-cases a planner.

Planners: `idm` (rule-based, routes through the lane graph to its goal) and
`bad_driver` (the frozen PufferDrive **PPO** policy; its
`conditioning.collision_factor` picks the driving style, 0 = aggressive,
2 = cautious, so the PPO_agg / PPO_norm / PPO_caution columns need no new code).

Metrics, all on the *driving-ego* subset (ego spawn->goal distance >= 10 m; a parked
ego reaches its goal for free):
`reached_goal_rate_driving` = Succ., `ego_offroad_rate_driving` = Off. (centerline
proxy at 2.75 m -- the maps carry no ROAD_EDGE entities), `ego_collision_rate_driving`
= Coll. (ego vs *any* vehicle, unlike the adversary-only DDPO reward hook).

```bash
# scene init = Log, SUT = IDM, traffic = IDM
python scripts/run_planner_matrix.py --sut idm --env idm \
    --num-scenes 1000 --batch-size 50 \
    --out-dir $DATASET_ROOT/critical_scene/planner_matrix_log_1000

# PPO x PPO control row, same scenes (--seed fixes the sample)
python scripts/run_planner_matrix.py --sut bad_driver --env bad_driver \
    --num-scenes 1000 --batch-size 50 \
    --out-dir $DATASET_ROOT/critical_scene/planner_matrix_log_1000

# an aggressive-PPO traffic column
python scripts/run_planner_matrix.py --sut idm --env bad_driver \
    --override planner.env.conditioning.collision_factor=0 \
    --num-scenes 1000 --batch-size 50 \
    --out-dir $DATASET_ROOT/critical_scene/planner_matrix_log_1000
```

Each cell writes `benchmark/<sut>__<env>__<source>/{per_scene.csv,summary.json}` and the
run rebuilds `table.{csv,md}` from every `summary.json` present, so cells accumulate into
one table across invocations.

Sanity check on the route search before trusting any IDM row (asserts routes start at the
agent, end at its goal, are evenly spaced, and bounds the straight-line fallback rate):

```bash
python test_scripts/test_routes.py --num-scenes 200
```
