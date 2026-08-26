# Before

```bash
source .venv/bin/activate
source ./scripts/define_env_variables.sh
```

# Repo layout

代码分层：仿真层与 RL 层分开，前者被 DDPO 和 planner benchmark 共用，两边都不知道对方存在。

```
sim/            仿真与度量，不含任何 RL / 扩散概念
  scenes.py     GeneratedScenes——任何场景来源交给 rollout 的唯一契约
  planners/     每角色一个 Planner(plan/apply)：idm（规则）/ ppo_*（冻结 PPO 网）
  runner.py     RolloutRunner + SimulatorConfig，按 sut/env/adv 分角色步进
  hooks.py      度量 hook（只测量，不判好坏；极性由调用方决定）
  world.py      SimScene，纯 numpy 复刻 PufferDrive rollout
  routes.py     车道图路径搜索（规则 planner 用）
  schema.py     agent-state 9/12 维 layout 的单一真源
nets/           冻结网络的架构移植（selfplay_drive/net.py = PufferDrive Drive 网）
checkpoints/planners/  planner 权重（不进源码目录）
ddpo/           只剩 RL：policy_ldm_adv.py 采样封装 / ddpo_loss.py / train_loop.py /
                reward.py（从 sim 的 metrics 组装对抗性标量，hook 集合在这里选）
```

两条约定：

* **同类能力共用父类和统一 API。** 所有 planner 都是 `Planner(cfg, *, role, device)`，
  所以任何 planner 都能填任何角色，benchmark 的一个 cell 纯靠配置组装决定。SUT 和
  traffic planner 理论上完全可互换，唯一的不对称在 hook —— 指标以 ego 为中心。
* **不用 fallback。** 配置读取一律严格，缺 key 直接报错，不走默认值
  （`sim/planners/base.py::require()`）。checkpoint / log / metrics 字典上的 `.get()`
  是真 Optional，不在此列。

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
python data_processing/waymo/preprocess_dataset_waymo_advscene.py \
    --split train --num-workers 64 \
    --waymo-dir /path/to/waymo/scenario/training \
    --output-dir $DATASET_ROOT/scene_goal_preprocess_waymo_v2
python data_processing/waymo/preprocess_dataset_waymo_advscene.py \
    --split val --num-workers 64 \
    --waymo-dir /path/to/waymo/scenario/validation \
    --output-dir $DATASET_ROOT/scene_goal_preprocess_waymo_v2

# temporary path: reuse the v1 pickles already on disk.
# --verify recomputes what v1 stored and checks the two agree.
python temp_scripts/convert_goal_v1_to_v2.py --split val --verify 200
python temp_scripts/convert_goal_v1_to_v2.py --split val   --num-workers 64
python temp_scripts/convert_goal_v1_to_v2.py --split train --num-workers 64
```

Everything downstream must then be rebuilt in this order:
`ae_goal` retrain -> latent cache (`advscene_ae_goal_latents_waymo_v2`) ->
`temp_scripts/create_waymo_goal_val_eval_set.py` -> `scripts/create_goal_init_prob_matrix.py`
-> `ldm_adv` retrain. The baseline LDM only needs its metrics recomputed against the new
reference set.

# Train

### Training of Adv Scene

完整流水线：训 goal autoencoder -> 缓存 latent -> 在 latent 上训 ldm_adv。

```bash
source scripts/define_env_variables.sh

# 1. 训 goal autoencoder（专用入口，配置在 cfgs/ae_goal/）
python train.py --config-name config_ae_goal

# 2. 缓存 latent（train/val 各跑一次）
python eval.py --config-name config_ae_goal \
    ae_goal.eval.cache_latents.enable_caching=True \
    ae_goal.eval.cache_latents.split_name=train
python eval.py --config-name config_ae_goal \
    ae_goal.eval.cache_latents.enable_caching=True \
    ae_goal.eval.cache_latents.split_name=val

# 3. 训 ldm_adv（在 goal-AE latent 上扩散 normal agents + 1 个 adversary）
python train.py --config-name config_ldm_adv_base
```

### critical_scene Stage 2 — DDPO

scenario-dreamer 内置，单 venv，无需 dump / .bin。

```bash
# 0. 一次性：把 PufferDrive 训好的 frozen PPO checkpoint 拷到 checkpoints/planners/ppo/
#    路径在 cfgs/planner/ppo_*.yaml 里配置（必须是绝对路径，用 ${project_root}/...）

source scripts/define_env_variables.sh

# 训练（统一走 train.py；只微调 adversary 分支，基座场景冻结）
# 二元名称 = SUT/ego planner - (background + generated adversary) planner；
# 这里的 PPO 固定为 ppo_normal。四个入口只声明「哪个 planner 驱动哪个角色」和
# 「从哪些 context 采样」，其余（reward=hierarchical_v2、DDIM-30、lr、k_steps、
# 有效性门控）全部继承自 cfgs/config_ldm_adv_ddpo.yaml。
python train.py --config-name config_ldm_adv_ddpo_idm_idm
python train.py --config-name config_ldm_adv_ddpo_idm_ppo
python train.py --config-name config_ldm_adv_ddpo_ppo_idm
python train.py --config-name config_ldm_adv_ddpo_ppo_ppo

# rollout 并行：把 rollout 分片到 N 个 worker 进程（sim/parallel.py）。rollout 占
# 一个 iteration 的 ~66%，且几乎全是 sim/world.py 里的纯 numpy 逐场景计算。
# 结果与单进程 BIT-EXACT（44 个指标 max|delta|=0，四种组合 × 4/8/16 workers 全部
# 验证过，见 scripts/rollout_fingerprint.py）。
# 约束：ceil(batch_size / 8) >= rollout_workers，保证每个 worker 拿到完整的 GRPO
# context group（batch 128 -> 最多 16）。
#
# 实测 rollout 阶段（H100 + 96 核，batch 128）：
#   idm-idm  11.6s -> 5.0s (8w, 2.3x) -> 3.1s (16w, 3.8x)   # 无中心前向，零同步
#   ppo-ppo  10.6s -> 5.8s (8w, 1.8x) -> 4.8s (16w, 2.2x)   # 三个角色的前向留在父进程
# 含 PPO 的组合上限更低：为了逐位一致，PPO 的批量前向必须留在父进程，workers 每步
# 要在 barrier 上会合（重batch 会让 logits 漂移 ~1e-5，足以在近似平局处翻转 argmax）。
python train.py --config-name config_ldm_adv_ddpo_ppo_ppo ddpo.rollout_workers=16

# 验证任何 rollout 改动仍然逐位一致：
python scripts/rollout_fingerprint.py --config-name config_ldm_adv_ddpo_idm_ppo --batch-size 128 \
    --workers 16 --selfcheck 4      # 同进程内单进程 vs 分片，连续多个 batch

# reward 是命令行上的一个轴，没有 per-reward 的入口文件；每个变体自带
# reward_run_tag，落在各自的 output_dir，resume 不会串。
python train.py --config-name config_ldm_adv_ddpo_idm_idm ddpo/reward=full

# 通用入口仍可用于临时组合/覆盖（默认是 legacy 的全 PPO 三角色）
python train.py --config-name config_ldm_adv_ddpo ddpo.sampler=ddpm  # 退回 DDPM 采样
```

入口配置 `cfgs/config_ldm_adv_ddpo.yaml`（`experiment.*` 派生 run_name / output_dir /
wandb）。五个正交的配置组由入口 defaults 组装：

* **flow**：`cfgs/ddpo/ldm_adv.yaml`（conditioning / ckpt / pool + simulator +
  优化与采样）
* **算法**：`cfgs/ddpo/algo/grpo.yaml`（group_size / 白化 / clip / KL 信任域；换 PPO 时
  加一个 sibling yaml，`ddpo/algo@ddpo.algo=ppo` 一键切换，代码读 `cfg.algo` 不变）
* **reward**：`cfgs/ddpo/reward/*.yaml`（标量奖励由哪些项、以什么权重组装。yaml 里的
  `name:` 选中变体，其余 key 1:1 映射该变体的 config，见 `ddpo/reward/<name>.py`）。
  `full` = 一直在用的那套（TTC + approach + lane/overlap
  约束 + collision bonus）；`ttc_only` = 只保留 min-TTC 的消融基线。切换：
  `ddpo/reward=ttc_only`。每个变体自带 `ddpo.reward_run_tag` 后缀（`full` 为空串，其余
  为 `_<变体名>`），所以消融跑进自己的 output_dir，不会被 `resume=true` 接到默认那次
  run 的 checkpoint 上。做“逐项加回来”的实验时，复制一份 yaml、只打开一个权重、改一
  下 `reward_run_tag` 即可
* **planner**：`cfgs/planner/ppo_*.yaml`（checkpoint / 网络结构 + 该策略的
  conditioning obs：collision/offroad factor、lane_width，标量或 `[lo,hi]` 采样），
  按角色组装 `planner@ddpo.planner.{sut,env,adv}`。通用入口当前组装：
  sut = `ppo_normal`，env / adv = `ppo_aggressive`；上面的四个 planner-pair 入口则把
  名称中的 `ppo` 固定为 `ppo_normal`。三个 PPO 变体的 `collision_factor` 分别为
  aggressive=0.5、normal=1.0、caution=2.0；也可按角色覆盖，例如：
  `ddpo.planner.adv.conditioning.collision_factor=0`。
* **rollout 动力学**：`cfgs/rollout/base.yaml`（dt / goal 行为 / map_extent；组装到
  `ddpo.planner.sim`，完整显式、无隐藏默认）

产物在 `${SCRATCH_ROOT}/critical_scene/<run_name>/`（checkpoints / generated / media）。
checkpoint 存为 Lightning 兼容格式（`diff_model.*` 前缀），可被 eval.py / viz 直接加载。

# Evaluation

### Evaluation of Autoencoder

Scenario Dreamer Baseline

```bash
# download data/checkpoints/scenario_dreamer_autoencoder_waymo/last.ckpt

# run on original dataset
python eval.py dataset_name=waymo model_name=autoencoder \
    ae.eval.run_name=scenario_dreamer_autoencoder_waymo

# run on my dataset
python eval.py --config-name config_ae_my_dataset
```

AdvScene

```bash
# run on my dataset
python eval.py --config-name config_ae_goal \
    +ae_goal.eval.split_name=val \
    ae_goal.eval.run_name=scenario_dreamer_ae_goal_waymo
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

#### Four-source critical-scene evaluation

四源对比 `original` / `base_gen` / `ddpo_gen` / `original_ddpo_adv`，都在 ppo
rollout 下打分（同一批 template 场景，配对可比）。

```bash
.venv/bin/python scripts/run_ldm_adv_ppo_table.py \
    --num-scenes 1000 --chunk-size 32 \
    --out-dir data/critical_scene/ldm_adv_ppo_eval

# 2x2 GIF 可视化（依赖上一步的 benchmark 输出）
.venv/bin/python scripts/render_ldm_adv_2x2_gifs.py \
    --out-dir data/critical_scene/ldm_adv_ppo_eval
```

# Planner benchmark (SUT x traffic planner x scene initialization)

Fills one cell of the evaluation table: an ego planner (the system under test) and a
traffic planner share a scene, and the ego's Succ. / Off. / Coll. rates are measured.
The table's two planner axes ARE the rollout's role axes (`planner.sut` = row,
`planner.env` = column), so a cell is selected purely by composing
`cfgs/planner/<name>.yaml` -- the harness never special-cases a planner.

Planners: `idm` (rule-based, routes through the lane graph to its goal) and the
**ppo family** -- `ppo_aggressive` / `ppo_normal` / `ppo_caution`, all the same
frozen PufferDrive PPO checkpoint at different `conditioning.collision_factor`
values (0 = aggressive, 2 = cautious). They are separate planner names, so the
PPO_agg / PPO_norm / PPO_caution columns are selected by `--env ppo_<variant>`
with no override and no new code.

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
python scripts/run_planner_matrix.py --sut ppo_normal --env ppo_normal \
    --num-scenes 1000 --batch-size 50 \
    --out-dir $DATASET_ROOT/critical_scene/planner_matrix_log_1000

# an aggressive-PPO traffic column
python scripts/run_planner_matrix.py --sut idm --env ppo_aggressive \
    --num-scenes 1000 --batch-size 50 \
    --out-dir $DATASET_ROOT/critical_scene/planner_matrix_log_1000
```

Each cell writes `benchmark/<sut>__<env>__<source>/{per_scene.csv,summary.json}` and the
run rebuilds `table.{csv,md}` from every `summary.json` present, so cells accumulate into
one table across invocations.

## Scene initialization axis (`--source`)

`--source log` reads the preprocessed Waymo pickles. Any other name is a directory of
samples cached by `utils.data_helpers.convert_batch_to_scenarios` (i.e. `eval.py` with
`<model>.eval.cache_samples=True`), declared under `benchmark.gen_dirs` in
`cfgs/config_planner_matrix.yaml` — adding a checkpoint's cache to the table is one yaml
line and no code. `ldm_adv_base` ships pointing at the 10k unconditional ldm_adv-base
samples.

```bash
# same IDM x IDM cell, but on generated scenes
python scripts/run_planner_matrix.py --sut idm --env idm --source ldm_adv_base \
    --num-scenes 1000 --batch-size 50 \
    --out-dir $DATASET_ROOT/critical_scene/planner_matrix_ldm_adv_base_1000

# drive the generated adversary with a distinct planner instead of folding it into traffic
python scripts/run_planner_matrix.py --sut idm --env ppo_normal --source ldm_adv_base \
    --adv ppo_aggressive \
    --num-scenes 1000 --batch-size 50 \
    --out-dir $DATASET_ROOT/critical_scene/planner_matrix_ldm_adv_base_1000
```

The cache is already sim-ready: the goal autoencoder decodes goals as part of the agent
state, so there is no `prepare_scene` step (and no trajectories to run one on). Only the
lane graph is reconstructed — the cache stores `road_connection_types` for every ordered
lane pair but not the edge index, which is the dense row-major enumeration
(`gen_scenes.dense_lane_edge_index`, verified against the `edge_index_lane_to_lane` real
scenes store next to the same array).

Each scene's LAST agent — where `convert_batch_to_scenarios` appends the generated
adversary — is always driven by `planner.adv`, which defaults to the same planner as
`--env` (so the traffic column means the same thing as on the log row) and can be set to
a distinct planner with `--adv`.

Reading the generated rows against the log row (measured on 400 val scenes vs the 10k
cache):

* **Not paired.** `init_scene` samples are unconditional (layout from the prior, all
  conditioning at the null token), so they correspond to no particular log scene. Only
  distribution-level comparison is meaningful, and `dataset_scene_idx` indexes the cache.
* **Generated egos have nearer goals** — spawn→goal median 15.3 m vs 29.5 m, driving-ego
  share 57% vs 66%. Success rates are inflated by the shorter drive, so always read them
  next to `ego_goal_dist_mean`.
* **The maps are generated**, so lane connectivity is only approximately consistent with
  lane geometry: an upstream lane's end sits a median 0.25 m (p90 1.0 m) from its `succ`
  lane's start, against exactly 0 in the log. The topological route search absorbs this —
  on 400 generated scenes ego route coverage is 99.6% (log: 97.8%), all-agent 88.5%
  (91.3%), detour p99 2.09 (1.81). Watch `route_unavailable_rate` per run anyway, since
  it is the generator's property, not the planner's:

  ```bash
  python test_scripts/test_routes.py --num-scenes 400 --gen-dir $DATASET_ROOT/adv_scene_ldm_adv_base
  ```
* **Agent sets are directly comparable**: 10.0 vs 9.2 agents/scene, 26.9 vs 27.8
  lanes/scene, and 0.5% vs 0.7% of egos spawning beyond the 2.75 m off-road proxy from
  every centerline — so `ego_offroad_rate_driving` needs no caveat.

## Visualization (GIF)

`--gif N` re-rolls N of the evaluated scenes with trajectory recording and writes one
animated GIF each to `benchmark/<cell>/gifs/`, named by outcome
(`collision_scene18629.gif`, `timeout_scene727.gif`, `reached_scene...gif`). Scenes are
picked stratified by outcome -- collisions first, then non-arrivals, then off-road, then
long-drive successes as the control -- because a random sample is mostly uneventful.
Rendering is a second pass on purpose: recording trajectories for a whole 1000-scene
sweep costs a lot of memory for frames nobody looks at.

```bash
# benchmark a cell and save 8 GIFs of the most informative scenes
python scripts/run_planner_matrix.py --sut idm --env idm \
    --num-scenes 1000 --batch-size 50 --gif 8 \
    --out-dir $DATASET_ROOT/critical_scene/planner_matrix_log_1000

# just look at some rollouts, no big sweep
python scripts/run_planner_matrix.py --sut idm --env idm \
    --num-scenes 64 --gif 6 --gif-fps 10 \
    --out-dir /tmp/idm_look
```

Ego is red, other vehicles blue; a moving agent's goal is a dotted line to an `x`,
parked/static agents get a bold black `x` at their centre (`ddpo/viz.py`).

`idm` routes ALWAYS follow lane centerlines -- there is no straight-line fallback. When
the lane graph has no path from spawn to goal the agent gets no route and coasts, and the
`route_unavailable_rate` column reports how often that happened, so "drove badly" stays
distinguishable from "was never given a path". Coverage on val: 97.8% of egos, 91.3% of
all agents. Three things carry that number, all of them worth ~4-12 points each: lateral
(left/right) lane neighbours join the candidate sets so a lane change is representable,
start and goal lanes are chosen *jointly* rather than independently, and trimming
projects onto the centerline by arc length instead of snapping to one of the 20 vertices.

Sanity check on the route search before trusting any IDM row (asserts routes start at the
agent, end at its goal, are evenly spaced, and bounds coverage + detour):

```bash
python test_scripts/test_routes.py --num-scenes 200
```

# Appendix: legacy PufferDrive-hosted DDPO

旧版流程，已被上面内置的 Stage 2 DDPO 替代，仅作记录。

```bash
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
