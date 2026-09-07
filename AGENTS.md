# AGENTS.md

## Project Overview

AdvScene is a research project built on [Scenario Dreamer](https://github.com/princeton-computational-imaging/scenario-dreamer).

Baseline pipeline:

```text
autoencoder / latent diffusion
→ initial scene generation
→ ctrl-sim rollout
```

AdvScene extends this by:

1. training a goal-conditioned autoencoder;
2. training a latent diffusion model with a dedicated `adv_agent` branch;
3. fine-tuning only the adversarial branch with DDPO while keeping the remaining generation components frozen;
4. evaluating scene criticality under different combinations of SUT, environment, and adversarial planners.

Currently the environment planner and adversarial planner use the same planner configuration.

`MYREADME.md` is the working command reference for this fork. Prefer it over the upstream `README.md` for project-specific workflows.

## Setup Commands

### Environment Setup

Run all commands from the repository root:

```bash
source .venv/bin/activate
source scripts/define_env_variables.sh
```

`define_env_variables.sh` sets:

- `PROJECT_ROOT`
- `SCRATCH_ROOT`
- `DATASET_ROOT`
- `CONFIG_PATH`
- `PYTHONPATH`

Hydra configs depend on these environment variables, so source the script before running training or evaluation.

### Run Commands

Main entry points:

```bash
python train.py --config-name <config>
python eval.py --config-name <config>
```

Main training pipeline:

```bash
python train.py --config-name config_ae_goal
python train.py --config-name config_ldm_adv_base

python train.py --config-name config_ldm_adv_ddpo_idm_idm
python train.py --config-name config_ldm_adv_ddpo_idm_ppo
python train.py --config-name config_ldm_adv_ddpo_ppo_idm
python train.py --config-name config_ldm_adv_ddpo_ppo_ppo
```

Current DDPO launch example (`ppo-ppo_aggressive`):

```bash
python train.py --config-name config_ldm_adv_ddpo \
  planner@ddpo.planner.sut=ppo_normal \
  planner@ddpo.planner.env=ppo_aggressive \
  planner@ddpo.planner.adv=ppo_aggressive \
  experiment.planner_name=ppo-ppo_aggressive \
  ddpo.context_prior.path=$PROJECT_ROOT/data/headroom_probe/context_prior_ppo-ppo_aggressive.json \
  ddpo.context_prior.focus_frac=0.7 \
  ddpo.rollout_workers=16 \
  ddpo.num_iterations=3000 \
  hydra.run.dir=$PROJECT_ROOT/slurm_logs/ddpo_ppo-ppo_aggressive
```

`ddpo.resume=true` resumes from an existing `<output_dir>/last.ckpt`; use a unique
`experiment.planner_name` for a new run.

Planner benchmark:

```bash
python scripts/run_planner_matrix.py --sut <planner> --env <planner> ...
```

For long Hydra override lists, explicitly set:

```text
hydra.run.dir=$PROJECT_ROOT/slurm_logs/<name>
```

to avoid overly long default output paths.

## Code Style

### 简洁优先

- 用最少的代码解决问题，拒绝冗余实现。
- 不使用 fallback、默认值或 defensive code 掩盖错误；非预期状态应直接报错。
- 不滥用 `if-else`、`try-except` 或 `dict.get(key, default)` 静默恢复错误。
- 只对真实、预期的业务条件进行显式处理。
- 不为一次性需求创建额外 abstraction 或复杂架构。
- 不为“未来可能需要”盲目增加扩展性或可配置性。
- 在当前任务范围内，如果实现明显可以简化，优先使用更简单的方案。
- 不以简化为理由扩大修改范围。
- 引入新 pattern、helper、dependency 或 abstraction 前，先检查项目已有实现。

### 精确修改

- 仅修改当前任务直接相关的代码。
- 不顺手优化相邻代码、注释或格式。
- 不重构与当前任务无关且正常工作的模块。
- 严格匹配项目现有代码风格。
- 可删除由本次修改直接产生的无效 import 或变量。
- 原有死代码或冗余内容仅提醒，不擅自删除。
- 当歧义会明显改变行为、接口、数据或修改范围时，再请求澄清。

## Architecture

Repository layout:

- `/sim`: simulation and measurement; should not depend on RL or diffusion internals.
- `/sim/planners`: planner implementations sharing the same planner API.
- `/ddpo`: DDPO sampling, loss, training loop, and reward implementations.
- `/critical_scene`: scene sources and evaluation harnesses.
- `/models`, `/nn_modules`, `/datasets`, `/datamodules`: Lightning / diffusion stack.
- `/checkpoints/planners`: planner checkpoints.
- `/tests`: automated test suite.
- `/test_scripts`: manual diagnostic scripts, not the formal test suite.
- `/temp_scripts`: temporary or one-off scripts.
- `/research`: LaTeX, references, experiment notes, and paper-related outputs.

### Simulation Roles

`RolloutRunner` treats controlled agents as three independent roles:

- `sut`: ego / system under test
- `adv`: generated adversarial agent
- `env`: remaining controlled agents

Each role uses the same `Planner` interface. Avoid role-specific branches when the behavior can be expressed through planner configuration.

Planning is two-phase: all planners observe the same pre-step state before actions are applied.

Current planner-pair convention:

| Pair | SUT | Environment / adversary |
| --- | --- | --- |
| `ppo-ppo_norm` | `ppo_normal` | `ppo_normal` |
| `ppo-ppo_aggressive` | `ppo_normal` | `ppo_aggressive` |
| `ppo-ppo_caution` | `ppo_normal` | `ppo_caution` |
| `idm-ppo_norm` | `idm` | `ppo_normal` |
| `idm-ppo_aggressive` | `idm` | `ppo_aggressive` |
| `idm-ppo_caution` | `idm` | `ppo_caution` |

For PPO-SUT experiments, keep the SUT on `ppo_normal`.

### Shared Interfaces

All planners use:

```text
(planner_cfg, *, role, device)
```

and are registered through `PLANNER_REGISTRY`.

All DDPO reward implementations subclass `RewardAssembler` and are registered through `ddpo/reward/registry.py`.

Hydra configuration should be preferred over introducing hard-coded planner, reward, or algorithm branches.

`plan(items)` is a BATCH entry point, not a per-agent one. Every planner decides
actions for all of its agents across all active scenes in one call, and the
rule-based ones are written that way:

- `sim.routes.LaneIndex` -- a scene's lane adjacency plus per-lane arc lengths,
  built once per `SimScene` and reused by every route search in it.
- `sim.routes.RoutePack` -- the driven agents' routes padded into `[A, Smax]`
  arrays, so one `project` call covers the whole batch. Built once per rollout
  and cached on each `SimScene` under a token; routes never change mid-rollout.
- `IDMPlanner._scene_state` -- every active scene's agent arrays padded to
  `[items, Nmax]`, so each agent reads its own scene's neighbours by index.

Padding must never change an answer: pad segments get `inf` distance before an
`argmin`, pad agent slots get a `False` candidate mask. That invariance is what
keeps `batched_across_scenes = False` correct for `idm`/`pdm` -- they still shard
with zero synchronisation, and a sharded rollout stays bit-exact even though each
worker pads to its own shard's dimensions.

`RoutePack.project` does NOT force a dtype; it promotes against whatever
`points` it is handed, because its two callers differ (idm projects float32
`SimScene` positions, pdm float64 proposal states). Forcing one would silently
change the other.

## Validation

Run the smallest relevant validation after modifying code.

Tests use `unittest`:

```bash
python -m unittest discover -s tests -v
python -m unittest <module.TestClass.test_method>
```

Do not assume pytest is available.

Manual diagnostics live under `test_scripts/` and are not substitutes for the automated test suite.

Do not modify unrelated behavior just to make tests pass.

If validation cannot be run, explicitly state why.

For DDPO:

```text
ceil(batch_size / 8) >= rollout_workers
```

must hold so each worker owns a complete GRPO context group. The same constraint
applies to any sharded rollout, so `score_paired_sources.py --workers 16` needs
`--batch-size 128`; below that `ParallelRolloutRunner` raises at construction.

After any change to `sim/`, check that the sharded rollout still agrees with the
single-process one:

```bash
python scripts/rollout_fingerprint.py --config-name <entry> \
  --override ddpo/reward=hierarchical_v3 --selfcheck 3 --workers 8 --batch-size 64
```

It runs several fresh batches through both runners in one process and diffs all
44 metric arrays; anything but `BIT-EXACT over all batches` is a regression.
Worker reuse and buffer reuse only misbehave on the SECOND rollout, which is why
it takes a batch count rather than a single batch.

When rewriting a planner, compare ACTIONS against the previous implementation on
the same states, not trajectories: one flipped `argmin` diverges everything
downstream, so a trajectory diff cannot tell a real regression from chaos. Drive
both implementations from the same states each step and count identical actions.

## Performance

Which phase dominates a DDPO iteration depends entirely on the planner pair.
Measured with `scripts/profile_ddpo.py` at batch 128, `ddpo/reward=hierarchical_v3`,
single process; JSONs in `data/critical_scene/profile_ddpo/`:

| pair | s/iter | `reward` (rollout) | `denoise_net` |
| --- | --- | --- | --- |
| `ppo-ppo_norm` | 9.0 | 34% | 61% |
| `ppo-idm` | 11.7 | 34% | 62% |
| `pdm-ppo_norm` | 24.8 | 78% | 21% |

`denoise_net` is `logprob_fwd` (58 forwards: `k_steps=32` x policy + reference)
plus `sample` (30 DDIM steps). For a neural traffic pair it is the long pole and
the rollout is not; for `pdm` it is the other way round. At 16 workers the same
runs measure 8.1 / 9.4 / 13.8 s per iteration.

Do NOT estimate rollout cost as `batch x sim_steps x max_agents`. With
`path_conflict.skip_rollout` on (every production reward config), a 128-scene
batch steps ~3200 scene-steps rather than 11648, and carries 8-13 agents per
scene rather than the configured 64. That arithmetic is 4-10x too high, and it
is what produced the stale "the rollout is ~66% of wall clock" claim that used
to head `sim/parallel.py`.

Things already fixed, so do not re-derive them:

- `shortest_lane_path` used to rebuild `[polyline_length(l) for l in lanes]` on
  every call (2.09 M calls per iteration). `LaneIndex` hoists it: route building
  went 12.87 s -> 1.17 s. This, not the step loop, was most of `idm`'s cost.
- `IDMPlanner.plan` / `PDMPlanner._proposal_actions` were per-agent Python
  loops. Batched: `plan_env` 16.12 -> 1.49 s, `plan_sut` 73.18 -> 16.04 s.
- `SimScene.update_metrics` tested one agent per `_sat_overlap` call; it now
  flattens every pair into one `sat_pairs` call (8-21x, identical output).
- `_role_items` used `np.intersect1d` on two already-sorted arrays 3 x scenes x
  steps times per rollout; role membership is a bool mask now (28 us -> 0.45 us).
- `scripts/profile_ddpo.py` omitted `train_batch_size`, so `--workers N > 0`
  always raised at construction. That is why no profile output existed in the
  repo before 2026-09-04.

Open leads, measured only as far as noted:

- `PDMPlanner` is still 65% of its pair's iteration: a 40-step horizon loop over
  an `[agents, 15]` grid, with `RoutePack.project` about two thirds of it.
- The denoiser's attention (`utils/dit_ex_layers.py:AttentionLayerDiT`) is a PyG
  `MessagePassing` layer that materialises `q_i * k_j` per EDGE over
  fully-connected within-scene edge sets, rather than a dense
  `scaled_dot_product_attention`. Unmeasured, but it is the plausible cause of
  both the `denoise_net` share above and the peak memory below.

## Git

- NEVER run `git add` or `git commit`.
- Only the user may stage or commit changes.
- Git operations should otherwise remain read-only unless explicitly requested.
- `git status`, `git diff`, `git log`, and `git show` are allowed.

## Research Workflow

Before spending a large training run, prefer cheap diagnostics when applicable:

- `scripts/headroom_probe.py`
- `scripts/reward_screen.py`
- `scripts/build_context_prior.py`
- `scripts/profile_ddpo.py`

Use measurements to validate that the base generator, reward, and rollout configuration have enough headroom before launching DDPO training.

A context prior is valid only for the exact `sut/env/adv` trio used by its
headroom probe. The standard probe is 1024 contexts x 32 samples with 16 workers;
the standard DDPO run uses batch size 128, 16 workers, 3000 iterations, and
`context_prior.focus_frac=0.7`.

Periodic DDPO validation uses 64 scenes and is diagnostic only. Compare the base
model and selected checkpoints on the same 1000 validation scenes before drawing
conclusions about collision-rate improvements.

Low GPU utilization during CPU rollout does not imply that another DDPO run fits.
Measure peak memory first; a batch-128 run peaks at 53-85 GB on the 96 GB H100
(`scripts/run_pdm_ddpo_pipeline.sh` sizes its sequential schedule on that), so
two never fit. An earlier "about 47 GB" figure here was an underestimate.

`scripts/profile_ddpo.py` is the phase profiler and now works at any worker
count; pass `--out` so the phase table is persisted rather than only printed.
Use `--workers 0` when you need the per-hook / per-method breakdown -- the
wrappers it installs only exist in the single-process path, and with workers the
parent's timers can only see the central forwards.

## Paper Evaluation

**`data/critical_scene/table_main_20260830/` is VOID for every rollout column.**
It was scored 2026-09-02, before the 09-04 sim boundary, and with
`score_paired_sources.py` (ego-vs-ANY) rather than the adversarial scorer. Its
`PROVENANCE.json` is still the right record of WHICH checkpoint and planner trio
produced each artifact, and its artifacts are still usable -- re-score them, do
not quote them. See `### The sim boundary of 2026-09-04`.

The current adversarial numbers are the table in
`## What DDPO actually moves, and on which metric`, backed by
`data/critical_scene/table_main_v6/`, `_v7/` and
`data/critical_scene/rescore_face_20260907/`.

Protocol for every cell: `--split val --num-scenes 1000`, the pair's checkpoint
(`_03000.ckpt` for the 20260830 run, `_00500.ckpt` for the v3-v7 sweeps),
`--workers 16`.

`scripts/score_adv_sources.py --reward <name>` is REQUIRED. It used to inherit
the entrypoint config's reward, which is `hierarchical_v2`, so every
`scored_adv.npz` written before that fix holds a reward/tier column from v2
rather than from the reward the run was trained with -- visible as `tier=5.0` in
files for a four-level reward. The `Coll.` / `Coll._f` / `minTTC` columns come
from rollout metrics and were never affected. The reward name is now written into
the emitted markdown so a stale file identifies itself.

Scene sources map to table rows as:

| Table row | Source | Produced by |
| --- | --- | --- |
| Log | `original` | `run_ldm_adv_ppo_table.py` |
| Log + proximity adversary | `proximity_adv` | `make_proximity_adv.py` |
| AdvScene-base (1 sample) | `base_gen` | `run_ldm_adv_ppo_table.py` |
| AdvScene-base (best-of-K) | `base_gen_bok{K}` | `run_best_of_k.py` |
| AdvScene | `ddpo_gen` | `run_ldm_adv_ppo_table.py` |
| Log + AdvScene adversary | `original_ddpo_adv` | `run_ldm_adv_ppo_table.py` |

`run_ldm_adv_ppo_table.py` benchmarks through `RewardModel`, whose collision is
ego-vs-ADVERSARY. Table numbers come from `score_paired_sources.py`, which is
ego-vs-ANY. Never mix the two in one table; state which one a figure uses.

Scene artifacts carry no SUT, so evaluating one model against another planner is
pure re-scoring with a different `--sut` -- the transfer table costs no generation.

### Evaluation traps

- `insert_adv_as_extra` appends adversaries after all base agents, so
  `agent_scene_idx` is NOT monotonic. Group scenes with a stable argsort, never
  `searchsorted`. Payloads produced by slicing are scene-major instead, so two
  payloads of the same scenes can differ row-by-row while being identical
  per scene -- compare per scene, not element-wise.
- `original` is bit-reproducible; `base_gen` is not (float-level kernel
  nondeterminism, max ~5 cm per agent). It is planner-independent and
  semantically stable, so cross-cell numbers are comparable, but do not expect
  identical bytes.
- Generated rows report ~983 driving egos against 1000 for `original`. That is
  autoencoder reconstruction jitter around the 10 m threshold; the ego moves a
  median of 3 cm. Not a bug, but say so if a caption claims identical scenes.
- **There is no EMA mismatch between `base_gen` and `ddpo_gen`, and an earlier
  note here claiming one was wrong.** DDPO initialises its policy with
  `use_ema_weights=true` and maintains no EMA of its own, so `ddpo_gen` is
  sampled from (base EMA weights + DDPO updates) and `base_gen` from the base
  EMA weights: the pair is exactly before/after, with nothing discarded. The
  raw `state_dict` in a DDPO checkpoint is not a lost shadow.
- **The EMA axis is nonetheless worth 2x on `Coll._f`**, so never evaluate a base
  model with raw weights by accident. Measured over the 12 cells by
  regenerating `base_gen` with `ddpo.use_ema_weights=false`
  (`data/critical_scene/base_noema_20260907/`): `flt&appr` 42 -> 21 and
  `Coll._f` 64 -> 57, while `TTC<3s` is unchanged (283 -> 285) and `Coll.` barely
  moves (280 -> 258). Weight smoothing decides whether the ego ends up the
  aggressor; it does not decide how many near misses there are.
- The proximity baseline's clearance is load-bearing. 8 m is the smallest value
  that leaves the spawn-overlap rate at the log distribution's own 6.8%; 5 m
  inflates it to 14.8% and turns the baseline into an overlap generator (16.50%
  vs 6.50% collisions). The result is flat from 8 m to 16 m.
- `set -eu` breaks `scripts/define_env_variables.sh` (`PYTHONPATH` unbound). Use
  `set -eo pipefail` in launcher scripts.
- `pkill -f <script>` matches the wrapper shell running the command and kills the
  session's own bash. Kill by PID.

### Findings that should shape further work

The first three of these were measured on the VOID 20260830 root. Their shape is
probably right and their numbers are not; re-score before quoting any of them.

- Most of the criticality gain comes from the generator, not from DDPO. Against
  logged scenes the base model gains +3.4 to +6.5 points; DDPO adds -0.19 to
  +7.67 on top, and that increment tracks how aggressive the traffic is
  (largest for `ppo_aggressive`, negative for `idm-idm`). [pre-boundary]
- Best-of-K from the frozen base overtakes AdvScene: one AdvScene sample is worth
  K=3 (IDM SUT) to K=6 (PPO SUT) base samples, and best-of-32 beats it outright.
  Report the strongest K, not a favourable one. [pre-boundary]
- Selecting best-of-K by reward recovers only ~45% of the oracle headroom
  (6.92% vs 15.97% at K=32), so the headroom probe's curve is a ceiling, not the
  baseline a practitioner achieves. [pre-boundary]
- Spawn overlap: log scenes 6.8%, `original_ddpo_adv` 12.8%, fully generated
  ~28%. Part of the fully-generated rows' collision rate is artifact, which is
  why `original_ddpo_adv` is the clean control. This one SURVIVES the boundary:
  it is a property of the initialisation, not of the rollout.
- Results in `data/critical_scene/table_main/` (2026-08-26) are void: they were
  measured with the broken `1.0 / 1.0` PPO config, which reported 9.80% ego
  success where the healthy planner reports 95.21%.

## CtRL-Sim as the Behavior-Driven Baseline

`sim/planners/ctrl_sim.py` runs the frozen CtRL-Sim checkpoint
(`data/checkpoints/ctrl_sim_waymo_1M_steps/last.ckpt`) as a rollout role. Two
registry names share it: `ctrl_sim` (tilt 0) and `ctrl_sim_adv` (negative tilt).
It exists to give the paper the behavior-driven adversary it otherwise never
compares against.

Three properties that constrain how it may be used:

- **It emits a k-disks token, not an entry of the 7x13 accel/steer table**, and is
  integrated by `utils.k_disks_helpers.forward_k_disks`. This is the only planner
  that does not share the common integrator, and the paper states it as an
  explicit exception.
- **It has no goal input.** Its agent state is
  `[x, y, vx, vy, heading, length, width, exist]`. So driving an AdvScene-placed
  adversary with CtRL-Sim discards the generated goal, i.e. half of what AdvScene
  produces. A placement x behavior 2x2 built this way does NOT isolate placement:
  the behavior axis silently removes goal-conditioning too.
- **It cannot be sharded.** `sim.parallel`'s shared memory is one flat
  `[rows, obs_dim]` matrix; this planner's input is agent-centric buffers plus
  lanes. Score it with `--workers 0` (~15 min per 1000 scenes). `sim/parallel.py`
  raises a named error rather than failing on a missing `obs_dim`.

### Measured, and not worth rediscovering

- **The RTG tilt is a weak knob here.** Forcing the return-to-go to its extremes
  (bin 349 vs bin 0) moves the action distribution by only TV ~= 0.08, at context
  depths 0/5/20/40. Since tilt only moves the *sampled* RTG inside that range,
  that is an upper bound on what any tilt can do, and the measured sweep is flat:
  ego collision 14.20 / 13.60 / 13.90 / 13.90 at tilt 0 / -2 / -5 / -10.
  The strength of this baseline comes from the model swap, not from the tilt.
- **It is not a neutral third SUT.** As traffic at tilt 0 on 1000 log scenes it
  yields ego `Succ. 89.97 / Coll. 10.78`, against `4.94` for `ppo_normal` traffic
  and `8.83` for `idm`. It imitates real drivers including bad ones, so using it
  as a system under test would not be comparable to the other two columns.
- **As an adversary it beats our method on the log-scene family**: inserted by the
  proximity rule it reaches `14.20`, against `7.65` for the AdvScene adversary and
  `6.50` for the same placement driven by `ppo_normal` (PPO SUT, 1000 val scenes).
  Report this; it is the comparison a reviewer will construct anyway.
- No behavior-realism metric exists in this repo. The realism proxy covers the
  INITIALIZATION (spawn overlap) only, so the objection "that baseline is strong
  because its behavior is implausible" currently cannot be answered with a number.

## What DDPO actually moves, and on which metric

Every number below is from ONE scoring pass with the current `sim/`, over the 12
`table_main` cells (1000 val scenes each, driving subset, n = 11 808). The v2/v3/v4
rows come from `data/critical_scene/rescore_face_20260907/`, which re-rolls those
versions' stored `ddpo_gen` artifacts under today's code; v6 and v7 are read from
their own roots, and `base_gen` / `proximity_adv` from `table_main_v6`, which was
verified bit-identical to today. Do NOT mix in a number from any root's own
`scored_adv.md` -- see `### The sim boundary of 2026-09-04`.

| source | TTC<3s | p | TTC<1.5s | Coll. | Coll._f | p | fault&appr | p | ram/Coll |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_gen` | 283 | | 178 | 280 | 64 | | 42 | | 81.8% |
| v3 | 443 | 5e-10 | 285 | 564 | 66 | 0.93 | 40 | 0.90 | 89.2% |
| v4 | **463** | 5e-12 | 288 | 398 | 65 | 1.00 | 37 | 0.64 | 85.7% |
| v6 | 379 | 8e-05 | 221 | 306 | 44 | 0.06 | 25 | 0.04 | 86.6% |
| v7 | 373 | 6e-05 | 216 | 293 | 50 | 0.17 | 31 | 0.20 | 86.0% |
| `proximity_adv` | 333 | 0.12 | 275 | 441 | 95 | 0.02 | 56 | 0.18 | 85.9% |

`p` is a paired McNemar against `base_gen` on the same scenes. `fault&appr` is
`ego_fault_collision AND finite ego_min_ttc` -- the ego both approached and made
the contact. `ram/Coll` is the share of collisions with `ego_min_ttc = inf`, i.e.
the ego never approached at all.

**DDPO buys approach events and does not convert them.** Of the scenes where the
ego did approach (`TTC<3s`), the fraction that end in a collision the ego caused:

| source | TTC<3s | fault&appr | conversion | p vs base |
| --- | ---: | ---: | ---: | ---: |
| `base_gen` | 283 | 42 | 14.8% | |
| v3 | 443 | 40 | 9.0% | 0.022 |
| v4 | 463 | 37 | 8.0% | 0.0045 |
| v6 | 379 | 25 | 6.6% | 0.00064 |
| v7 | 373 | 31 | 8.3% | 0.012 |
| `proximity_adv` | 333 | 56 | 16.8% | 0.58 |

The two effects cancel: 463 x 8.0% is the same 37 as 283 x 14.8% is 42. That is
why the absolute fault count looks flat -- not because nothing happened.
`proximity_adv` is the control that matters: it does NOT lose conversion.

The mechanism is measured, not inferred. DDPO leaves the spawn distance alone
(median 16.7 m, same as base) and instead drives `ego_adv_min_dist` down
(12.54 -> 10.99 m) while cutting contacts before 1 s (80 -> 45 scenes) and spawn
overlap (6.1% -> 5.2%). That is exactly what the reward asks for: the TTC band is
reachable, the fault band is rare, and `hard_collision_t = 1.0` makes an early
contact -1. But a configuration tuned to minimise TTC gives the ego the whole
approach to react, and `ppo_normal` / `idm` / `pdm` all brake -- ego `Succ.` only
falls 86.1% -> 85.1%. `proximity_adv` optimises nothing and simply parks a car
8.2 m ahead (`path_conflict` 98.0%), which the ego cannot recover from.

So the TTC band is a proxy for the fault collision, and optimising the proxy
hard selects scenes that are critical BY THAT PROXY and recoverable in fact.
This is the sharper form of "the policy sits at the TTC ceiling": the scenes on
that ceiling do not structurally lead to contact.

**Near misses are the metric with power, and DDPO raises them.** `Coll._f` is 1 to
12 events per cell out of ~984 driving scenes, so per cell it measures nothing;
pooled it resolves, and only for v6/v7. Report event counts (`50/11808`), pool
before testing, never average per-cell percentages. The earlier "0 of 12 cells
positive" reading of the v4 sweep was noise.

**Restricting the reward costs near misses without buying fault collisions.** v3
and v4 hold `Coll._f` at the base model's level (66 and 65 against 64) while
raising near misses most; v6 and v7 significantly LOWER `fault&appr` (25 and 31
against 42) and land lower on TTC too. So "DDPO lowers ego-fault collisions" is a
property of v6/v7, not of DDPO.

This is the answer to the open risk `hierarchical_v4`'s docstring names: the TTC
ceiling is 0.85 and a fault collision is 1.0, but crossing between them means
turning a ~9% event into a <1% one, so the policy sits at the ceiling. Raising
the fault level's value cannot fix it; only making fault collisions less rare
would.

**The ram share is not a DDPO pathology.** `ram/Coll` is 82% for the FROZEN base
model and 86% for `proximity_adv`; every DDPO version sits in the same 86-89%
band. It is a property of the traffic planners, concentrated by cell: 93-96% in
the three `ppo_aggressive` cells against 40-46% in `ppo-idm` / `idm-idm`. A
reward that penalises it is deleting signal, not fixing a hack -- which is what
v5/v6/v7 measured.

`proximity_adv` is the baseline to beat, not a weak control: 95 fault collisions
against v4's 65 and v6's 44, at the base model's own fault share -- it simply
produces more collisions of every kind, by putting a car where the ego is going.
It is the comparison a reviewer will construct.

Two more traps in the same family:

- **Do not compare a training-log metric to an eval metric.** Training runs on
  `context_prior` with `focus_frac=0.7`, measured 4.5x more collision-prone than
  uniform val (train `coll` 0.132 against eval 0.0259); the denominators differ
  (whole batch of 128 against the driving subset); and `base_gen` is generated
  with EMA weights while `ddpo_gen` is not. A +35% move in the training `fault`
  is consistent with no move at all in eval.
- **TTC alone overstates `proximity_adv`.** In `ppo-ppo_norm` it reaches 92
  scenes under 3 s -- the highest of any source -- with 2 collisions and 0 fault
  collisions, because a car parked 8 m ahead guarantees a small TTC and
  `ppo_normal` simply brakes. Always print the collision count beside TTC.

The `driving` subset (`ego_goal_dist >= 10`) is a scene property, not a rollout
outcome -- it is spawn-to-goal distance, identical to 4 mm across SUTs -- so it
is exactly 984 scenes in all 12 cells and the pairing across cells is exact.

### The sim boundary of 2026-09-04

`sim/` changed materially at 09-04 12:07-12:08, and nothing measured or trained
before it is comparable with anything after:

- `d880e2f` controls EVERY agent, not only those spawned away from their goal.
  This changes which cars drive, so it changes every collision and every TTC.
- `fd327bc` gave PDM lateral-offset proposals, changing every `pdm-*` rollout.
- `5fda69d` made the ego-fault predicate geometric; `1062449` (09-05 22:44) then
  moved it to the front-face contact test.

What that invalidates, checked by re-scoring rather than assumed:

| root | scored | status |
| --- | --- | --- |
| `table_main_20260830` (v2) | 09-02 | void; also reports ego-vs-ANY from `score_paired_sources.py` |
| `table_main_v3` | 09-04 02:06 | void in EVERY rollout column -- re-scoring moves one scene's `ego_min_ttc` by 5.7 s |
| `table_main_v4` | 09-05 02:17 | TTC and Coll. bit-exact against today; fault column stale |
| `table_main_v5/v6/v7` | 09-05 -> 09-06 | current |

It also splits the CHECKPOINTS. v2 (trained 08-28) and v3 (trained 09-04 07:46,
four hours before the commit) optimised a different simulator; v4 onward did not.
Their artifacts can still be scored fairly -- that is what the re-score does --
but a v2-or-v3-versus-v4 gap is reward x sim-version, not a reward result. Only
v4/v5/v6/v7 form a clean reward comparison. Settling whether a fault-agnostic
collision band beats v4 needs v3 RETRAINED under the current sim, not re-scored.

## The reward series: use v4

**`hierarchical_v4` is the version to use.** v5, v6 and v7 are a closed negative
result: each removed a further way to score, and each cost near misses without
buying ego-fault collisions (the eval table above). Do not extend the series in
that direction, and do not revive v5's or v6's premise without new evidence.

`hierarchical_v3` is described below; v4 through v7 each change exactly one
thing, and all four trained under the same `sim/`, so they form a clean
comparison (v2 and v3 do not -- see `### The sim boundary of 2026-09-04`). The
per-version outcomes below are TRAINING-log metrics on prior-focused contexts,
not eval metrics -- see the trap above before comparing any of them to a table
column. What each change did to the EVAL numbers is the table above, not this
one: a version can look good here and lose there, which is exactly what v6 did.

| version | change from the previous | measured outcome |
| --- | --- | --- |
| v4 | collision level admits only ego-fault collisions | ram falls to `d_min`, where a contact scores that band's maximum: ramming paid ~4x a quiet scene in the three highest-ram cells |
| v5 | ram and off-lane join `invalid` (-1) | ramming stops (collisions 97 -> 29 scenes in idm-ppo_aggressive) but the invalid band never comes down: 22.4% flat over 500 iterations, 22.6%, 18.3%, 14.9% across four cells |
| v6 | ram scored 0 instead of -1; off-lane stays -1 | invalid halves to 11%, collisions return to base level, `tier3` rises in all 12 cells (+13% to +82%) |
| v7 | a ram that followed a real ego approach keeps the TTC band | rescores 0.15% of scenes, 9 of 12 of them in ppo-idm; recovers about half of v6's `fault&appr` loss (25 -> 31 against base 42) and nothing on TTC |

The carve-out v7 exists for did NOT pay off where it was designed to. Its
target is the `ppo_aggressive` family, the only cells with `tier4 ram` above
10%; pooled over those three cells it is not significant on any column
(p = 0.34 to 1.00). Two of the three looked positive on their own -- that is
what single cells of a 5-to-15-event metric do.

Five things measured across all 12 cells of each batch:

- **`tier0 invalid` and `init_invalid` are different quantities and move in
  OPPOSITE directions.** `init_invalid` (the placement flag) falls in 36 cells
  out of 36 across the v4, v6 and v7 batches, 7-9% down to 6-8%. `tier0` (the
  reward's invalid BAND, which also absorbs rejects and pre-`hard_collision_t`
  contact) rises. Do not read one as the other; both are called "invalid".
- **How much `tier0` rises separates v4 from v6/v7.** v4 is flat: mean 8.2% ->
  8.3%, up in only 8 of 12 cells. v6 and v7 climb 11.6% -> 14.1% and 11.8% ->
  14.0%, up in 12 of 12. So the versions that lose on eval are exactly the ones
  paying placement legality for tier2/tier3 mass; the winning one does not. An
  earlier note here recorded the 12-of-12 rise as universal -- it is v6/v7 only.
- **`grp_std` never collapsed.** It runs 0.23-0.40 and is HIGHER under the
  versions with a bigger invalid band, not lower -- a bimodal -1/positive reward
  has more spread, not less. The starvation worry that shaped v5's design was
  wrong.
- **`tier4 ram` moves in whichever direction its starting value implies**: cells
  starting above 10% fall (-9%, -20%), cells starting below 6% rise. A reward
  that removes the ram incentive can only act where there are rams to remove.
- **A negative mean training reward is not a failure.** Four cells ran negative
  for all 500 iterations (`pdm-idm`, `pdm-ppo_norm`, `pdm-ppo_caution`,
  `idm-ppo_norm`) and every one of them scored at or above `base_gen` on eval.
  GRPO whitens within the group, so the mean carries no information about
  whether the policy is improving; read `grp_std`, `tier2`/`tier3` and the eval
  pass instead.

## Reward: `hierarchical_v3`

Four levels, strictly ordered, selected with `ddpo/reward=hierarchical_v3`:

```text
invalid  ->  collision  ->  min TTC_ego  ->  d_min
```

```text
R = -1                              invalid, or contact before hard_collision_t
  = 0.9 + 0.1 * 1[ego at fault]     valid collision
  = 0.4 + 0.45 * g_ttc              ego TTC risk
  = 0.3 * g_d                       otherwise
```

with `g_ttc = clip(1 - minTTC_ego/tau, 0, 1)` and
`g_d = clip((d_far - dmin)/(d_far - d_near), 0, 1)`, the latter zeroed unless
`closed_in > close_delta`.

Every level above `invalid` measures ONE phenomenon at a different severity: the
ego running into the adversary. `ego_min_ttc` and `ego_fault_collision` gate on
the same idea at two distances -- `SimScene._ego_approaching_mask` (the forward
cone, for the approach) and `_ego_front_contact_mask` (the front face against the
other's box, for the contact) -- so the near miss and the crash are the same
event seen earlier or later. `hierarchical` (v2) did not have this property: its
collision level was fault-agnostic while its TTC level was ego-gated.

Four things that are load-bearing, each with the measurement behind it. The
first of them is what v4 overturned, and the reason it could be overturned is
that the fault predicate changed under it: the measurement below assumes
`fault | collision` = 44.4%, which the cone predicate produced. See
`### The ego-fault predicate`.

- **Fault is a bonus, not a level.** Making it its own top level inverts the
  expected ordering. `fault | collision` is 44.4% (ppo-ppo_norm `base_gen`, 984
  driving scenes), and whether the ego ends up the aggressor is mostly decided by
  the frozen ego, so a fault-only top level makes a reliable near miss worth more
  in expectation than causing a crash. At 0.9/1.0 a collision is worth
  `0.444*1.0 + 0.556*0.9 = 0.944` against 0.85 for the best possible near miss.
- **The TTC ceiling must stay below the collision base**, for the same reason.
  Raising the fault gap is possible but must be bought by LOWERING the TTC
  ceiling, not by lowering the non-fault collision value: at `R_nonfault = 0.75`
  the margin is 0.011, inside the drift of the 44.4% estimate.
- **`hard_collision_t = 1.0` makes an early contact INVALID, not merely
  uncredited.** Its post-contact `dmin` is large (the cars separated), which is
  what let spawn artifacts outrank quiet samples in v2. The cost is a cliff:
  1.05 s scores +1, 0.95 s scores -1.
- **`d_min` is absolute, gated on `closed_in`.** The relative form `1 - dmin/d0`
  hides nothing but scores a 40->20 m approach the same as 5->2.5 m; absolute
  distance alone is farmable by spawning alongside the ego and driving parallel
  (the hack `EgoAdvMinDistHook` warns about). Both are needed.

`lane_penalty` is 0 in v3: there is no band for it, so realism guarding falls
entirely to the `invalid` level.

### The ego-fault predicate

`ego_fault_collision` tests whether the ego's FRONT FACE -- the segment between
the two corners at `+length/2` -- overlaps the other's box
(`SimScene._ego_front_contact_mask`), plus an absolute speed gate. It used to ask
where the other agent's CENTRE fell inside a cone (`|y|/x <= W/L`), which asks
about the wrong point: two idm-idm scenes that read as head-on were scored
not-at-fault, one of them a two-degree miss with the ego closing at 6.6 m/s. The
face test raised the fault share of that cell's collisions from 33.3% to 47.1%.

The predicate is deliberately SPLIT. A face-vs-box overlap is only true at the
instant of contact, so gating `EgoMinTTCHook` on it puts `ego_min_ttc` at +inf
everywhere and collapses the TTC level. `_ego_approaching_mask` keeps the cone
for the approach -- it is the same front face swept forward. A scene can satisfy
the cone during the approach and have the contact land on the ego's flank: a
near miss the ego caused, but not a collision it caused.

`Coll._f` measured before this change is stale; `minTTC` is NOT, because the
cone half is untouched -- so v3-era and v4-era TTC columns compare directly.
Re-scoring needs no regeneration; the scene artifacts are unchanged.

### Why v2 failed, and what changed under it

v2's ego-fault bonus never produced gradient because `_ego_aggressor_mask` used
to test the ego's velocity PROJECTED onto the ego->other direction, under which
81% of collisions counted as ego-fault. A bonus present on 81% of the collision
band is nearly a constant offset, and GRPO's per-group whitening removes constant
offsets. The predicate is now geometric -- contact inside the cone the ego's own
front face subtends (`|y|/x <= W/L`) plus an absolute speed gate -- and the share
drops to 44.4%, so the flag discriminates.

That change moves `ego_fault_collision`, `ego_fault_collision_any` (the tables'
`Coll_f`) AND `ego_min_ttc`, since the TTC hook gates on the same predicate. Every
`Coll_f` and TTC number measured before it is stale.

## Current PPO Setup

The active planner configs are:

| Planner | Checkpoint | Collision / offroad conditioning |
| --- | --- | --- |
| `ppo_normal` | `cond_drive_178774809225.pt` | `0.5 / 0.5` |
| `ppo_aggressive` | `sut_drive_178776072918_aggressive.pt` | `0.1 / 0.1` |
| `ppo_caution` | `sut_drive_178777020497_caution.pt` | `3.0 / 3.0` |

These are distinct checkpoints, not a conditioning-only ablation. PPO results made
with the old `1.0 / 1.0` setup or a mismatched context prior must be regenerated.
