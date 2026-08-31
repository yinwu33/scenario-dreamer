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
Measure peak memory first; a batch-128 run has used about 47 GB on the 96 GB H100.

## Paper Evaluation

Results for the paper's tables live in `data/critical_scene/table_main_20260830/`,
one directory per planner pair plus `PROVENANCE.json`, which records every printed
number with its checkpoint, planner trio and denominator. Read that file rather
than re-deriving numbers.

Protocol for every cell: `--split val --num-scenes 1000`, the pair's `_03000.ckpt`,
`--workers 16`. Scene sources map to table rows as:

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
- DDPO checkpoints store raw `state_dict` with no EMA shadow, so the base model
  is evaluated with EMA weights and AdvScene without. This understates the
  AdvScene-vs-base delta rather than inflating it.
- The proximity baseline's clearance is load-bearing. 8 m is the smallest value
  that leaves the spawn-overlap rate at the log distribution's own 6.8%; 5 m
  inflates it to 14.8% and turns the baseline into an overlap generator (16.50%
  vs 6.50% collisions). The result is flat from 8 m to 16 m.
- `set -eu` breaks `scripts/define_env_variables.sh` (`PYTHONPATH` unbound). Use
  `set -eo pipefail` in launcher scripts.
- `pkill -f <script>` matches the wrapper shell running the command and kills the
  session's own bash. Kill by PID.

### Findings that should shape further work

- Most of the criticality gain comes from the generator, not from DDPO. Against
  logged scenes the base model gains +3.4 to +6.5 points; DDPO adds -0.19 to
  +7.67 on top, and that increment tracks how aggressive the traffic is
  (largest for `ppo_aggressive`, negative for `idm-idm`).
- Best-of-K from the frozen base overtakes AdvScene: one AdvScene sample is worth
  K=3 (IDM SUT) to K=6 (PPO SUT) base samples, and best-of-32 beats it outright.
  Report the strongest K, not a favourable one.
- Selecting best-of-K by reward recovers only ~45% of the oracle headroom
  (6.92% vs 15.97% at K=32), so the headroom probe's curve is a ceiling, not the
  baseline a practitioner achieves.
- Spawn overlap: log scenes 6.8%, `original_ddpo_adv` 12.8%, fully generated
  ~28%. Part of the fully-generated rows' collision rate is artifact, which is
  why `original_ddpo_adv` is the clean control.
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

## Current PPO Setup

The active planner configs are:

| Planner | Checkpoint | Collision / offroad conditioning |
| --- | --- | --- |
| `ppo_normal` | `cond_drive_178774809225.pt` | `0.5 / 0.5` |
| `ppo_aggressive` | `sut_drive_178776072918_aggressive.pt` | `0.1 / 0.1` |
| `ppo_caution` | `sut_drive_178777020497_caution.pt` | `3.0 / 3.0` |

These are distinct checkpoints, not a conditioning-only ablation. PPO results made
with the old `1.0 / 1.0` setup or a mismatched context prior must be regenerated.
