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

must hold so each worker owns a complete GRPO context group.

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

## Current Blocker

The frozen PPO planner checkpoint appears to require conditioning values around:

```text
collision_factor = 2.0
offroad_factor = 2.0
```

Recent configs using `1.0 / 1.0` produced severely degraded planner behavior and invalidate experiments involving PPO roles.

Before rerunning affected experiments:

1. verify the conditioning distribution used to train the original PufferDrive checkpoint;
2. sweep the relevant conditioning values;
3. correct the planner configs;
4. rebuild/rescore experiments that used the incorrect PPO conditioning.

Do not treat previous PPO-based DDPO or planner-evaluation results as valid until this is resolved.