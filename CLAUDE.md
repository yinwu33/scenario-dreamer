# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Introduction

This is a research repo. The original repo is clone from [Scenario Dreamer](https://github.com/princeton-computational-imaging/scenario-dreamer.git). Scenario Dreamer provides an autoencoder and latent diffusion model to geenrate initial scene. Then use ctrl-sim to rollout the scenario. This is the baseline of this work. This work is called AdvScene, aiming to generate more challenging initial scene, rollout with pretrained planner and increase criticality.

The idea of AdvScene is to train autoencoder with goal firstly, then train latent diffusion model with a dedicated adv_agent network, then using DDPO to finetune this adv_agent network only with maintaining the other generation components fixed.

The evaluation should consider different combinations of SUT (system under test) ego planner, the environment's planners and the adversarial planner. Currently environment's planners and adv planner are same entity.

## Setup

Every command assumes both of these, run from the repo root (`define_env_variables.sh` uses `$(pwd)`):

```bash
source .venv/bin/activate
source scripts/define_env_variables.sh   # PROJECT_ROOT / SCRATCH_ROOT / DATASET_ROOT / CONFIG_PATH / PYTHONPATH
```

`SCRATCH_ROOT` and `DATASET_ROOT` both point at `<repo>/data`. The configs read them via
`${oc.env:...}` (`cfgs/_base.yaml`), so an unsourced shell fails at config resolution, not later.

`MYREADME.md` is the working command reference (bilingual EN/中文) and is kept current — read it
before inventing a command line. `README.md` is the upstream Scenario Dreamer readme.

## Commands

Everything trains through `train.py` and evaluates through `eval.py`, both Hydra entrypoints over
`cfgs/`. `--config-name` picks the entry config; `model_name` inside it selects the pipeline
(`model_registry.py` maps it to a Lightning module, except `model_name: ddpo`, which train.py
dispatches to its own RL loop before the registry collapse).

```bash
# --- generative pipeline (in order) ---
python train.py --config-name config_ae_goal                       # goal autoencoder
python eval.py  --config-name config_ae_goal \
    ae_goal.eval.cache_latents.enable_caching=True \
    ae_goal.eval.cache_latents.split_name=train                    # latent cache (train, then val)
python train.py --config-name config_ldm_adv_base                  # ldm_adv on the goal latents

# --- Stage 2: DDPO fine-tuning of the adversary branch ---
# Entry name = <sut planner>_<env+adv planner>; everything else is inherited from
# config_ldm_adv_ddpo.yaml (reward=hierarchical_v2, DDIM-30, lr, k_steps, validity gates).
python train.py --config-name config_ldm_adv_ddpo_idm_idm
python train.py --config-name config_ldm_adv_ddpo_idm_ppo
python train.py --config-name config_ldm_adv_ddpo_ppo_idm
python train.py --config-name config_ldm_adv_ddpo_ppo_ppo

python train.py --config-name config_ldm_adv_ddpo_idm_idm ddpo/reward=full   # reward is a CLI axis
python train.py --config-name config_ldm_adv_ddpo_ppo_ppo ddpo.rollout_workers=16

# --- planner benchmark (one table cell per invocation, cells accumulate into table.{csv,md}) ---
python scripts/run_planner_matrix.py --sut idm --env ppo_aggressive \
    --num-scenes 1000 --batch-size 50 --gif 8 \
    --out-dir $DATASET_ROOT/critical_scene/planner_matrix_log_1000

# --- four-source ldm_adv evaluation ---
.venv/bin/python scripts/run_ldm_adv_ppo_table.py --num-scenes 1000 --chunk-size 32 \
    --out-dir data/critical_scene/ldm_adv_ppo_eval
```

**Tests.** `unittest`, no pytest in the venv:

```bash
python -m unittest discover -s tests -v          # whole suite
python -m unittest tests.test_hierarchical_reward.HierarchicalV2Test.test_six_levels_strictly_ordered
```

`test_scripts/` holds manual diagnostic scripts (gitignored), not the suite — e.g.
`python test_scripts/test_routes.py --num-scenes 200` sanity-checks lane-graph route search before
trusting any IDM row.

**Pin `hydra.run.dir` on any multi-override run.** The default run dir embeds
`${hydra.job.override_dirname}` (`cfgs/_base.yaml`), which blows up on long override lists. Add
`hydra.run.dir=$PROJECT_ROOT/slurm_logs/<name>`.

## Architecture

The repo is layered so simulation knows nothing about RL and vice versa; DDPO and the planner
benchmark are two consumers of the same simulation layer.

```
sim/          simulation + measurement, zero RL/diffusion concepts
  scenes.py     GeneratedScenes — the ONLY contract any scene source hands the rollout
  runner.py     RolloutRunner + SimulatorConfig — steps the three roles, fires hooks
  planners/     one Planner(cfg, *, role, device) per policy; base.py holds the contract
  hooks.py      metric hooks — measure only, never judge; polarity is the caller's
  world.py      SimScene, a pure-numpy reimplementation of the PufferDrive rollout
  routes.py     lane-graph path search (rule-based planners only)
  parallel.py   bit-exact sharded rollout across worker processes
  schema.py     single source of truth for the 9/12-dim agent-state layout
nets/         frozen network architectures (selfplay_drive/net.py = the PufferDrive Drive net)
checkpoints/planners/   planner weights (kept out of source dirs)
ddpo/         RL only: policy_ldm_adv.py (sampling) / ddpo_loss.py / train_loop.py / reward/
critical_scene/  scene sources + eval harnesses (log_scenes, gen_scenes, ldm_adv_eval,
                 planner_matrix_eval) that feed sim/ and score its metrics
models/, nn_modules/, datasets/, datamodules/   the Lightning/diffusion stack
```

The repo aims to give a paper. In folder `research/`, latex files, references can be found.

### Three roles, one runner

`RolloutRunner` partitions every scene's agents into `sut` (the ego / system under test, local
index 0), `adv` (THE generated adversary, `scenes.adv_local_idx`), and `env` (all remaining
controlled agents). Each is driven by its own `Planner`, and the runner never special-cases which
planner sits in which role. Planning is two-phase: every role's `plan` runs off the same pre-step
state, then every `apply` integrates — so no role observes another's same-step movement.

The only asymmetry in the whole rollout is that the hooks score the ego. That is what lets the
benchmark's table axes literally be the rollout's role axes: a cell is selected by config
composition alone (`planner@ddpo.planner.sut` = row, `.env` = column).

### Config composition (Hydra groups)

A DDPO entry config composes five orthogonal groups; adding a variant is a yaml plus (for
planners/rewards) one registry line, never a code branch:

* `cfgs/ddpo/ldm_adv.yaml` — flow: conditioning, checkpoints, pools, `simulator:`, optimizer/sampler
* `cfgs/ddpo/algo/*.yaml` — RL algorithm (`grpo`); code reads `cfg.algo`, so swapping is one override
* `cfgs/ddpo/reward/*.yaml` — the `name:` key selects the assembler in `ddpo/reward/<name>.py`, and
  the remaining keys map **1:1** onto that variant's dataclass fields
* `cfgs/planner/*.yaml` — per-role planner: checkpoint/net plus that policy's `conditioning:` obs.
  The `ppo_*` family is one frozen checkpoint at three `collision_factor` values
  (aggressive 0.5 / normal 1.0 / caution 2.0), so they are three planners, not three policies.
* `cfgs/rollout/base.yaml` — shared rollout dynamics (dt, goal lifecycle, map extent), composed to
  `ddpo.planner.sim`

Every reward variant carries a `ddpo.reward_run_tag` suffix so ablations land in their own
`output_dir` and `resume: true` never picks up a checkpoint trained under a different reward.
Run artifacts go to `${SCRATCH_ROOT}/critical_scene/<run_name>/`; DDPO checkpoints are saved in
Lightning-compatible form (`diff_model.*` prefix) so `eval.py` and the viz scripts can load them.

## Conventions

**No fallbacks.** Do not write defensive code that hides errors or invalid states. Avoid fallbacks, silent recovery, and default-value access such as `dict.get(key, default)`. Do not use `try/except` or defensive `if/else` to mask unexpected failures; let errors propagate and fail loudly. Handle only genuinely expected business conditions explicitly.

**Shared capability, shared base class and API.** All planners take `(planner_cfg, *, role, device)`
and are looked up in `PLANNER_REGISTRY`, so any planner can fill any role. All rewards subclass
`RewardAssembler` and are looked up in `ddpo/reward/registry.py`.




Constraint: `ceil(batch_size / 8) >= rollout_workers`, so every worker owns a whole GRPO context group.

**Measure before spending a training run.** `scripts/headroom_probe.py` (best-of-N from the frozen
base: DDPO anchored by KL can only sharpen what the base can already sample),
`scripts/reward_screen.py` (re-scores one rollout dump under every candidate reward),
`scripts/build_context_prior.py` (turns a probe into the `ddpo.context_prior` manifest — the probe
found only ~25% of contexts attackable, so uniform sampling wastes most of a batch), and
`scripts/profile_ddpo.py` (per-phase wall clock; rollout is ~66%).

