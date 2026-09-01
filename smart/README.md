# `smart/` — a SMART-style learned traffic model

The rest of this repository is a diffusion pipeline. This is a behavior model,
so it lives on its own: network, observation spec, rollout planner, data
processing and training are all here, and nothing was added to `models/`,
`nets/`, `datasets/` or `datamodules/`.

It drives the **traffic** (`env`) role, never the SUT. `Succ.` is a goal check in
`SimScene.goal_step`, and this model has no goal input, so a goal-free ego would
produce a number that is not comparable to the IDM and PPO columns.

## Seam with the main repo

Three lines outside this directory, by design:

| where | what | why it cannot live here |
| --- | --- | --- |
| `sim/planners/__init__.py` | imports `smart.planner.SMARTPlanner`, one `PLANNER_REGISTRY` entry | the registry is how any role's planner is resolved |
| `cfgs/planner/smart_probe.yaml` | the planner config | the rollout composes roles from the `planner` hydra group |

The dependency runs one way. This package imports from `sim` — the planner
contract, the world's scaling constants, and the shared accel/steer table with
its integrator. That last one is load bearing: action labels are produced by
inverting exactly the dynamics `SimScene.step_dynamics` will run, so a training
target cannot drift from what the simulator does.

## Why it shards, when `ctrl_sim` does not

`sim/parallel.py` shards a rollout by shuttling one flat `[rows, obs_dim]` matrix
through shared memory. `ctrl_sim` cannot use it because its input is a set of
agent-centric buffers plus lanes — an artifact of its shape, not of
agent-centric models. This planner's `gather` returns a flat 570-wide row, so
workers run the CPU halves and the parent runs one batched forward, exactly like
`sim/planners/ppo.py`.

Measured (128 cached scenes, `config_ldm_adv_ddpo_ppo_ppo`, only the `env` role
swapped): rollout phase 2.164 s → 2.619 s at 16 workers, **+21%**, and bit-exact
against the single-process runner (`max|delta| = 0.000e+00`). Inside DDPO that is
about 23 extra minutes per 3000-iteration run.

Read the ratio, not the seconds: the baseline rollout was 4.8 s when
`sim/parallel.py` was first profiled and is 2.164 s now, so absolute figures here
age quickly.

## Files

| file | role |
| --- | --- |
| `net.py` | observation layout constants, `SMARTTrafficNet`, `load_net` |
| `observation.py` | the observation builder, shared by training and rollout |
| `planner.py` | `SMARTPlanner`: `gather` / `forward` / `scatter` + rolling history |
| `actions.py` | the shared action table, its integrator, and label search |
| `records.py` | Waymo records → arrays, on the simulator's agent set |
| `preprocess.py` | caches a closed-loop action label per logged transition |
| `dataset.py` | labelled action CHAINS with randomly masked history |
| `trajectory.py` | differentiable K-step integration + the trajectory loss |
| `train.py` | training loop (cross-entropy + trajectory) |
| `evaluate.py` | closed-loop behavior check: ADE/FDE vs the log, collisions, off-road |
| `viz.py` | rollout pictures: logged track vs model track. Package-local on purpose |
| `check_action_table.py` | diagnostic: can the action table express logged motion? |

## Workflow

```bash
source .venv/bin/activate && source scripts/define_env_variables.sh

# 0. diagnostic: is the shared action table expressive enough?  (already run)
python smart/check_action_table.py --scenes 150

# 1. cache the action labels (~0.2 s per scene, so run it once, in parallel)
python smart/preprocess.py --split val   --workers 32
python smart/preprocess.py --split train --workers 32

# 2. train (cross-entropy on the action labels + a trajectory loss on the
#    integrated chain; --chain-steps is the horizon in 0.1 s sim steps)
python smart/train.py --steps 20000 --chain-steps 10 --traj-weight 1.0 \
    --wandb --run-name v1 --out checkpoints/planners/smart/v1.pt

#    --wandb logs ce / traj / lr per 100 steps, the val metrics, and a rollout
#    PICTURE at every validation. Project defaults to "smart-traffic", kept apart
#    from the diffusion repo's "scenario-dreamer".

# 3. does it drive like logged traffic?  ADE/FDE over 8 s with EVERY agent
#    driven by the model, plus the cold-start vs primed-history gap
python smart/evaluate.py --weights checkpoints/planners/smart/v1.pt --scenes 300

#    and to look at it rather than read numbers:
python smart/viz.py --weights checkpoints/planners/smart/v1.pt --scenes 6 --out rollout.png

# 4. serve: point a planner config at the checkpoint, then score it like any
#    other planner. Copy cfgs/planner/smart_probe.yaml, set
#    `weights: ${project_root}/checkpoints/planners/smart/v1.pt`, register the
#    name in sim/planners/__init__.py, then:
python scripts/score_paired_sources.py --sut ppo_normal --env smart --workers 16
```

## Measured

**The shared action table is expressive enough** (150 val scenes, 90,485
transitions, 1,286 replayed tracks with a median 8.4 s run):

| labelling | ADE | FDE | note |
| --- | --- | --- | --- |
| teacher-forced, per step | — | median 0.006 m | 97.1% within 5 cm |
| greedy per-step, replayed open loop | median 0.453 m | median 1.206 m | residuals accumulate |
| **closed-loop (`chase`)** | **median 0.022 m** | **median 0.023 m** | 99.2% of tracks under 0.5 m |

So the 1.2 m of the naive labeller is a labelling artifact, not a limit of the
action space. `preprocess.py` therefore labels with `chase`, and the paper keeps
a **single** integrator exception (`ctrl_sim`) rather than adding a second.

## Two losses, and why both

Cross-entropy alone optimises one action at a time and is blind to the way
per-step errors compound once the actions are integrated. So the model's own
action distribution is integrated K steps forward (`trajectory.py`) and the
resulting poses are scored in metres against the logged ones. The next pose is
the probability-weighted average of all 91 candidate poses -- `integrate` is
closed form and already vectorised over the table, so this is differentiable with
no sampling and no variance.

Context is teacher-forced (observations come from the logged state at each step)
but the agent's own pose free-runs, so drift accumulates across the chain and the
loss sees it.

Chain length is a real trade-off, measured on val:

| K | label actions | uniform policy | range |
| --- | --- | --- | --- |
| 5 | 0.058 m | 0.145 m | 0.087 |
| **10** | **0.086 m** | **0.259 m** | **0.173** |
| 20 | 0.154 m | 0.545 m | 0.392 |
| 40 | 0.188 m | 1.518 m | 1.330 |

Longer chains give a much stronger signal, but expected-action integration
averages poses, and over 2-4 s genuinely distinct futures (turn vs. straight)
start being averaged into a physically meaningless middle. The default is 1 s:
signal-to-floor 2.0, short enough that multimodality is not yet the dominant
effect. Cross-entropy stays primary and keeps the distribution sharp, since it is
a classification loss and does not blur modes together.

## Evaluating behavior

`evaluate.py` drives EVERY agent of a logged scene with the model for 8 s and
compares against what those agents really did. Training loss says nothing about
this; a model can have a fine cross-entropy and still drive into walls.

It reports the same rollout twice, with an empty history and with the 1 s of real
past handed to `SMARTPlanner.prime_history`. That gap is the direct test of
whether the random history masking worked, because a generated scene only ever
offers the cold case — if cold is much worse than primed, the model leans on a
past it will not have where it matters.

## Map coverage: measured, and why not simply "more segments"

The FOV is a 64 x 64 m box. Measured over val scenes: a map holds a median of 32
lane polylines (max 85) and 608 segments (max 1615). The observation takes the 64
nearest segments, which is **10.2% of the map**, drawn from a median of 9 lanes.

Raising that to 1024 segment tokens is the wrong lever: tokens go 91 -> 1051 and
attention is O(T^2), so ~130x the compute, which would also destroy the +21%
sharding result. The cheap route is a change of GRANULARITY -- one token per lane
polyline instead of one per segment:

| | now | 1024 segments | one token per lane |
| --- | --- | --- | --- |
| tokens | 91 | 1051 | 91 (cap 64 lanes) |
| obs_dim | 570 | 6330 | 2746 (still under PPO's 4036) |
| map covered | 10.2% | ~100% | ~100% |

The trade to watch: a pooled lane token summarises 20 points, and the model also
needs the precise lateral offset to the nearest centreline. A hybrid -- a few
nearest SEGMENTS for local geometry plus lane tokens for global structure --
keeps both. Not yet implemented.

## Known gaps

- `smart_probe` has untrained weights. It exists only to measure rollout cost and
  must never appear in a results table.
- Training reports top-1 over 91 actions, which understates quality because
  neighbouring accel/steer cells are nearly the same motion. Watch the val
  trajectory distance instead; the evaluation that counts is a closed-loop
  rollout scored like every other planner.
- This is single-step behaviour cloning with a trajectory correction, NOT SMART's
  joint spatio-temporal transformer. That was a deliberate trade: a joint model
  takes per-scene token grids and would land back where `ctrl_sim` is, losing the
  sharding result above. It trains on one target per row where SMART gets A x T,
  so expect it to need more data for the same quality.
- Expect this model to RAISE the Log baseline's collision rate and shrink
  AdvScene's margin: `ctrl_sim` as traffic already gives ego `Coll. 10.78`
  against `4.94` for `ppo_normal`. That is the honest cost of a realistic traffic
  column, and the reason to report it rather than avoid it.
