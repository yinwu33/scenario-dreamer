#!/bin/bash
# table_main_v3: hierarchical_v3 DDPO at 500 iterations, scored in the
# ADVERSARIAL scope (ego vs the generated adversary) to match the reward.
#
# Rows: original / proximity_adv / base_gen / ddpo_gen / original_ddpo_adv.
# best-of-K is deliberately absent -- reusing the v2-selected artifacts would
# understate the baseline, and re-selecting under v3 costs ~18 h. TODO.
#
# Every stage is guarded by its own output file, so a killed run resumes.
# `set -eu` breaks define_env_variables.sh (PYTHONPATH unbound); use -eo.
set -eo pipefail
cd /home/tjhu78u/workspace/scenario-dreamer
source scripts/define_env_variables.sh

ITERS=500
NSCENES=1000
OLD=data/critical_scene/table_main_20260830
OUT=data/critical_scene/table_main_v3
mkdir -p $OUT

# ppo-ppo_norm first: its 500-iteration checkpoint already exists (the 1000-it
# run is deterministic up to it 500 -- nothing schedules off num_iterations), so
# it exercises the generate + score stages in minutes before any training runs.
CELLS="
ppo-ppo_norm|ppo_normal|ppo_normal
idm-idm|idm|idm
idm-ppo_norm|idm|ppo_normal
idm-ppo_aggressive|idm|ppo_aggressive
idm-ppo_caution|idm|ppo_caution
ppo-idm|ppo_normal|idm
ppo-ppo_aggressive|ppo_normal|ppo_aggressive
ppo-ppo_caution|ppo_normal|ppo_caution
pdm-idm|pdm|idm
pdm-ppo_norm|pdm|ppo_normal
pdm-ppo_aggressive|pdm|ppo_aggressive
pdm-ppo_caution|pdm|ppo_caution
"

for spec in $CELLS; do
  CELL=${spec%%|*}; rest=${spec#*|}; SUT=${rest%%|*}; ENV=${rest#*|}
  RUN=${CELL}_v3
  CKDIR=data/critical_scene/critical_scene_ddpo_ldm_adv_ddim_${RUN}_hier_v3
  CKPT=$CKDIR/critical_scene_ddpo_ldm_adv_ddim_${RUN}_hier_v3_00$(printf %03d $ITERS).ckpt
  CDIR=$OUT/$CELL
  echo "[cell] ===== $CELL  sut=$SUT  env=adv=$ENV ====="
  # Each cell runs in its own subshell so one failure does not abort the
  # remaining cells. It must NOT be written as `if ! ( ... )`: bash disables
  # errexit inside a compound command used as an if-condition, and propagates
  # that into the subshell, so every failure is swallowed and the cell still
  # reports COMPLETE. Capture the exit status explicitly instead.
  #
  # A truncated checkpoint is what a full disk looks like here (818 MB per
  # save), so refuse to start a cell without room for one.
  FREE=$(df --output=avail -BG /home | tail -1 | tr -dc '0-9')
  if [ "${FREE:-0}" -lt 5 ]; then
    echo "[cell] $CELL FAILED -- only ${FREE}G free on /home, need >=5G"
    continue
  fi
  set +e
  (
  set -eo pipefail

  # -- 1. train -------------------------------------------------------------
  if [ -f "$CKPT" ]; then
    echo "[cell] $CELL train: skip, $CKPT exists"
  else
    .venv/bin/python train.py --config-name config_ldm_adv_ddpo \
      ddpo/reward=hierarchical_v3 \
      planner@ddpo.planner.sut=$SUT \
      planner@ddpo.planner.env=$ENV \
      planner@ddpo.planner.adv=$ENV \
      experiment.planner_name=$RUN \
      ddpo.algo.kl_target=0.2 ddpo.save_every=$ITERS ddpo.resume=false \
      ddpo.context_prior.path=$PROJECT_ROOT/data/headroom_probe/context_prior_${CELL}.json \
      ddpo.context_prior.focus_frac=0.7 ddpo.rollout_workers=16 \
      ddpo.num_iterations=$ITERS \
      hydra.run.dir=$PROJECT_ROOT/slurm_logs/tmv3_$CELL
    echo "[cell] $CELL train done"
  fi

  # -- 2. generate the four paired sources ----------------------------------
  if [ -f "$CDIR/artifacts/ddpo_gen.pt" ]; then
    echo "[cell] $CELL generate: skip"
  else
    .venv/bin/python scripts/run_ldm_adv_ppo_table.py \
      --out-dir $CDIR --num-scenes $NSCENES --split val --seed 0 --chunk-size 32 \
      --skip-benchmark \
      --overrides planner@ddpo.planner.sut=$SUT \
                  planner@ddpo.planner.env=$ENV \
                  planner@ddpo.planner.adv=$ENV \
      --ddpo-ckpt "$CKPT"
    echo "[cell] $CELL generate done"
  fi
  # proximity_adv is a geometric rule on log scenes, not a model output: reuse.
  ln -sf $PROJECT_ROOT/$OLD/$CELL/artifacts/proximity_adv.pt $CDIR/artifacts/proximity_adv.pt \
    || echo "[cell] $CELL: no proximity_adv artifact to reuse"

  # -- 3. score in the adversarial scope ------------------------------------
  # skip_rollout=false: the path-conflict screen retires non-conflicting scenes
  # before step 0, which leaves reached_goal at 0 and makes Succ. meaningless.
  # Measured: it does not change Coll./Coll._f at all.
  if [ -f "$CDIR/scored_adv.md" ]; then
    echo "[cell] $CELL score: skip"
  else
    .venv/bin/python scripts/score_adv_sources.py \
      --artifacts $CDIR/artifacts \
      --sut $SUT --env $ENV --adv $ENV \
      --workers 16 --batch-size 128 \
      --override ddpo.simulator.path_conflict.skip_rollout=false \
      --sources original proximity_adv base_gen ddpo_gen original_ddpo_adv \
      --out $CDIR/scored_adv.md
    echo "[cell] $CELL score done"
  fi
  )
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "[cell] $CELL FAILED (exit $rc) -- continuing with the next cell"
    continue
  fi
  echo "[cell] $CELL COMPLETE"
done
echo "table_main_v3 all done"
