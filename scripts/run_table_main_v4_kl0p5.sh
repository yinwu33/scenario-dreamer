#!/bin/bash
# table_main_v4_kl0p5: the SAME reward and iteration count as table_main_v4, with
# kl_target raised from 0.2 to 0.5. One variable, so table_main_v4 is the control.
#
# Why 0.5: a five-arm KL ablation on ppo-ppo_norm (0.1 / 0.2 / 0.5 / 1.0 / 2.0,
# 1000 iterations each, hierarchical_v4) put P(minTTC<3s) at 59 / 65 / 74 / 68 /
# 61 against a base of 34, with no arm collapsing and init_invalid BELOW base in
# all five (4.1-4.7% against 6.10%). The peak at 0.5 is single-humped but 0.5
# does NOT separate from 0.2 head to head (p=0.14 at tau=3s, n=984), so this
# table is what decides it. Everything left of the sweep is settled: kl_coef=0
# collapses by iteration 77 and freezes at exactly zero gradient by 240.
#
# 500 iterations, matching table_main_v4 exactly. 500 is not under-training:
# v2's _00500 and _03000 artifacts are statistically identical on held-out data
# (6 cells, n=5903, p>=0.19 on every column), and the ablation arms plateau by
# iteration 300.
#
# Cell order is v4's. Every stage is guarded by its own output file, so a killed
# run resumes. `set -eu` breaks define_env_variables.sh (PYTHONPATH unbound).
set -eo pipefail
cd /home/tjhu78u/workspace/scenario-dreamer
source scripts/define_env_variables.sh

ITERS=500
NSCENES=1000
# v3 has a proximity_adv for all 12 cells now, and it is derived from the
# planner-independent `original` scenes, so reusing it keeps the baseline
# row IDENTICAL between the v3 and v4 tables.
OLD=data/critical_scene/table_main_v4
OUT=data/critical_scene/table_main_v4_kl0p5
mkdir -p $OUT

# ppo-ppo_norm first: its 500-iteration checkpoint already exists (the 1000-it
# run is deterministic up to it 500 -- nothing schedules off num_iterations), so
# it exercises the generate + score stages in minutes before any training runs.
CELLS="
ppo-idm|ppo_normal|idm
idm-idm|idm|idm
pdm-idm|pdm|idm
ppo-ppo_norm|ppo_normal|ppo_normal
idm-ppo_norm|idm|ppo_normal
pdm-ppo_norm|pdm|ppo_normal
ppo-ppo_caution|ppo_normal|ppo_caution
idm-ppo_caution|idm|ppo_caution
pdm-ppo_caution|pdm|ppo_caution
ppo-ppo_aggressive|ppo_normal|ppo_aggressive
idm-ppo_aggressive|idm|ppo_aggressive
pdm-ppo_aggressive|pdm|ppo_aggressive
"

for spec in $CELLS; do
  CELL=${spec%%|*}; rest=${spec#*|}; SUT=${rest%%|*}; ENV=${rest#*|}
  RUN=${CELL}_v4kl5
  CKDIR=data/critical_scene/critical_scene_ddpo_ldm_adv_ddim_${RUN}_hier_v4
  CKPT=$CKDIR/critical_scene_ddpo_ldm_adv_ddim_${RUN}_hier_v4_00$(printf %03d $ITERS).ckpt
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
      ddpo/reward=hierarchical_v4 \
      planner@ddpo.planner.sut=$SUT \
      planner@ddpo.planner.env=$ENV \
      planner@ddpo.planner.adv=$ENV \
      experiment.planner_name=$RUN \
      ddpo.algo.kl_target=0.5 ddpo.save_every=$ITERS ddpo.resume=false \
      ddpo.eval_every=$ITERS \
      ddpo.context_prior.path=$PROJECT_ROOT/data/headroom_probe/context_prior_${CELL}.json \
      ddpo.context_prior.focus_frac=0.7 ddpo.rollout_workers=16 \
      ddpo.num_iterations=$ITERS \
      hydra.run.dir=$PROJECT_ROOT/slurm_logs/tmv4kl5_$CELL
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
  # proximity_adv is a geometric rule, but over THIS cell's own `original`
  # scenes, so it cannot simply be shared between cells.
  #
  # The previous form -- an unconditional `ln -sf` with an `|| echo` guard --
  # could not work: ln succeeds on a missing target, it just leaves a dangling
  # symlink, so the guard never fired and score_adv_sources.py died in
  # torch.load instead. That is what it did for every cell the 20260830 table
  # has no artifact for, i.e. all four pdm ones.
  PROX=$CDIR/artifacts/proximity_adv.pt
  OLD_PROX=$PROJECT_ROOT/$OLD/$CELL/artifacts/proximity_adv.pt
  if [ -e "$PROX" ]; then
    echo "[cell] $CELL proximity_adv: already present"
  elif [ -f "$OLD_PROX" ]; then
    ln -sf "$OLD_PROX" "$PROX"
    echo "[cell] $CELL proximity_adv: reused from $OLD"
  else
    # -e above is false for a dangling link, but writing through one would land
    # the file in the OLD table's directory, so drop it first.
    rm -f "$PROX"
    .venv/bin/python scripts/make_proximity_adv.py \
      --original $CDIR/artifacts/original.pt \
      --reference $CDIR/artifacts/ddpo_gen.pt \
      --out "$PROX"
    echo "[cell] $CELL proximity_adv: generated at the default 8 m clearance"
  fi

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
      --reward hierarchical_v4 \
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
echo "table_main_v4_kl0p5 all done"
