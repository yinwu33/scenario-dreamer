#!/bin/bash
# Overnight driver: headroom probe -> context prior -> DDPO, for each PDM pair.
#
# PDM is the SUT on every pair; traffic and adversary vary. Runs strictly
# sequentially -- one DDPO run peaks at 53-85 GB on a 96 GB card, so two never
# fit. A pair that fails is logged and SKIPPED rather than aborting the batch,
# so one bad probe cannot cost the other three runs a night.
#
#   bash scripts/run_pdm_ddpo_pipeline.sh
#
# Progress markers go to stdout prefixed '@@@' for the monitor to pick up;
# per-stage output goes to <LOG_ROOT>/<stage>_<pair>.log.

# NOTE: 'set -u' breaks scripts/define_env_variables.sh (PYTHONPATH unbound).
set -o pipefail

cd /home/tjhu78u/workspace/scenario-dreamer || exit 1
source .venv/bin/activate
source scripts/define_env_variables.sh

LOG_ROOT="$PROJECT_ROOT/slurm_logs/pdm_pipeline_20260902"
mkdir -p "$LOG_ROOT"

# Training budget. 1000 rather than the 3000 the earlier IDM/PPO cells used:
# pdm-idm's reward curve was flat from about it 600, so the extra 2000 buys
# noise. The PDM cells are therefore NOT budget-matched to the rest of
# table_main -- say so in the caption.
ITERS=1000

stamp() { date +%F_%T; }

run_pair() {
    local ENV_P="$1" NAME="$2" CFG="$3"
    local PROBE="$PROJECT_ROOT/data/headroom_probe/pdm-${ENV_P}-${ENV_P}_1024x32.json"
    local PRIOR="$PROJECT_ROOT/data/headroom_probe/context_prior_${NAME}.json"

    echo "@@@ [$(stamp)] $NAME PROBE start (sut=pdm env=$ENV_P adv=$ENV_P)"
    if [ -f "$PROBE" ]; then
        echo "@@@ [$(stamp)] $NAME PROBE reused existing $PROBE"
    else
        python scripts/headroom_probe.py \
            --sut pdm --env "$ENV_P" --adv "$ENV_P" \
            --num-contexts 1024 --samples-per-context 32 --workers 16 \
            --out "$PROBE" > "$LOG_ROOT/probe_${NAME}.log" 2>&1
        if [ $? -ne 0 ]; then
            echo "@@@ [$(stamp)] $NAME PROBE FAILED -- see $LOG_ROOT/probe_${NAME}.log; skipping pair"
            return 1
        fi
        echo "@@@ [$(stamp)] $NAME PROBE done"
    fi

    python scripts/build_context_prior.py "$PROBE" --out "$PRIOR" \
        > "$LOG_ROOT/prior_${NAME}.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "@@@ [$(stamp)] $NAME PRIOR FAILED -- see $LOG_ROOT/prior_${NAME}.log; skipping pair"
        return 1
    fi
    local NCTX
    NCTX=$(python -c "import json;print(len(json.load(open('$PRIOR'))['scene_idx']))" 2>/dev/null)
    echo "@@@ [$(stamp)] $NAME PRIOR done -- $NCTX / 1024 contexts attackable"
    if [ "$NCTX" = "0" ]; then
        echo "@@@ [$(stamp)] $NAME PRIOR is EMPTY -- no headroom, skipping training"
        return 1
    fi

    # Idempotent: a pair whose final checkpoint already exists is skipped, so
    # the driver can be re-launched after an interruption without redoing work.
    local RUN="critical_scene_ddpo_ldm_adv_ddim_${NAME}_hier_v2"
    local FINAL
    FINAL="$SCRATCH_ROOT/critical_scene/$RUN/${RUN}_$(printf '%05d' "$ITERS").ckpt"
    if [ -f "$FINAL" ]; then
        echo "@@@ [$(stamp)] $NAME TRAIN skipped -- $(basename "$FINAL") already exists"
        return 0
    fi

    echo "@@@ [$(stamp)] $NAME TRAIN start (bs=128 workers=16 iters=$ITERS focus=0.7)"
    python train.py --config-name "$CFG" \
        ddpo.rollout_workers=16 \
        ddpo.num_iterations="$ITERS" \
        hydra.run.dir="$LOG_ROOT/ddpo_${NAME}" \
        > "$LOG_ROOT/train_${NAME}.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "@@@ [$(stamp)] $NAME TRAIN FAILED -- see $LOG_ROOT/train_${NAME}.log"
        return 1
    fi
    echo "@@@ [$(stamp)] $NAME TRAIN done -- $(df -h /home | awk 'NR==2{print $4}') disk left"
}

echo "@@@ [$(stamp)] PIPELINE start, logs in $LOG_ROOT"
run_pair idm            pdm-idm            config_ldm_adv_ddpo_pdm_idm
run_pair ppo_normal     pdm-ppo_norm       config_ldm_adv_ddpo_pdm_ppo_normal
run_pair ppo_aggressive pdm-ppo_aggressive config_ldm_adv_ddpo_pdm_ppo_aggressive
run_pair ppo_caution    pdm-ppo_caution    config_ldm_adv_ddpo_pdm_ppo_caution
echo "@@@ [$(stamp)] PIPELINE done"
