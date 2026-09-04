#!/bin/bash
# Companion watcher for run_pdm_ddpo_pipeline.sh.
#
# Two event streams on one stdout:
#   * every '@@@' progress marker the driver prints, immediately;
#   * an hourly heartbeat: active stage, newest log line, GPU memory, disk.
#
#   bash scripts/watch_pdm_ddpo_pipeline.sh <driver-log>

set -o pipefail
DRIVER_LOG="$1"
LOG_ROOT="/home/tjhu78u/workspace/scenario-dreamer/slurm_logs/pdm_pipeline_20260902"
CKPT_ROOT="/home/tjhu78u/workspace/scenario-dreamer/data/critical_scene"

# Stage transitions and failures, as they happen.
tail -n +1 -f "$DRIVER_LOG" 2>/dev/null | grep --line-buffered -E "@@@" &
TAIL_PID=$!
trap 'kill $TAIL_PID 2>/dev/null' EXIT

while true; do
    sleep 3600

    ACTIVE=$(ls -t "$LOG_ROOT"/train_*.log "$LOG_ROOT"/probe_*.log 2>/dev/null | head -1)
    STAGE=$(basename "${ACTIVE:-none}" .log)
    LAST=$(grep -v '^[[:space:]]*$' "$ACTIVE" 2>/dev/null | tail -1 | cut -c1-160)
    GPU=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
            --format=csv,noheader 2>/dev/null | head -1)
    DISK=$(df -h /home | awk 'NR==2{print $4" free ("$5" used)"}')
    CKPTS=$(ls -t "$CKPT_ROOT"/critical_scene_ddpo_ldm_adv_ddim_pdm-*/*.ckpt 2>/dev/null \
            | head -1 | xargs -r basename)
    ALIVE=$(pgrep -c -f "[t]rain.py --config-name config_ldm_adv_ddpo_pdm" 2>/dev/null)
    ERR=$(grep -cE "Traceback|CUDA out of memory|RuntimeError" "$ACTIVE" 2>/dev/null)

    echo "HEARTBEAT $(date +%F_%T) | stage=$STAGE | train_procs=${ALIVE:-0}" \
         "| gpu=[$GPU] | disk=$DISK | newest_ckpt=${CKPTS:-none} | errors_in_log=${ERR:-0}"
    echo "  last: ${LAST:-<no output yet>}"

    if [ "${ERR:-0}" != "0" ]; then
        echo "  !! error text found in $ACTIVE -- inspect it"
    fi
    if grep -q "@@@.*PIPELINE done" "$DRIVER_LOG" 2>/dev/null; then
        echo "PIPELINE COMPLETE -- watcher exiting"
        exit 0
    fi
done
