#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Extra CLI args are forwarded to the Python
# analyzer, so this wrapper can launch either the default pilot or a full run.
#
# Examples:
#   bash data_analysis/run_ldm_adv_ddpo_support_analysis.sh \
#     --splits train val --num-scenes 1000 --samples-per-scene 16
#
#   bash data_analysis/run_ldm_adv_ddpo_support_analysis.sh \
#     --splits train --num-scenes 8 --samples-per-scene 2 \
#     --override ddpo.sampler=ddim --override ddpo.ddim_steps=10

cd "$(dirname "$0")/.."
set +u
source scripts/define_env_variables.sh
set -u
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

.venv/bin/python -u data_analysis/analyze_ldm_adv_ddpo_support.py "$@"
