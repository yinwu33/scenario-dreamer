#!/usr/bin/env bash
set -euo pipefail

# Stage launcher for a faithful Scenario Dreamer Base reproduction.
# Run it inside a GPU allocation from the repository root. Extra arguments are
# passed through as Hydra overrides, e.g.:
#   scripts/run_scenario_dreamer_base_waymo.sh train-ldm ldm.train.devices=1

SD_BASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SD_BASE_ROOT"
# define_env_variables.sh appends to PYTHONPATH; initialize it for shells where
# the variable has not been exported yet (this launcher uses nounset).
export PYTHONPATH="${PYTHONPATH:-}"
source scripts/define_env_variables.sh

SD_BASE_PYTHON_BIN="${SD_BASE_PYTHON_BIN:-python}"
SD_BASE_CONFIG="config_scenario_dreamer_base_waymo"
SD_BASE_OFFICIAL_AE="scenario_dreamer_autoencoder_waymo"
SD_BASE_RETRAIN_AE="scenario_dreamer_autoencoder_base_waymo_seed0"
SD_BASE_OFFICIAL_LATENTS="${SCRATCH_ROOT}/scenario_dreamer_autoencoder_latents_waymo"
SD_BASE_RETRAIN_LATENTS="${SCRATCH_ROOT}/scenario_dreamer_autoencoder_latents_base_waymo_seed0"

usage() {
  echo "Usage: $0 STAGE [hydra overrides...]"
  echo
  echo "Recommended route (reuse the official AE):"
  echo "  cache-official-train   Cache the missing train latents"
  echo "  cache-official-val     Rebuild val latents if provenance is uncertain"
  echo "  train-ldm              Train the original 376.68M Base LDM"
  echo
  echo "Optional full from-scratch route:"
  echo "  train-ae"
  echo "  cache-retrained-train"
  echo "  cache-retrained-val"
  echo "  train-ldm-retrained-ae"
  echo
  echo "Evaluation:"
  echo "  generate"
  echo "  metrics"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

SD_BASE_STAGE="$1"
shift

if ! "$SD_BASE_PYTHON_BIN" -c "import hydra, pytorch_lightning, torch" >/dev/null 2>&1; then
  echo "Python environment is missing Scenario Dreamer dependencies: $SD_BASE_PYTHON_BIN" >&2
  echo "Activate the scenario-dreamer environment or set SD_BASE_PYTHON_BIN." >&2
  exit 1
fi

# All stages except metric aggregation perform neural-network inference or
# training. Refuse to silently run those multi-hour jobs on CPU.
if [[ "$SD_BASE_STAGE" != "metrics" ]]; then
  if ! "$SD_BASE_PYTHON_BIN" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    echo "No CUDA GPU is visible; refusing to run stage '$SD_BASE_STAGE' on CPU." >&2
    exit 1
  fi
fi

case "$SD_BASE_STAGE" in
  cache-official-train)
    exec "$SD_BASE_PYTHON_BIN" eval.py \
      --config-name "$SD_BASE_CONFIG" \
      model_name=autoencoder \
      ae.eval.run_name="$SD_BASE_OFFICIAL_AE" \
      ae.eval.cache_latents.enable_caching=True \
      ae.eval.cache_latents.split_name=train \
      ae.eval.cache_latents.latent_dir="$SD_BASE_OFFICIAL_LATENTS" \
      "$@"
    ;;
  cache-official-val)
    exec "$SD_BASE_PYTHON_BIN" eval.py \
      --config-name "$SD_BASE_CONFIG" \
      model_name=autoencoder \
      ae.eval.run_name="$SD_BASE_OFFICIAL_AE" \
      ae.eval.cache_latents.enable_caching=True \
      ae.eval.cache_latents.split_name=val \
      ae.eval.cache_latents.latent_dir="$SD_BASE_OFFICIAL_LATENTS" \
      "$@"
    ;;
  train-ldm)
    exec "$SD_BASE_PYTHON_BIN" train.py \
      --config-name "$SD_BASE_CONFIG" \
      model_name=ldm \
      "$@"
    ;;
  train-ae)
    exec "$SD_BASE_PYTHON_BIN" train.py \
      --config-name "$SD_BASE_CONFIG" \
      model_name=autoencoder \
      ae.train.run_name="$SD_BASE_RETRAIN_AE" \
      "$@"
    ;;
  cache-retrained-train)
    exec "$SD_BASE_PYTHON_BIN" eval.py \
      --config-name "$SD_BASE_CONFIG" \
      model_name=autoencoder \
      ae.eval.run_name="$SD_BASE_RETRAIN_AE" \
      ae.eval.cache_latents.enable_caching=True \
      ae.eval.cache_latents.split_name=train \
      ae.eval.cache_latents.latent_dir="$SD_BASE_RETRAIN_LATENTS" \
      "$@"
    ;;
  cache-retrained-val)
    exec "$SD_BASE_PYTHON_BIN" eval.py \
      --config-name "$SD_BASE_CONFIG" \
      model_name=autoencoder \
      ae.eval.run_name="$SD_BASE_RETRAIN_AE" \
      ae.eval.cache_latents.enable_caching=True \
      ae.eval.cache_latents.split_name=val \
      ae.eval.cache_latents.latent_dir="$SD_BASE_RETRAIN_LATENTS" \
      "$@"
    ;;
  train-ldm-retrained-ae)
    exec "$SD_BASE_PYTHON_BIN" train.py \
      --config-name "$SD_BASE_CONFIG" \
      model_name=ldm \
      ldm.model.autoencoder_run_name="$SD_BASE_RETRAIN_AE" \
      ldm.dataset.dataset_path="$SD_BASE_RETRAIN_LATENTS" \
      ldm.train.run_name=scenario_dreamer_ldm_base_waymo_full_retrain_seed0 \
      "$@"
    ;;
  generate)
    exec "$SD_BASE_PYTHON_BIN" eval.py \
      --config-name "$SD_BASE_CONFIG" \
      model_name=ldm \
      ldm.eval.mode=initial_scene \
      ldm.eval.num_samples=10000 \
      ldm.eval.cache_samples=True \
      ldm.eval.visualize=False \
      "$@"
    ;;
  metrics)
    exec "$SD_BASE_PYTHON_BIN" eval.py \
      --config-name "$SD_BASE_CONFIG" \
      model_name=ldm \
      ldm.eval.mode=metrics \
      +ldm.eval.metrics.num_samples=10000 \
      "$@"
    ;;
  *)
    echo "Unknown stage: $SD_BASE_STAGE" >&2
    usage >&2
    exit 2
    ;;
esac
