#!/bin/bash
# Compare the conditional ppo-ppo_norm v4kl5 DDPO generator against fully
# unconditional frozen-base best-of-K sampling on the same 1000 val slots.
set -eo pipefail
cd /home/tjhu78u/workspace/scenario-dreamer
source scripts/define_env_variables.sh

OUT=data/critical_scene/ppo-ppo_norm_v4kl5_vs_allnull_bok16
CKPT=data/critical_scene/critical_scene_ddpo_ldm_adv_ddim_ppo-ppo_norm_v4kl5_hier_v4/critical_scene_ddpo_ldm_adv_ddim_ppo-ppo_norm_v4kl5_hier_v4_00500.ckpt

.venv/bin/python scripts/run_ldm_adv_ppo_table.py \
  --out-dir "$OUT" --num-scenes 1000 --split val --seed 0 --chunk-size 32 \
  --sources ddpo_gen --skip-benchmark --generation-conditioning ego_far \
  --overrides ddpo/reward=hierarchical_v4 \
              planner@ddpo.planner.sut=ppo_normal \
              planner@ddpo.planner.env=ppo_normal \
              planner@ddpo.planner.adv=ppo_normal \
  --ddpo-ckpt "$CKPT"

.venv/bin/python scripts/run_best_of_k.py \
  --out-dir "$OUT" --num-scenes 1000 --split val --seed 0 --chunk-size 32 \
  --num-draws 16 --all-null --workers 16 --benchmark-batch-size 256 \
  --overrides ddpo/reward=hierarchical_v4 \
              planner@ddpo.planner.sut=ppo_normal \
              planner@ddpo.planner.env=ppo_normal \
              planner@ddpo.planner.adv=ppo_normal

.venv/bin/python scripts/score_adv_sources.py \
  --artifacts "$OUT/artifacts" \
  --sut ppo_normal --env ppo_normal --adv ppo_normal \
  --reward hierarchical_v4 --workers 16 --batch-size 128 \
  --override ddpo.simulator.path_conflict.skip_rollout=false \
  --sources ddpo_gen \
            base_gen_allnull_bok1 base_gen_allnull_bok2 \
            base_gen_allnull_bok4 base_gen_allnull_bok8 \
            base_gen_allnull_bok16 \
  --out "$OUT/scored_adv.md"
