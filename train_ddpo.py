"""Deprecated entry point for DDPO fine-tuning.

DDPO is now part of the unified train.py entry. Prefer:
    python train.py --config-name config_ddpo_dm_goal
    python train.py --config-name config_ddpo_ldm_goal
    python train.py --config-name config_ddpo_dm_goal ddpo.mode=goal   # goal | init_goal | all

This shim is kept for backward compatibility; it composes config_ddpo and calls
the dm_goal loop (ddpo.train_loop.run_ddpo) that train.py dispatches to.
"""

import hydra

from cfgs.config import CONFIG_PATH
from ddpo.train_loop import run_ddpo


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config_ddpo")
def main(cfg):
    run_ddpo(cfg)


if __name__ == "__main__":
    main()
