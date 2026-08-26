"""DDPO reward: roll generated scenes out with a planner, score the ego.

  build_reward_config(cfg.reward) -> the variant's config (yaml ``name`` picks it)
  build_reward(config, gen_invalid_enabled) -> assembler: metrics -> (reward, components)
  RewardModel(planner_cfg, simulator_cfg, config) -> rollout + assembly

Variants live one per module (``flat``, ``tiered``, ``hierarchical``); adding one
means a new module plus an entry in ``registry.REWARDS``.
"""

from ddpo.reward.base import (
    ApproachRewardConfig,
    BaseRewardConfig,
    RewardAssembler,
)
from ddpo.reward.flat import FlatReward, FlatRewardConfig
from ddpo.reward.hierarchical import HierarchicalReward, HierarchicalRewardConfig
from ddpo.reward.model import RewardModel
from ddpo.reward.registry import REWARDS, build_reward, build_reward_config
from ddpo.reward.tiered import TieredReward, TieredRewardConfig

__all__ = [
    "ApproachRewardConfig",
    "BaseRewardConfig",
    "FlatReward",
    "FlatRewardConfig",
    "HierarchicalReward",
    "HierarchicalRewardConfig",
    "REWARDS",
    "RewardAssembler",
    "RewardModel",
    "TieredReward",
    "TieredRewardConfig",
    "build_reward",
    "build_reward_config",
]
