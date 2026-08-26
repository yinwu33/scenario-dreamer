"""Reward variant registry: the yaml ``name`` selects config + assembler."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from ddpo.reward.base import BaseRewardConfig, RewardAssembler
from ddpo.reward.flat import FlatReward
from ddpo.reward.hierarchical import HierarchicalReward
from ddpo.reward.tiered import TieredReward

REWARDS: dict[str, type[RewardAssembler]] = {
    cls.name: cls for cls in (FlatReward, TieredReward, HierarchicalReward)
}


def _assembler_cls(name: str) -> type[RewardAssembler]:
    if name not in REWARDS:
        raise KeyError(f"unknown reward '{name}'; known: {sorted(REWARDS)}")
    return REWARDS[name]


def build_reward_config(node) -> BaseRewardConfig:
    """Parse a ``cfgs/ddpo/reward/<name>.yaml`` ``reward:`` node into its config."""
    if isinstance(node, DictConfig):
        node = OmegaConf.to_container(node, resolve=True)
    values = dict(node)
    return _assembler_cls(values.pop("name")).config_cls(**values)


def build_reward(cfg: BaseRewardConfig, gen_invalid_enabled: bool) -> RewardAssembler:
    """The assembler for ``cfg``: a pure metrics -> (reward, components) function."""
    return _assembler_cls(cfg.name)(cfg, gen_invalid_enabled)
