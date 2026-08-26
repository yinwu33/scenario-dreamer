"""Reward interface: config bases, assembler base, shared component dict."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass
class BaseRewardConfig:
    """Fields every variant reads. Strict: yaml keys map 1:1 onto the fields of
    the variant selected by ``name``, and every field is required."""

    name: ClassVar[str]

    ttc_tau: float
    lane_soft: float
    lane_hard: float
    lane_penalty: float
    collision_warmup: float
    collision_window: float
    trivial_collision_t: float
    invalid_grade_scale: float

    def __post_init__(self):
        self.collision_window = max(float(self.collision_window), 1e-6)


@dataclass
class ApproachRewardConfig(BaseRewardConfig):
    """Adds the dense approach/risk term and its annealing schedule."""

    risk_coef: float
    approach_d_safe: float
    approach_d_scale: float
    approach_close_delta: float
    approach_close_scale: float
    approach_coef: float
    approach_coef_final: float
    approach_anneal_begin: int
    approach_anneal_end: int

    def __post_init__(self):
        super().__post_init__()
        self.approach_d_scale = max(float(self.approach_d_scale), 1e-6)
        self.approach_close_scale = max(float(self.approach_close_scale), 1e-6)


class RewardAssembler(ABC):
    """Scalar reward assembled from one rollout's per-scene metric table."""

    name: ClassVar[str]
    config_cls: ClassVar[type[BaseRewardConfig]]
    requires_path_conflict: ClassVar[bool] = False

    def __init__(self, cfg: BaseRewardConfig, gen_invalid_enabled: bool):
        self.cfg = cfg
        self.gen_invalid_enabled = bool(gen_invalid_enabled)

    @abstractmethod
    def assemble(self, m: dict) -> tuple[np.ndarray, dict]:
        """(per-scene reward, per-component diagnostics)."""

    def set_train_iteration(self, it: int) -> None:
        """Advance annealing schedules; variants without one do nothing."""

    @property
    def approach_coef(self) -> float:
        return 0.0


class AnnealedApproachAssembler(RewardAssembler):
    """Assembler whose approach weight anneals linearly over train iterations."""

    def __init__(self, cfg: ApproachRewardConfig, gen_invalid_enabled: bool):
        super().__init__(cfg, gen_invalid_enabled)
        self._approach_coef = float(cfg.approach_coef)

    @property
    def approach_coef(self) -> float:
        return self._approach_coef

    def set_train_iteration(self, it: int) -> None:
        cfg = self.cfg
        if cfg.approach_anneal_end <= cfg.approach_anneal_begin:
            self._approach_coef = float(cfg.approach_coef)
            return
        t = (it - cfg.approach_anneal_begin) / (
            cfg.approach_anneal_end - cfg.approach_anneal_begin
        )
        t = min(max(t, 0.0), 1.0)
        self._approach_coef = float(
            cfg.approach_coef + t * (cfg.approach_coef_final - cfg.approach_coef)
        )


def component_dict(
    *,
    r_ttc: np.ndarray,
    r_approach: np.ndarray,
    r_risk: np.ndarray,
    r_collision: np.ndarray,
    r_bonus: np.ndarray,
    criticality: np.ndarray,
    c_spawn_lane: np.ndarray,
    c_goal_lane: np.ndarray,
    c_parking: np.ndarray,
    c_invalid: np.ndarray,
    c_invalid_sev: np.ndarray,
    c_invalid_reason: np.ndarray,
    c_trivial: np.ndarray,
    c_overlap: np.ndarray,
    constraint: np.ndarray,
) -> dict:
    """Diagnostics every variant emits (train/eval logging and viz read these)."""
    return {
        "r_ttc": r_ttc,
        "r_approach": r_approach,
        "r_risk": r_risk,
        "r_collision": r_collision,
        "r_bonus": r_bonus,
        "criticality": criticality,
        "c_lane": c_spawn_lane + c_goal_lane,
        "c_spawn_lane": c_spawn_lane,
        "c_goal_lane": c_goal_lane,
        "c_parking": c_parking,
        "c_invalid": c_invalid,
        "c_invalid_sev": c_invalid_sev,
        "c_invalid_reason": c_invalid_reason,
        "c_trivial": c_trivial,
        "c_overlap": c_overlap,
        "constraint": constraint,
    }
