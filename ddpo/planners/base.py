"""Rollout-planner interface and registry for DDPO reward evaluation.

A ``RolloutPlanner`` owns one rollout: given a batch of generated scenes it
advances them, and returns the per-scene reward metrics (and optional
trajectories). The reward module (``ddpo.reward``) stays planner-agnostic - it
just asks the configured planner for metrics and assembles the scalar reward.

Two extension points:

  * ``NumpyPlanner`` - shared base for everything that drives the in-repo numpy
    ``SimScene`` (the frozen ``selfplay_drive`` net, the rule-based ``dummy``
    goal-seeker, future ``simpl``). Subclasses implement a single ``_advance``;
    the SimScene step loop, metric hooks and trajectory recording are shared.
  * ``RolloutPlanner`` directly - for backends that own their whole loop and do
    not use ``SimScene`` (e.g. ``puffer_drive``'s C environment).

Planners are selected by name through a registry; each name has its own
``cfgs/planner/<name>.yaml``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from ..interfaces import GeneratedScenes
from ..pufferdrive_sim import SimScene, load_sim_config
from ..reward_hooks import (
    ControlledParkingHook,
    EgoAdvMinDistHook,
    EgoCollisionHook,
    EgoMinTTCHook,
    EgoOffroadHook,
    GoalOfflaneHook,
    InitOverlapHook,
    ParkingMismatchHook,
    ReachedGoalHook,
    RolloutContext,
    TrajectoryHook,
)


@dataclass
class RolloutParams:
    """Rollout / metric parameters shared by every planner.

    These are the non-reward-scalar knobs a planner needs to reproduce the
    metric set the reward consumes. Reward weights (ttc_tau, penalties, ...)
    stay in ``ddpo.reward`` and never reach the planner.
    """

    sim_steps: int = 91
    seed: int = 0
    init_overlap_margin: float = 0.0
    goal_offlane_threshold: float = 3.0
    goal_onroad_threshold: float = 2.0
    pufferdrive_root: str | None = None


@dataclass
class RolloutResult:
    """Output of a planner rollout consumed by the reward."""

    metrics: dict[str, np.ndarray]
    trajectories: list[dict[str, Any]] | None = None


class RolloutPlanner(ABC):
    """Advance generated scenes and return per-scene reward metrics."""

    @abstractmethod
    def rollout(
        self, scenes: GeneratedScenes, *, record_trajectories: bool = False
    ) -> RolloutResult:
        """Roll ``scenes`` out and return the populated metrics / trajectories."""


class NumpyPlanner(RolloutPlanner):
    """Shared base for numpy ``SimScene`` rollouts (selfplay_drive, dummy, ...).

    The SimScene step loop, metric-hook lifecycle and trajectory recording are
    fixed here; a subclass only implements ``_advance`` to decide and apply one
    step of motion to the controlled agents of the active scenes.
    """

    def __init__(self, planner_cfg, params: RolloutParams, *, device: str | None = None):
        self.params = params
        self.sim_cfg = load_sim_config()
        self.rng = np.random.default_rng(int(params.seed))

    # ------------------------------------------------------------------ build
    def _build_scenes(self, scenes: GeneratedScenes) -> list[SimScene]:
        """Group a batched ``GeneratedScenes`` into per-scene ``SimScene`` objects."""
        states = scenes.agent_states.detach().cpu().numpy()
        types = scenes.agent_types.detach().cpu().numpy()
        a_idx = scenes.agent_scene_idx.detach().cpu().numpy()
        lanes = scenes.lane_polylines
        if isinstance(lanes, torch.Tensor):
            lanes = lanes.detach().cpu().numpy()
        l_idx = scenes.meta["lane_scene_idx"]
        if isinstance(l_idx, torch.Tensor):
            l_idx = l_idx.detach().cpu().numpy()

        sims = []
        for s in range(scenes.num_scenes):
            sims.append(
                SimScene(
                    states[a_idx == s],
                    types[a_idx == s],
                    lanes[l_idx == s],
                    rng=self.rng,
                )
            )
        return sims

    def _make_hooks(self) -> list:
        """Metric hooks shared by all numpy planners (identical metric set)."""
        p = self.params
        return [
            InitOverlapHook(p.init_overlap_margin),
            EgoCollisionHook(),
            EgoOffroadHook(),
            EgoMinTTCHook(),
            EgoAdvMinDistHook(),
            TrajectoryHook(),
            ReachedGoalHook(self.sim_cfg.goal_radius),
            GoalOfflaneHook(p.goal_offlane_threshold, p.goal_onroad_threshold),
            ParkingMismatchHook(),
            ControlledParkingHook(),
        ]

    # --------------------------------------------------------------- advance
    @abstractmethod
    def _advance(self, sims: list[SimScene], active: list[int]) -> None:
        """Decide + apply one step of motion to every active scene (in place).

        The runner has already fired ``before_step_scene`` hooks; afterwards it
        runs ``update_metrics`` / ``goal_step`` / ``after_step_scene`` per scene.
        """

    # --------------------------------------------------------------- rollout
    @torch.no_grad()
    def rollout(
        self, scenes: GeneratedScenes, *, record_trajectories: bool = False
    ) -> RolloutResult:
        sims = self._build_scenes(scenes)
        hooks = self._make_hooks()
        m = len(sims)
        ctx = RolloutContext(
            scenes=scenes,
            sims=sims,
            finished=np.zeros(m, dtype=bool),
            metrics={
                "ego_collision": np.zeros(m, dtype=np.float32),
                "ego_offroad": np.zeros(m, dtype=np.float32),
                "init_invalid": np.zeros(m, dtype=np.float32),
                "reached_goal": np.zeros(m, dtype=np.float32),
            },
            record_trajectories=record_trajectories,
        )

        for hook in hooks:
            hook.before_rollout(ctx)

        for t in range(self.params.sim_steps):
            ctx.t = t

            # Score current state before stepping, matching the original reward.
            for s, sim in enumerate(sims):
                if ctx.finished[s]:
                    continue
                for hook in hooks:
                    hook.before_step_scene(ctx, s, sim)

            active = [s for s in range(m) if not ctx.finished[s]]
            if not active:
                break

            self._advance(sims, active)

            for s in active:
                sims[s].update_metrics()
                ego_reached, _ = sims[s].goal_step()
                for hook in hooks:
                    hook.after_step_scene(ctx, s, sims[s], ego_reached=ego_reached)

        for hook in hooks:
            hook.after_rollout(ctx)

        return RolloutResult(metrics=ctx.metrics, trajectories=ctx.trajectories)


# ---------------------------------------------------------------- registry
_REGISTRY: dict[str, Callable[..., RolloutPlanner]] = {}


def register_planner(name: str):
    """Class decorator registering a planner under ``name``."""

    def deco(cls):
        _REGISTRY[name] = cls
        return cls

    return deco


def build_planner(planner_cfg, params: RolloutParams, *, device: str | None = None) -> RolloutPlanner:
    """Instantiate the planner named by ``planner_cfg['name']``.

    ``planner_cfg`` is any mapping with a ``name`` key (an OmegaConf node from
    ``cfgs/planner/<name>.yaml`` or a plain dict). Importing this module's
    package registers the built-in planners.
    """
    name = planner_cfg.get("name")
    if name is None:
        raise ValueError("planner config must define a 'name'")
    name = str(name)
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown planner '{name}'; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](planner_cfg, params, device=device)
