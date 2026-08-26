"""Rollout + scoring: generated scenes -> per-scene DDPO reward.

Three strict configs, one concern each (see ``cfgs/ddpo/ldm_adv.yaml``):
  planner_cfg    WHICH policy drives each role (sut / env / adv) + shared sim
                 dynamics;
  simulator_cfg  HOW the rollout measures metrics (``sim.runner``);
  reward_cfg     HOW the scalar is assembled from them (``cfgs/ddpo/reward``).

This module owns the metric set: it builds the rollout hooks and hands their
output to the reward assembler. Metric access is strict -- a missing key raises.
Only the ego (scene agent 0) is scored, and ``ego_offroad`` is always 0 because
generated maps carry no road edges (see ``sim.world``).
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ddpo.reward.base import BaseRewardConfig
from ddpo.reward.registry import build_reward
from sim.parallel import ParallelRolloutRunner
from sim.scenes import GeneratedScenes
from sim.runner import RolloutRunner, SimulatorConfig
from sim.hooks import (
    GenAgentInvalidHook,
    GenAgentParkingHook,
    EgoAdvMinDistHook,
    EgoCollisionHook,
    EgoMinTTCHook,
    EgoOffroadProxyHook,
    GoalOfflaneHook,
    InitOverlapHook,
    ParkingMismatchHook,
    PathConflictHook,
    ReachedGoalHook,
    TrajectoryHook,
)


class RewardModel:
    """Planner rollout of generated scenes plus the reward assembled from it."""

    def __init__(
        self,
        planner_cfg,
        simulator_cfg: SimulatorConfig,
        reward_cfg: BaseRewardConfig,
        num_workers: int = 0,
        train_batch_size: int = 0,
    ):
        self.cfg = reward_cfg
        self.simulator_cfg = simulator_cfg
        # With a condition-violation check configured, the realized-vs-target
        # metric supersedes the plain parked-adv gate.
        self.gen_invalid_enabled = simulator_cfg.gen_invalid is not None
        self.assembler = build_reward(reward_cfg, self.gen_invalid_enabled)
        if self.assembler.requires_path_conflict and simulator_cfg.path_conflict is None:
            raise ValueError(
                f"reward '{reward_cfg.name}' requires simulator.path_conflict"
            )
        # num_workers > 0 shards the rollout across processes (sim.parallel).
        # The result is bit-exact, so this is a pure throughput knob.
        self.num_workers = int(num_workers)
        self.runner = (
            ParallelRolloutRunner(planner_cfg, simulator_cfg, num_workers=self.num_workers,
                                  train_batch_size=train_batch_size)
            if self.num_workers > 0
            else RolloutRunner(planner_cfg, simulator_cfg)
        )

    def close(self) -> None:
        """Shut the rollout workers down (no-op for the single-process runner)."""
        if self.num_workers > 0:
            self.runner.close()

    def assemble(self, metrics: dict) -> tuple:
        return self.assembler.assemble(metrics)

    @property
    def approach_coef(self) -> float:
        return self.assembler.approach_coef

    def set_train_iteration(self, it: int) -> None:
        """Advance the reward's annealing schedule (called once per iteration;
        eval calls in between reuse the latest weight)."""
        self.assembler.set_train_iteration(it)

    def _make_hooks(self, record_trajectories: bool = False) -> list:
        """Exactly the metric set the reward and ``evaluate`` consume.

        ``record_trajectories`` disables the path-conflict skip: a skipped scene
        has no movie to render, and eval rates must not inherit the predicate's
        recall.
        """
        p = self.simulator_cfg
        hooks = [
            InitOverlapHook(p.init_overlap_margin),
            EgoCollisionHook(),
            EgoMinTTCHook(),
            EgoOffroadProxyHook(p.ego_offroad_threshold),
            EgoAdvMinDistHook(p.approach_warmup_time),
            TrajectoryHook(),
            ReachedGoalHook(self.runner.sim_cfg.goal_radius),
            GoalOfflaneHook(p.goal_offlane_threshold, p.goal_onroad_threshold),
            ParkingMismatchHook(),
            GenAgentParkingHook(),
        ]
        if p.gen_invalid is not None:
            hooks.append(GenAgentInvalidHook.from_check(p.gen_invalid))
        # First, so its skip decision is taken before any stepping hook runs.
        if p.path_conflict is not None:
            check = p.path_conflict
            if record_trajectories and check.skip_rollout:
                check = replace(check, skip_rollout=False)
            hooks.insert(0, PathConflictHook(check))
        return hooks

    @torch.no_grad()
    def measure(self, scenes: GeneratedScenes, record_trajectories: bool = False):
        """Roll the scenes out and return the raw ``RolloutResult``, no reward.

        Split out so one sample+rollout sweep can be re-scored offline under any
        number of reward configs (see ``scripts/reward_screen.py``).
        """
        return self.runner.rollout(
            scenes,
            hooks=self._make_hooks(record_trajectories),
            record_trajectories=record_trajectories,
        )

    @torch.no_grad()
    def evaluate(
        self, scenes: GeneratedScenes, record_trajectories: bool = False
    ) -> dict:
        """Roll out ``scenes`` and return per-scene numpy arrays: the assembled
        ``reward``, the raw rollout metrics, and every reward component."""
        result = self.measure(scenes, record_trajectories=record_trajectories)
        metrics = result.metrics

        rewards, components = self.assembler.assemble(metrics)

        out = {
            "reward": rewards,
            "ego_collision": metrics["ego_collision"],
            "ego_fault_collision": metrics["ego_fault_collision"],
            "ego_collision_time": metrics["ego_collision_time"],
            "ego_min_ttc": metrics["ego_min_ttc"],
            "ego_offroad": metrics["ego_offroad"],
            "ego_offroad_proxy": metrics["ego_offroad_proxy"],
            "ego_offroad_frac": metrics["ego_offroad_frac"],
            "ego_lane_dist_max": metrics["ego_lane_dist_max"],
            "init_invalid": metrics["init_invalid"],
            "init_overlap_frac": metrics["init_overlap_frac"],
            "reached_goal": metrics["reached_goal"],
            "goal_offlane_frac": metrics["goal_offlane_frac"],
            "goal_lane_dist": metrics["goal_lane_dist"],
            "spawn_lane_dist": metrics["spawn_lane_dist"],
            "parking_mismatch_frac": metrics["parking_mismatch_frac"],
            "ego_adv_min_dist": metrics["ego_adv_min_dist"],
            "ego_adv_init_dist": metrics["ego_adv_init_dist"],
            "ego_adv_min_dist_warmup": metrics["ego_adv_min_dist_warmup"],
            "gen_agent_is_parked": metrics["gen_agent_is_parked"],
        }
        if self.gen_invalid_enabled:
            out["gen_agent_is_invalid"] = metrics["gen_agent_is_invalid"]
        if self.simulator_cfg.path_conflict is not None:
            for k in ("path_conflict", "path_conflict_dist", "path_conflict_pet",
                      "ego_adv_spawn_dist"):
                out[k] = metrics[k]
        out.update(components)
        if result.trajectories is not None:
            out["trajectories"] = result.trajectories
        return out
