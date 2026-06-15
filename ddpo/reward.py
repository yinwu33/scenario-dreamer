"""DDPO reward: roll generated scenes out with the frozen planner, score the ego.

Same reward semantics as PufferDrive/scene_init_ddpo/reward.py, but the rollout
runs on the in-repo numpy port of the simulator (``pufferdrive_sim``) - no C env,
no .bin files, no second venv:

  * only the ego (scene agent 0) is scored;
  * the base reward is a DENSE criticality term ``clip(1 - min_TTC/tau, 0, 1)``
    over the ego's min time-to-collision along the rollout, so near-misses give
    gradient even without an actual crash; an ego collision caps it at 1;
  * ``init_invalid`` flags scenes with overlapping vehicles at t=0 (degenerate
    init / reward hacking) -> strong negative reward;
  * ``ego_offroad`` is kept for interface compatibility but is always 0: the
    generated maps carry no road edges (see pufferdrive_sim docstring);
  * a scene stops being stepped/scored once its ego reaches its goal.
"""

from __future__ import annotations

import numpy as np
import torch

from .interfaces import GeneratedScenes
from planner.selfplay_drive.planner import load_planner, load_planner_config
from .pufferdrive_sim import SimScene, load_sim_config
from .reward_hooks import (
    EgoCollisionHook,
    EgoMinTTCHook,
    EgoOffroadHook,
    GoalOfflaneHook,
    InitOverlapHook,
    ParkingMismatchHook,
    ReachedGoalHook,
    TrajectoryHook,
)
from .rollout_runner import PlannerRolloutRunner


class PufferDriveReward:
    """Evaluate generated scenes by rolling them out with the frozen planner.

    The reward object converts batched ``GeneratedScenes`` into independent
    simulator scenes, executes the planner for each active ego, and returns the
    collision reward plus goal validity penalties used by DDPO training.
    """

    def __init__(
        self,
        *,
        sim_steps: int = 91,
        deterministic: bool | None = None,
        ttc_tau: float = 3.0,
        init_overlap_margin: float = 0.0,
        goal_offlane_threshold: float = 3.0,
        goal_onroad_threshold: float = 2.0,
        goal_offlane_penalty: float = 0.5,
        parking_mismatch_penalty: float = 0.5,
        seed: int = 0,
    ):
        """Initialize planner-backed reward evaluation.

        Args:
            sim_steps: Maximum number of simulator steps per scene.
            deterministic: Whether planner actions should be deterministic. If
                ``None``, use the planner config default.
            ttc_tau: Time-to-collision horizon (seconds) normalising the dense
                criticality reward ``clip(1 - min_TTC/ttc_tau, 0, 1)``.
            init_overlap_margin: Box-inflation margin (metres) for the t=0
                vehicle-overlap (init_invalid) check; 0 rejects only true overlap
                and allows bumper-to-bumper traffic-jam spawns.
            goal_offlane_threshold: Lane-centerline distance in meters above
                which a moving car's goal is considered off-lane.
            goal_onroad_threshold: Lane-centerline distance in meters within
                which a car is considered to have *spawned* on the road (only
                such cars are required to keep an on-lane goal).
            goal_offlane_penalty: Penalty scale applied to the off-lane goal
                fraction for each scene.
            parking_mismatch_penalty: Penalty scale applied when generated
                parking/static state disagrees with ``meta["gt_parking_mask"]``.
            seed: RNG seed passed into simulator scene construction.
        """
        planner_cfg = load_planner_config()
        sim_cfg = load_sim_config()
        self.planner = load_planner()
        self.device = str(next(self.planner.parameters()).device)
        self.sim_steps = int(sim_steps)
        self.deterministic = (
            planner_cfg.deterministic if deterministic is None else deterministic
        )
        self.ttc_tau = float(ttc_tau)
        self.init_overlap_margin = float(init_overlap_margin)
        self.goal_onroad_threshold = float(goal_onroad_threshold)
        self.goal_radius = sim_cfg.goal_radius
        # Goals of moving (controlled) agents must lie on the road: the planner was
        # trained with goals taken from real on-lane trajectories, so an off-lane
        # goal sends it out of distribution (cheap reward hacking). Static agents
        # (goal within 2 m of spawn, e.g. parked cars) are exempt - their "goal"
        # legitimately sits in a driveway/parking spot.
        self.goal_offlane_threshold = float(goal_offlane_threshold)
        self.goal_offlane_penalty = float(goal_offlane_penalty)
        # Parking-state consistency (goal mode only; needs meta["gt_parking_mask"]):
        # an agent is "parking" when its goal lies within 2 m of its spawn. Penalise
        # the fraction of agents whose generated parking state differs from GT -
        # blocks both "force a parked car to drive" (off-lane penalty would otherwise
        # be satisfiable by moving its goal onto the lane) and "freeze all traffic
        # into a static obstacle field" (cheap +1 by ramming the ego through it).
        self.parking_mismatch_penalty = float(parking_mismatch_penalty)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ build
    def _build_scenes(self, scenes: GeneratedScenes) -> list[SimScene]:
        """Convert a batched ``GeneratedScenes`` object into simulator scenes.

        Agent and lane tensors are grouped by their scene-index metadata, moved
        to CPU numpy arrays, and wrapped in ``SimScene`` instances that share
        this reward object's RNG.
        """
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

    def _reward(self, ego_collision, ego_min_ttc, init_invalid):
        """Dense critical-scene reward.

        Base term is continuous time-to-collision criticality in [0, 1]
        (``clip(1 - min_TTC/tau, 0, 1)``; ``min_TTC=inf`` -> 0). An actual ego
        collision caps it at 1; degenerate init (overlapping vehicles) -> -1.
        """
        ttc_term = np.clip(1.0 - ego_min_ttc / self.ttc_tau, 0.0, 1.0)
        r = np.where(ego_collision > 0, 1.0, ttc_term)
        r = np.where(init_invalid > 0, -1.0, r)
        return r.astype(np.float32)

    # --------------------------------------------------------------- evaluate
    @torch.no_grad()
    def evaluate(
        self, scenes: GeneratedScenes, record_trajectories: bool = False
    ) -> dict:
        """Roll out generated scenes and return reward metrics.

        Args:
            scenes: Batched generated scenes containing agent states, types,
                agent-to-scene indices, lane polylines, and lane-to-scene
                metadata.
            record_trajectories: If true, include per-scene trajectory arrays
                suitable for rollout visualization.

        Returns:
            Dictionary of per-scene numpy arrays for reward, collision, offroad,
            initial invalid state, goal completion, goal off-lane fraction, and
            parking mismatch fraction. When ``record_trajectories`` is true, the
            dictionary also contains a ``trajectories`` list.
        """
        sims = self._build_scenes(scenes)
        hooks = [
            InitOverlapHook(self.init_overlap_margin),
            EgoCollisionHook(),
            EgoOffroadHook(),
            EgoMinTTCHook(),
            TrajectoryHook(),
            ReachedGoalHook(self.goal_radius),
            GoalOfflaneHook(self.goal_offlane_threshold, self.goal_onroad_threshold),
            ParkingMismatchHook(),
        ]
        runner = PlannerRolloutRunner(
            planner=self.planner,
            device=self.device,
            deterministic=self.deterministic,
            sim_steps=self.sim_steps,
            hooks=hooks,
        )
        ctx = runner.rollout(scenes, sims, record_trajectories=record_trajectories)
        metrics = ctx.metrics

        rewards = self._reward(
            metrics["ego_collision"],
            metrics["ego_min_ttc"],
            metrics["init_invalid"],
        )
        # All penalty terms are per-scene FRACTIONS in [0, 1] (count-normalised),
        # so scenes with many agents are not penalised more than sparse ones; the
        # coefficients set the scale relative to the +/-1 collision reward.
        rewards = rewards - self.goal_offlane_penalty * metrics["goal_offlane_frac"]
        rewards = (
            rewards - self.parking_mismatch_penalty * metrics["parking_mismatch_frac"]
        )
        out = {
            "reward": rewards,
            "ego_collision": metrics["ego_collision"],
            "ego_min_ttc": metrics["ego_min_ttc"],
            "ego_offroad": metrics["ego_offroad"],
            "init_invalid": metrics["init_invalid"],
            "reached_goal": metrics["reached_goal"],
            "goal_offlane_frac": metrics["goal_offlane_frac"],
            "parking_mismatch_frac": metrics["parking_mismatch_frac"],
        }
        if ctx.trajectories is not None:
            out["trajectories"] = ctx.trajectories
        return out
