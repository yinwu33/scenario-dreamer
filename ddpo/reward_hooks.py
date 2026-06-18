"""Hook components for planner-backed DDPO reward rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .interfaces import GeneratedScenes
from .pufferdrive_sim import MIN_DISTANCE_TO_GOAL, TYPE_VEHICLE, SimScene


@dataclass
class RolloutContext:
    """Mutable state shared by rollout hooks.

    The runner owns rollout ordering and lifecycle. Hooks update per-scene
    metrics, optional trajectories, and terminal flags where explicitly needed.
    """

    scenes: GeneratedScenes
    sims: list[SimScene]
    finished: np.ndarray
    metrics: dict[str, np.ndarray]
    record_trajectories: bool = False
    t: int = 0
    trajectories: list[dict[str, Any]] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def num_scenes(self) -> int:
        return len(self.sims)


class RewardHook:
    """No-op base class for rollout metric hooks."""

    def before_rollout(self, ctx: RolloutContext) -> None:
        pass

    def before_step_scene(self, ctx: RolloutContext, scene_idx: int, sim: SimScene) -> None:
        pass

    def after_step_scene(
        self,
        ctx: RolloutContext,
        scene_idx: int,
        sim: SimScene,
        *,
        ego_reached: bool,
    ) -> None:
        pass

    def after_rollout(self, ctx: RolloutContext) -> None:
        pass


class InitOverlapHook(RewardHook):
    """Flag degenerate generated init states with overlapping vehicles.

    Replaces the old ego-only t=0 overlap check: any two active vehicle boxes
    overlapping at spawn (within ``margin``) marks the scene init_invalid, which
    the reward floors to -1. ``margin=0`` allows bumper-to-bumper traffic-jam
    spawns (the planner is expected to brake), only rejecting true overlap.
    """

    def __init__(self, margin: float = 0.0):
        self.margin = float(margin)

    def before_rollout(self, ctx: RolloutContext) -> None:
        for s, sim in enumerate(ctx.sims):
            if sim.any_vehicle_overlap(self.margin):
                ctx.metrics["init_invalid"][s] = 1.0


class EgoCollisionHook(RewardHook):
    """Track ego collisions and the time of the first collision over the rollout.

    Collision was previously disabled outright (it rewarded the policy for
    teleporting an adversary into the ego at t=0). It is now ``enabled``-gated
    and, crucially, time-stamped: ``ego_collision_time`` lets the reward reward
    only *meaningful* late collisions and penalise *trivial* early ones, instead
    of treating every contact as +1. When disabled, ``ego_collision`` stays 0
    (current default behaviour) but ``ego_collision_time`` is still populated as
    +inf so downstream reward code is uniform.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)

    def before_rollout(self, ctx: RolloutContext) -> None:
        ctx.metrics["ego_collision_time"] = np.full(
            ctx.num_scenes, np.inf, dtype=np.float32
        )

    def before_step_scene(self, ctx: RolloutContext, scene_idx: int, sim: SimScene) -> None:
        if not self.enabled:
            ctx.metrics["ego_collision"][scene_idx] = 0.0
            return
        if sim.ego_collides_now():
            ctx.metrics["ego_collision"][scene_idx] = 1.0
            t = float(ctx.t * sim.dt)
            if t < ctx.metrics["ego_collision_time"][scene_idx]:
                ctx.metrics["ego_collision_time"][scene_idx] = t


class EgoMinTTCHook(RewardHook):
    """Dense criticality feature: min ego time-to-collision over the rollout."""

    def before_rollout(self, ctx: RolloutContext) -> None:
        ctx.metrics["ego_min_ttc"] = np.full(ctx.num_scenes, np.inf, dtype=np.float32)

    def before_step_scene(self, ctx: RolloutContext, scene_idx: int, sim: SimScene) -> None:
        ttc = sim.ego_min_ttc_now()
        if ttc < ctx.metrics["ego_min_ttc"][scene_idx]:
            ctx.metrics["ego_min_ttc"][scene_idx] = ttc


class EgoOffroadHook(RewardHook):
    """Compatibility hook for generated maps without road-edge offroad checks."""

    def before_rollout(self, ctx: RolloutContext) -> None:
        ctx.metrics.setdefault("ego_offroad", np.zeros(ctx.num_scenes, dtype=np.float32))


class ReachedGoalHook(RewardHook):
    """Track ego goal completion and stop finished scenes."""

    def __init__(self, goal_radius: float):
        self.goal_radius = float(goal_radius)

    def before_rollout(self, ctx: RolloutContext) -> None:
        # Egos whose generated goal is already inside the static-agent threshold
        # are never controlled by PufferDrive; their scene is trivially complete.
        for s, sim in enumerate(ctx.sims):
            if sim.n == 0 or 0 not in sim.controlled:
                ctx.metrics["reached_goal"][s] = 1.0
                ctx.finished[s] = True

    def before_step_scene(self, ctx: RolloutContext, scene_idx: int, sim: SimScene) -> None:
        # State-based fallback, including ego spawned near its goal.
        d = float(np.hypot(sim.goal[0, 0] - sim.x[0], sim.goal[0, 1] - sim.y[0]))
        if d < self.goal_radius:
            ctx.metrics["reached_goal"][scene_idx] = 1.0
            ctx.finished[scene_idx] = True
            if ctx.trajectories is not None:
                ctx.trajectories[scene_idx]["done"].append(True)

    def after_step_scene(
        self,
        ctx: RolloutContext,
        scene_idx: int,
        sim: SimScene,
        *,
        ego_reached: bool,
    ) -> None:
        if ego_reached:
            ctx.metrics["reached_goal"][scene_idx] = 1.0
            ctx.finished[scene_idx] = True


class TrajectoryHook(RewardHook):
    """Record per-scene rollout trajectories for visualization."""

    def before_rollout(self, ctx: RolloutContext) -> None:
        if not ctx.record_trajectories:
            return
        ctx.trajectories = [
            {
                "x": [],
                "y": [],
                "heading": [],
                "respawn": [],
                "done": [],
                "length": sim.length.copy(),
                "width": sim.width.copy(),
            }
            for sim in ctx.sims
        ]

    def before_step_scene(self, ctx: RolloutContext, scene_idx: int, sim: SimScene) -> None:
        if ctx.trajectories is None:
            return
        tr = ctx.trajectories[scene_idx]
        tr["x"].append(sim.x.copy())
        tr["y"].append(sim.y.copy())
        tr["heading"].append(sim.heading.copy())
        tr["respawn"].append(sim.respawned.copy())

    def after_step_scene(
        self,
        ctx: RolloutContext,
        scene_idx: int,
        sim: SimScene,
        *,
        ego_reached: bool,
    ) -> None:
        if ctx.trajectories is not None and ctx.trajectories[scene_idx]["x"]:
            ctx.trajectories[scene_idx]["done"].append(bool(ego_reached))

    def after_rollout(self, ctx: RolloutContext) -> None:
        if ctx.trajectories is None:
            return
        for tr in ctx.trajectories:
            tr["x"] = np.asarray(tr["x"], dtype=np.float32) if tr["x"] else np.zeros((0, 0), np.float32)
            tr["y"] = np.asarray(tr["y"], dtype=np.float32) if tr["y"] else np.zeros((0, 0), np.float32)
            tr["heading"] = (
                np.asarray(tr["heading"], dtype=np.float32)
                if tr["heading"]
                else np.zeros((0, 0), np.float32)
            )
            tr["respawn"] = (
                np.asarray(tr["respawn"], dtype=bool)
                if tr["respawn"]
                else np.zeros((0, 0), bool)
            )
            tr["done"] = np.asarray(tr["done"], dtype=bool)


class GoalOfflaneHook(RewardHook):
    """Penalty feature for moving cars placed off the lane graph.

    Every moving car (generated goal >= 2 m from spawn, i.e. in
    ``initial_controlled``) is required to keep BOTH its spawn and its goal on a
    lane: a moving vehicle is flagged off-lane when its spawn is farther than
    ``onroad_threshold`` from the nearest lane centerline OR its goal is farther
    than ``threshold``. This closes the reward-hacking hole where the policy
    spawned an adversary off-road to dodge the goal-off-lane penalty (an off-road
    spawn used to exempt the agent entirely). Pedestrians/cyclists are exempt
    (they do not follow the lane graph). The penalty is the fraction of moving
    cars that are off-lane by either criterion. Distances that cannot be measured
    (scene has no lane geometry -> +inf) do not count as off-lane.
    """

    def __init__(self, threshold: float, onroad_threshold: float = 1.0):
        self.threshold = float(threshold)
        self.onroad_threshold = float(onroad_threshold)

    def after_rollout(self, ctx: RolloutContext) -> None:
        frac = np.zeros(ctx.num_scenes, dtype=np.float32)
        # Continuous worst-case lane distances over eligible moving cars, for the
        # smoothstep lane penalty in reward.py (replaces the binary fraction,
        # which jumps 0->1 in the one-agent setup). Non-finite (no lane geometry)
        # distances are ignored -> 0, matching the fraction's "doesn't count".
        goal_lane_dist = np.zeros(ctx.num_scenes, dtype=np.float32)
        spawn_lane_dist = np.zeros(ctx.num_scenes, dtype=np.float32)
        for s, sim in enumerate(ctx.sims):
            idx = sim.initial_controlled
            if len(idx) == 0:
                continue
            eligible = sim.ptype[idx] == TYPE_VEHICLE
            if not eligible.any():
                continue
            spawn_d = sim.dist_to_lane_centerline(sim.spawn[idx, :2])
            goal_d = sim.dist_to_lane_centerline(sim.goal[idx])
            offlane = eligible & (
                (np.isfinite(spawn_d) & (spawn_d > self.onroad_threshold))
                | (np.isfinite(goal_d) & (goal_d > self.threshold))
            )
            frac[s] = float(offlane.sum() / eligible.sum())
            gd = goal_d[eligible & np.isfinite(goal_d)]
            sd = spawn_d[eligible & np.isfinite(spawn_d)]
            if gd.size:
                goal_lane_dist[s] = float(gd.max())
            if sd.size:
                spawn_lane_dist[s] = float(sd.max())
        ctx.metrics["goal_offlane_frac"] = frac
        ctx.metrics["goal_lane_dist"] = goal_lane_dist
        ctx.metrics["spawn_lane_dist"] = spawn_lane_dist


class ParkingMismatchHook(RewardHook):
    """Penalty feature for generated-vs-ground-truth parking state mismatch."""

    def after_rollout(self, ctx: RolloutContext) -> None:
        frac = np.zeros(ctx.num_scenes, dtype=np.float32)
        gt_parking = ctx.scenes.meta.get("gt_parking_mask")
        if gt_parking is not None:
            if isinstance(gt_parking, torch.Tensor):
                gt_parking = gt_parking.detach().cpu().numpy()
            a_idx = ctx.scenes.agent_scene_idx.detach().cpu().numpy()
            for s, sim in enumerate(ctx.sims):
                gt_p = gt_parking[a_idx == s]
                gen_dist = np.hypot(
                    sim.goal[:, 0] - sim.spawn[:, 0],
                    sim.goal[:, 1] - sim.spawn[:, 1],
                )
                gen_p = gen_dist < MIN_DISTANCE_TO_GOAL
                if len(gt_p):
                    frac[s] = float((gen_p != gt_p).mean())
        ctx.metrics["parking_mismatch_frac"] = frac


def controlled_nonego_local_indices(scenes, num_scenes):
    """Per-scene sim-local indices of DDPO-controlled non-ego agents.

    ``_build_scenes`` keeps the GeneratedScenes agent order when slicing each
    scene, so an agent's local index within its scene equals its index in the
    corresponding SimScene (ego is always local 0). ``meta['controlled_mask']``
    (set by the policy decode) flags the generated nodes; the ego is dropped.
    Returns empty arrays when no mask is present (e.g. raw conditioning scenes).
    """
    controlled = scenes.meta.get("controlled_mask")
    if controlled is None:
        return [np.zeros(0, dtype=np.int64) for _ in range(num_scenes)]
    if isinstance(controlled, torch.Tensor):
        controlled = controlled.detach().cpu().numpy()
    a_idx = scenes.agent_scene_idx
    if isinstance(a_idx, torch.Tensor):
        a_idx = a_idx.detach().cpu().numpy()
    out = []
    for s in range(num_scenes):
        local = np.nonzero(controlled[a_idx == s])[0]
        out.append(local[local > 0].astype(np.int64))
    return out


class EgoAdvMinDistHook(RewardHook):
    """Dense shaping feature: min same-step ego<->controlled-adversary distance.

    Complements EgoMinTTCHook, which sweeps only the ego forward (an adversary
    closing on a slow/stationary ego yields TTC=inf, hence no gradient). This
    symmetric centre distance gives signal at any range and regardless of which
    party is moving. Only DDPO-controlled non-ego agents are measured, so the
    metric is attributable to the policy (fixed GT neighbours never move it).

    Three features are exposed so the reward can score *closing* interactions
    rather than mere spatial proximity (which the policy can trivially hack by
    spawning the adversary on top of the ego):

      * ``ego_adv_min_dist``        - min over the whole rollout (legacy);
      * ``ego_adv_init_dist`` (d0)  - clearance at t=0 (before any motion);
      * ``ego_adv_min_dist_warmup`` - min ignoring the first ``warmup_time``
                                      seconds, so an adversary that only starts
                                      close (large d0 - dmin == 0) scores low.
    """

    def __init__(self, warmup_time: float = 0.5):
        self.warmup_time = float(warmup_time)

    def before_rollout(self, ctx: RolloutContext) -> None:
        n = ctx.num_scenes
        ctx.metrics["ego_adv_min_dist"] = np.full(n, np.inf, dtype=np.float32)
        ctx.metrics["ego_adv_init_dist"] = np.full(n, np.inf, dtype=np.float32)
        ctx.metrics["ego_adv_min_dist_warmup"] = np.full(n, np.inf, dtype=np.float32)
        self._adv = controlled_nonego_local_indices(ctx.scenes, ctx.num_scenes)

    def before_step_scene(self, ctx: RolloutContext, scene_idx: int, sim: SimScene) -> None:
        adv = self._adv[scene_idx]
        if len(adv) == 0 or sim.respawned[0]:
            return
        adv = adv[~sim.respawned[adv]]  # drop removed / respawned adversaries
        if len(adv) == 0:
            return
        dx = sim.x[adv] - sim.x[0]
        dy = sim.y[adv] - sim.y[0]
        d = float(np.sqrt(dx * dx + dy * dy).min())
        if d < ctx.metrics["ego_adv_min_dist"][scene_idx]:
            ctx.metrics["ego_adv_min_dist"][scene_idx] = d
        if ctx.t == 0:
            ctx.metrics["ego_adv_init_dist"][scene_idx] = d
        if ctx.t * sim.dt >= self.warmup_time and d < ctx.metrics["ego_adv_min_dist_warmup"][scene_idx]:
            ctx.metrics["ego_adv_min_dist_warmup"][scene_idx] = d


class ControlledParkingHook(RewardHook):
    """Penalty feature: fraction of controlled non-ego agents generated parked.

    A generated adversary whose goal sits within MIN_DISTANCE_TO_GOAL of its
    spawn is static (the sim never controls it - see ``set_active_agents``). To
    push the policy to make the adversary actually drive, penalise the fraction
    of controlled non-ego agents that are parked. Unlike ParkingMismatchHook this
    is independent of GT, so it works in agent_only mode (no gt_parking_mask).
    """

    def after_rollout(self, ctx: RolloutContext) -> None:
        frac = np.zeros(ctx.num_scenes, dtype=np.float32)
        adv = controlled_nonego_local_indices(ctx.scenes, ctx.num_scenes)
        for s, sim in enumerate(ctx.sims):
            idx = adv[s]
            if len(idx) == 0:
                continue
            gen_dist = np.hypot(
                sim.goal[idx, 0] - sim.spawn[idx, 0],
                sim.goal[idx, 1] - sim.spawn[idx, 1],
            )
            frac[s] = float((gen_dist < MIN_DISTANCE_TO_GOAL).mean())
        ctx.metrics["controlled_parking_frac"] = frac

