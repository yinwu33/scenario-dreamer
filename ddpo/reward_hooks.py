"""Hook components for planner-backed DDPO reward rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .geometry import _corners, _obb_overlap_frac, sat_first_contact_time
from .interfaces import GeneratedScenes
from .pufferdrive_sim import (
    COLLISION_DIST2_GATE,
    MIN_DISTANCE_TO_GOAL,
    TYPE_PEDESTRIAN,
    TYPE_VEHICLE,
    SimScene,
)

TTC_SWEEP_HORIZON = 10.0

# Per-field null-token id for the adv conditioning ([type, motion, goal_dist,
# ego_dist]) = that field's bucket count (the LabelEmbedder's unconditional row).
# A target equal to this means the field was set to null in adv_cond_target (fed
# the model's null token, genuinely unconstrained), so the realized-vs-target
# check skips it. MUST match ``_ADV_COND_NULL`` in ddpo/conditioning.py.
_ADV_COND_NULL = (3, 2, 3, 3)  # (type, motion, goal_dist, ego_dist)


@dataclass
class GenInvalidCheck:
    """Config for ``RewardHookGenAgentInvalid``.

    Bucket thresholds mirror the dataset's adv-conditioning discretization
    (``WaymoDatasetLDMAdv._adv_condition`` / ``cfgs/ldm_adv/dataset.yaml``); they
    MUST match or valid samples get wrongly rejected. Each ``check_*`` toggles
    one field of the ``[type, motion, goal_dist, ego_dist]`` condition; disable a
    field the model cannot reliably control (or that the target does not pin).
    """

    goaldist_near: float
    goaldist_far: float
    egodist_near: float
    egodist_far: float
    min_distance_to_goal: float = MIN_DISTANCE_TO_GOAL
    check_type: bool = True
    check_motion: bool = True
    check_goal_dist: bool = True
    check_ego_dist: bool = True


def _bucket_gap(dist: float, bucket: int, near: float, far: float) -> float:
    """Metres from ``dist`` to the near(0)/middle(1)/far(2) target ``bucket``.

    0.0 when ``dist`` already falls inside the bucket's interval
    ([0, near) / [near, far) / [far, inf))."""
    if bucket == 0:
        return max(dist - near, 0.0)
    if bucket == 1:
        return max(near - dist, dist - far, 0.0)
    return max(far - dist, 0.0)


def gen_invalid_gap(
    check,               # GenInvalidCheck or any object with the same fields
    tgt,                 # [type, motion, goal_dist, ego_dist] target bucket ids
    realized_type: int,  # condition-space type id (vehicle=0 / ped=1 / cyc=2)
    goal_dist: float,    # ||goal - spawn|| of the adversary (m)
    ego_dist: float,     # ||adv spawn - ego spawn|| (m)
) -> tuple[bool, str, float]:
    """Realized-vs-target condition check with a graded distance-to-valid gap.

    Used by ``RewardHookGenAgentInvalid`` to emit the
    ``gen_agent_is_invalid`` / ``gen_agent_invalid_reason`` /
    ``gen_agent_invalid_gap`` metrics. A field whose target is the null token
    was generated unconditionally (``adv_cond_target: null``) and is skipped.

    Returns ``(invalid, reason, gap)`` where ``gap`` is how many metres the
    realized scalar sits outside the target bucket, maximised over the violated
    distance checks (motion / goal_dist / ego_dist); a wrong TYPE is categorical
    -- no metric distance exists -- and yields ``inf`` (full severity). A valid
    adversary returns ``(False, "", 0.0)``. The reward turns the gap into the
    graded reject penalty (``RewardConfig.invalid_grade_scale``) so GRPO keeps a
    "how far past the boundary" ordering instead of a flat -1 cliff.
    """
    bad = False
    reason = []
    gap = 0.0
    if check.check_type and int(tgt[0]) != _ADV_COND_NULL[0]:
        is_wrong_type = int(realized_type) != int(tgt[0])
        bad |= is_wrong_type
        if is_wrong_type:
            reason.append(f"type:{int(realized_type)}!={int(tgt[0])}")
            gap = float("inf")
    if check.check_motion and int(tgt[1]) != _ADV_COND_NULL[1]:
        realized_motion = 0 if goal_dist < check.min_distance_to_goal else 1
        is_wrong_motion = realized_motion != int(tgt[1])
        bad |= is_wrong_motion
        if is_wrong_motion:
            reason.append(f"motion:{realized_motion}!={int(tgt[1])}")
            gap = max(gap, abs(goal_dist - check.min_distance_to_goal))
    if check.check_goal_dist and int(tgt[2]) != _ADV_COND_NULL[2]:
        realized_gd = _bucket_id(goal_dist, check.goaldist_near, check.goaldist_far)
        is_wrong_goal_dist = realized_gd != int(tgt[2])
        bad |= is_wrong_goal_dist
        if is_wrong_goal_dist:
            reason.append(f"goal_dist:{realized_gd}!={int(tgt[2])}")
            gap = max(gap, _bucket_gap(goal_dist, int(tgt[2]), check.goaldist_near, check.goaldist_far))
    if check.check_ego_dist and int(tgt[3]) != _ADV_COND_NULL[3]:
        realized_ed = _bucket_id(ego_dist, check.egodist_near, check.egodist_far)
        is_wrong_ego_dist = realized_ed != int(tgt[3])
        bad |= is_wrong_ego_dist
        if is_wrong_ego_dist:
            reason.append(f"ego_dist:{realized_ed}!={int(tgt[3])}")
            gap = max(gap, _bucket_gap(ego_dist, int(tgt[3]), check.egodist_near, check.egodist_far))
    return bad, " ".join(reason), gap if bad else 0.0


def _bucket_id(dist: float, near: float, far: float) -> int:
    """Discretize a physical distance into near(0)/middle(1)/far(2). Matches
    ``WaymoDatasetLDMAdv._bucket``."""
    if dist < near:
        return 0
    if dist < far:
        return 1
    return 2


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

    def before_step_scene(
        self, ctx: RolloutContext, scene_idx: int, sim: SimScene
    ) -> None:
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


class RewardHookInitOverlap(RewardHook):
    """Measure how much a generated adversary overlaps a vehicle at spawn.

    Adversary-only: only overlaps that involve the generated adversary (vs the ego
    or any real neighbour) count, so a degenerate init is always the adversary's
    fault and real Waymo neighbours never penalise the reward. ``margin=0`` allows
    bumper-to-bumper spawns (the planner is expected to brake), only flagging true
    interpenetration.

    Emits a *continuous* ``init_overlap_frac`` (max over neighbours of intersection
    area / adversary area, in [0,1]) that the reward turns into a soft penalty and a
    criticality gate, instead of a hard -1 floor. Normalising by the adversary's own
    footprint (not the union, as IoU would) makes the signal independent of neighbour
    size: a half-buried adversary reads 0.5 whether it overlaps a car or a bus.
    ``init_invalid`` is kept as a boolean (frac > ``invalid_frac``) for logging / eval
    and for the RewardHookEgoCollision gate, not to floor the reward.
    """

    def __init__(self, margin: float = 0.0, invalid_frac: float = 0.0):
        self.margin = float(margin)
        self.invalid_frac = float(invalid_frac)

    def _adv_overlap_frac(self, sim: SimScene, adv_idx: int) -> float:
        """Max over neighbours of (intersection area / adversary area) at spawn."""
        if adv_idx < 0 or sim.removed[adv_idx]:
            return 0.0
        a = int(adv_idx)
        others = sim.slot_order[sim.ptype[sim.slot_order] != TYPE_PEDESTRIAN]
        rest = others[others != a]
        if len(rest) == 0:
            return 0.0
        a_box = _corners(
            sim.x[a:a + 1],
            sim.y[a:a + 1],
            sim.heading[a:a + 1],
            sim.length[a:a + 1] + 2.0 * self.margin,
            sim.width[a:a + 1] + 2.0 * self.margin,
        )[0]
        rest_box = _corners(
            sim.x[rest],
            sim.y[rest],
            sim.heading[rest],
            sim.length[rest] + 2.0 * self.margin,
            sim.width[rest] + 2.0 * self.margin,
        )
        dx = sim.x[rest] - sim.x[a]
        dy = sim.y[rest] - sim.y[a]
        gate = (dx * dx + dy * dy) <= COLLISION_DIST2_GATE
        if not gate.any():
            return 0.0
        frac = _obb_overlap_frac(a_box, rest_box[gate])
        return float(frac.max()) if frac.size else 0.0

    def before_rollout(self, ctx: RolloutContext) -> None:
        ctx.metrics.setdefault(
            "init_overlap_frac", np.zeros(ctx.num_scenes, dtype=np.float32)
        )
        adv = adv_local_indices(ctx.scenes, ctx.num_scenes)
        for s, sim in enumerate(ctx.sims):
            frac = self._adv_overlap_frac(sim, adv[s])
            ctx.metrics["init_overlap_frac"][s] = frac
            # TODO: remove -- init_invalid is now diagnostic/viz-only (the reward
            # gates on the continuous init_overlap_frac, not this boolean).
            if frac > self.invalid_frac:
                ctx.metrics["init_invalid"][s] = 1.0


class RewardHookEgoCollision(RewardHook):
    """Track ego<->adversary collisions and the time of the first one.

    Two collision notions are kept distinct:
      * a *general* collision (``crashed[0]`` from any ego<->vehicle overlap)
        stops the scene (physical realism + anti pass-through spoof) - this never
        enters the reward;
      * the rewarded ``ego_collision`` event: the ego contacted the ADVERSARY.
        It is restricted to the adversary so the reward is about the adversary
        (the ego hitting a real neighbour is not credited) but is *fault-agnostic*
        - the adversary ramming a passive ego counts just as much as the ego
        driving into the adversary, since both are critical scenes. The companion
        ``ego_fault_collision`` records the subset where the ego was the aggressor,
        for responsibility analysis (it does not enter the reward).

    Timing. The collision is created mid-step inside ``_advance`` and immediately
    frozen by ``latch_ego_crash`` (which zeroes both vehicles' velocity). A
    velocity-based re-check on the *next* observation would therefore always read
    the ego as passive and miss the event entirely. Instead the rewarded event is
    consumed from ``sim.last_ego_collision_partners`` in ``after_step_scene`` - the
    same step it is latched, before the contact response is observed.

    All three outputs (``ego_collision``, ``ego_collision_time``,
    ``ego_fault_collision``) are diagnostic-only: collision no longer enters the
    reward (a contact still surfaces via the dense TTC term). ``ego_collision_time``
    is time-stamped so eval can separate *meaningful* late collisions from
    *trivial* early ones. The hook is self-contained -- it consumes only
    ``sim.last_ego_collision_partners`` and does not read any other hook's metric;
    overlap discounting of a degenerate init is applied once, in the reward, via
    the continuous ``init_overlap_frac`` gate.
    """

    def before_rollout(self, ctx: RolloutContext) -> None:
        ctx.metrics["ego_collision_time"] = np.full(
            ctx.num_scenes, np.inf, dtype=np.float32
        )
        ctx.metrics.setdefault(
            "ego_fault_collision", np.zeros(ctx.num_scenes, dtype=np.float32)
        )
        # Per-scene sim-local index of the generated adversary (the only agent
        # the rewarded collision is scored against).
        self._adv = adv_local_indices(ctx.scenes, ctx.num_scenes)

    def before_step_scene(
        self, ctx: RolloutContext, scene_idx: int, sim: SimScene
    ) -> None:
        # General collision: any ego<->vehicle overlap latched crashed[0] in the
        # previous step's latch_ego_crash. Stop the scene so a pass-through
        # adversary cannot re-emerge ahead of the ego and spoof min-TTC -
        # regardless of fault.
        if bool(sim.crashed[0]):
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
        adv = self._adv[scene_idx]
        if adv < 0:
            return
        # Consume this step's latch event (recorded before the contact response
        # froze the ego). focal: ego<->adversary contact regardless of fault.
        if not np.any(sim.last_ego_collision_partners == adv):
            return
        ctx.metrics["ego_collision"][scene_idx] = 1.0
        # The contact occurred during this step's advance (t -> t+1).
        t = float((ctx.t + 1) * sim.dt)
        if t < ctx.metrics["ego_collision_time"][scene_idx]:
            ctx.metrics["ego_collision_time"][scene_idx] = t
        if np.any(sim.last_ego_fault_partners == adv):
            ctx.metrics["ego_fault_collision"][scene_idx] = 1.0


class RewardHookEgoAnyCollision(RewardHook):
    """Ego collision with ANY vehicle -- the planner-quality collision notion.

    Companion to ``RewardHookEgoCollision``, which scores the ego against the
    *generated adversary only* (``if adv < 0: return``) because DDPO is asking
    "did the adversary create a critical scene". A planner benchmark asks the
    opposite question -- "did this planner crash" -- and the scenes it runs on
    (log scenes) have no adversary at all, so the adversarial hook would report a
    flat zero. Here every ego<->vehicle contact counts, whoever caused it.

    Same timing constraint as the adversarial hook: the event is consumed from
    ``sim.last_ego_collision_partners`` in ``after_step_scene``, the step it is
    latched, because ``latch_ego_crash`` zeroes both vehicles' velocities
    immediately and a later re-check would read the ego as passive.

    Metrics:
      * ``ego_collision_any``       -- 1.0 if the ego contacted any vehicle
      * ``ego_fault_collision_any`` -- subset where the ego was the aggressor
      * ``ego_collision_time``      -- time (s) of the first contact, else inf
    """

    def before_rollout(self, ctx: RolloutContext) -> None:
        n = ctx.num_scenes
        ctx.metrics["ego_collision_any"] = np.zeros(n, dtype=np.float32)
        ctx.metrics["ego_fault_collision_any"] = np.zeros(n, dtype=np.float32)
        ctx.metrics["ego_collision_time"] = np.full(n, np.inf, dtype=np.float32)

    def before_step_scene(
        self, ctx: RolloutContext, scene_idx: int, sim: SimScene
    ) -> None:
        # The ego was frozen by the previous step's latch_ego_crash: the run is
        # over, further steps would only measure a stationary wreck.
        if sim.n and bool(sim.crashed[0]):
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
        if not len(sim.last_ego_collision_partners):
            return
        ctx.metrics["ego_collision_any"][scene_idx] = 1.0
        t = float((ctx.t + 1) * sim.dt)   # the contact happened during t -> t+1
        if t < ctx.metrics["ego_collision_time"][scene_idx]:
            ctx.metrics["ego_collision_time"][scene_idx] = t
        if len(sim.last_ego_fault_partners):
            ctx.metrics["ego_fault_collision_any"][scene_idx] = 1.0


class RewardHookEgoMinTTC(RewardHook):
    """Dense criticality feature: min ego time-to-collision over the rollout.

    Restricted to the generated adversary, so the criticality TTC measures only
    the adversary closing on the ego (not any real neighbour in the scene).
    """

    def before_rollout(self, ctx: RolloutContext) -> None:
        ctx.metrics["ego_min_ttc"] = np.full(ctx.num_scenes, np.inf, dtype=np.float32)
        self._adv = adv_local_indices(ctx.scenes, ctx.num_scenes)

    def _ego_min_ttc_now(self, sim: SimScene, others=None) -> float:
        """Relative-velocity TTC for the ego converging with another active agent."""
        if sim.n <= 1 or sim.crashed[0]:
            return float(np.inf)
        if others is None:
            others = sim.slot_order[sim.slot_order != 0]
        else:
            # An externally supplied adversary may itself have been retired.
            others = np.asarray(others, dtype=np.int64)
            others = others[(others != 0) & (~sim.removed[others])]
        others = others[
            (sim.ptype[others] != TYPE_PEDESTRIAN) & (~sim.crashed[others])
        ]
        if not len(others):
            return float(np.inf)

        # Only score agents the ego is actively driving toward; a car bearing down
        # on a passive ego is not an ego-caused near miss.
        others = others[sim._ego_aggressor_mask(others)]
        if not len(others):
            return float(np.inf)

        rvx = sim.vx[others] - sim.vx[0]
        rvy = sim.vy[others] - sim.vy[0]
        closing = (rvx * rvx + rvy * rvy) >= 1e-6
        if not closing.any():
            return float(np.inf)
        others, rvx, rvy = others[closing], rvx[closing], rvy[closing]

        ego_box = _corners(
            np.asarray([sim.x[0]]),
            np.asarray([sim.y[0]]),
            np.asarray([sim.heading[0]]),
            np.asarray([sim.length[0]]),
            np.asarray([sim.width[0]]),
        )[0]
        base_boxes = _corners(
            sim.x[others],
            sim.y[others],
            sim.heading[others],
            sim.length[others],
            sim.width[others],
        )
        # Closed-form first-contact time over the same dt grid, replacing the
        # per-step SAT sweep (bit-exact; see geometry.sat_first_contact_time). The
        # sweep returned the first grid step where ANY box overlapped == the min
        # over boxes of their individual first-contact times.
        rv = np.stack([rvx, rvy], axis=1)
        contact = sat_first_contact_time(
            ego_box, base_boxes, rv, sim.dt, TTC_SWEEP_HORIZON
        )
        return float(contact.min())

    def before_step_scene(
        self, ctx: RolloutContext, scene_idx: int, sim: SimScene
    ) -> None:
        adv = self._adv[scene_idx]
        if adv < 0:
            return
        ttc = self._ego_min_ttc_now(sim, others=np.array([adv], dtype=np.int64))
        if ttc < ctx.metrics["ego_min_ttc"][scene_idx]:
            ctx.metrics["ego_min_ttc"][scene_idx] = ttc


class RewardHookReachedGoal(RewardHook):
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

    def before_step_scene(
        self, ctx: RolloutContext, scene_idx: int, sim: SimScene
    ) -> None:
        if ctx.finished[scene_idx]:
            return
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


class RewardHookTrajectory(RewardHook):
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

    def before_step_scene(
        self, ctx: RolloutContext, scene_idx: int, sim: SimScene
    ) -> None:
        if ctx.trajectories is None:
            return
        tr = ctx.trajectories[scene_idx]
        tr["x"].append(sim.x.copy())
        tr["y"].append(sim.y.copy())
        tr["heading"].append(sim.heading.copy())
        # The "respawn" trajectory key is the shared cross-backend "hide this agent"
        # viz mask; the numpy sim has no respawn, so it is fed from the removed mask.
        tr["respawn"].append(sim.removed.copy())

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
            tr["x"] = (
                np.asarray(tr["x"], dtype=np.float32)
                if tr["x"]
                else np.zeros((0, 0), np.float32)
            )
            tr["y"] = (
                np.asarray(tr["y"], dtype=np.float32)
                if tr["y"]
                else np.zeros((0, 0), np.float32)
            )
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


def dist_to_lane_centerline(sim: SimScene, points: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest lane-centerline segment."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float32))
    if sim.seg_start.shape[0] == 0:
        return np.full(points.shape[0], np.inf, dtype=np.float32)
    a, b = sim.seg_start, sim.seg_end
    ab = b - a
    denom = np.maximum((ab * ab).sum(-1), 1e-9)
    ap = points[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(-1) / denom[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    d = points[:, None, :] - proj
    return np.sqrt((d * d).sum(-1)).min(axis=1)


class RewardHookGoalOfflane(RewardHook):
    """Penalty feature for the DDPO-generated adversary placed off the lane graph.

    Only the generated adversary (``scenes.adv_local_idx``) is scored here, and a
    pedestrian/cyclist adversary is exempt. Ego and fixed GT agents are
    deliberately excluded so the lane constraint is attributable to the policy.
    The adversary is flagged off-lane when its spawn is farther than
    ``onroad_threshold`` from the nearest lane centerline OR its goal is farther
    than ``threshold``. Distances that cannot be measured (scene has no lane
    geometry -> +inf) do not count as off-lane.
    """

    def __init__(self, threshold: float, onroad_threshold: float = 1.0):
        self.threshold = float(threshold)
        self.onroad_threshold = float(onroad_threshold)

    def _dist_to_lane_centerline(self, sim: SimScene, points: np.ndarray) -> np.ndarray:
        return dist_to_lane_centerline(sim, points)

    def after_rollout(self, ctx: RolloutContext) -> None:
        frac = np.zeros(ctx.num_scenes, dtype=np.float32)
        # Continuous worst-case lane distances over eligible generated vehicles,
        # for the smoothstep lane penalty in reward.py. Non-finite (no lane
        # geometry) distances are ignored -> 0, matching the fraction's
        # "doesn't count".
        goal_lane_dist = np.zeros(ctx.num_scenes, dtype=np.float32)
        spawn_lane_dist = np.zeros(ctx.num_scenes, dtype=np.float32)
        adv = adv_local_indices(ctx.scenes, ctx.num_scenes)
        for s, sim in enumerate(ctx.sims):
            a = adv[s]
            if a < 0 or sim.ptype[a] != TYPE_VEHICLE:
                continue
            spawn_d = float(self._dist_to_lane_centerline(sim, sim.spawn[a:a + 1, :2])[0])
            goal_d = float(self._dist_to_lane_centerline(sim, sim.goal[a:a + 1])[0])
            offlane = (np.isfinite(spawn_d) and spawn_d > self.onroad_threshold) or (
                np.isfinite(goal_d) and goal_d > self.threshold
            )
            frac[s] = 1.0 if offlane else 0.0
            if np.isfinite(goal_d):
                goal_lane_dist[s] = goal_d
            if np.isfinite(spawn_d):
                spawn_lane_dist[s] = spawn_d
        ctx.metrics["goal_offlane_frac"] = frac
        ctx.metrics["goal_lane_dist"] = goal_lane_dist
        ctx.metrics["spawn_lane_dist"] = spawn_lane_dist


class RewardHookEgoOffroadProxy(RewardHook):
    """Ego off-road proxy: per-step ego distance to the nearest lane centerline.

    The maps carry lane centerlines only (no ROAD_EDGE entities), so the sim's
    real off-road check never fires (``ego_offroad`` is always 0). This hook is
    the cross-source-comparable substitute: the ego counts as off-road on a step
    when its centre is farther than ``threshold`` metres from every lane
    centerline. Steps after the scene finishes (goal reached) are not measured;
    non-finite distances (scene without lane geometry) are skipped.

    Metrics:
      * ``ego_offroad_proxy``  -- 1.0 if any measured step was off-road
      * ``ego_offroad_frac``   -- off-road fraction of the measured steps
      * ``ego_lane_dist_max``  -- max centerline distance over measured steps
    """

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def before_rollout(self, ctx: RolloutContext) -> None:
        n = ctx.num_scenes
        ctx.metrics["ego_offroad_proxy"] = np.zeros(n, dtype=np.float32)
        ctx.metrics["ego_offroad_frac"] = np.zeros(n, dtype=np.float32)
        ctx.metrics["ego_lane_dist_max"] = np.zeros(n, dtype=np.float32)
        ctx.extras["ego_offroad_steps"] = np.zeros(n, dtype=np.int64)
        ctx.extras["ego_offroad_measured"] = np.zeros(n, dtype=np.int64)

    def before_step_scene(
        self, ctx: RolloutContext, scene_idx: int, sim: SimScene
    ) -> None:
        if sim.n == 0:
            return
        d = float(dist_to_lane_centerline(sim, np.array([[sim.x[0], sim.y[0]]]))[0])
        if not np.isfinite(d):
            return
        ctx.extras["ego_offroad_measured"][scene_idx] += 1
        if d > ctx.metrics["ego_lane_dist_max"][scene_idx]:
            ctx.metrics["ego_lane_dist_max"][scene_idx] = d
        if d > self.threshold:
            ctx.extras["ego_offroad_steps"][scene_idx] += 1
            ctx.metrics["ego_offroad_proxy"][scene_idx] = 1.0

    def after_rollout(self, ctx: RolloutContext) -> None:
        measured = np.maximum(ctx.extras["ego_offroad_measured"], 1)
        ctx.metrics["ego_offroad_frac"] = (
            ctx.extras["ego_offroad_steps"] / measured
        ).astype(np.float32)


class RewardHookParkingMismatch(RewardHook):
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


def adv_local_indices(scenes, num_scenes):
    """Per-scene sim-local index of THE generated adversary (-1 if none).

    Single-adversary contract: each scene has at most one generated non-ego
    agent. ``GeneratedScenes.adv_local_idx`` (set by the policy decode) is that
    index; the order it was computed in matches how ``_build_scenes`` slices each
    ``SimScene`` (ego is local 0). Returns ``-1`` for scenes with no adversary
    (e.g. raw conditioning scenes, or a policy that sets no adv index).
    """
    adv = getattr(scenes, "adv_local_idx", None)
    if adv is None:
        return np.full(num_scenes, -1, dtype=np.int64)
    if isinstance(adv, torch.Tensor):
        adv = adv.detach().cpu().numpy()
    return np.asarray(adv, dtype=np.int64)


class RewardHookEgoAdvMinDist(RewardHook):
    """Dense shaping feature: min same-step ego<->adversary distance.

    Complements RewardHookEgoMinTTC, which sweeps only the ego forward (an adversary
    closing on a slow/stationary ego yields TTC=inf, hence no gradient). This
    symmetric centre distance gives signal at any range and regardless of which
    party is moving. Only the generated adversary is measured, so the metric is
    attributable to the policy (fixed GT neighbours never move it).

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
        self._adv = adv_local_indices(ctx.scenes, ctx.num_scenes)

    def before_step_scene(
        self, ctx: RolloutContext, scene_idx: int, sim: SimScene
    ) -> None:
        adv = self._adv[scene_idx]
        if adv < 0 or sim.removed[0] or sim.crashed[0]:
            return
        # Drop a removed / crashed adversary. A crashed one is frozen against the
        # ego by the collision response; it must not keep pinning the min distance
        # after contact.
        if sim.removed[adv] or sim.crashed[adv]:
            return
        dx = sim.x[adv] - sim.x[0]
        dy = sim.y[adv] - sim.y[0]
        d = float(np.hypot(dx, dy))
        if d < ctx.metrics["ego_adv_min_dist"][scene_idx]:
            ctx.metrics["ego_adv_min_dist"][scene_idx] = d
        if ctx.t == 0:
            ctx.metrics["ego_adv_init_dist"][scene_idx] = d
        if (
            ctx.t * sim.dt >= self.warmup_time
            and d < ctx.metrics["ego_adv_min_dist_warmup"][scene_idx]
        ):
            ctx.metrics["ego_adv_min_dist_warmup"][scene_idx] = d


class RewardHookGenAgentParking(RewardHook):
    """Penalty feature: whether the generated adversary is parked (1.0 / 0.0).

    A generated adversary whose goal sits within MIN_DISTANCE_TO_GOAL of its
    spawn is static (the sim never controls it - see ``set_active_agents``). To
    push the policy to make the adversary actually drive, penalise a parked
    adversary. The metric (``gen_agent_is_parked``) is now per-scene 0/1 for
    the single adversary. Unlike RewardHookParkingMismatch this is independent of
    GT, so it works in agent_only mode (no gt_parking_mask).
    """

    def before_rollout(self, ctx: RolloutContext) -> None:
        is_parked = np.zeros(ctx.num_scenes, dtype=np.float32)
        adv = adv_local_indices(ctx.scenes, ctx.num_scenes)
        for s, sim in enumerate(ctx.sims):
            a = adv[s]
            if a < 0:
                continue
            gen_dist = float(np.hypot(
                sim.goal[a, 0] - sim.spawn[a, 0],
                sim.goal[a, 1] - sim.spawn[a, 1],
            ))
            is_parked[s] = 1.0 if gen_dist < MIN_DISTANCE_TO_GOAL else 0.0
        ctx.metrics["gen_agent_is_parked"] = is_parked


class RewardHookGenAgentInvalid(RewardHook):
    """Penalty feature: whether the generated adversary violates its condition.

    The adversary is generated from a discretized ``[type, motion, goal_dist,
    ego_dist]`` condition (see ``WaymoDatasetLDMAdv._adv_condition``). Conditional
    generation is imperfect, so the realized adversary sometimes lands in a
    different bucket than requested. This recomputes the realized labels from the
    decoded scene (physical metres, same thresholds as the dataset) and flags a
    mismatch on any *enabled* field whose target is a concrete bucket (a field
    whose target is the null token -- ``adv_cond_target: null`` -- was generated
    unconditionally and is skipped), writing the per-scene 0/1 metric
    ``gen_agent_is_invalid`` plus ``gen_agent_invalid_gap`` (metres outside the
    target bucket; feeds the graded reject penalty, see ``gen_invalid_gap``).
    Like the parked-adv gate it generalises: when ``check_motion`` is on and the
    target is motion=moving, a parked adversary is one such violation.

    The condition target is carried in ``ctx.scenes.meta['adv_cond']`` (a
    ``[num_scenes, 4]`` long array set by the policy decode); scenes with no
    adversary carry ``-1`` and are skipped. When the target is absent the metric
    is all-zeros (no-op, e.g. non-conditional flows).
    """

    def __init__(
        self,
        *,
        goaldist_near: float,
        goaldist_far: float,
        egodist_near: float,
        egodist_far: float,
        min_distance_to_goal: float = MIN_DISTANCE_TO_GOAL,
        check_type: bool = True,
        check_motion: bool = True,
        check_goal_dist: bool = True,
        check_ego_dist: bool = True,
    ):
        self.goaldist_near = float(goaldist_near)
        self.goaldist_far = float(goaldist_far)
        self.egodist_near = float(egodist_near)
        self.egodist_far = float(egodist_far)
        self.min_distance_to_goal = float(min_distance_to_goal)
        self.check_type = bool(check_type)
        self.check_motion = bool(check_motion)
        self.check_goal_dist = bool(check_goal_dist)
        self.check_ego_dist = bool(check_ego_dist)

    @classmethod
    def from_check(cls, check: GenInvalidCheck) -> "RewardHookGenAgentInvalid":
        """Build from the typed ``GenInvalidCheck`` carried on ``SimulatorConfig``."""
        from dataclasses import asdict

        return cls(**asdict(check))

    def before_rollout(self, ctx: RolloutContext) -> None:
        n = ctx.num_scenes
        is_invalid = np.zeros(n, dtype=np.float32)
        reasons = np.full(n, "", dtype=object)
        gaps = np.zeros(n, dtype=np.float32)
        cond = ctx.scenes.meta.get("adv_cond")
        if cond is None:
            ctx.metrics["gen_agent_is_invalid"] = is_invalid
            ctx.metrics["gen_agent_invalid_reason"] = reasons
            ctx.metrics["gen_agent_invalid_gap"] = gaps
            return
        if isinstance(cond, torch.Tensor):
            cond = cond.detach().cpu().numpy()
        cond = np.asarray(cond)
        adv = adv_local_indices(ctx.scenes, ctx.num_scenes)
        for s, sim in enumerate(ctx.sims):
            a = adv[s]
            if a < 0 or int(cond[s, 0]) < 0:
                continue
            tgt = cond[s]  # [type, motion, goal_dist, ego_dist] target bucket ids
            goal_dist = float(np.hypot(
                sim.goal[a, 0] - sim.spawn[a, 0],
                sim.goal[a, 1] - sim.spawn[a, 1],
            ))
            ego_dist = float(np.hypot(
                sim.spawn[a, 0] - sim.spawn[0, 0],
                sim.spawn[a, 1] - sim.spawn[0, 1],
            ))
            # sim type ids are 1/2/3 (veh/ped/cyc); condition ids are 0/1/2.
            bad, reason, gap = gen_invalid_gap(
                self, tgt, int(sim.ptype[a]) - 1, goal_dist, ego_dist
            )
            is_invalid[s] = 1.0 if bad else 0.0
            reasons[s] = reason
            gaps[s] = gap
        ctx.metrics["gen_agent_is_invalid"] = is_invalid
        ctx.metrics["gen_agent_invalid_reason"] = reasons  # for logging / eval, not the reward
        ctx.metrics["gen_agent_invalid_gap"] = gaps  # feeds the graded reject penalty
