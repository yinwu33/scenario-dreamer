"""Differentiable scene costs for SceneControl-style guided sampling.

Everything here operates on the diffusion model's **predicted clean scene** ``x̂₀``
and must therefore be differentiable end to end: guidance works by pushing ``x̂₀``
down the gradient of :class:`CriticalSceneCost`, so any term that detaches or
hard-thresholds simply contributes nothing.

This is deliberately an *analytic surrogate* for the DDPO reward in ``ddpo/reward.py``.
That reward is measured by rolling the scene out in PufferDrive, which is not
differentiable and cannot appear inside a sampling loop. The surrogate replaces the
rollout with a constant-speed, goal-directed extrapolation (:func:`rollout_states`)
and mirrors the reward's structure term by term:

===========================  =========================================================
DDPO reward term             surrogate here
===========================  =========================================================
``r_approach`` gate          :meth:`_closure` -- the adversary must *close* on the ego
``d_safe`` proximity         :meth:`_proximity` -- soft-min gap driven to a near-miss band
trivial-collision gate       :meth:`_ttc_window` -- conflict must not happen at t~0
``offlane_pen``              :meth:`_offroad` -- spawn and goal distance to the lane graph
(scene validity)             :meth:`_overlap` -- no footprint overlap at t = 0
===========================  =========================================================

Units: every cost is computed in **physical** units (metres, m/s, seconds), so the
weights below are interpretable. :func:`unnormalize_agent_states` converts from the
model's [-1, 1] space.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

#: Below this goal-spawn distance an agent is treated as parked and extrapolated
#: along its heading instead of toward its goal. Matches ddpo.goal_schema.
MIN_DISTANCE_TO_GOAL = 2.0

#: Columns of the 9-D agent state.
X, Y, SPEED, COS, SIN, LENGTH, WIDTH, GOAL_X, GOAL_Y = range(9)


def unnormalize_agent_states(x: torch.Tensor, cfg_dataset) -> torch.Tensor:
    """``[..., >=9]`` in the model's [-1, 1] space -> physical units. Differentiable.

    A functional (out-of-place) counterpart to ``utils.data_helpers.unnormalize_scene``,
    which writes in place and would break autograd.

    Deliberately does **not** clip to [-1, 1] the way ``unnormalize_scene`` does: a
    clip zeroes the gradient for any component that has drifted outside the range,
    which is exactly when guidance most needs to pull it back. ``x̂₀`` from a trained
    model sits near [-1, 1] anyway, and the trust-region term keeps it there.
    """
    fov = float(cfg_dataset.fov)
    half = fov / 2.0

    pos_x = (x[..., X] + 1.0) / 2.0 * fov - half
    pos_y = (x[..., Y] + 1.0) / 2.0 * fov - half
    speed = (x[..., SPEED] + 1.0) / 2.0 * (
        float(cfg_dataset.max_speed) - float(cfg_dataset.min_speed)
    ) + float(cfg_dataset.min_speed)
    cos_t = x[..., COS]
    sin_t = x[..., SIN]
    length = (x[..., LENGTH] + 1.0) / 2.0 * (
        float(cfg_dataset.max_length) - float(cfg_dataset.min_length)
    ) + float(cfg_dataset.min_length)
    width = (x[..., WIDTH] + 1.0) / 2.0 * (
        float(cfg_dataset.max_width) - float(cfg_dataset.min_width)
    ) + float(cfg_dataset.min_width)
    goal_x = (x[..., GOAL_X] + 1.0) / 2.0 * fov - half
    goal_y = (x[..., GOAL_Y] + 1.0) / 2.0 * fov - half

    return torch.stack(
        [pos_x, pos_y, speed, cos_t, sin_t, length, width, goal_x, goal_y], dim=-1
    )


def unnormalize_lane_points(lane: torch.Tensor, cfg_dataset) -> torch.Tensor:
    """``[L, P, 2]`` normalized lane polylines -> metres. Differentiable."""
    min_x, max_x = float(cfg_dataset.min_lane_x), float(cfg_dataset.max_lane_x)
    min_y, max_y = float(cfg_dataset.min_lane_y), float(cfg_dataset.max_lane_y)
    px = (lane[..., 0] + 1.0) / 2.0 * (max_x - min_x) + min_x
    py = (lane[..., 1] + 1.0) / 2.0 * (max_y - min_y) + min_y
    return torch.stack([px, py], dim=-1)


def footprint_radius(states: torch.Tensor) -> torch.Tensor:
    """Circumscribed-circle radius of each agent box, ``[N]``.

    A circle is a coarse stand-in for the oriented box, but it keeps the gap smooth
    (an exact OBB distance has non-differentiable corners) and errs conservative:
    two circles touch slightly before the boxes do, so the guided scenes are
    near-misses rather than interpenetrations.
    """
    return 0.5 * torch.sqrt(states[..., LENGTH] ** 2 + states[..., WIDTH] ** 2)


def rollout_states(states: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """Constant-speed, goal-directed extrapolation. ``[N, 9]``, ``[T]`` -> ``[N, T, 2]``.

    The scene the model generates is a single frame plus a goal, so a trajectory has
    to be assumed. Driving toward the goal (rather than straight along the heading)
    is what makes the *goal* columns part of the control surface: guidance can move
    an adversary's intent, not just where it is parked. Agents whose goal is within
    ``MIN_DISTANCE_TO_GOAL`` are parked, and fall back to their heading direction so
    the normalization stays well-conditioned.

    Travel is clamped at the goal so agents stop on arrival instead of flying past it.
    """
    pos = states[..., :2]
    goal = states[..., GOAL_X:GOAL_Y + 1]
    speed = states[..., SPEED].clamp_min(0.0)

    to_goal = goal - pos
    goal_dist = torch.linalg.norm(to_goal, dim=-1)
    safe_dist = goal_dist.clamp_min(1e-6)

    heading = torch.stack([states[..., COS], states[..., SIN]], dim=-1)
    heading = heading / torch.linalg.norm(heading, dim=-1, keepdim=True).clamp_min(1e-6)

    moving = (goal_dist > MIN_DISTANCE_TO_GOAL).unsqueeze(-1)
    direction = torch.where(moving, to_goal / safe_dist.unsqueeze(-1), heading)

    travel = speed.unsqueeze(-1) * timesteps.unsqueeze(0)          # [N, T]
    # parked agents have no goal to stop at, so only clamp the goal-directed ones
    capped = torch.minimum(travel, goal_dist.unsqueeze(-1))
    travel = torch.where(moving, capped, travel)

    return pos.unsqueeze(1) + direction.unsqueeze(1) * travel.unsqueeze(-1)


def point_to_polyline_dist(query: torch.Tensor, polylines: torch.Tensor) -> torch.Tensor:
    """Distance from each query point to the nearest lane *segment*. Differentiable.

    ``query``: ``[Q, 2]``; ``polylines``: ``[M, P, 2]``; returns ``[Q]``. The torch
    counterpart of ``utils.goal_runtime.point_to_polyline_dist`` (which is numpy and
    used for the offline off-road filter), so the guidance penalty and the dataset
    filter measure the same quantity.
    """
    if polylines.numel() == 0 or query.numel() == 0:
        return query.new_zeros(query.shape[0])

    starts = polylines[:, :-1, :].reshape(-1, 2)
    ends = polylines[:, 1:, :].reshape(-1, 2)
    seg = ends - starts
    seg_len_sq = (seg ** 2).sum(-1).clamp_min(1e-12)

    rel = query.unsqueeze(1) - starts.unsqueeze(0)                  # [Q, S, 2]
    t = ((rel * seg.unsqueeze(0)).sum(-1) / seg_len_sq.unsqueeze(0)).clamp(0.0, 1.0)
    closest = starts.unsqueeze(0) + t.unsqueeze(-1) * seg.unsqueeze(0)
    return torch.linalg.norm(query.unsqueeze(1) - closest, dim=-1).min(dim=1).values


def huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Quadratic near 0, linear beyond ``delta``. ``x`` is expected non-negative.

    The distance terms are in squared metres, so a plain ``x**2`` at a 13 m gap
    contributes ~144 while the TTC and overlap terms top out around 1. The
    optimiser then buys proximity with arbitrarily large constraint violations --
    negative speeds, adversaries parked inside other cars. Growing linearly far
    from the target keeps every term in the same order of magnitude, so the
    weights in the config mean what they say.
    """
    return torch.where(x <= delta, 0.5 * x ** 2, delta * (x - 0.5 * delta))


def soft_min(values: torch.Tensor, tau: float, dim: int = -1) -> torch.Tensor:
    """``-tau * logsumexp(-values / tau)``: a smooth minimum.

    A hard ``min`` routes the whole gradient to a single timestep, so guidance
    chatters between adjacent steps as the argmin flips. The soft version spreads it
    over every timestep near the closest approach, which is what makes the conflict
    move smoothly in time rather than snapping.
    """
    return -tau * torch.logsumexp(-values / tau, dim=dim)


def first_index_per_scene(batch: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Global index of each scene's first node. ``[batch_size]``.

    PyG concatenates scenes in order, so a scene's first agent is its ego (the
    datasets put the ego at local index 0 and ``reorder_indices`` never moves it).
    """
    counts = torch.zeros(batch_size, dtype=torch.long, device=batch.device)
    counts.scatter_add_(0, batch, torch.ones_like(batch))
    return torch.cumsum(counts, 0) - counts


class ProximityGoalCost:
    """The baseline objective: three purely geometric constraints, no motion model.

    A guidance *baseline* should encode as little of our own prior as possible.
    :class:`CriticalSceneCost` assumes a dynamics model (constant speed toward the
    goal) in order to talk about closest approach and time-to-conflict -- and then
    needs a closure gate, a TTC window and a Huber knee to keep that assumption from
    producing degenerate scenes. All of that is *our* modelling, not SceneControl's.

    This objective instead constrains only quantities the model emits directly, with
    no dynamics in between:

    1. the adversary spawns within ``adv_ego_spawn_radius`` of the ego,
    2. the adversary's goal lies within ``adv_ego_goal_radius`` of the ego's goal,
    3. the ego's own goal is at least ``ego_min_goal_dist`` away from the ego.

    Together they make a conflict *structural* rather than simulated: two agents that
    start near each other and are headed for the same place must interact, whatever
    dynamics you later roll out. (3) is what stops the scenario from being trivial --
    the undertrained generator emits a stationary ego most of the time (measured
    goal distances of 0.4 / 5.0 / 31.1 / 0.1 m on a sample of four), and a parked ego
    cannot be threatened. Real Waymo egos travel ~30 m over the same window, so the
    25 m default pulls the generated distribution back toward the data rather than
    away from it.

    Each term is a squared hinge, so a constraint that is already satisfied
    contributes exactly zero gradient and the scene is left alone.
    """

    #: Columns this cost is allowed to move on the adversary / on the ego.
    adv_columns = (X, Y, GOAL_X, GOAL_Y)
    ego_columns = (GOAL_X, GOAL_Y)  # never the ego's position: the frame is ego-centred

    def __init__(self, cfg, cfg_dataset):
        self.cfg = cfg
        self.cfg_dataset = cfg_dataset

    def __call__(
        self,
        agent_states_norm: torch.Tensor,
        lane_states_norm: torch.Tensor,
        agent_batch: torch.Tensor,
        lane_batch: torch.Tensor,
        adv_idx: torch.Tensor,
        batch_size: int,
        reference_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        states = unnormalize_agent_states(agent_states_norm, self.cfg_dataset)
        ego_idx = first_index_per_scene(agent_batch, batch_size)

        adv_pos, adv_goal = states[adv_idx, :2], states[adv_idx, GOAL_X:GOAL_Y + 1]
        ego_pos, ego_goal = states[ego_idx, :2], states[ego_idx, GOAL_X:GOAL_Y + 1]

        # Each constraint owns exactly the variable it is about, enforced by detaching
        # the other side. (1) and (2) constrain the ADVERSARY relative to the ego, so
        # the ego is a fixed reference in them; (3) constrains the EGO's goal.
        #
        # Without this the terms fight: (2) is equally well satisfied by dragging the
        # ego's goal toward the adversary's, which is often the cheaper direction, and
        # (3) cannot resist because a satisfied hinge has exactly zero gradient. It
        # only wakes up once the ego's trip has already been broken below the
        # threshold. Measured on four scenes, that regressed a scene whose ego trip
        # was a healthy 31.4 m down to 18.5 m.
        spawn_gap = torch.linalg.norm(adv_pos - ego_pos.detach(), dim=-1)
        goal_gap = torch.linalg.norm(adv_goal - ego_goal.detach(), dim=-1)
        ego_trip = torch.linalg.norm(ego_goal - ego_pos.detach(), dim=-1)

        terms = {
            "adv_spawn": self.cfg.adv_spawn_coef
            * torch.relu(spawn_gap - self.cfg.adv_ego_spawn_radius) ** 2,
            "adv_goal": self.cfg.adv_goal_coef
            * torch.relu(goal_gap - self.cfg.adv_ego_goal_radius) ** 2,
            "ego_trip": self.cfg.ego_trip_coef
            * torch.relu(self.cfg.ego_min_goal_dist - ego_trip) ** 2,
        }

        total = torch.stack(list(terms.values()), dim=0).sum()
        logged = {f"cost_{k}": float(v.mean().detach()) for k, v in terms.items()}
        # Raw distances, prefixed so they cannot collide with the cost keys above --
        # `ego_trip` previously named both, and the dict update silently dropped the
        # cost term, hiding constraint (3) from every report.
        logged.update(
            m_spawn_gap=float(spawn_gap.mean().detach()),
            m_goal_gap=float(goal_gap.mean().detach()),
            m_ego_trip=float(ego_trip.mean().detach()),
        )
        return total, logged


class CriticalSceneCost:
    """Analytic, differentiable criticality objective over a generated scene.

    Call it with the **normalized** ``x̂₀`` tensors; it unnormalizes internally so all
    the weights in ``cfg`` are in physical units.

    Returns ``(cost, terms)`` where ``cost`` is a scalar (summed over the batch, so
    one backward pass covers every scene -- the scenes are independent, so the
    per-scene gradients do not mix) and ``terms`` is a detached dict for logging.
    """

    #: Columns this cost is allowed to move. The ego is frozen entirely here.
    adv_columns = (X, Y, SPEED, COS, SIN, GOAL_X, GOAL_Y)
    ego_columns = ()

    def __init__(self, cfg, cfg_dataset):
        self.cfg = cfg
        self.cfg_dataset = cfg_dataset
        self.timesteps = None  # lazily built on the right device

    # ------------------------------------------------------------------ terms --
    def _proximity(self, dmin: torch.Tensor) -> torch.Tensor:
        """Drive the closest approach into a near-miss band.

        ``relu(dmin - d_target)**2`` pulls the adversary in until the gap reaches
        ``d_target`` and then stops -- driving it to zero instead would just produce
        interpenetrating boxes, which score well on any collision metric and are
        worthless as scenarios. ``interpenetration_coef`` pushes back if the gap goes
        negative, keeping the result a near-miss rather than an overlap.
        """
        too_far = huber(torch.relu(dmin - self.cfg.d_target), self.cfg.huber_delta)
        overlap = torch.relu(-dmin) ** 2  # stays quadratic: interpenetration must bite
        return too_far + self.cfg.interpenetration_coef * overlap

    def _closure(self, dmin: torch.Tensor, d0: torch.Tensor) -> torch.Tensor:
        """Require the conflict to *develop*: ``dmin <= d0 - min_closure``.

        Without this the cheapest way to satisfy :meth:`_proximity` is to spawn the
        adversary already next to the ego, which is the degenerate scenario the DDPO
        reward gates against with its ``sigma((d0 - dmin - delta)/s)`` factor.
        """
        return huber(torch.relu(dmin - (d0 - self.cfg.min_closure)), self.cfg.huber_delta)

    def _ttc_window(self, gap: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Keep the time of closest approach inside ``[ttc_lo, ttc_hi]`` seconds.

        ``t_star`` is a soft arg-min over the horizon (softmax weights on ``-gap``).
        Penalising ``t_star`` near 0 rules out spawn-overlap scenarios; penalising it
        near the horizon rules out conflicts too far out to matter -- the same two
        failure modes the reward's trivial-collision gate handles.
        """
        weights = torch.softmax(-gap / self.cfg.soft_min_tau, dim=-1)
        t_star = (weights * timesteps.unsqueeze(0)).sum(-1)
        early = torch.relu(self.cfg.ttc_lo - t_star) ** 2
        late = torch.relu(t_star - self.cfg.ttc_hi) ** 2
        return early + late

    def _offroad(self, adv: torch.Tensor, lanes_per_scene) -> torch.Tensor:
        """Hinge penalty on spawn and goal distance to the lane graph.

        Measured with the torch port of the same point-to-polyline distance the
        offline filter uses (``utils.goal_runtime``), so "on-road" means the same
        thing to guidance as it does to the training distribution.
        """
        costs = []
        for i, lanes in enumerate(lanes_per_scene):
            query = torch.stack([adv[i, :2], adv[i, GOAL_X:GOAL_Y + 1]], dim=0)
            dist = point_to_polyline_dist(query, lanes)
            costs.append((torch.relu(dist - self.cfg.offroad_threshold) ** 2).sum())
        return torch.stack(costs)

    def _overlap(self, adv: torch.Tensor, adv_r: torch.Tensor, others_per_scene) -> torch.Tensor:
        """No footprint overlap at t = 0 with any other agent in the scene.

        A scene that starts already in collision is invalid regardless of how
        critical it looks later, and nothing else in the objective forbids it.
        """
        costs = []
        for i, (others, others_r) in enumerate(others_per_scene):
            if others.numel() == 0:
                costs.append(adv.new_zeros(()))
                continue
            d = torch.linalg.norm(others[:, :2] - adv[i, :2].unsqueeze(0), dim=-1)
            need = adv_r[i] + others_r + self.cfg.overlap_margin
            costs.append((torch.relu(need - d) ** 2).sum())
        return torch.stack(costs)

    # ------------------------------------------------------------------- call --
    def __call__(
        self,
        agent_states_norm: torch.Tensor,
        lane_states_norm: torch.Tensor,
        agent_batch: torch.Tensor,
        lane_batch: torch.Tensor,
        adv_idx: torch.Tensor,
        batch_size: int,
        reference_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = agent_states_norm.device
        if self.timesteps is None or self.timesteps.device != device:
            self.timesteps = torch.arange(
                0.0, self.cfg.horizon + 1e-6, self.cfg.dt, device=device
            )

        states = unnormalize_agent_states(agent_states_norm, self.cfg_dataset)
        lanes = unnormalize_lane_points(lane_states_norm, self.cfg_dataset)

        ego_idx = first_index_per_scene(agent_batch, batch_size)
        adv = states[adv_idx]                                   # [B, 9]
        ego = states[ego_idx]                                   # [B, 9]

        traj_adv = rollout_states(adv, self.timesteps)          # [B, T, 2]
        traj_ego = rollout_states(ego, self.timesteps)
        centre_gap = torch.linalg.norm(traj_adv - traj_ego, dim=-1)
        adv_r, ego_r = footprint_radius(adv), footprint_radius(ego)
        gap = centre_gap - (adv_r + ego_r).unsqueeze(-1)        # [B, T]

        dmin = soft_min(gap, self.cfg.soft_min_tau, dim=-1)
        d0 = gap[:, 0]

        # per-scene slices for the two terms that need the whole scene
        lanes_per_scene = [lanes[lane_batch == i] for i in range(batch_size)]
        others_per_scene = []
        for i in range(batch_size):
            mask = agent_batch == i
            mask = mask.clone()
            mask[adv_idx[i]] = False
            others_per_scene.append((states[mask], footprint_radius(states[mask])))

        terms = {
            "proximity": self.cfg.proximity_coef * self._proximity(dmin),
            "closure": self.cfg.closure_coef * self._closure(dmin, d0),
            "ttc": self.cfg.ttc_coef * self._ttc_window(gap, self.timesteps),
            "offroad": self.cfg.offroad_coef * self._offroad(adv, lanes_per_scene),
            "overlap": self.cfg.overlap_coef * self._overlap(adv, adv_r, others_per_scene),
        }
        if reference_norm is not None and self.cfg.trust_region_coef > 0:
            # Anchor the guided adversary to what the model would have predicted on
            # its own. Guidance is an off-manifold force; without a leash it will
            # happily trade realism for criticality, which is the standard failure
            # mode of guided sampling and exactly what the realism metrics catch.
            drift = agent_states_norm[adv_idx] - reference_norm[adv_idx]
            terms["trust"] = self.cfg.trust_region_coef * (drift ** 2).sum(-1)

        total = torch.stack(list(terms.values()), dim=0).sum()
        logged = {k: float(v.mean().detach()) for k, v in terms.items()}
        logged["dmin"] = float(dmin.mean().detach())
        logged["d0"] = float(d0.mean().detach())
        return total, logged
