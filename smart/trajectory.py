"""Differentiable K-step rollout through the shared action table.

Cross-entropy on one action per step is not the objective we care about. What a
traffic model has to get right is the TRAJECTORY it produces once its actions are
integrated, and a single-step loss is blind to the way small per-step errors
compound over a rollout.

This module closes that gap. From the logged state at ``t`` it integrates K steps
forward using the model's own action distribution, and scores the resulting poses
against the logged ones. Two design points make it cheap and stable:

  * **Expected-action integration.** ``integrate`` is closed form and already
    vectorised over all 91 actions, so the next pose is the probability-weighted
    average of the 91 candidate poses. Differentiable in the logits, with no
    sampling, no straight-through estimator and no variance. Averaging the
    candidate headings is safe because they differ only by ``yaw_rate * dt``, a
    band far narrower than a wrap.
  * **Teacher-forced context, free-running pose.** Observations come from the
    LOGGED state at each step, because the observation builder is numpy and the
    neighbours are other agents we are not predicting. The agent's own pose,
    however, is the one the integrator actually reached, so drift accumulates
    across the chain and the loss sees it. That is the whole point: this term
    penalises accumulated error, which the per-step cross-entropy cannot.

The error is box-corner distance in metres -- the same criterion
``smart.actions`` labels with, so the loss and the targets agree on what "close"
means.
"""

from __future__ import annotations

import torch

from .actions import ACCEL, BETA, COS_BETA, MAX_SPEED, STEER, TAN_STEER


class ActionTable:
    """The shared accel/steer table as tensors on one device."""

    def __init__(self, device):
        f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        self.accel, self.beta = f(ACCEL), f(BETA)
        self.cos_beta, self.tan_steer = f(COS_BETA), f(TAN_STEER)
        self.steer = f(STEER)


def expected_step(x, y, heading, signed_speed, length, probs, table, dt):
    """One step of ``SimScene.step_dynamics``, averaged under ``probs``.

    All state arguments are ``[B]``; ``probs`` is ``[B, 91]``. Returns the next
    ``(x, y, heading, signed_speed)``, each ``[B]`` and differentiable in ``probs``.
    """
    s = torch.clamp(
        signed_speed[:, None] + table.accel[None, :] * dt, -MAX_SPEED, MAX_SPEED
    )                                                                   # [B, 91]
    yaw_rate = s * table.cos_beta[None, :] * table.tan_steer[None, :] / length[:, None]
    ang = heading[:, None] + table.beta[None, :]
    nx = x[:, None] + s * torch.cos(ang) * dt
    ny = y[:, None] + s * torch.sin(ang) * dt
    nh = heading[:, None] + yaw_rate * dt
    return (
        (probs * nx).sum(-1),
        (probs * ny).sum(-1),
        (probs * nh).sum(-1),
        (probs * s).sum(-1),
    )


# torch.hypot has a NaN gradient at exactly (0, 0), which a stationary agent hits
# every step -- and roughly a fifth of Waymo scenes have a stationary ego. A
# smoothed norm is finite there; 1e-12 is 1 micrometre of slack.
_EPS = 1e-12


def corner_distance(x, y, heading, ref_x, ref_y, ref_heading, length, width):
    """Mean distance over the 4 box corners between a pose and a reference pose."""
    half_l, half_w = length / 2.0, width / 2.0
    ox = torch.stack([-half_l, -half_l, half_l, half_l], dim=-1)
    oy = torch.stack([-half_w, half_w, half_w, -half_w], dim=-1)
    c, s = torch.cos(heading)[:, None], torch.sin(heading)[:, None]
    rc, rs = torch.cos(ref_heading)[:, None], torch.sin(ref_heading)[:, None]
    dx = (x[:, None] + ox * c - oy * s) - (ref_x[:, None] + ox * rc - oy * rs)
    dy = (y[:, None] + ox * s + oy * c) - (ref_y[:, None] + ox * rs + oy * rc)
    return torch.sqrt(dx * dx + dy * dy + _EPS).mean(-1)


def rollout_loss(logits, state0, ref, length, width, table, dt):
    """Mean per-step box distance (m) of the integrated chain from the log.

    ``logits`` [N, K, 91], ``state0`` [N, 4] = (x, y, heading, signed_speed),
    ``ref`` [N, K, 3] = the logged (x, y, heading) at steps 1..K.
    Returns ``(loss, final_distance)``; the second is diagnostic only.
    """
    x, y, h, s = (state0[:, i] for i in range(4))
    probs = logits.softmax(-1)
    total, dist = 0.0, None
    for k in range(logits.shape[1]):
        x, y, h, s = expected_step(x, y, h, s, length, probs[:, k], table, dt)
        dist = corner_distance(x, y, h, ref[:, k, 0], ref[:, k, 1], ref[:, k, 2],
                               length, width)
        total = total + dist.mean()
    return total / logits.shape[1], dist.mean().detach()
