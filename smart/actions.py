"""Label logged motion with the SHARED accel/steer action table.

The model predicts an index into the same 7x13 table every other planner emits,
so ``SimScene.step_dynamics`` integrates it and the simulator keeps ONE
integrator (``sim.planners.ctrl_sim`` is the sole exception, and this design
avoids adding a second). That only works if the table can actually express how
logged agents move, which is what this module measures.

Labels are produced by inverting the exact forward model of
``SimScene.step_dynamics``: for each transition, every one of the 91 actions is
integrated one step and the one whose resulting BOX POSE is closest to the
observed one wins. Box-corner distance rather than centre distance, so a label
that gets the position right but the heading wrong is penalised -- the same
criterion ``ctrl_sim`` uses to tokenise observed motion against its k-disks
vocabulary.

Two errors matter and they are very different:

  * **teacher-forced** -- per-step error with the true state restored each step.
    An optimistic bound: it never accumulates.
  * **replay drift** -- integrate the whole labelled action sequence from the
    initial state only. This is what the model actually has to live with,
    because at rollout time nothing restores the true state.
"""

from __future__ import annotations

import numpy as np

from sim.world import ACCELERATION_VALUES, MAX_SPEED, NUM_STEER, STEERING_VALUES

# The full action table, flattened exactly as step_dynamics indexes it:
# accel = ACCELERATION_VALUES[a // NUM_STEER], steer = STEERING_VALUES[a % NUM_STEER].
ACCEL = np.repeat(ACCELERATION_VALUES, NUM_STEER).astype(np.float64)
STEER = np.tile(STEERING_VALUES, len(ACCELERATION_VALUES)).astype(np.float64)
NUM_ACTIONS = len(ACCEL)

# Precomputed steering geometry (bicycle slip angle), constant per action.
BETA = np.tanh(0.5 * np.tan(STEER))
TAN_STEER = np.tan(STEER)
COS_BETA = np.cos(BETA)

# Tie-break toward the least aggressive action.
#
# The yaw rate is ``s' * cos(beta) * tan(steer) / length``, so at ``s' ~ 0``
# steering has NO effect on the resulting pose and all 13 steering bins tie
# EXACTLY. A plain argmin then returns bin 0, which is full left lock. Measured:
# 38.7% of logged transitions are slower than 0.1 m/s, and 58.4% of those were
# being labelled ``steer = -1.0`` -- against 3.2% among moving ones. A model
# trained on that learns to hold full lock for a large class of inputs, and turns
# straight off the road the moment such an agent starts moving.
#
# The penalty is 1e-6 m against pose errors that are typically 1e-3 to 1e-1 m, so
# it decides ties and nothing else.
TIE_BREAK = 1e-6
NEUTRALITY = (np.abs(STEER) / np.abs(STEER).max()
              + np.abs(ACCEL) / np.abs(ACCEL).max())


def integrate(x, y, heading, signed_speed, length, actions, dt):
    """One step of ``SimScene.step_dynamics``, for arbitrary leading shape.

    Every argument broadcasts together; ``actions`` indexes the shared table.
    Returns ``(x, y, heading, signed_speed)`` after the step.
    """
    s = np.clip(signed_speed + ACCEL[actions] * dt, -MAX_SPEED, MAX_SPEED)
    yaw_rate = s * COS_BETA[actions] * TAN_STEER[actions] / length
    new_vx = s * np.cos(heading + BETA[actions])
    new_vy = s * np.sin(heading + BETA[actions])
    return x + new_vx * dt, y + new_vy * dt, heading + yaw_rate * dt, s


def _corner_offsets(length, width):
    """[..., 4, 2] body-frame corners of each box."""
    half_l, half_w = length / 2.0, width / 2.0
    return np.stack(
        [
            np.stack([-half_l, -half_w], axis=-1),
            np.stack([-half_l, half_w], axis=-1),
            np.stack([half_l, half_w], axis=-1),
            np.stack([half_l, -half_w], axis=-1),
        ],
        axis=-2,
    )


def pose_error(x, y, heading, ref_x, ref_y, ref_heading, length, width):
    """Mean distance over the 4 box corners between a pose and a reference pose.

    Shapes broadcast; the corner axis is reduced away.
    """
    off = _corner_offsets(length, width)                       # [..., 4, 2]
    c, s = np.cos(heading)[..., None], np.sin(heading)[..., None]
    rc, rs = np.cos(ref_heading)[..., None], np.sin(ref_heading)[..., None]
    dx = (x[..., None] + off[..., 0] * c - off[..., 1] * s) - (
        ref_x[..., None] + off[..., 0] * rc - off[..., 1] * rs
    )
    dy = (y[..., None] + off[..., 0] * s + off[..., 1] * c) - (
        ref_y[..., None] + off[..., 0] * rs + off[..., 1] * rc
    )
    return np.hypot(dx, dy).mean(axis=-1)


def label_transitions(state, next_state, length, width, dt, chunk=8192):
    """Best action per transition, and the residual pose error it leaves.

    ``state`` / ``next_state`` are ``[N, 4]`` ``[x, y, heading, signed_speed]``.
    Returns ``(actions [N], error_m [N])``. Chunked over N because the search is
    dense over all 91 actions.
    """
    n = len(state)
    actions = np.empty(n, dtype=np.int64)
    errors = np.empty(n, dtype=np.float64)
    all_actions = np.arange(NUM_ACTIONS)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        sl = slice(lo, hi)
        px, py, ph, _ = integrate(
            state[sl, 0][:, None], state[sl, 1][:, None], state[sl, 2][:, None],
            state[sl, 3][:, None], length[sl][:, None], all_actions[None, :], dt,
        )
        err = pose_error(
            px, py, ph,
            next_state[sl, 0][:, None], next_state[sl, 1][:, None], next_state[sl, 2][:, None],
            np.broadcast_to(length[sl][:, None], px.shape),
            np.broadcast_to(width[sl][:, None], px.shape),
        )
        best = (err + TIE_BREAK * NEUTRALITY[None, :]).argmin(axis=1)
        actions[sl] = best
        errors[sl] = err[np.arange(hi - lo), best]
    return actions, errors


def replay(state0, length, actions, dt):
    """Integrate an action SEQUENCE from the initial state, open loop.

    ``state0`` is ``[N, 4]``, ``actions`` is ``[N, T]``. Returns the resulting
    ``[N, T, 3]`` ``[x, y, heading]`` -- what the simulator would produce if a
    model emitted exactly these actions, with nothing restoring the true state.
    """
    x, y, h, s = (state0[:, i].astype(np.float64).copy() for i in range(4))
    out = np.empty((len(state0), actions.shape[1], 3), dtype=np.float64)
    for t in range(actions.shape[1]):
        x, y, h, s = integrate(x, y, h, s, length, actions[:, t], dt)
        out[:, t, 0], out[:, t, 1], out[:, t, 2] = x, y, h
    return out


def chase(reference, length, width, dt):
    """Greedy CLOSED-LOOP labelling: track a reference from the integrated pose.

    ``label_transitions`` picks each action from the TRUE state, so its residuals
    accumulate when the sequence is replayed. This instead picks, at every step,
    the action that best reaches the next reference pose FROM THE POSE THE
    INTEGRATOR IS ACTUALLY AT. Nothing restores the truth, so the resulting drift
    measures what the action space can do rather than what greedy per-step
    labelling leaves behind.

    ``reference`` is ``[T, 4]``. Returns ``(actions [T-1], poses [T-1, 3])``.
    """
    all_actions = np.arange(NUM_ACTIONS)
    t_steps = len(reference) - 1
    actions = np.empty(t_steps, dtype=np.int64)
    poses = np.empty((t_steps, 3), dtype=np.float64)
    x, y, h, s = (float(reference[0, i]) for i in range(4))
    ln = np.full(NUM_ACTIONS, length)
    wd = np.full(NUM_ACTIONS, width)
    for t in range(t_steps):
        px, py, ph, ps = integrate(x, y, h, s, length, all_actions, dt)
        ref = reference[t + 1]
        err = pose_error(
            px, py, ph,
            np.full(NUM_ACTIONS, ref[0]), np.full(NUM_ACTIONS, ref[1]),
            np.full(NUM_ACTIONS, ref[2]), ln, wd,
        )
        a = int((err + TIE_BREAK * NEUTRALITY).argmin())
        actions[t] = a
        x, y, h, s = px[a], py[a], ph[a], ps[a]
        poses[t] = (x, y, h)
    return actions, poses
