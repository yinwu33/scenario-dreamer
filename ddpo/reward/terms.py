"""Numeric terms shared by the reward variants."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """0 at/below ``lo``, 1 at/above ``hi``, Hermite ramp in between."""
    t = np.clip((np.asarray(x, dtype=np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def ttc_grade(m: dict, ttc_tau: float) -> np.ndarray:
    """clip(1 - min_TTC / tau, 0, 1)."""
    ttc = np.asarray(m["ego_min_ttc"], dtype=np.float32)
    return np.clip(1.0 - ttc / ttc_tau, 0.0, 1.0).astype(np.float32)


def collision_ramp(m: dict, warmup: float, window: float) -> np.ndarray:
    """Collision indicator ramped over [warmup, warmup + window] seconds."""
    collision = np.asarray(m["ego_collision"], dtype=np.float32)
    ctime = np.asarray(m["ego_collision_time"], dtype=np.float32)
    return (collision * smoothstep(ctime, warmup, warmup + window)).astype(np.float32)


def trivial_collision(m: dict, trivial_t: float) -> np.ndarray:
    collision = np.asarray(m["ego_collision"], dtype=np.float32)
    ctime = np.asarray(m["ego_collision_time"], dtype=np.float32)
    return (collision * (ctime < trivial_t)).astype(np.float32)


def lane_costs(m: dict, lane_soft: float, lane_hard: float) -> tuple[np.ndarray, np.ndarray]:
    """(spawn, goal) adversary lane-centerline distance costs."""
    return (
        smoothstep(m["spawn_lane_dist"], lane_soft, lane_hard),
        smoothstep(m["goal_lane_dist"], lane_soft, lane_hard),
    )


def _ego_adv_distances(m: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Both distances are inf on a skipped rollout, and inf - inf is a nan, so
    # mask before any arithmetic rather than after it.
    d0 = np.asarray(m["ego_adv_init_dist"], dtype=np.float32)
    dmin = np.asarray(m["ego_adv_min_dist_warmup"], dtype=np.float32)
    finite = np.isfinite(d0) & np.isfinite(dmin)
    return np.where(finite, d0, 0.0), np.where(finite, dmin, 0.0), finite


def closed_in(m: dict) -> np.ndarray:
    """How far the adversary closed in over the rollout (d0 - dmin)."""
    d0, dmin, finite = _ego_adv_distances(m)
    return np.where(finite, d0 - dmin, 0.0).astype(np.float32)


def approach_grade(
    m: dict, d_safe: float, d_scale: float, close_delta: float, close_scale: float
) -> np.ndarray:
    """Proximity gate x closing gate: fires only if the adversary got close AND
    actually closed in, so spawning it next to the ego scores nothing."""
    d0, dmin, finite = _ego_adv_distances(m)
    prox = sigmoid((d_safe - dmin) / d_scale)
    closing = sigmoid((d0 - dmin - close_delta) / close_scale)
    return np.where(finite, prox * closing, 0.0).astype(np.float32)


def reject_terms(
    m: dict, *, gen_invalid_enabled: bool, grade_scale: float, parked_when_disabled: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(is_reject, reason, severity) of the condition-violation gate.

    Severity is the metric distance to the nearest valid bucket, normalised by
    ``grade_scale``; 0 disables grading and severity collapses to the flag.
    """
    n = len(m["init_overlap_frac"])
    if not gen_invalid_enabled:
        parked = np.asarray(m["gen_agent_is_parked"], dtype=np.float32)
        c_reject = parked if parked_when_disabled else np.zeros(n, dtype=np.float32)
        return c_reject, np.full(n, "", dtype=object), c_reject

    c_reject = np.asarray(m["gen_agent_is_invalid"], dtype=np.float32)
    if grade_scale > 0.0:
        gap = np.asarray(m["gen_agent_invalid_gap"], dtype=np.float32)
        sev = np.clip(gap / grade_scale, 0.0, 1.0).astype(np.float32)
    else:
        sev = c_reject
    return c_reject, m["gen_agent_invalid_reason"], sev
