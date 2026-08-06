"""Per-scene metric helpers shared by the critical-scene evaluation scripts.

Small, metric-agnostic pieces used by both the four-source ldm_adv benchmark
(``critical_scene.ldm_adv_eval``) and the SUT x scene-initialization planner
matrix (``critical_scene.planner_matrix_eval``). They live here rather than in
either script so the lightweight one does not have to import the LDM / AE stack
just to compute a rate.

Both callers pass a mapping with ``agent_states`` / ``agent_scene_idx`` /
``num_scenes`` -- an artifact payload on one side, a ``GeneratedScenes`` adapter
on the other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rate(values: np.ndarray) -> float:
    """Fraction of scenes where the (0/1 or count) metric fired."""
    arr = np.asarray(values, dtype=np.float32)
    return float((arr > 0).mean()) if arr.size else float("nan")


def mean_finite(values: np.ndarray) -> float:
    """Mean over the finite entries only (inf marks 'never happened')."""
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    return float(arr[finite].mean()) if finite.any() else float("nan")


def _as_numpy(t) -> np.ndarray:
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def ego_goal_dist(payload: Any) -> np.ndarray:
    """Per-scene ego spawn->goal distance (agent row 0).

    States layout ``[x, y, speed, cos, sin, length, width, goal_x, goal_y]``.
    """
    states = _as_numpy(payload["agent_states"])
    scene_idx = _as_numpy(payload["agent_scene_idx"])
    n = int(payload["num_scenes"])
    out = np.full(n, np.nan, dtype=np.float32)
    for s in range(n):
        rows = np.flatnonzero(scene_idx == s)
        if rows.size == 0:
            continue
        ego = states[rows[0]]
        out[s] = float(np.hypot(ego[7] - ego[0], ego[8] - ego[1]))
    return out


def num_agents_per_scene(payload: Any) -> np.ndarray:
    return np.bincount(
        _as_numpy(payload["agent_scene_idx"]), minlength=int(payload["num_scenes"])
    ).astype(np.int64)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
