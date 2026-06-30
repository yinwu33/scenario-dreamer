"""Native PufferDrive C rollout backend for DDPO rewards.

This module keeps the DDPO policy and frozen planner in Python/Torch, but moves
the simulator hot path (observations, dynamics, collisions, goal lifecycle) into
PufferDrive's vectorized C environment through a small scene-init binding.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from planner..planner import OBS_DIM

from .interfaces import GeneratedScenes
from .pufferdrive_sim import load_sim_config


GOAL_BEHAVIOR = {"respawn": 0, "continue": 1, "stop": 2, "remove": 3}
OFFROAD_MODE = {"road_edge": 0, "lane_center": 1}
# drive.h teleports goal-reached agents (goal_behavior="remove") and collided/
# offroad-removed agents to INVALID_POSITION=-10000. Any returned pose whose
# magnitude exceeds this (generated scenes are local, |coord|<~100 m) is a
# removed agent, not a real position.
INVALID_POSITION = -10000.0
_GONE_DISTANCE = 5000.0
CONTROL_MODE = {
    "control_vehicles": 0,
    "control_agents": 1,
    "control_wosac": 2,
    "control_sdc_only": 3,
    "control_mixed_play": 4,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_binding(pufferdrive_root: str | Path | None):
    root = Path(pufferdrive_root) if pufferdrive_root is not None else _repo_root() / "PufferDrive"
    root = root.resolve()
    drive_dir = root / "pufferlib" / "pacific" / "drive"
    candidates = sorted(drive_dir.glob("binding*.so"))
    if not candidates:
        raise RuntimeError(
            "Could not find PufferDrive native binding. Build it with "
            "`cd PufferDrive && python setup.py build_ext --inplace --force`."
        )

    try:
        spec = importlib.util.spec_from_file_location("binding", candidates[0])
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {candidates[0]}")
        binding = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(binding)
    except Exception as exc:  # pragma: no cover - exercised in integration envs
        raise RuntimeError(
            "Could not import PufferDrive native binding. Build it with "
            "`cd PufferDrive && python setup.py build_ext --inplace --force`."
        ) from exc
    return binding


def _as_numpy(value: Any, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


@dataclass
class _NativeScene:
    handle: int
    active_count: int
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    truncations: np.ndarray


class PufferBackend:
    """Roll generated scenes through PufferDrive's C simulator."""

    def __init__(
        self,
        *,
        planner,
        device: str,
        deterministic: bool,
        sim_steps: int,
        seed: int = 0,
        pufferdrive_root: str | Path | None = None,
        offroad_mode: str = "road_edge",
        control_mode: str = "control_agents",
    ):
        self.binding = _load_binding(pufferdrive_root)
        self.planner = planner
        self.device = device
        self.deterministic = bool(deterministic)
        self.sim_steps = int(sim_steps)
        self.seed = int(seed)
        self.sim_cfg = load_sim_config()
        self.offroad_mode = OFFROAD_MODE[offroad_mode]
        self.control_mode = CONTROL_MODE[control_mode]

    def _scene_kwargs(self, s: int, states, types, lanes):
        return dict(
            agent_states=states,
            agent_types=types,
            lane_polylines=lanes,
            map_id=int(s),
            action_type=0,          # discrete
            dynamics_model=0,       # classic
            condition_sample_mode=1,  # fixed
            fixed_collision_factor=float(self.sim_cfg.fixed_collision_factor),
            fixed_offroad_factor=float(self.sim_cfg.fixed_offroad_factor),
            fixed_lane_width=float(self.sim_cfg.fixed_lane_width),
            reward_goal=1.0,
            reward_goal_post_respawn=0.25,
            reward_steer_jitter=0.0,
            reward_time_penalty=0.0,
            goal_radius=float(self.sim_cfg.goal_radius),
            # Current numpy reward treats goal completion as distance-only.
            goal_speed=100.0,
            goal_behavior=GOAL_BEHAVIOR.get(self.sim_cfg.goal_behavior, 3),
            goal_target_distance=30.0,
            collision_behavior=0,
            offroad_behavior=0,
            offroad_mode=self.offroad_mode,
            centerline_only=1,
            lane_width=float(self.sim_cfg.fixed_lane_width),
            dt=float(self.sim_cfg.dt),
            episode_length=int(self.sim_steps),
            termination_mode=1,
            init_steps=0,
            init_mode=0,
            control_mode=self.control_mode,
            max_controlled_agents=int(self.sim_cfg.max_controlled_agents),
            max_agents=int(states.shape[0]),
            render_mode=1,
        )

    def _build_native_scenes(self, scenes: GeneratedScenes) -> tuple[list[_NativeScene], int]:
        states = _as_numpy(scenes.agent_states, np.float32)
        types = _as_numpy(scenes.agent_types, np.int32)
        agent_scene_idx = _as_numpy(scenes.agent_scene_idx, np.int64)
        lanes = _as_numpy(scenes.lane_polylines, np.float32)
        lane_scene_idx = _as_numpy(scenes.meta["lane_scene_idx"], np.int64)

        native: list[_NativeScene] = []
        total_active = 0
        for s in range(scenes.num_scenes):
            s_states = np.ascontiguousarray(states[agent_scene_idx == s], dtype=np.float32)
            s_types = np.ascontiguousarray(types[agent_scene_idx == s], dtype=np.int32)
            s_lanes = np.ascontiguousarray(lanes[lane_scene_idx == s], dtype=np.float32)
            capacity = max(1, int(s_states.shape[0]))
            obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
            actions = np.zeros(capacity, dtype=np.int32)
            rewards = np.zeros(capacity, dtype=np.float32)
            terminals = np.zeros(capacity, dtype=np.uint8)
            truncations = np.zeros(capacity, dtype=np.uint8)
            handle, active_count = self.binding.scene_init_env_init(
                obs,
                actions,
                rewards,
                terminals,
                truncations,
                self.seed + s,
                **self._scene_kwargs(s, s_states, s_types, s_lanes),
            )
            active_count = int(active_count)
            total_active += active_count
            native.append(
                _NativeScene(
                    handle=int(handle),
                    active_count=active_count,
                    obs=obs,
                    actions=actions,
                    rewards=rewards,
                    terminals=terminals,
                    truncations=truncations,
                )
            )
        return native, total_active

    def _metrics(self, vec, num_scenes: int) -> dict[str, np.ndarray]:
        out = {
            "ego_collision": np.zeros(num_scenes, dtype=np.float32),
            "ego_offroad": np.zeros(num_scenes, dtype=np.float32),
            "reached_goal": np.zeros(num_scenes, dtype=np.float32),
            "ego_min_ttc": np.zeros(num_scenes, dtype=np.float32),
            "init_invalid": np.zeros(num_scenes, dtype=np.float32),
        }
        self.binding.scene_init_vec_metrics(
            vec,
            out["ego_collision"],
            out["ego_offroad"],
            out["reached_goal"],
            out["ego_min_ttc"],
            out["init_invalid"],
        )
        return out

    def _snapshot(self, vec, native: list[_NativeScene], trajectories, finished):
        total = sum(ns.active_count for ns in native)
        x = np.zeros(total, dtype=np.float32)
        y = np.zeros(total, dtype=np.float32)
        z = np.zeros(total, dtype=np.float32)
        heading = np.zeros(total, dtype=np.float32)
        ids = np.zeros(total, dtype=np.int32)
        length = np.zeros(total, dtype=np.float32)
        width = np.zeros(total, dtype=np.float32)
        respawn = np.zeros(total, dtype=np.int32)
        self.binding.vec_get_global_agent_state(vec, x, y, z, heading, ids, length, width, respawn)
        # goal_behavior="remove" deletes goal-reached agents by teleporting them to
        # INVALID_POSITION and leaves respawn=0 (that flag is RESPAWN-only). Mirror
        # the numpy backend's SimScene._remove_agent (respawned=True, no sentinel
        # pose): flag them inactive so the viz masks them, and NaN their pose so the
        # -10000 sentinel does not blow up the plot window in viz._view.
        respawn = respawn.astype(bool)
        gone = ~(np.isfinite(x) & np.isfinite(y)) | (np.abs(x) > _GONE_DISTANCE) | (np.abs(y) > _GONE_DISTANCE)
        if gone.any():
            respawn = respawn | gone
            x = np.where(gone, np.nan, x)
            y = np.where(gone, np.nan, y)
        off = 0
        for s, ns in enumerate(native):
            n = ns.active_count
            if not finished[s]:
                tr = trajectories[s]
                tr["x"].append(x[off : off + n].copy())
                tr["y"].append(y[off : off + n].copy())
                tr["heading"].append(heading[off : off + n].copy())
                tr["respawn"].append(respawn[off : off + n].astype(bool).copy())
                if tr["length"] is None:
                    tr["length"] = length[off : off + n].copy()
                    tr["width"] = width[off : off + n].copy()
            off += n

    @staticmethod
    def _pack_obs(native: list[_NativeScene], active: list[int], obs_batch: np.ndarray) -> int:
        off = 0
        for s in active:
            n = native[s].active_count
            obs_batch[off : off + n] = native[s].obs[:n]
            off += n
        return off

    @torch.no_grad()
    def rollout(
        self,
        scenes: GeneratedScenes,
        *,
        record_trajectories: bool = False,
        profile: bool = False,
    ) -> dict:
        t_total = time.perf_counter()
        prof = {
            "build_env": 0.0,
            "metrics": 0.0,
            "trajectory": 0.0,
            "pack_obs": 0.0,
            "planner": 0.0,
            "scatter_actions": 0.0,
            "vec_step": 0.0,
            "close": 0.0,
            "steps": 0,
            "planner_rows": 0,
        }
        t0 = time.perf_counter()
        native, _ = self._build_native_scenes(scenes)
        prof["build_env"] += time.perf_counter() - t0
        vec = self.binding.vectorize(*[ns.handle for ns in native])
        num_scenes = len(native)
        total_active_capacity = sum(ns.active_count for ns in native)
        obs_batch = np.empty((max(1, total_active_capacity), OBS_DIM), dtype=np.float32)
        finished = np.zeros(num_scenes, dtype=bool)
        metrics = {
            "ego_collision": np.zeros(num_scenes, dtype=np.float32),
            "ego_offroad": np.zeros(num_scenes, dtype=np.float32),
            "init_invalid": np.zeros(num_scenes, dtype=np.float32),
            "reached_goal": np.zeros(num_scenes, dtype=np.float32),
            "ego_min_ttc": np.full(num_scenes, np.inf, dtype=np.float32),
        }
        trajectories = (
            [
                {"x": [], "y": [], "heading": [], "respawn": [], "done": [], "length": None, "width": None}
                for _ in range(num_scenes)
            ]
            if record_trajectories
            else None
        )

        try:
            for t in range(self.sim_steps):
                prof["steps"] += 1
                t0 = time.perf_counter()
                cur = self._metrics(vec, num_scenes)
                prof["metrics"] += time.perf_counter() - t0
                if t == 0:
                    metrics["init_invalid"] = np.maximum(metrics["init_invalid"], cur["init_invalid"])
                active_mask = ~finished
                metrics["ego_collision"][active_mask] = np.maximum(
                    metrics["ego_collision"][active_mask], cur["ego_collision"][active_mask]
                )
                metrics["ego_offroad"][active_mask] = np.maximum(
                    metrics["ego_offroad"][active_mask], cur["ego_offroad"][active_mask]
                )
                metrics["ego_min_ttc"][active_mask] = np.minimum(
                    metrics["ego_min_ttc"][active_mask], cur["ego_min_ttc"][active_mask]
                )

                if trajectories is not None:
                    t0 = time.perf_counter()
                    self._snapshot(vec, native, trajectories, finished)
                    prof["trajectory"] += time.perf_counter() - t0

                pre_reached = active_mask & (cur["reached_goal"] > 0)
                if pre_reached.any():
                    metrics["reached_goal"][pre_reached] = 1.0
                    finished[pre_reached] = True
                    if trajectories is not None:
                        for s in np.nonzero(pre_reached)[0]:
                            trajectories[s]["done"].append(True)

                active = [s for s in range(num_scenes) if not finished[s]]
                if not active:
                    break

                t0 = time.perf_counter()
                n_obs = self._pack_obs(native, active, obs_batch)
                prof["pack_obs"] += time.perf_counter() - t0
                prof["planner_rows"] += int(n_obs)

                t0 = time.perf_counter()
                obs = torch.as_tensor(obs_batch[:n_obs], device=self.device)
                actions = self.planner.act(obs, deterministic=self.deterministic).cpu().numpy().astype(np.int32)
                prof["planner"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                off = 0
                for s in active:
                    n = native[s].active_count
                    native[s].actions[:n] = actions[off : off + n]
                    off += n
                prof["scatter_actions"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                self.binding.vec_step(vec)
                prof["vec_step"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                post = self._metrics(vec, num_scenes)
                prof["metrics"] += time.perf_counter() - t0
                post_reached = np.zeros(num_scenes, dtype=bool)
                post_reached[active] = post["reached_goal"][active] > 0
                if post_reached.any():
                    metrics["reached_goal"][post_reached] = 1.0
                    finished[post_reached] = True
                if trajectories is not None:
                    for s in active:
                        trajectories[s]["done"].append(bool(post_reached[s]))
        finally:
            t0 = time.perf_counter()
            self.binding.vec_close(vec)
            prof["close"] += time.perf_counter() - t0

        if trajectories is not None:
            for tr in trajectories:
                tr["x"] = np.asarray(tr["x"], dtype=np.float32) if tr["x"] else np.zeros((0, 0), np.float32)
                tr["y"] = np.asarray(tr["y"], dtype=np.float32) if tr["y"] else np.zeros((0, 0), np.float32)
                tr["heading"] = (
                    np.asarray(tr["heading"], dtype=np.float32) if tr["heading"] else np.zeros((0, 0), np.float32)
                )
                tr["respawn"] = np.asarray(tr["respawn"], dtype=bool) if tr["respawn"] else np.zeros((0, 0), bool)
                tr["done"] = np.asarray(tr["done"], dtype=bool)
                if tr["length"] is None:
                    tr["length"] = np.zeros(0, dtype=np.float32)
                    tr["width"] = np.zeros(0, dtype=np.float32)
            metrics["trajectories"] = trajectories

        if profile:
            prof["total"] = time.perf_counter() - t_total
            metrics["_profile"] = prof

        return metrics
