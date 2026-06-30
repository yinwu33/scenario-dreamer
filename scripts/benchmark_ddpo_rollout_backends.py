#!/usr/bin/env python3
"""Benchmark DDPO rollout backends on the same synthetic GeneratedScenes batch.

This isolates reward rollout speed from LDM/AE sampling so it can run without
scene-generator checkpoints:

    python scripts/benchmark_ddpo_rollout_backends.py --batch-size 16 --iters 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpo.geometry import _corners, _sat_overlap
from ddpo.interfaces import GeneratedScenes
from ddpo.native_pufferdrive import PufferBackend
from ddpo.planners.type_utils import to_puffer_agent_types
from ddpo.pufferdrive_sim import COLLISION_DIST2_GATE, TYPE_PEDESTRIAN, SimScene
from ddpo.reward import PufferDriveReward

TTC_SWEEP_HORIZON = 10.0


def make_scenes(
    *,
    num_scenes: int,
    num_agents: int,
    lanes_per_scene: int,
    lane_points: int,
    device: str,
    seed: int,
) -> GeneratedScenes:
    rng = np.random.default_rng(seed)
    all_agents = []
    all_types = []
    agent_scene_idx = []
    all_lanes = []
    lane_scene_idx = []

    xs = np.linspace(-40.0, 80.0, lane_points, dtype=np.float32)
    for s in range(num_scenes):
        lanes = []
        for lane_id in range(lanes_per_scene):
            y = (lane_id - lanes_per_scene // 2) * 3.5
            lanes.append(np.stack([xs, np.full_like(xs, y)], axis=1))
        lanes = np.asarray(lanes, dtype=np.float32)
        all_lanes.append(lanes)
        lane_scene_idx.extend([s] * lanes_per_scene)

        states = np.zeros((num_agents, 9), dtype=np.float32)
        states[0] = [0.0, 0.0, 8.0, 1.0, 0.0, 4.8, 2.0, 45.0, 0.0]
        for a in range(1, num_agents):
            lane_y = float(rng.choice(lanes[:, 0, 1]))
            x = float(rng.uniform(8.0, 55.0))
            speed = float(rng.uniform(3.0, 11.0))
            states[a] = [x, lane_y, speed, 1.0, 0.0, 4.8, 2.0, x + rng.uniform(20.0, 45.0), lane_y]
        all_agents.append(states)
        all_types.append(np.zeros(num_agents, dtype=np.int64))
        agent_scene_idx.extend([s] * num_agents)

    agents = np.concatenate(all_agents, axis=0)
    types = np.concatenate(all_types, axis=0)
    lanes = np.concatenate(all_lanes, axis=0)
    return GeneratedScenes(
        agent_states=torch.as_tensor(agents, device=device),
        agent_types=torch.as_tensor(types, device=device),
        agent_scene_idx=torch.as_tensor(agent_scene_idx, dtype=torch.long, device=device),
        lane_polylines=torch.as_tensor(lanes, device=device),
        num_scenes=num_scenes,
        meta={"lane_scene_idx": torch.as_tensor(lane_scene_idx, dtype=torch.long, device=device)},
    )


def time_backend(name: str, reward: PufferDriveReward, scenes: GeneratedScenes, warmup: int, iters: int):
    for _ in range(warmup):
        reward.evaluate(scenes)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    last = None
    for _ in range(iters):
        last = reward.evaluate(scenes)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "backend": name,
        "seconds_total": dt,
        "seconds_per_eval": dt / max(iters, 1),
        "reward_mean": float(last["reward"].mean()) if last is not None else float("nan"),
        "collision_rate": float(last["ego_collision"].mean()) if last is not None else float("nan"),
    }


def _to_numpy(value, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def build_python_sims(scenes: GeneratedScenes, seed: int) -> list[SimScene]:
    states = _to_numpy(scenes.agent_states, np.float32)
    types = _to_numpy(scenes.agent_types, np.int64)
    ptypes = to_puffer_agent_types(types)
    agent_scene_idx = _to_numpy(scenes.agent_scene_idx, np.int64)
    lanes = _to_numpy(scenes.lane_polylines, np.float32)
    lane_scene_idx = _to_numpy(scenes.meta["lane_scene_idx"], np.int64)

    rng = np.random.default_rng(seed)
    sims = []
    for s in range(scenes.num_scenes):
        sims.append(
            SimScene(
                states[agent_scene_idx == s],
                ptypes[agent_scene_idx == s],
                lanes[lane_scene_idx == s],
                rng=rng,
            )
        )
    return sims


def any_vehicle_overlap(sim: SimScene, margin: float = 0.0) -> bool:
    """Benchmark-only initial overlap statistic for active non-pedestrian agents."""
    active = sim.slot_order[sim.ptype[sim.slot_order] != TYPE_PEDESTRIAN]
    if len(active) < 2:
        return False
    boxes = _corners(
        sim.x[active],
        sim.y[active],
        sim.heading[active],
        sim.length[active] + 2.0 * margin,
        sim.width[active] + 2.0 * margin,
    )
    for k in range(len(active) - 1):
        dx = sim.x[active[k + 1:]] - sim.x[active[k]]
        dy = sim.y[active[k + 1:]] - sim.y[active[k]]
        gate = (dx * dx + dy * dy) <= COLLISION_DIST2_GATE
        if gate.any() and _sat_overlap(boxes[k], boxes[k + 1:][gate]).any():
            return True
    return False


def ego_collides_now(sim: SimScene, others=None) -> bool:
    """Benchmark-only current ego-fault overlap statistic."""
    if sim.n <= 1:
        return False
    if others is None:
        others = sim.slot_order[sim.slot_order != 0]
    else:
        others = np.asarray(others, dtype=np.int64)
        others = others[(others != 0) & (~sim.removed[others])]
    if not len(others):
        return False
    boxes = _corners(sim.x, sim.y, sim.heading, sim.length, sim.width)
    overlap = _sat_overlap(boxes[0], boxes[others])
    if not overlap.any():
        return False
    return bool((overlap & sim._ego_aggressor_mask(others)).any())


def ego_min_ttc_now(sim: SimScene, others=None) -> float:
    """Benchmark-only relative-velocity TTC sweep."""
    if sim.n <= 1 or sim.crashed[0]:
        return float(np.inf)
    if others is None:
        others = sim.slot_order[sim.slot_order != 0]
    else:
        others = np.asarray(others, dtype=np.int64)
        others = others[(others != 0) & (~sim.removed[others])]
    others = others[(sim.ptype[others] != TYPE_PEDESTRIAN) & (~sim.crashed[others])]
    if not len(others):
        return float(np.inf)
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
    rv = np.stack([rvx, rvy], axis=1).astype(np.float64)
    steps = int(np.ceil(TTC_SWEEP_HORIZON / max(sim.dt, 1e-3)))
    for step in range(0, steps + 1):
        t = step * sim.dt
        moved = base_boxes + (rv * t)[:, None, :]
        if _sat_overlap(ego_box, moved).any():
            return float(t)
    return float(np.inf)


def python_sim_eval(scenes: GeneratedScenes, sim_steps: int, seed: int):
    sims = build_python_sims(scenes, seed)
    collisions = 0
    for sim in sims:
        collisions += int(any_vehicle_overlap(sim, 0.0))
    for _ in range(sim_steps):
        for sim in sims:
            obs = sim.compute_obs()
            actions = np.zeros(obs.shape[0], dtype=np.int32)
            sim.step_dynamics(actions)
            sim.update_metrics()
            collisions += int(ego_collides_now(sim))
            ego_min_ttc_now(sim)
            sim.goal_step()
    return collisions


def native_sim_eval(
    backend: PufferBackend,
    scenes: GeneratedScenes,
    sim_steps: int,
):
    native, _ = backend._build_native_scenes(scenes)
    vec = backend.binding.vectorize(*[ns.handle for ns in native])
    num_scenes = len(native)
    collision = np.zeros(num_scenes, dtype=np.float32)
    offroad = np.zeros(num_scenes, dtype=np.float32)
    reached = np.zeros(num_scenes, dtype=np.float32)
    min_ttc = np.zeros(num_scenes, dtype=np.float32)
    init_invalid = np.zeros(num_scenes, dtype=np.float32)
    try:
        for _ in range(sim_steps):
            backend.binding.scene_init_vec_metrics(vec, collision, offroad, reached, min_ttc, init_invalid)
            backend.binding.vec_step(vec)
    finally:
        backend.binding.vec_close(vec)
    return float(collision.sum() + init_invalid.sum())


def time_sim_backend(name: str, fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    last = None
    for _ in range(iters):
        last = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "backend": name,
        "seconds_total": dt,
        "seconds_per_eval": dt / max(iters, 1),
        "evals_per_second": max(iters, 1) / dt if dt > 0 else float("inf"),
        "last": last,
    }


def print_rows(title: str, rows: list[dict], metric_cols: tuple[str, ...] = ()):
    base = rows[0]["seconds_per_eval"]
    print(f"\n{title}")
    print("backend              sec/eval    eval/s    speedup" + "".join(f"  {c:>8}" for c in metric_cols))
    for row in rows:
        sec = row["seconds_per_eval"]
        speedup = base / sec if sec > 0 else float("inf")
        extra = "".join(f"  {row[c]:8.3f}" for c in metric_cols)
        print(f"{row['backend']:<20} {sec:8.4f}  {1.0/sec:8.2f}  {speedup:8.2f}{extra}")


def print_profile(profile: dict):
    total = max(float(profile.get("total", 0.0)), 1e-12)
    print("\nNative rollout profile")
    print("part                  seconds   percent")
    for key in (
        "build_env",
        "metrics",
        "pack_obs",
        "planner",
        "scatter_actions",
        "vec_step",
        "trajectory",
        "close",
    ):
        sec = float(profile.get(key, 0.0))
        print(f"{key:<20} {sec:8.4f}  {100.0 * sec / total:7.2f}")
    print(f"{'total':<20} {total:8.4f}  {100.0:7.2f}")
    print(f"steps={int(profile.get('steps', 0))} planner_rows={int(profile.get('planner_rows', 0))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-agents", type=int, default=30)
    ap.add_argument("--lanes-per-scene", type=int, default=8)
    ap.add_argument("--lane-points", type=int, default=20)
    ap.add_argument("--sim-steps", type=int, default=91)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pufferdrive-root", default=str(Path(__file__).resolve().parents[1] / "PufferDrive"))
    ap.add_argument("--mode", choices=("end_to_end", "sim_only", "both"), default="both")
    ap.add_argument("--profile-native", action="store_true")
    args = ap.parse_args()

    scenes = make_scenes(
        num_scenes=args.batch_size,
        num_agents=args.num_agents,
        lanes_per_scene=args.lanes_per_scene,
        lane_points=args.lane_points,
        device=args.device,
        seed=args.seed,
    )

    print(
        f"Benchmark: batch={args.batch_size}, agents/scene={args.num_agents}, "
        f"sim_steps={args.sim_steps}, device={args.device}, mode={args.mode}"
    )

    if args.mode in ("end_to_end", "both"):
        common = dict(
            sim_steps=args.sim_steps,
            deterministic=True,
            ttc_tau=3.0,
            init_overlap_margin=0.0,
            goal_offlane_threshold=1.5,
            goal_onroad_threshold=2.0,
            goal_offlane_penalty=0.0,
            parking_mismatch_penalty=0.0,
            seed=args.seed,
        )
        rewards = [
            ("numpy", PufferDriveReward(backend="numpy", **common)),
            (
                "puffer",
                PufferDriveReward(
                    backend="puffer",
                    pufferdrive_root=args.pufferdrive_root,
                    **common,
                ),
            ),
        ]
        rows = []
        for name, reward in rewards:
            rows.append(time_backend(name, reward, scenes, args.warmup, args.iters))
        print_rows("End-to-end reward rollout", rows, ("reward_mean", "collision_rate"))
        if args.profile_native:
            native_reward = rewards[1][1]
            prof_metrics = native_reward.native_backend.rollout(scenes, profile=True)
            print_profile(prof_metrics["_profile"])

    if args.mode in ("sim_only", "both"):
        native_backend = PufferBackend(
            planner=None,
            device=args.device,
            deterministic=True,
            sim_steps=args.sim_steps,
            seed=args.seed,
            pufferdrive_root=args.pufferdrive_root,
        )
        rows = [
            time_sim_backend(
                "numpy",
                lambda: python_sim_eval(scenes, args.sim_steps, args.seed),
                args.warmup,
                args.iters,
            ),
            time_sim_backend(
                "puffer",
                lambda: native_sim_eval(native_backend, scenes, args.sim_steps),
                args.warmup,
                args.iters,
            ),
        ]
        print_rows("Simulator only, fixed zero actions", rows)


if __name__ == "__main__":
    main()
