#!/usr/bin/env python
"""Does the trained model actually drive like logged traffic?

Everything else in this package measures machinery -- cost, sharding,
label quality, training loss. This measures behavior: it drives EVERY agent of a
logged scene with the model for 8 s and compares the result against what those
agents really did.

Three numbers, and the third is the one the paper is missing:

  * **ADE / FDE against the log** -- displacement error, in metres. A traffic
    model that imitates well stays close; one that has collapsed to "drive
    straight" does not.
  * **collision and off-road rates** -- driving close to the log is worthless if
    the model achieves it by driving through other cars.
  * **cold start vs primed history** -- the same rollout with an EMPTY history
    and with the 1 s of real past handed over. The gap is the direct test of
    whether the random history masking in training did its job, because a
    generated scene only ever offers the cold case.

Rollouts use ``SimScene`` and the shared ``step_dynamics``, so this is the same
integrator the benchmark runs, not a private replay.

Usage:
    python smart/evaluate.py --weights checkpoints/planners/smart/v1.pt --scenes 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sim.planners import build_planner, parse_conditioning
from sim.scenes import lane_graph_edges
from sim.world import SimConfig, SimScene
from utils.goal_runtime import point_to_polyline_dist

from smart.observation import POSE_SLOTS
from smart.records import load_scene, scene_paths

DT = 0.1
SCENE_T0 = 10          # the record's scene_timestep: where agent_states is taken
LOG_STEPS = 90         # last trajectory index
OFFROAD_M = 1.5        # same threshold the dataset filter uses


def sim_config(goal_behavior: str) -> SimConfig:
    """The shared rollout dynamics (cfgs/rollout/base.yaml), with the goal
    lifecycle overridable: behavior realism wants agents to keep driving rather
    than retire at a goal this model cannot even see."""
    return SimConfig(dt=DT, goal_radius=2.0, goal_speed=100.0,
                     goal_behavior=goal_behavior, map_extent=72.0,
                     max_controlled_agents=64, ego_crash_freeze=False)


def build_sim(scene: dict, cfg: SimConfig) -> tuple[SimScene, np.ndarray]:
    """A SimScene started from the logged state at ``SCENE_T0``."""
    state, valid = scene["state"], scene["valid"]
    live = np.flatnonzero(valid[:, SCENE_T0])
    s = state[live, SCENE_T0]
    last = np.array([np.flatnonzero(valid[a])[-1] for a in live])
    agent_states = np.zeros((len(live), 9), dtype=np.float32)
    agent_states[:, 0], agent_states[:, 1] = s[:, 0], s[:, 1]
    agent_states[:, 2] = s[:, 3]
    agent_states[:, 3], agent_states[:, 4] = np.cos(s[:, 2]), np.sin(s[:, 2])
    agent_states[:, 5] = scene["length"][live]
    agent_states[:, 6] = scene["width"][live]
    agent_states[:, 7:9] = state[live, last, :2]        # goal: last logged position
    ptypes = np.argmax(scene["types"][live], axis=1) + 1
    sim = SimScene(agent_states, ptypes, scene["lanes"], sim_cfg=cfg)
    # Route planners read connectivity off the scene; the log records store it.
    sim.lane_graph = lane_graph_edges(scene["lane_edges"], scene["lane_conn"])
    return sim, live


def rollout(scenes: list[dict], planner, cfg: SimConfig, prime: bool,
            ablate_history: bool = False, conditioning: dict | None = None) -> list[dict]:
    """Drive every agent of every scene for the logged horizon.

    Scenes are stepped TOGETHER, in one ``plan`` call per step, because that is
    what the planner is built for: batching across scenes is the whole reason its
    gather returns a flat matrix. One scene at a time makes this a few thousand
    tiny forwards and is unusably slow.
    """
    sims, lives = [], []
    for scene in scenes:
        sim, live = build_sim(scene, cfg)
        if prime and hasattr(planner, "prime_history"):
            # Stop BEFORE the current step: gather() opens with _advance(), which
            # appends the current pose itself. Including it here writes it twice
            # and fabricates a zero-motion step right before the first prediction
            # -- i.e. tells a moving car it just stopped.
            lo = SCENE_T0 - (POSE_SLOTS - 1)
            planner.prime_history(
                sim, scene["state"][live, lo:SCENE_T0, :3].astype(np.float32)
            )
        if conditioning:
            for field, value in conditioning.items():
                getattr(sim, field)[:] = np.float32(
                    value if not isinstance(value, tuple) else 0.5 * (value[0] + value[1])
                )
        sims.append(sim)
        lives.append(live)

    items = [(sim, np.arange(sim.n)) for sim in sims]
    steps = LOG_STEPS - SCENE_T0
    tracks = [np.empty((steps, sim.n, 3), dtype=np.float64) for sim in sims]
    for k in range(steps):
        if ablate_history:
            for sim in sims:
                planner._poses_for(sim)["filled"] = 0
        planner.apply(items, planner.plan(items))
        for j, sim in enumerate(sims):
            sim.update_metrics()
            tracks[j][k, :, 0], tracks[j][k, :, 1] = sim.x, sim.y
            tracks[j][k, :, 2] = sim.heading

    out = []
    for scene, sim, live, track in zip(scenes, sims, lives, tracks):
        ref = scene["state"][live, SCENE_T0 + 1 : LOG_STEPS + 1, :2].transpose(1, 0, 2)
        ok = scene["valid"][live, SCENE_T0 + 1 : LOG_STEPS + 1].T
        # Off-road is scored at each agent's LAST LOGGED step, not at the last
        # rollout step. The lane graph only covers the 64 x 64 m FOV, and 36% of
        # moving agents leave it inside 8 s (96 m of travel at 12 m/s against a
        # 72 m box), so scoring the final rollout position measures "drove out of
        # the crop" for a third of the population -- and the log side cannot even
        # be scored there, because its track has ended. That asymmetry pinned the
        # metric near 70% no matter what the model did.
        last = np.where(ok.any(axis=0), ok.shape[0] - 1 - np.argmax(ok[::-1], axis=0), -1)
        rows = np.arange(len(live))
        final = np.where((last >= 0)[:, None],
                         track[np.maximum(last, 0), rows, :2], np.nan)
        lane_d = point_to_polyline_dist(np.nan_to_num(final).astype(np.float32),
                                        scene["lanes"])
        scored = last >= 0
        offroad = np.where(scored, lane_d > OFFROAD_M, np.nan)
        out.append({
            # the raw rolled-out track and which record rows it maps to, so
            # smart.viz can draw the rollout without reimplementing it
            "track": track,
            "live": live,
            "dist": np.linalg.norm(track[:, :, :2] - ref, axis=-1),
            "ok": ok,
            "collided": np.where(scored, sim.collision_state > 0, np.nan),
            "offroad": offroad,
            "moving": scene["moving"][live],
        })
    return out


def log_reference(scenes: list[dict], cfg: SimConfig) -> list[dict]:
    """The LOGGED trajectories pushed through the same collision / off-road check.

    Without this the absolute rates mean nothing. "52% off-road" sounds alarming
    until you learn what fraction of REAL agents this proxy also calls off-road:
    it measures distance to a lane CENTRELINE, and pedestrians, cyclists and
    parked cars are legitimately far from one.
    """
    out = []
    for scene in scenes:
        sim, live = build_sim(scene, cfg)
        state, valid = scene["state"], scene["valid"]
        for k in range(SCENE_T0 + 1, LOG_STEPS + 1):
            ok = valid[live, k]
            # A track that has ended is stored at a sentinel coordinate (-1e4),
            # and EVERY ended track shares it. Writing those positions puts all of
            # them on one point, where they register mutual collisions -- which is
            # a measurement artifact, not anything the log did. Retire them
            # instead, exactly as the rollout retires an agent that leaves.
            for i in np.flatnonzero(~ok & ~sim.removed):
                sim._remove_agent(int(i))
            sim.x[ok], sim.y[ok] = state[live, k, 0][ok], state[live, k, 1][ok]
            sim.heading[ok] = state[live, k, 2][ok]
            sim.heading_x, sim.heading_y = np.cos(sim.heading), np.sin(sim.heading)
            sim.update_metrics()
        lane_d = point_to_polyline_dist(
            np.stack([sim.x, sim.y], axis=1).astype(np.float32), scene["lanes"]
        )
        scored = valid[live, SCENE_T0 + 1 : LOG_STEPS + 1].any(axis=1)
        out.append({
            "dist": np.zeros((1, sim.n)), "ok": np.ones((1, sim.n), dtype=bool),
            "collided": np.where(scored, sim.collision_state > 0, np.nan),
            "offroad": np.where(scored, lane_d > OFFROAD_M, np.nan),
            "moving": scene["moving"][live],
        })
    return out


def summarize(runs: list[dict], label: str) -> None:
    """Report moving and parked agents SEPARATELY.

    About 30% of agents never move 2 m. Their displacement error is trivially
    near zero, so pooling them with the rest drags every ADE/FDE down and hides
    how the model actually drives. They are still worth reporting: a traffic
    model has to keep a parked car parked, and at rollout time it drives those
    agents too.
    """
    ade, fde, mv = [], [], []
    for r in runs:
        for a in np.flatnonzero(r["ok"].any(axis=0)):
            d, ok = r["dist"][:, a], r["ok"][:, a]
            ade.append(d[ok].mean())
            fde.append(d[ok][-1])
            mv.append(r["moving"][a])
    ade, fde, mv = np.asarray(ade), np.asarray(fde), np.asarray(mv, dtype=bool)
    coll = np.concatenate([r["collided"] for r in runs])
    off = np.concatenate([r["offroad"] for r in runs])
    amv = np.concatenate([r["moving"] for r in runs]).astype(bool)
    for tag, sel_a, sel_b in (("moving", mv, amv), ("parked", ~mv, ~amv)):
        if not sel_a.any():
            continue
        print(f"  {label:<22} {tag:<7} n={int(sel_a.sum()):>5}  "
              f"ADE {ade[sel_a].mean():6.3f} (p50 {np.percentile(ade[sel_a], 50):6.3f})  "
              f"FDE {fde[sel_a].mean():6.3f} (p50 {np.percentile(fde[sel_a], 50):6.3f})  "
              f"coll {100 * np.nanmean(coll[sel_b]):5.2f}%  "
              f"offroad {100 * np.nanmean(off[sel_b]):5.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default=None, help="'random' or a checkpoint path (smart only)")
    ap.add_argument("--planner", default="smart_probe",
                    help="any name in PLANNER_REGISTRY, so this model can be scored "
                         "against ppo_normal / idm on the SAME metric")
    ap.add_argument("--split", default="val")
    ap.add_argument("--scenes", type=int, default=300)
    ap.add_argument("--goal-behavior", default="remove_off_map",
                    choices=["stop", "remove", "remove_off_map", "continue"])
    ap.add_argument("--batch", type=int, default=32, help="scenes stepped together")
    ap.add_argument("--sample", action="store_true",
                    help="sample the action instead of taking the argmax. Behaviour "
                         "cloning + argmax collapses onto the majority action: measured "
                         "78.9%% of rollout decisions are zero-accel zero-steer, i.e. "
                         "constant-velocity straight line, while the policy entropy is "
                         "1.8 of a possible 4.5 nats -- the distribution is fine, the "
                         "argmax is what throws it away.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cpu",
                    help="the joint planner's per-scene attention is far too heavy for "
                         "cpu; give it cuda")
    ap.add_argument("--ablate-history", action="store_true",
                    help="diagnostic: hold the history EMPTY for the whole rollout. If this "
                         "changes nothing, the history block is not being used and the "
                         "cold-vs-primed gap has nothing to do with history quality.")
    args = ap.parse_args()

    # ${project_root} is resolved by the main hydra tree, which this package does
    # not compose. Substitute it on the RAW string: touching cfg.checkpoint first
    # would trigger the interpolation and raise.
    raw = OmegaConf.to_container(OmegaConf.load(f"cfgs/planner/{args.planner}.yaml"),
                                 resolve=False)
    raw = {k: (v.replace("${project_root}", str(REPO)) if isinstance(v, str) else v)
           for k, v in raw.items()}
    cfg = OmegaConf.create(raw)
    cfg.device = args.device
    if "weights" in cfg:
        cfg.weights = (args.weights if args.weights == "random"
                       else str(Path(args.weights).resolve()))
    if args.sample and "deterministic" in cfg:
        cfg.deterministic = False
        cfg.temperature = args.temperature
    planner = build_planner(cfg, role="env", device=args.device)
    # A conditioned policy reads its driving style out of each agent's own obs
    # slots, which RolloutRunner normally fills. Without this the ppo family runs
    # at collision_factor 0, i.e. its most reckless setting, and the comparison
    # would be against a planner nobody uses.
    conditioning = parse_conditioning("env", cfg.get("conditioning"))
    sim_cfg = sim_config(args.goal_behavior)

    paths = scene_paths(args.split)
    paths = paths[:: max(1, len(paths) // args.scenes)][: args.scenes]

    print(f"planner={args.planner}  weights={args.weights}  "
          f"{'sample T=%.2f' % args.temperature if args.sample else 'argmax'}  "
          f"scenes={len(paths)}  "
          f"horizon={(LOG_STEPS - SCENE_T0) * DT:.1f} s  goal_behavior={args.goal_behavior}")
    loaded = [load_scene(p) for p in paths]
    ref = []
    for lo in range(0, len(loaded), args.batch):
        ref += log_reference(loaded[lo : lo + args.batch], sim_cfg)
    summarize(ref, "LOG (reference)")
    for prime in (False, True):
        runs = []
        for lo in range(0, len(loaded), args.batch):
            runs += rollout(loaded[lo : lo + args.batch], planner, sim_cfg, prime,
                            args.ablate_history, conditioning)
        tag = "primed history" if prime else "cold start"
        summarize(runs, tag + (" [no hist]" if args.ablate_history else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
