"""DDPO reward: roll generated scenes out with the frozen planner, score the ego.

Same reward semantics as PufferDrive/scene_init_ddpo/reward.py, but the rollout
runs on the in-repo numpy port of the simulator (``pufferdrive_sim``) - no C env,
no .bin files, no second venv:

  * only the ego (scene agent 0) is scored;
  * ``init_invalid`` flags scenes whose ego already overlaps another agent at
    t=0 (reward hacking by spawning a doomed ego) -> strong negative reward;
  * ``ego_collision`` is the planner-rollout collision flag (+1 reward);
  * ``ego_offroad`` is kept for interface compatibility but is always 0: the
    generated maps carry no road edges (see pufferdrive_sim docstring);
  * a scene stops being stepped/scored once its ego reaches its goal.
"""

from __future__ import annotations

import numpy as np
import torch

from .interfaces import GeneratedScenes
from planner.selfplay_drive.planner import load_planner, load_planner_config
from .pufferdrive_sim import MIN_DISTANCE_TO_GOAL, SimScene, load_sim_config


def default_reward_fn(ego_collision, ego_offroad_, init_invalid):
    """Critical-scene reward: +1 if the planner collides, -1 for degenerate scenes."""
    r = np.where(ego_collision > 0, 1.0, 0.0)
    r = np.where(init_invalid > 0, -1.0, r)
    return r.astype(np.float32)


class PufferDriveReward:
    def __init__(
        self,
        *,
        sim_steps: int = 91,
        deterministic: bool | None = None,
        reward_fn=default_reward_fn,
        goal_offlane_threshold: float = 3.0,
        goal_offlane_penalty: float = 0.5,
        parking_mismatch_penalty: float = 0.5,
        seed: int = 0,
    ):
        planner_cfg = load_planner_config()
        sim_cfg = load_sim_config()
        self.planner = load_planner()
        self.device = str(next(self.planner.parameters()).device)
        self.sim_steps = int(sim_steps)
        self.deterministic = planner_cfg.deterministic if deterministic is None else deterministic
        self.reward_fn = reward_fn
        self.goal_radius = sim_cfg.goal_radius
        # Goals of moving (controlled) agents must lie on the road: the planner was
        # trained with goals taken from real on-lane trajectories, so an off-lane
        # goal sends it out of distribution (cheap reward hacking). Static agents
        # (goal within 2 m of spawn, e.g. parked cars) are exempt - their "goal"
        # legitimately sits in a driveway/parking spot.
        self.goal_offlane_threshold = float(goal_offlane_threshold)
        self.goal_offlane_penalty = float(goal_offlane_penalty)
        # Parking-state consistency (goal mode only; needs meta["gt_parking_mask"]):
        # an agent is "parking" when its goal lies within 2 m of its spawn. Penalise
        # the fraction of agents whose generated parking state differs from GT -
        # blocks both "force a parked car to drive" (off-lane penalty would otherwise
        # be satisfiable by moving its goal onto the lane) and "freeze all traffic
        # into a static obstacle field" (cheap +1 by ramming the ego through it).
        self.parking_mismatch_penalty = float(parking_mismatch_penalty)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ build
    def _build_scenes(self, scenes: GeneratedScenes) -> list[SimScene]:
        states = scenes.agent_states.detach().cpu().numpy()
        types = scenes.agent_types.detach().cpu().numpy()
        a_idx = scenes.agent_scene_idx.detach().cpu().numpy()
        lanes = scenes.lane_polylines
        if isinstance(lanes, torch.Tensor):
            lanes = lanes.detach().cpu().numpy()
        l_idx = scenes.meta["lane_scene_idx"]
        if isinstance(l_idx, torch.Tensor):
            l_idx = l_idx.detach().cpu().numpy()

        sims = []
        for s in range(scenes.num_scenes):
            sims.append(
                SimScene(
                    states[a_idx == s],
                    types[a_idx == s],
                    lanes[l_idx == s],
                    rng=self.rng,
                )
            )
        return sims

    # --------------------------------------------------------------- evaluate
    @torch.no_grad()
    def evaluate(self, scenes: GeneratedScenes, record_trajectories: bool = False) -> dict:
        sims = self._build_scenes(scenes)
        m = len(sims)
        ego_collision = np.zeros(m, dtype=np.float32)
        ego_offroad = np.zeros(m, dtype=np.float32)
        init_invalid = np.zeros(m, dtype=np.float32)
        reached_goal = np.zeros(m, dtype=np.float32)
        finished = np.zeros(m, dtype=bool)

        traj = None
        if record_trajectories:
            traj = [
                {"x": [], "y": [], "heading": [], "respawn": [], "done": [],
                 "length": sim.length.copy(), "width": sim.width.copy()}
                for sim in sims
            ]

        # Egos whose generated goal is already inside the 2 m radius are static in
        # PufferDrive (never controlled); the scene is trivially over.
        for s, sim in enumerate(sims):
            if sim.n == 0 or 0 not in sim.controlled:
                reached_goal[s] = 1.0
                finished[s] = True

        for t in range(self.sim_steps):
            # ---- score current state (mirrors reward.py: read state, then step)
            for s, sim in enumerate(sims):
                if finished[s]:
                    continue
                collided = sim.ego_collides_now()
                if t == 0 and collided:
                    init_invalid[s] = 1.0
                ego_collision[s] = max(ego_collision[s], float(collided))
                if traj is not None:
                    traj[s]["x"].append(sim.x.copy())
                    traj[s]["y"].append(sim.y.copy())
                    traj[s]["heading"].append(sim.heading.copy())
                    traj[s]["respawn"].append(sim.respawned.copy())
                # state-based fallback (covers ego spawned near goal)
                d = float(np.hypot(sim.goal[0, 0] - sim.x[0], sim.goal[0, 1] - sim.y[0]))
                if d < self.goal_radius:
                    reached_goal[s] = 1.0
                    finished[s] = True
                    if traj is not None:
                        traj[s]["done"].append(True)

            active = [s for s in range(m) if not finished[s]]
            if not active:
                break

            # ---- planner forward over all controlled agents of active scenes
            obs_list = [sims[s].compute_obs() for s in active]
            obs = torch.as_tensor(np.concatenate(obs_list), device=self.device)
            actions = self.planner.act(obs, deterministic=self.deterministic).cpu().numpy()

            off = 0
            for s, ob in zip(active, obs_list):
                n_ctrl = ob.shape[0]
                sims[s].step_dynamics(actions[off : off + n_ctrl])
                sims[s].update_metrics()
                ego_reached, _ = sims[s].goal_step()
                off += n_ctrl
                if ego_reached:
                    reached_goal[s] = 1.0
                    finished[s] = True
                if traj is not None and traj[s]["x"]:
                    traj[s]["done"].append(bool(ego_reached))

        # ---- off-lane goal penalty (initial controlled agents only; see __init__)
        # "initial_controlled" implements the parking exemption: agents whose goal
        # is within 2 m of spawn are static in PufferDrive and never enter this
        # penalty (their goal may legitimately sit off-lane in a parking spot).
        # Keep this fixed across the rollout so goal_behavior="remove" cannot erase
        # agents from the penalty after they reach their goal.
        goal_offlane_frac = np.zeros(m, dtype=np.float32)
        for s, sim in enumerate(sims):
            if len(sim.initial_controlled) == 0:
                continue
            d = sim.dist_to_lane_centerline(sim.goal[sim.initial_controlled])
            offlane = np.isfinite(d) & (d > self.goal_offlane_threshold)
            goal_offlane_frac[s] = float(offlane.mean())

        # ---- parking-state mismatch penalty (goal mode only) -------------------
        parking_mismatch_frac = np.zeros(m, dtype=np.float32)
        gt_parking = scenes.meta.get("gt_parking_mask")
        if gt_parking is not None:
            if isinstance(gt_parking, torch.Tensor):
                gt_parking = gt_parking.detach().cpu().numpy()
            a_idx = scenes.agent_scene_idx.detach().cpu().numpy()
            for s, sim in enumerate(sims):
                gt_p = gt_parking[a_idx == s]
                gen_dist = np.hypot(sim.goal[:, 0] - sim.spawn[:, 0], sim.goal[:, 1] - sim.spawn[:, 1])
                gen_p = gen_dist < MIN_DISTANCE_TO_GOAL
                if len(gt_p):
                    parking_mismatch_frac[s] = float((gen_p != gt_p).mean())

        rewards = self.reward_fn(ego_collision, ego_offroad, init_invalid)
        # All penalty terms are per-scene FRACTIONS in [0, 1] (count-normalised),
        # so scenes with many agents are not penalised more than sparse ones; the
        # coefficients set the scale relative to the +/-1 collision reward.
        rewards = rewards - self.goal_offlane_penalty * goal_offlane_frac
        rewards = rewards - self.parking_mismatch_penalty * parking_mismatch_frac
        out = {
            "reward": rewards,
            "ego_collision": ego_collision,
            "ego_offroad": ego_offroad,
            "init_invalid": init_invalid,
            "reached_goal": reached_goal,
            "goal_offlane_frac": goal_offlane_frac,
            "parking_mismatch_frac": parking_mismatch_frac,
        }
        if traj is not None:
            for tr in traj:
                tr["x"] = np.asarray(tr["x"], dtype=np.float32) if tr["x"] else np.zeros((0, 0), np.float32)
                tr["y"] = np.asarray(tr["y"], dtype=np.float32) if tr["y"] else np.zeros((0, 0), np.float32)
                tr["heading"] = np.asarray(tr["heading"], dtype=np.float32) if tr["heading"] else np.zeros((0, 0), np.float32)
                tr["respawn"] = np.asarray(tr["respawn"], dtype=bool) if tr["respawn"] else np.zeros((0, 0), bool)
                tr["done"] = np.asarray(tr["done"], dtype=bool)
            out["trajectories"] = traj
        return out
