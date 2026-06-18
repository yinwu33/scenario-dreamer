"""Pure-numpy re-implementation of the PufferDrive rollout used as DDPO reward.

Faithful port of the pieces of PufferDrive's ``drive.h`` that the frozen planner
checkpoint depends on, so the reward can be computed inside the scenario-dreamer
venv without the C env / .bin round-trip:

  * classic dynamics (``move_dynamics``): signed-speed integrator, dt = 0.1,
    discrete 7x13 accel/steer table;
  * observation builder (``compute_observations``): ego(11) + partners(63*7) +
    road(512*7), same normalisation constants, partner gate 64 m, road segments
    collected from a 5 m grid in the same 21x21-cell spiral order, truncated at
    512;
  * agent lifecycle (``set_active_agents`` / ``respawn_agent`` / ``c_step``):
    agents whose goal is closer than 2 m at spawn are static (not controlled);
    an agent that reaches its goal (dist < 2, speed <= goal_speed) follows the configured
    goal behavior: stop, keep being controlled, respawn, or be removed;
  * vehicle collision (``collision_check``): oriented-box SAT, 15 m gate,
    pedestrians never collide, inactive agents excluded.

Deliberate deviations (match how the previous PufferDrive-hosted DDPO actually
behaved with generated scenes):
  * the generated maps carry lane centerlines only (no ROAD_EDGE entities), so
    the off-road check never fires - exactly as with ``scene_codec`` .bin maps;
  * once the ego (scene agent 0) reaches its goal the scene is finished and no
    longer stepped (the C env would respawn the ego; the reward stopped scoring
    the scene at that point anyway).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .geometry import _corners, _sat_overlap
from .goal_schema import MIN_DISTANCE_TO_GOAL
from planner.selfplay_drive.planner import (
    EGO_FEATURES,
    MAX_AGENTS,
    MAX_PARTNER_OBJECTS,
    MAX_ROAD_OBJECTS,
    OBS_DIM,
    PARTNER_FEATURES,
    ROAD_FEATURES,
)

# ----------------------------------------------------------------- constants
# drive.h: classic action space
ACCELERATION_VALUES = np.array([-4.0, -2.667, -1.333, -0.0, 1.333, 2.667, 4.0], dtype=np.float32)
STEERING_VALUES = np.array(
    [-1.0, -0.833, -0.667, -0.5, -0.333, -0.167, 0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0],
    dtype=np.float32,
)
NUM_STEER = len(STEERING_VALUES)

MAX_SPEED = 100.0
MAX_VEH_LEN = 30.0
MAX_VEH_WIDTH = 15.0
MAX_ROAD_SEGMENT_LENGTH = 100.0
MAX_ROAD_SCALE = 100.0
GRID_CELL_SIZE = 5.0
VISION_RANGE = 21                  # drive.h init_grid_map (hardcoded)
MAX_CONTROLLED_AGENTS = 32         # config/pacific/selfplay_drive.ini max_controlled_agents
# MIN_DISTANCE_TO_GOAL (static-agent threshold at spawn) imported from .goal_schema
# and re-exported here for the planner/metric modules that import it from this file.
PARTNER_DIST2_GATE = 4096.0        # 64 m
COLLISION_DIST2_GATE = 225.0       # 15 m
TTC_SWEEP_HORIZON = 10.0           # seconds; reward clips all values >= ttc_tau to 0
CONFIG_SIM_PATH = Path(__file__).with_name("config_sim.yaml")

TYPE_VEHICLE, TYPE_PEDESTRIAN, TYPE_CYCLIST = 1, 2, 3
ROAD_LANE_TYPE_FEATURE = 0.0       # entity type 4 (ROAD_LANE) - 4


@dataclass(frozen=True)
class SimConfig:
    dt: float
    goal_radius: float
    goal_speed: float
    goal_behavior: str
    max_controlled_agents: int
    condition_sample_mode: str
    fixed_collision_factor: float
    fixed_offroad_factor: float
    fixed_lane_width: float
    collision_factor_range: tuple[float, float]
    offroad_factor_range: tuple[float, float]
    lane_width_range: tuple[float, float]


def _float_pair(value, name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value!r}")
    return float(value[0]), float(value[1])


@lru_cache(maxsize=1)
def load_sim_config() -> SimConfig:
    raw = OmegaConf.load(CONFIG_SIM_PATH)
    return SimConfig(
        dt=float(raw.get("dt", 0.1)),
        goal_radius=float(raw.get("goal_radius", 2.0)),
        goal_speed=float(raw.get("goal_speed", 5.0)),
        goal_behavior=str(raw.get("goal_behavior", "stop")),
        max_controlled_agents=int(raw.get("max_controlled_agents", MAX_CONTROLLED_AGENTS)),
        condition_sample_mode=str(raw.get("condition_sample_mode", "random")),
        fixed_collision_factor=float(raw.get("fixed_collision_factor", 2.0)),
        fixed_offroad_factor=float(raw.get("fixed_offroad_factor", 2.0)),
        fixed_lane_width=float(raw.get("fixed_lane_width", 3.5)),
        collision_factor_range=_float_pair(raw.get("collision_factor_range", (0.0, 2.0)), "collision_factor_range"),
        offroad_factor_range=_float_pair(raw.get("offroad_factor_range", (0.0, 2.0)), "offroad_factor_range"),
        lane_width_range=_float_pair(raw.get("lane_width_range", (1.0, 5.0)), "lane_width_range"),
    )


def spiral_offsets(vision_range: int = VISION_RANGE) -> np.ndarray:
    """Cell-visit order of the road-obs neighbor cache (drive.h init_neighbor_offsets)."""
    offs = [(0, 0)]
    dx, dy = [1, 0, -1, 0], [0, 1, 0, -1]
    x = y = 0
    d, steps_to_take, steps_taken, segments = 0, 1, 0, 0
    max_offsets = vision_range * vision_range
    while len(offs) < max_offsets:
        x += dx[d]
        y += dy[d]
        if abs(x) <= vision_range // 2 and abs(y) <= vision_range // 2:
            offs.append((x, y))
        steps_taken += 1
        if steps_taken == steps_to_take:
            steps_taken = 0
            d = (d + 1) % 4
            segments += 1
            if segments % 2 == 0:
                steps_to_take += 1
    return np.asarray(offs, dtype=np.int64)


_SPIRAL = spiral_offsets()


class SimScene:
    """One generated scene rolled out with the frozen planner.

    Parameters are physical units in the scene frame (dm_goal decode layout).
    """

    def __init__(
        self,
        agent_states: np.ndarray,    # [N, 9] x, y, speed, cos, sin, length, width, goal_x, goal_y
        agent_types: np.ndarray,     # [N] 0 veh / 1 ped / 2 cyc (dm_goal ids)
        lane_polylines: np.ndarray,  # [L, P, 2]
        *,
        rng: np.random.Generator,
    ):
        cfg = load_sim_config()
        s = np.asarray(agent_states, dtype=np.float32)
        n = s.shape[0]
        self.n = n
        self.dt = cfg.dt
        self.goal_radius = cfg.goal_radius
        self.goal_speed = cfg.goal_speed
        if cfg.goal_behavior not in ("stop", "continue", "respawn", "remove"):
            raise ValueError(
                "goal_behavior must be one of 'stop', 'continue', 'respawn', or "
                f"'remove', got {cfg.goal_behavior!r}"
            )
        self.goal_behavior = cfg.goal_behavior

        self.x = s[:, 0].copy()
        self.y = s[:, 1].copy()
        self.heading = np.arctan2(s[:, 4], s[:, 3]).astype(np.float32)
        self.heading_x = np.cos(self.heading)
        self.heading_y = np.sin(self.heading)
        speed = s[:, 2]
        # same spawn velocities scene_codec wrote into the .bin trajectories
        self.vx = (speed * s[:, 3]).astype(np.float32)
        self.vy = (speed * s[:, 4]).astype(np.float32)
        # Generated spawn-speed magnitude, kept constant by the dummy goal-seek
        # planner (step_goal_seek); always >= 0 regardless of the sign of speed.
        self.speed0 = np.hypot(self.vx, self.vy).astype(np.float32)
        self.length = np.maximum(s[:, 5], 0.5).astype(np.float32)
        self.width = np.maximum(s[:, 6], 0.5).astype(np.float32)
        self.goal = s[:, 7:9].copy()
        self.ptype = (np.asarray(agent_types, dtype=np.int64) + 1).clip(TYPE_VEHICLE, TYPE_CYCLIST)

        self.spawn = np.stack([self.x, self.y, self.heading, self.vx, self.vy], axis=1)
        self.respawned = np.zeros(n, dtype=bool)
        self.removed = np.zeros(n, dtype=bool)
        self.stopped = np.zeros(n, dtype=bool)   # latched by GOAL_STOP; frozen in place
        self.collision_state = np.zeros(n, dtype=np.int64)

        # Per-agent conditioning inputs fed into the trailing ego observation slots.
        if cfg.condition_sample_mode == "fixed":
            self.collision_factor = np.full(n, cfg.fixed_collision_factor, dtype=np.float32)
            self.offroad_factor = np.full(n, cfg.fixed_offroad_factor, dtype=np.float32)
            self.lane_width = np.full(n, cfg.fixed_lane_width, dtype=np.float32)
        elif cfg.condition_sample_mode == "random":
            lo, hi = cfg.collision_factor_range
            self.collision_factor = rng.uniform(lo, hi, n).astype(np.float32)
            lo, hi = cfg.offroad_factor_range
            self.offroad_factor = rng.uniform(lo, hi, n).astype(np.float32)
            lo, hi = cfg.lane_width_range
            self.lane_width = rng.uniform(lo, hi, n).astype(np.float32)
        else:
            raise ValueError(
                "condition_sample_mode must be one of 'random' or 'fixed', "
                f"got {cfg.condition_sample_mode!r}"
            )

        # ---- set_active_agents: controlled vs static --------------------------
        goal_dist0 = np.hypot(self.goal[:, 0] - self.x, self.goal[:, 1] - self.y)
        controlled, static = [], []
        for i in range(n):
            if len(controlled) < cfg.max_controlled_agents and goal_dist0[i] >= MIN_DISTANCE_TO_GOAL:
                controlled.append(i)
            else:
                static.append(i)
        self.controlled = np.asarray(controlled, dtype=np.int64)
        self.static = np.asarray(static, dtype=np.int64)
        self.initial_controlled = self.controlled.copy()
        # Partner/collision scan order mirrors drive.h: active slots first, then
        # static slots, capped by MAX_AGENTS observation capacity.
        slot_order = np.concatenate([self.controlled, self.static]) if n else np.zeros(0, np.int64)
        self.slot_order = slot_order[:MAX_AGENTS]

        self._build_grid(np.asarray(lane_polylines, dtype=np.float32))
        self.update_metrics()  # c_reset computes metrics before the first observation

    # ------------------------------------------------------------------ grid
    def _build_grid(self, lanes: np.ndarray) -> None:
        """Register lane segments into 5 m cells and precompute the spiral cache."""
        mids, half_len, dirs = [], [], []
        starts, ends = [], []
        pts_all = []
        for poly in lanes:
            valid = np.isfinite(poly).all(axis=1)
            p = poly[valid]
            if p.shape[0] >= 2:
                pts_all.append(p)
                start, end = p[:-1], p[1:]
                mid = (start + end) / 2.0
                d = end - mid
                h = np.hypot(d[:, 0], d[:, 1])
                dn = np.where(h[:, None] > 0, d / np.maximum(h[:, None], 1e-12), d)
                mids.append(mid)
                half_len.append(h)
                dirs.append(dn)
                starts.append(start)
                ends.append(end)
        if not mids:
            self.seg_mid = np.zeros((0, 2), np.float32)
            self.seg_half_len = np.zeros(0, np.float32)
            self.seg_dir = np.zeros((0, 2), np.float32)
            self.seg_start = np.zeros((0, 2), np.float32)
            self.seg_end = np.zeros((0, 2), np.float32)
            self._grid_ok = False
            self._cell_cache: dict[int, np.ndarray] = {}
            return
        self.seg_mid = np.concatenate(mids).astype(np.float32)
        self.seg_half_len = np.concatenate(half_len).astype(np.float32)
        self.seg_dir = np.concatenate(dirs).astype(np.float32)
        self.seg_start = np.concatenate(starts).astype(np.float32)
        self.seg_end = np.concatenate(ends).astype(np.float32)

        pts = np.concatenate(pts_all)
        self.min_x, self.max_x = float(pts[:, 0].min()), float(pts[:, 0].max())
        self.min_y, self.max_y = float(pts[:, 1].min()), float(pts[:, 1].max())
        self._grid_ok = self.min_x < self.max_x and self.min_y < self.max_y
        if not self._grid_ok:
            self._cell_cache = {}
            return
        self.grid_cols = int(np.ceil((self.max_x - self.min_x) / GRID_CELL_SIZE)) + 1
        self.grid_rows = int(np.ceil((self.max_y - self.min_y) / GRID_CELL_SIZE)) + 1

        # per-cell segment lists in registration order (lane idx, then point idx)
        gx = ((self.seg_mid[:, 0] - self.min_x) / GRID_CELL_SIZE).astype(np.int64)
        gy = ((self.seg_mid[:, 1] - self.min_y) / GRID_CELL_SIZE).astype(np.int64)
        in_bounds = (gx >= 0) & (gx < self.grid_cols) & (gy >= 0) & (gy < self.grid_rows)
        cell_of_seg = gy * self.grid_cols + gx
        cells: dict[int, list[int]] = {}
        for seg_idx in np.nonzero(in_bounds)[0]:
            cells.setdefault(int(cell_of_seg[seg_idx]), []).append(int(seg_idx))

        # neighbor cache: for each cell, all segments of the 21x21 spiral neighborhood
        self._cell_cache = {}
        for cy in range(self.grid_rows):
            for cx in range(self.grid_cols):
                acc: list[int] = []
                for ox, oy in _SPIRAL:
                    nx_, ny_ = cx + int(ox), cy + int(oy)
                    if 0 <= nx_ < self.grid_cols and 0 <= ny_ < self.grid_rows:
                        lst = cells.get(ny_ * self.grid_cols + nx_)
                        if lst:
                            acc.extend(lst)
                            if len(acc) >= MAX_ROAD_OBJECTS:
                                break
                if acc:
                    self._cell_cache[cy * self.grid_cols + cx] = np.asarray(
                        acc[:MAX_ROAD_OBJECTS], dtype=np.int64
                    )

    def _agent_cell(self, i: int) -> int:
        if not self._grid_ok:
            return -1
        # C-style (int) cast truncates toward zero, matching getGridIndex
        gx = int((self.x[i] - self.min_x) / GRID_CELL_SIZE)
        gy = int((self.y[i] - self.min_y) / GRID_CELL_SIZE)
        if gx < 0 or gx >= self.grid_cols or gy < 0 or gy >= self.grid_rows:
            return -1
        return gy * self.grid_cols + gx

    # ----------------------------------------------------------- observations
    def compute_obs(self) -> np.ndarray:
        """[n_controlled, OBS_DIM] in active-agent order (drive.h compute_observations)."""
        obs = np.zeros((len(self.controlled), OBS_DIM), dtype=np.float32)
        for k, i in enumerate(self.controlled):
            o = obs[k]
            ch, sh = self.heading_x[i], self.heading_y[i]
            speed_mag = float(np.hypot(self.vx[i], self.vy[i]))
            v_dot_h = self.vx[i] * ch + self.vy[i] * sh
            signed_speed = np.copysign(speed_mag, v_dot_h)

            gx, gy = self.goal[i, 0] - self.x[i], self.goal[i, 1] - self.y[i]
            o[0] = (gx * ch + gy * sh) * 0.005
            o[1] = (-gx * sh + gy * ch) * 0.005
            o[2] = signed_speed / MAX_SPEED
            o[3] = self.width[i] / MAX_VEH_WIDTH
            o[4] = self.length[i] / MAX_VEH_LEN
            o[5] = 1.0 if self.collision_state[i] > 0 else 0.0
            o[6] = 1.0 if self.respawned[i] else 0.0
            o[7] = self.ptype[i] / 3.0
            o[EGO_FEATURES - 3] = self.collision_factor[i]
            o[EGO_FEATURES - 2] = self.offroad_factor[i]
            o[EGO_FEATURES - 1] = self.lane_width[i]

            # ---- partners (skipped entirely while the ego is inactive) --------
            base = EGO_FEATURES
            if not self.respawned[i]:
                cand = self.slot_order[(self.slot_order != i) & (~self.respawned[self.slot_order])]
                if len(cand):
                    dx = self.x[cand] - self.x[i]
                    dy = self.y[cand] - self.y[i]
                    near = (dx * dx + dy * dy) <= PARTNER_DIST2_GATE
                    cand, dx, dy = cand[near], dx[near], dy[near]
                    cand = cand[:MAX_PARTNER_OBJECTS]
                    dx, dy = dx[:MAX_PARTNER_OBJECTS], dy[:MAX_PARTNER_OBJECTS]
                    m = len(cand)
                    if m:
                        rel = np.empty((m, PARTNER_FEATURES), dtype=np.float32)
                        rel[:, 0] = (dx * ch + dy * sh) * 0.02
                        rel[:, 1] = (-dx * sh + dy * ch) * 0.02
                        rel[:, 2] = self.width[cand] / MAX_VEH_WIDTH
                        rel[:, 3] = self.length[cand] / MAX_VEH_LEN
                        rel[:, 4] = self.heading_x[cand] * ch + self.heading_y[cand] * sh
                        rel[:, 5] = self.heading_y[cand] * ch - self.heading_x[cand] * sh
                        sp = np.hypot(self.vx[cand], self.vy[cand])
                        vdh = self.vx[cand] * self.heading_x[cand] + self.vy[cand] * self.heading_y[cand]
                        rel[:, 6] = np.copysign(sp, vdh) / MAX_SPEED
                        o[base : base + m * PARTNER_FEATURES] = rel.reshape(-1)

            # ---- road segments ------------------------------------------------
            base = EGO_FEATURES + MAX_PARTNER_OBJECTS * PARTNER_FEATURES
            cell = self._agent_cell(i)
            segs = self._cell_cache.get(cell) if cell >= 0 else None
            if segs is not None and len(segs):
                mid = self.seg_mid[segs]
                rx = mid[:, 0] - self.x[i]
                ry = mid[:, 1] - self.y[i]
                dn = self.seg_dir[segs]
                feat = np.empty((len(segs), ROAD_FEATURES), dtype=np.float32)
                feat[:, 0] = (rx * ch + ry * sh) * 0.02
                feat[:, 1] = (-rx * sh + ry * ch) * 0.02
                feat[:, 2] = self.seg_half_len[segs] / MAX_ROAD_SEGMENT_LENGTH
                feat[:, 3] = 0.1 / MAX_ROAD_SCALE
                feat[:, 4] = dn[:, 0] * ch + dn[:, 1] * sh
                feat[:, 5] = -dn[:, 0] * sh + dn[:, 1] * ch
                feat[:, 6] = ROAD_LANE_TYPE_FEATURE
                o[base : base + len(segs) * ROAD_FEATURES] = feat.reshape(-1)
        return obs

    # -------------------------------------------------------------- dynamics
    def step_dynamics(self, actions: np.ndarray) -> None:
        """Classic dynamics for all controlled agents (actions in [0, 91)).

        Stopped agents (GOAL_STOP) are frozen: drive.h move_dynamics zeroes their
        velocity and returns before integrating.
        """
        moving = ~self.stopped[self.controlled]
        self.vx[self.controlled[~moving]] = 0.0
        self.vy[self.controlled[~moving]] = 0.0
        idx = self.controlled[moving]
        actions = np.asarray(actions)[moving]
        if len(idx) == 0:
            return
        a = ACCELERATION_VALUES[actions // NUM_STEER]
        steer = STEERING_VALUES[actions % NUM_STEER]

        speed_mag = np.hypot(self.vx[idx], self.vy[idx])
        v_dot_h = self.vx[idx] * self.heading_x[idx] + self.vy[idx] * self.heading_y[idx]
        signed_speed = np.copysign(speed_mag, v_dot_h) + a * self.dt
        signed_speed = np.clip(signed_speed, -MAX_SPEED, MAX_SPEED)

        beta = np.tanh(0.5 * np.tan(steer))
        yaw_rate = signed_speed * np.cos(beta) * np.tan(steer) / self.length[idx]
        new_vx = signed_speed * np.cos(self.heading[idx] + beta)
        new_vy = signed_speed * np.sin(self.heading[idx] + beta)

        self.x[idx] += new_vx * self.dt
        self.y[idx] += new_vy * self.dt
        self.heading[idx] += yaw_rate * self.dt
        self.heading_x[idx] = np.cos(self.heading[idx])
        self.heading_y[idx] = np.sin(self.heading[idx])
        self.vx[idx] = new_vx
        self.vy[idx] = new_vy

    def step_goal_seek(self) -> None:
        """Dummy rule-based motion: translate each controlled agent toward its
        goal at its generated spawn speed, with NO acceleration/steering
        integration (the ``DummyPlanner`` rollout).

        Heading (and thus the collision box orientation) is left at the generated
        value, so an agent whose goal is not straight ahead slides diagonally
        toward it. The step is not clamped to the goal (it may overshoot);
        arrival is handled afterwards by ``goal_step``. Stopped / respawned agents
        are frozen, matching ``step_dynamics``.
        """
        idx = self.controlled[~self.stopped[self.controlled]]
        idx = idx[~self.respawned[idx]]
        if len(idx) == 0:
            return
        gx = self.goal[idx, 0] - self.x[idx]
        gy = self.goal[idx, 1] - self.y[idx]
        dist = np.hypot(gx, gy)
        moving = dist > 1e-6
        sel = idx[moving]
        if len(sel) == 0:
            return
        inv_speed = self.speed0[sel] / dist[moving]
        vx = gx[moving] * inv_speed
        vy = gy[moving] * inv_speed
        self.vx[sel] = vx
        self.vy[sel] = vy
        self.x[sel] += vx * self.dt
        self.y[sel] += vy * self.dt

    # --------------------------------------------------------------- metrics
    def update_metrics(self) -> None:
        """Vehicle-collision state per controlled agent (collision_check port).

        No ROAD_EDGE entities exist in generated maps, so the off-road branch of
        compute_agent_metrics can never fire and is omitted.
        """
        self.collision_state[:] = 0
        idx = self.controlled
        if len(idx) == 0:
            return
        boxes = _corners(self.x, self.y, self.heading, self.length, self.width)
        for i in idx:
            if self.ptype[i] == TYPE_PEDESTRIAN or self.respawned[i]:
                continue
            cand = self.slot_order[(self.slot_order != i) & (~self.respawned[self.slot_order])]
            if not len(cand):
                continue
            dx = self.x[cand] - self.x[i]
            dy = self.y[cand] - self.y[i]
            cand = cand[(dx * dx + dy * dy) <= COLLISION_DIST2_GATE]
            if len(cand) and _sat_overlap(boxes[i], boxes[cand]).any():
                self.collision_state[i] = 2  # VEHICLE_COLLISION

    def _remove_agent(self, i: int) -> None:
        """Deactivate an agent after goal arrival for goal_behavior='remove'."""
        self.removed[i] = True
        self.respawned[i] = True  # existing trajectory/viz mask: hide inactive agents
        self.stopped[i] = False
        self.vx[i], self.vy[i] = 0.0, 0.0
        self.collision_state[i] = 0
        self.controlled = self.controlled[self.controlled != i]
        self.static = self.static[self.static != i]
        self.slot_order = self.slot_order[self.slot_order != i]

    def goal_step(self) -> tuple[bool, np.ndarray]:
        """Goal handling after a dynamics step (c_step goal phase).

        Returns ``(ego_reached, reached_indices)``. The ego (agent 0) is only
        reported - the caller finishes the scene. Non-ego agents that reach
        their goal (dist < goal_radius, speed <= goal_speed):

          * ``goal_behavior="stop"`` (default, drive.h GOAL_STOP): freeze in
            place with zero velocity; they stay in the world as parked
            obstacles (collisions + partner observations still see them);
          * ``goal_behavior="continue"``: leave the agent active and controlled;
            no state is changed when it enters the goal radius;
          * ``goal_behavior="respawn"`` (drive.h GOAL_RESPAWN): teleport back to
            the spawn pose and latch as respawned (excluded from collisions and
            partner observations) - the original PufferDrive default;
          * ``goal_behavior="remove"``: delete the agent from subsequent control,
            collision checks, and partner observations.
        """
        idx = self.controlled
        if len(idx) == 0:
            return False, np.zeros(0, np.int64)
        dist = np.hypot(self.goal[idx, 0] - self.x[idx], self.goal[idx, 1] - self.y[idx])
        speed = np.hypot(self.vx[idx], self.vy[idx])
        # reached = idx[(dist < self.goal_radius) & (speed <= self.goal_speed)]
        reached = idx[dist < self.goal_radius]
        ego_reached = bool(0 in reached)
        for i in reached:
            if i == 0:
                continue
            if self.goal_behavior == "stop":
                self.stopped[i] = True
                self.vx[i], self.vy[i] = 0.0, 0.0
            elif self.goal_behavior == "continue":
                continue
            elif self.goal_behavior == "respawn":
                self.x[i], self.y[i], self.heading[i] = self.spawn[i, 0], self.spawn[i, 1], self.spawn[i, 2]
                self.heading_x[i] = np.cos(self.heading[i])
                self.heading_y[i] = np.sin(self.heading[i])
                self.vx[i], self.vy[i] = self.spawn[i, 3], self.spawn[i, 4]
                self.respawned[i] = True
            else:  # remove
                self._remove_agent(int(i))
        return ego_reached, reached

    # ------------------------------------------------------------- inspection
    def dist_to_lane_centerline(self, points: np.ndarray) -> np.ndarray:
        """Min distance of each point [M, 2] to any lane centerline segment.

        Returns +inf entries when the scene has no lane geometry.
        """
        points = np.atleast_2d(np.asarray(points, dtype=np.float32))
        if self.seg_start.shape[0] == 0:
            return np.full(points.shape[0], np.inf, dtype=np.float32)
        a, b = self.seg_start, self.seg_end                       # [S, 2]
        ab = b - a
        denom = np.maximum((ab * ab).sum(-1), 1e-9)               # [S]
        ap = points[:, None, :] - a[None]                         # [M, S, 2]
        t = np.clip((ap * ab[None]).sum(-1) / denom[None], 0.0, 1.0)
        proj = a[None] + t[..., None] * ab[None]
        d = points[:, None, :] - proj
        return np.sqrt((d * d).sum(-1)).min(axis=1)

    def ego_collides_now(self) -> bool:
        """Oriented-box overlap of the ego (agent 0) with any active agent."""
        if self.n <= 1 or self.respawned[0]:
            return False
        others = self.slot_order[(self.slot_order != 0) & (~self.respawned[self.slot_order])]
        if not len(others):
            return False
        boxes = _corners(self.x, self.y, self.heading, self.length, self.width)
        return bool(_sat_overlap(boxes[0], boxes[others]).any())

    def ego_min_ttc_now(self) -> float:
        """Box-aware TTC for the ego driving into another active agent.

        This deliberately uses ego-only motion: the ego box is swept forward with
        its current velocity and heading, while other boxes stay at their current
        poses. That matches the reward intent of scoring "ego would hit another
        car" and avoids rewarding cases where another actor is merely closing on
        the ego.
        """
        if self.n <= 1 or self.respawned[0]:
            return float(np.inf)
        others = self.slot_order[(self.slot_order != 0) & (~self.respawned[self.slot_order])]
        others = others[self.ptype[others] != TYPE_PEDESTRIAN]
        if not len(others):
            return float(np.inf)
        ego_vx = float(self.vx[0])
        ego_vy = float(self.vy[0])
        if ego_vx * ego_vx + ego_vy * ego_vy < 1e-6:
            return float(np.inf)

        other_boxes = _corners(
            self.x[others],
            self.y[others],
            self.heading[others],
            self.length[others],
            self.width[others],
        )
        steps = int(np.ceil(TTC_SWEEP_HORIZON / max(self.dt, 1e-3)))
        for step in range(0, steps + 1):
            t = step * self.dt
            ego_box = _corners(
                np.asarray([self.x[0] + ego_vx * t]),
                np.asarray([self.y[0] + ego_vy * t]),
                np.asarray([self.heading[0]]),
                np.asarray([self.length[0]]),
                np.asarray([self.width[0]]),
            )[0]
            if _sat_overlap(ego_box, other_boxes).any():
                return float(t)
        return float(np.inf)

    def any_vehicle_overlap(self, margin: float = 0.0) -> bool:
        """True if any two active (non-pedestrian, non-respawned) agent boxes overlap.

        Boxes are grown by ``margin`` metres per side before the SAT test, so a
        positive margin also rejects near-touching spawns; ``margin=0`` is exact
        overlap. Used to flag degenerate generated init states (init_invalid).
        """
        active = self.slot_order[~self.respawned[self.slot_order]]
        active = active[self.ptype[active] != TYPE_PEDESTRIAN]
        if len(active) < 2:
            return False
        boxes = _corners(
            self.x[active],
            self.y[active],
            self.heading[active],
            self.length[active] + 2.0 * margin,
            self.width[active] + 2.0 * margin,
        )
        for k in range(len(active) - 1):
            dx = self.x[active[k + 1:]] - self.x[active[k]]
            dy = self.y[active[k + 1:]] - self.y[active[k]]
            gate = (dx * dx + dy * dy) <= COLLISION_DIST2_GATE
            if gate.any() and _sat_overlap(boxes[k], boxes[k + 1:][gate]).any():
                return True
        return False
