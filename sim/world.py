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
  * agent lifecycle (``set_active_agents`` / ``c_step``):
    agents whose goal is closer than 2 m at spawn are static (not controlled);
    an agent that reaches its goal (dist < 2, speed <= goal_speed) follows the configured
    goal behavior: stop, keep being controlled (continue), or be removed;
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

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from omegaconf import OmegaConf

from .geometry import _corners, _sat_overlap
from .schema import MIN_DISTANCE_TO_GOAL
from nets.selfplay_drive.net import (
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
# MIN_DISTANCE_TO_GOAL (static-agent threshold at spawn) imported from .schema
# and re-exported here for the planner/metric modules that import it from this file.
PARTNER_DIST2_GATE = 4096.0        # 64 m
COLLISION_DIST2_GATE = 225.0       # 15 m
EGO_AGGRESSOR_MIN_SPEED = 0.5      # m/s; ego counts as the aggressor only when its
                                   # velocity projected onto the ego->other direction
                                   # exceeds this (a passive/slow ego is never at fault)
TYPE_VEHICLE, TYPE_PEDESTRIAN, TYPE_CYCLIST = 1, 2, 3
ROAD_LANE_TYPE_FEATURE = 0.0       # entity type 4 (ROAD_LANE) - 4


def to_puffer_agent_types(agent_types) -> np.ndarray:
    """Convert dataset/model ids (0 veh, 1 ped, 2 cyc) to PufferDrive ids.

    ``GeneratedScenes.agent_types`` deliberately keeps the model-side convention.
    PufferDrive observations and collision logic use entity ids 1..3, so callers
    convert at this boundary before constructing a sim scene.
    """
    return (
        np.asarray(agent_types, dtype=np.int64) + 1
    ).clip(TYPE_VEHICLE, TYPE_CYCLIST)


@dataclass(frozen=True)
class SimConfig:
    dt: float
    goal_radius: float
    goal_speed: float
    goal_behavior: str
    map_extent: float
    max_controlled_agents: int


def load_sim_config(cfg) -> SimConfig:
    """Build the strict SimConfig from the composed ``cfgs/rollout`` group node.

    The yaml must be COMPLETE: a missing or unknown key raises (same contract
    as SimulatorConfig / RewardConfig). The per-agent conditioning obs
    (collision_factor / offroad_factor / lane_width) are NOT here: they are
    policy inputs owned by each role's planner config (``conditioning:`` in
    ``cfgs/planner/<name>.yaml``, applied by ``RolloutRunner``).
    """
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    raw = dict(cfg)
    raw["goal_behavior"] = str(raw["goal_behavior"])
    return SimConfig(**raw)


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


# --- lane-grid cache -------------------------------------------------------
# The 5 m cell grid + spiral neighbour cache depends ONLY on the (frozen) lane
# polylines, but SimScene rebuilds it per scene per rollout -- the single biggest
# reward cost. In the DDPO flow the base scene is fixed and only the adversary
# latent changes, and group replication repeats each context group_size times, so
# a batch of B scenes has only ~B/group_size distinct lane sets. This module-level
# LRU (keyed on the lane bytes) collapses that redundancy within a rollout and
# across iterations for recurring pool contexts. The cached dict is shared by
# reference; every field it holds is read-only during a rollout (only compute_obs
# / _agent_cell read them, never mutate), so sharing is safe. Not thread-safe;
# fork-based workers get independent module globals, which is fine.
_GRID_CACHE_MAXSIZE = 1024
_grid_cache: "OrderedDict[tuple, dict]" = OrderedDict()


def _grid_cache_get(key):
    g = _grid_cache.get(key)
    if g is not None:
        _grid_cache.move_to_end(key)
    return g


def _grid_cache_put(key, g):
    _grid_cache[key] = g
    _grid_cache.move_to_end(key)
    while len(_grid_cache) > _GRID_CACHE_MAXSIZE:
        _grid_cache.popitem(last=False)


def _compute_grid(lanes: np.ndarray) -> dict:
    """Pure lane-grid build (segments + 5 m cell spiral cache). Returns a dict of
    every field ``SimScene._apply_grid`` assigns; degenerate scenes get empty
    segments / ``grid_ok=False`` (grid_cols/rows/min_* are then never read)."""
    empty = dict(
        seg_mid=np.zeros((0, 2), np.float32),
        seg_half_len=np.zeros(0, np.float32),
        seg_dir=np.zeros((0, 2), np.float32),
        seg_start=np.zeros((0, 2), np.float32),
        seg_end=np.zeros((0, 2), np.float32),
        min_x=0.0, max_x=0.0, min_y=0.0, max_y=0.0,
        grid_ok=False, grid_cols=0, grid_rows=0, cell_cache={},
    )

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
        return empty

    seg_mid = np.concatenate(mids).astype(np.float32)
    seg_half_len = np.concatenate(half_len).astype(np.float32)
    seg_dir = np.concatenate(dirs).astype(np.float32)
    seg_start = np.concatenate(starts).astype(np.float32)
    seg_end = np.concatenate(ends).astype(np.float32)

    pts = np.concatenate(pts_all)
    min_x, max_x = float(pts[:, 0].min()), float(pts[:, 0].max())
    min_y, max_y = float(pts[:, 1].min()), float(pts[:, 1].max())
    grid = dict(
        seg_mid=seg_mid, seg_half_len=seg_half_len, seg_dir=seg_dir,
        seg_start=seg_start, seg_end=seg_end,
        min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
        grid_ok=(min_x < max_x and min_y < max_y),
        grid_cols=0, grid_rows=0, cell_cache={},
    )
    if not grid["grid_ok"]:
        return grid

    grid_cols = int(np.ceil((max_x - min_x) / GRID_CELL_SIZE)) + 1
    grid_rows = int(np.ceil((max_y - min_y) / GRID_CELL_SIZE)) + 1

    # per-cell segment lists in registration order (lane idx, then point idx)
    gx = ((seg_mid[:, 0] - min_x) / GRID_CELL_SIZE).astype(np.int64)
    gy = ((seg_mid[:, 1] - min_y) / GRID_CELL_SIZE).astype(np.int64)
    in_bounds = (gx >= 0) & (gx < grid_cols) & (gy >= 0) & (gy < grid_rows)
    cell_of_seg = gy * grid_cols + gx
    cells: dict[int, list[int]] = {}
    for seg_idx in np.nonzero(in_bounds)[0]:
        cells.setdefault(int(cell_of_seg[seg_idx]), []).append(int(seg_idx))

    # neighbor cache: for each cell, all segments of the 21x21 spiral neighborhood
    cell_cache: dict[int, np.ndarray] = {}
    for cy in range(grid_rows):
        for cx in range(grid_cols):
            acc: list[int] = []
            for ox, oy in _SPIRAL:
                nx_, ny_ = cx + int(ox), cy + int(oy)
                if 0 <= nx_ < grid_cols and 0 <= ny_ < grid_rows:
                    lst = cells.get(ny_ * grid_cols + nx_)
                    if lst:
                        acc.extend(lst)
                        if len(acc) >= MAX_ROAD_OBJECTS:
                            break
            if acc:
                cell_cache[cy * grid_cols + cx] = np.asarray(
                    acc[:MAX_ROAD_OBJECTS], dtype=np.int64
                )
    grid["grid_cols"] = grid_cols
    grid["grid_rows"] = grid_rows
    grid["cell_cache"] = cell_cache
    return grid


class SimScene:
    """One generated scene rolled out with the frozen planner.

    Parameters are physical units in the scene frame (dm_goal decode layout).
    """

    def __init__(
        self,
        agent_states: np.ndarray,    # [N, 9] x, y, speed, cos, sin, length, width, goal_x, goal_y
        agent_ptypes: np.ndarray,    # [N] 1 vehicle / 2 pedestrian / 3 cyclist (PufferDrive ids)
        lane_polylines: np.ndarray,  # [L, P, 2]
        *,
        sim_cfg: SimConfig,
    ):
        cfg = sim_cfg
        s = np.asarray(agent_states, dtype=np.float32)
        n = s.shape[0]
        self.n = n
        self.dt = cfg.dt
        self.goal_radius = cfg.goal_radius
        self.goal_speed = cfg.goal_speed
        if cfg.goal_behavior not in ("stop", "continue", "remove"):
            raise ValueError(
                "goal_behavior must be one of 'stop', 'continue', or 'remove', "
                f"got {cfg.goal_behavior!r}"
            )
        self.goal_behavior = cfg.goal_behavior
        # Half-extent of the square map. Non-ego agents whose centre leaves
        # [-map_half, map_half] in x or y are removed (remove_out_of_bounds).
        self.map_half = float(cfg.map_extent) / 2.0

        self.x = s[:, 0].copy()
        self.y = s[:, 1].copy()
        self.heading = np.arctan2(s[:, 4], s[:, 3]).astype(np.float32)
        self.heading_x = np.cos(self.heading)
        self.heading_y = np.sin(self.heading)
        speed = s[:, 2]
        # same spawn velocities scene_codec wrote into the .bin trajectories
        self.vx = (speed * s[:, 3]).astype(np.float32)
        self.vy = (speed * s[:, 4]).astype(np.float32)
        self.length = np.maximum(s[:, 5], 0.5).astype(np.float32)
        self.width = np.maximum(s[:, 6], 0.5).astype(np.float32)
        self.goal = s[:, 7:9].copy()
        self.ptype = np.asarray(agent_ptypes, dtype=np.int64).copy()
        if self.ptype.shape[0] != n:
            raise ValueError(
                f"agent_ptypes length must match agent_states rows ({n}), "
                f"got {self.ptype.shape[0]}"
            )
        if self.ptype.size and (
            self.ptype.min() < TYPE_VEHICLE or self.ptype.max() > TYPE_CYCLIST
        ):
            raise ValueError(
                "agent_ptypes must use PufferDrive ids "
                f"[{TYPE_VEHICLE}, {TYPE_CYCLIST}]"
            )

        # Spawn pose, kept for the lane-distance / parking reward hooks (it is no
        # longer used for a respawn teleport, which has been removed).
        self.spawn = np.stack([self.x, self.y, self.heading, self.vx, self.vy], axis=1)
        # Inactive mask: an agent retired by ``_remove_agent`` (goal_behavior
        # 'remove' or leaving the map) is dropped from controlled/static/slot_order
        # and flagged here so the trajectory/viz layer hides it from that step on.
        self.removed = np.zeros(n, dtype=bool)
        self.stopped = np.zeros(n, dtype=bool)   # latched by GOAL_STOP; frozen in place
        # Collision response (see latch_ego_crash): the ego and any vehicle it
        # drives into are latched here and frozen, so boxes can no longer pass
        # through each other. Crashed agents are excluded from TTC / min-dist.
        self.crashed = np.zeros(n, dtype=bool)
        # Sticky fault flag: set by latch_ego_crash when the ego drove *into*
        # another car. Decouples the rewarded ego collision from the general
        # crashed[0] freeze (which fires on any overlap to stop pass-through).
        self.ego_caused_collision = False
        # Per-step ego collision event, recorded by latch_ego_crash *before* the
        # contact response zeroes the ego velocity and consumed the same step by
        # EgoCollisionHook.after_step_scene. ``last_ego_collision_partners`` is the
        # set of vehicles the ego contacted this step (focal event, fault-agnostic);
        # ``last_ego_fault_partners`` is the subset the ego was driving *into*.
        # Recording the event here is what lets the rewarded collision fire at all:
        # by the next observation the contacted boxes are frozen with zero velocity,
        # so a velocity-based aggressor re-check would always read the ego as passive.
        self.last_ego_collision_partners = np.empty(0, dtype=np.int64)
        self.last_ego_fault_partners = np.empty(0, dtype=np.int64)
        self.collision_state = np.zeros(n, dtype=np.int64)

        # Per-agent conditioning inputs fed into the trailing ego observation
        # slots. They parameterize the CONDITIONED frozen policy's driving
        # style, so their values are owned by each role's planner config
        # (``conditioning:`` in cfgs/planner/<name>.yaml) and filled in by
        # ``RolloutRunner._apply_conditioning`` right after role assignment;
        # a planner that ignores the slots leaves them at 0.
        self.collision_factor = np.zeros(n, dtype=np.float32)
        self.offroad_factor = np.zeros(n, dtype=np.float32)
        self.lane_width = np.zeros(n, dtype=np.float32)

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

        lanes = np.asarray(lane_polylines, dtype=np.float32)
        # Kept alongside the grid decomposition: the grid flattens the lanes into
        # anonymous segments (seg_start/seg_end), which answers "how far is the
        # nearest centerline" but not "which lane is this and where does it go".
        # A route-planning planner needs the per-lane polylines back.
        self.lane_polylines = lanes
        # Lane GRAPH: {"succ": [2, E], "lateral": [2, E]} over lane rows, or None
        # when the scene source did not supply connectivity (most do not -- see
        # GeneratedScenes.meta['lane_graph']). Filled in by RolloutRunner.
        self.lane_graph: dict[str, np.ndarray] | None = None
        self._build_grid(lanes)
        self.update_metrics()  # c_reset computes metrics before the first observation

    # ------------------------------------------------------------------ grid
    def _build_grid(self, lanes: np.ndarray) -> None:
        """Register lane segments into 5 m cells + precompute the spiral cache.

        The grid depends only on ``lanes``, which are frozen in the DDPO flow, so
        the heavy build is memoised on the lane bytes (see ``_grid_cache``). The
        cached dict is shared by reference across scenes; ``_apply_grid`` only
        reads its fields, never mutates them, so sharing is safe."""
        key = (lanes.shape, lanes.tobytes())
        g = _grid_cache_get(key)
        if g is None:
            g = _compute_grid(lanes)
            _grid_cache_put(key, g)
        self._apply_grid(g)

    def _apply_grid(self, g: dict) -> None:
        self.seg_mid = g["seg_mid"]
        self.seg_half_len = g["seg_half_len"]
        self.seg_dir = g["seg_dir"]
        self.seg_start = g["seg_start"]
        self.seg_end = g["seg_end"]
        self.min_x, self.max_x = g["min_x"], g["max_x"]
        self.min_y, self.max_y = g["min_y"], g["max_y"]
        self._grid_ok = g["grid_ok"]
        self.grid_cols = g["grid_cols"]
        self.grid_rows = g["grid_rows"]
        self._cell_cache = g["cell_cache"]

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
    def compute_obs(self, indices=None) -> np.ndarray:
        """[n_agents, OBS_DIM] in ``indices`` order (drive.h compute_observations).

        ``indices`` defaults to every controlled agent; a per-role planner passes
        the subset of agents it drives. Each row only depends on world state, so
        splitting one call into subset calls yields identical observations.
        """
        agents = self.controlled if indices is None else np.asarray(indices, dtype=np.int64)
        obs = np.zeros((len(agents), OBS_DIM), dtype=np.float32)
        for k, i in enumerate(agents):
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
            o[6] = 1.0 if self.removed[i] else 0.0
            o[7] = self.ptype[i] / 3.0
            o[EGO_FEATURES - 3] = self.collision_factor[i]
            o[EGO_FEATURES - 2] = self.offroad_factor[i]
            o[EGO_FEATURES - 1] = self.lane_width[i]

            # ---- partners (slot_order already excludes retired agents) --------
            base = EGO_FEATURES
            cand = self.slot_order[self.slot_order != i]
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
    def step_dynamics(self, actions: np.ndarray, indices=None) -> None:
        """Classic dynamics (actions in [0, 91)) for ``indices`` agents.

        ``indices`` defaults to every controlled agent; a per-role planner passes
        the subset it drives, with ``actions`` aligned to it. Integration is
        per-agent independent, so disjoint subset calls compose exactly.

        Stopped agents (GOAL_STOP) are frozen: drive.h move_dynamics zeroes their
        velocity and returns before integrating. Crashed agents (collision
        response, see latch_ego_crash) are frozen the same way.
        """
        agents = self.controlled if indices is None else np.asarray(indices, dtype=np.int64)
        moving = ~(self.stopped | self.crashed)[agents]
        self.vx[agents[~moving]] = 0.0
        self.vy[agents[~moving]] = 0.0
        idx = agents[moving]
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
            if self.ptype[i] == TYPE_PEDESTRIAN:
                continue
            cand = self.slot_order[self.slot_order != i]
            if not len(cand):
                continue
            dx = self.x[cand] - self.x[i]
            dy = self.y[cand] - self.y[i]
            cand = cand[(dx * dx + dy * dy) <= COLLISION_DIST2_GATE]
            if len(cand) and _sat_overlap(boxes[i], boxes[cand]).any():
                self.collision_state[i] = 2  # VEHICLE_COLLISION

    def latch_ego_crash(self) -> None:
        """General collision response: freeze the ego and any car it contacts.

        The numpy sim integrates dynamics without contact forces, so boxes pass
        straight through each other. A DDPO policy exploits this by sending an
        adversary through the ego from behind: it overlaps the ego, re-emerges in
        front, and farms a near-zero ego time-to-collision. To remove the exploit
        at its source, *any* ego<->vehicle overlap (regardless of fault) latches
        both as ``crashed`` - frozen in place (zero velocity, never re-integrated
        by ``step_dynamics``) and excluded from TTC /
        min-distance scoring. A contacting adversary therefore can never pass
        through to the ego's forward path, so it cannot spoof min-TTC.

        Fault is recorded separately: ``ego_caused_collision`` is set only when the
        ego was driving *into* the contacted car (``_ego_aggressor_mask``), so the
        general freeze/stop and the rewarded ego-collision event are decoupled - a
        car ramming a passive ego stops the scene but is not the ego's collision.

        Called once per step by the rollout loop after ``update_metrics``; a
        no-op once the ego has already crashed or is inactive.
        """
        # Clear last step's event first, so a collision-free step (including the
        # early returns below) reports no event to the hook.
        self.last_ego_collision_partners = np.empty(0, dtype=np.int64)
        self.last_ego_fault_partners = np.empty(0, dtype=np.int64)
        if self.n <= 1 or self.crashed[0]:
            return
        if 0 not in self.controlled:
            return
        others = self.slot_order[self.slot_order != 0]
        others = others[(self.ptype[others] != TYPE_PEDESTRIAN) & (~self.crashed[others])]
        if not len(others):
            return
        # Broad phase: only test boxes within the collision gate (matches
        # update_metrics), then the oriented-box SAT narrow phase.
        dx = self.x[others] - self.x[0]
        dy = self.y[others] - self.y[0]
        others = others[(dx * dx + dy * dy) <= COLLISION_DIST2_GATE]
        if not len(others):
            return
        boxes = _corners(self.x, self.y, self.heading, self.length, self.width)
        overlap = _sat_overlap(boxes[0], boxes[others])
        hit = others[overlap]
        if not len(hit):
            return
        # Record the event + fault attribution while the ego velocity is still the
        # genuine pre-collision one (it is zeroed below). The hook consumes these
        # the same step instead of re-deriving contact from the frozen ego.
        fault_mask = self._ego_aggressor_mask(hit)
        self.last_ego_collision_partners = hit.copy()
        self.last_ego_fault_partners = hit[fault_mask].copy()
        if fault_mask.any():
            self.ego_caused_collision = True
        crashed_now = np.concatenate(([0], hit)).astype(np.int64)
        self.crashed[crashed_now] = True
        self.vx[crashed_now] = 0.0
        self.vy[crashed_now] = 0.0

    def _remove_agent(self, i: int) -> None:
        """Retire an agent (goal_behavior='remove' or out-of-bounds): drop it from
        control / collision / observation and flag it so the viz hides it."""
        self.removed[i] = True
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
            else:  # remove
                self._remove_agent(int(i))
        return ego_reached, reached

    def remove_out_of_bounds(self) -> np.ndarray:
        """Remove non-ego controlled agents whose centre left the map square.

        Called once per step after ``goal_step``. With ``goal_behavior='continue'``
        agents keep driving past their goal, so the map boundary
        (``|x| > map_half`` or ``|y| > map_half``) is what retires them. The ego
        (agent 0) is exempt - its scene is finished by ``ReachedGoalHook`` when it
        reaches its goal, never by leaving the map. Returns the removed indices.
        """
        idx = self.controlled
        if len(idx) == 0:
            return np.zeros(0, np.int64)
        oob = idx[(np.abs(self.x[idx]) > self.map_half) | (np.abs(self.y[idx]) > self.map_half)]
        oob = oob[oob != 0]  # ego is never removed by the bounds check
        for i in oob:
            self._remove_agent(int(i))
        return oob

    # ------------------------------------------------------ collision attribution
    def _ego_aggressor_mask(self, others: np.ndarray) -> np.ndarray:
        """Per-``others`` mask: True where the ego is driving *toward* that agent.

        Credits only ego-caused contact / closing. The ego is the aggressor w.r.t.
        an agent when its own velocity has a positive component along the
        ego->other direction above ``EGO_AGGRESSOR_MIN_SPEED``; a stopped/slow ego,
        or one moving across or away from the other, is never at fault. Used by
        the crash-latch check so a car ramming a passive ego still stops the
        rollout but is not recorded as ego-fault.
        """
        dx = self.x[others] - self.x[0]
        dy = self.y[others] - self.y[0]
        dist = np.sqrt(dx * dx + dy * dy)
        ego_closing = np.zeros(len(others), dtype=np.float64)
        safe = dist > 1e-6
        ego_closing[safe] = (self.vx[0] * dx[safe] + self.vy[0] * dy[safe]) / dist[safe]
        return ego_closing > EGO_AGGRESSOR_MIN_SPEED
