"""The rollout: step generated scenes with one planner per role, fire hooks.

``RolloutRunner`` owns the simulation. It groups a batch of ``GeneratedScenes``
into numpy ``SimScene`` objects, partitions each scene's agents into three
roles, and steps them while firing externally supplied metric hooks -- the
caller decides WHAT to measure (``ddpo.reward`` for the adversarial reward,
``critical_scene.planner_matrix_eval`` for the planner benchmark), the runner
only decides WHEN.

The three roles are the axes every consumer measures along:

  * ``sut`` -- the system under test: the ego, local agent 0 of every scene;
  * ``adv`` -- THE generated adversary (``scenes.adv_local_idx``), if any;
  * ``env`` -- every other controlled agent (background traffic).

Each is driven by its own ``Planner``, composed per role from
``cfgs/planner/<name>.yaml``. The runner treats the three symmetrically: it
never special-cases which planner sits in which role, which is what lets the
SUT x traffic benchmark fill a table cell by config alone. The only asymmetry
in the whole rollout is that the hooks score the ego.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .hooks import (
    GenInvalidCheck,
    PathConflictCheck,
    RolloutContext,
    adv_local_indices,
)
from .planners import PlanItem, Planner, build_planner, parse_conditioning
from .scenes import GeneratedScenes
from .world import SimScene, load_sim_config, to_puffer_agent_types

ROLES = ("sut", "env", "adv")


@dataclass
class SimulatorConfig:
    """Rollout / metric-measurement parameters shared by every planner.

    These are the knobs the rollout needs while stepping to produce the metric
    set its consumer scores (the ``simulator:`` section of an entry config).
    Reward weights (ttc_tau, penalties, ...) stay in ``ddpo.reward`` and never
    reach the rollout. All fields are required so a missing config key fails
    loudly at construction.
    """

    sim_steps: int
    seed: int
    # Box-inflation margin (metres) for the adversary spawn-overlap check; 0
    # rejects only true interpenetration and allows bumper-to-bumper spawns.
    init_overlap_margin: float
    # Lane-centerline distances (m) above which the adversary's goal / spawn is
    # flagged off-lane by the diagnostic ``goal_offlane_frac`` metric.
    goal_offlane_threshold: float
    goal_onroad_threshold: float
    # The approach bonus ignores the first ``approach_warmup_time`` seconds when
    # measuring how much the adversary closed in on the ego.
    approach_warmup_time: float
    # Ego off-road proxy: the ego counts as off-road on a step when its centre is
    # farther than this from every lane centerline (the maps carry no ROAD_EDGE,
    # so the real off-road check never fires). Diagnostic only.
    ego_offroad_threshold: float
    # When set, the conditionally-generated adversary is rejected (hard -1) if its
    # realized labels violate the requested condition; see GenAgentInvalidHook.
    # ``None`` keeps the plain parked-adv gate (GenAgentParkingHook) only, which
    # is what scene sources without a generated adversary use. Injected by the
    # caller (thresholds come from the dataset config), never from yaml, so it is
    # spelled out at every call site rather than defaulted here.
    gen_invalid: GenInvalidCheck | None
    # Pre-rollout ego/adversary path-conflict screen (PathConflictHook). Unlike
    # gen_invalid this one IS spelled in yaml (``simulator.path_conflict``): it
    # has no dataset-derived thresholds, and with ``skip_rollout`` it changes how
    # much of the batch is stepped, which belongs next to sim_steps. ``None``
    # skips the check entirely -- the tiered reward requires it and will fail
    # loudly on the missing metric.
    path_conflict: PathConflictCheck | dict | None = None

    def __post_init__(self):
        if isinstance(self.path_conflict, dict):
            self.path_conflict = PathConflictCheck(**self.path_conflict)


@dataclass
class RolloutResult:
    """Output of a runner rollout consumed by the caller's scoring layer."""

    metrics: dict[str, np.ndarray]
    trajectories: list[dict[str, Any]] | None = None


class RolloutRunner:
    """Advance generated scenes with per-role planners and return metrics.

    The SimScene step loop, role partition and trajectory recording are fixed
    here; the metric hooks are injected per rollout by the caller.
    """

    def __init__(self, planner_cfg, params: SimulatorConfig, *, device: str | None = None):
        self.params = params
        # Simulation dynamics shared by the whole rollout (the cfgs/rollout
        # group, composed at <entry>.planner.sim) -- a property of the sim, not
        # of any single role's policy. Strict: must be present and complete.
        if "sim" not in planner_cfg:
            raise KeyError(
                "planner config must define 'sim' (the shared rollout dynamics); "
                "compose it via rollout@<...>.planner.sim: base in the entry config"
            )
        self.sim_cfg = load_sim_config(planner_cfg["sim"])
        self.rng = np.random.default_rng(int(params.seed))
        self.planners: dict[str, Planner] = {}
        self.conditioning: dict[str, dict | None] = {}
        for role in ROLES:
            if role not in planner_cfg:
                raise KeyError(
                    f"planner config must define '{role}' (per-role planners: {ROLES}); "
                    "compose it via planner@<...>.planner.<role>: <name> in the entry config"
                )
            role_cfg = planner_cfg[role]
            if "conditioning" not in role_cfg:
                raise KeyError(
                    f"planner config for role '{role}' must define 'conditioning' "
                    "(set conditioning: null if the policy ignores the conditioning obs)"
                )
            self.conditioning[role] = parse_conditioning(role, role_cfg["conditioning"])
            self.planners[role] = build_planner(role_cfg, role=role, device=device)

    # ------------------------------------------------------------------ build
    def _build_scenes(self, scenes: GeneratedScenes) -> list[SimScene]:
        """Group a batched ``GeneratedScenes`` into per-scene ``SimScene`` objects."""
        states = scenes.agent_states.detach().cpu().numpy()
        types = scenes.agent_types.detach().cpu().numpy()
        ptypes = to_puffer_agent_types(types)
        a_idx = scenes.agent_scene_idx.detach().cpu().numpy()
        lanes = scenes.lane_polylines
        if isinstance(lanes, torch.Tensor):
            lanes = lanes.detach().cpu().numpy()
        l_idx = scenes.meta["lane_scene_idx"]
        if isinstance(l_idx, torch.Tensor):
            l_idx = l_idx.detach().cpu().numpy()

        # Optional lane graph, one scene-local {"succ", "lateral"} dict per scene.
        # Only route-planning planners (idm) consume it; absent for every scene
        # source that ships geometry only, hence a genuine Optional rather than
        # a defaulted key.
        lane_graph = scenes.meta.get("lane_graph")

        sims = []
        for s in range(scenes.num_scenes):
            sim = SimScene(
                states[a_idx == s],
                ptypes[a_idx == s],
                lanes[l_idx == s],
                sim_cfg=self.sim_cfg,
            )
            if lane_graph is not None:
                sim.lane_graph = lane_graph[s]
            sims.append(sim)
        return sims

    def _apply_conditioning(
        self, sims: list[SimScene], role_ids: list[dict[str, np.ndarray]]
    ) -> None:
        """Fill each agent's conditioning obs slots from its role's planner cfg.

        The conditioned frozen policy reads the factors from each agent's own
        observation slot, so per-role values change only that role's driving
        style; rule-based planners ignore the slots (conditioning: null). E.g.
        the adv role can run the reckless ppo variant (collision_factor 0)
        so ONLY the adversary stops yielding -- two mutually avoidant agents
        almost never collide (8rlw8ay8: collision rate pinned at ~2% with
        adv_dist plateauing at ~7 m -- the planner's separation, not the
        model's choice).
        """
        for s, sim in enumerate(sims):
            for role, cond in self.conditioning.items():
                if cond is None:
                    continue
                ids = role_ids[s][role]
                if not len(ids):
                    continue
                for field, value in cond.items():
                    arr = getattr(sim, field)
                    if isinstance(value, tuple):
                        lo, hi = value
                        arr[ids] = self.rng.uniform(lo, hi, len(ids)).astype(np.float32)
                    else:
                        arr[ids] = np.float32(value)

    def _assign_roles(
        self, scenes: GeneratedScenes, sims: list[SimScene]
    ) -> list[dict[str, np.ndarray]]:
        """Static per-scene agent-id sets for each role.

        The sets partition ALL agent ids; per step they are intersected with the
        scene's current ``controlled`` (agents can retire mid-rollout).
        """
        adv = adv_local_indices(scenes, scenes.num_scenes)
        out = []
        for s, sim in enumerate(sims):
            a = int(adv[s])
            sut = np.array([0], dtype=np.int64)
            adv_ids = (
                np.array([a], dtype=np.int64) if a > 0 else np.empty(0, dtype=np.int64)
            )
            env = np.setdiff1d(np.arange(sim.n, dtype=np.int64), np.concatenate([sut, adv_ids]))
            out.append({"sut": sut, "env": env, "adv": adv_ids})
        return out

    # ------------------------------------------------- parallel hook points
    # The step loop below is shared verbatim with ``sim.parallel``, whose worker
    # subclass owns only a SHARD of the batch. Two decisions differ there and
    # nowhere else, so they are the only two things it overrides:
    #
    #   _should_stop  -- a shard running dry does not end the rollout; the
    #                    workers stop together when EVERY shard is dry.
    #   _stage_plans  -- a worker settles every centrally batched role in ONE
    #                    barrier round, and must join that round even when its
    #                    own shard contributes zero agents to a role.
    #
    # Keeping them as overrides (rather than forking the loop) is what makes the
    # parallel rollout bit-exact by construction: every other line of stepping,
    # planning and hook firing is the SAME code object in both paths.

    def _should_stop(self, active: list[int]) -> bool:
        """Single process: the rollout ends when no scene is still running."""
        return not active

    def _role_items(self, role, active, sims, role_ids) -> list[PlanItem]:
        """The (scene, agent ids) work units this role drives right now."""
        items: list[PlanItem] = []
        for s in active:
            ids = np.intersect1d(role_ids[s][role], sims[s].controlled)
            if len(ids):
                items.append((sims[s], ids))
        return items

    def _stage_plans(self, active, sims, role_ids) -> list:
        """Plan EVERY role from the same pre-step state; the caller applies after.

        Staged as one list so ``sim.parallel`` can settle all of its centrally
        batched roles in a single barrier round instead of one round per role --
        each round costs a full rendezvous plus the slowest shard's tail.
        """
        staged = []
        for role, planner in self.planners.items():
            items = self._role_items(role, active, sims, role_ids)
            if items:
                staged.append((planner, items, planner.plan(items)))
        return staged

    # --------------------------------------------------------------- rollout
    @torch.no_grad()
    def rollout(
        self,
        scenes: GeneratedScenes,
        *,
        hooks: list,
        record_trajectories: bool = False,
    ) -> RolloutResult:
        """Roll ``scenes`` out under ``hooks`` and return metrics / trajectories."""
        sims = self._build_scenes(scenes)
        role_ids = self._assign_roles(scenes, sims)
        self._apply_conditioning(sims, role_ids)
        m = len(sims)
        ctx = RolloutContext(
            scenes=scenes,
            sims=sims,
            finished=np.zeros(m, dtype=bool),
            metrics={
                "ego_collision": np.zeros(m, dtype=np.float32),
                "ego_offroad": np.zeros(m, dtype=np.float32),
                "init_invalid": np.zeros(m, dtype=np.float32),
                "init_overlap_frac": np.zeros(m, dtype=np.float32),
                "reached_goal": np.zeros(m, dtype=np.float32),
            },
            record_trajectories=record_trajectories,
        )

        for hook in hooks:
            hook.before_rollout(ctx)

        for t in range(self.params.sim_steps):
            ctx.t = t

            # Score current state before stepping, matching the original reward.
            for s, sim in enumerate(sims):
                if ctx.finished[s]:
                    continue
                for hook in hooks:
                    hook.before_step_scene(ctx, s, sim)

            active = [s for s in range(m) if not ctx.finished[s]]
            if self._should_stop(active):
                break

            # Two-phase advance: every role plans from the same pre-step state,
            # then all actions are applied (no role sees another role's
            # same-step movement).
            staged = self._stage_plans(active, sims, role_ids)
            for planner, items, plans in staged:
                planner.apply(items, plans)

            for s in active:
                sims[s].update_metrics()
                # Collision response: freeze the ego + any car it drove into so
                # boxes cannot pass through each other (kills the "rear-end and
                # re-emerge in front" TTC exploit at its source).
                sims[s].latch_ego_crash()
                ego_reached, _ = sims[s].goal_step()
                # Retire non-ego agents once their centre leaves the map square.
                # A no-op under goal_behavior='continue', which retires nobody;
                # the method owns that gate so every caller agrees.
                sims[s].remove_out_of_bounds()
                for hook in hooks:
                    hook.after_step_scene(ctx, s, sims[s], ego_reached=ego_reached)

        for hook in hooks:
            hook.after_rollout(ctx)

        return RolloutResult(metrics=ctx.metrics, trajectories=ctx.trajectories)
