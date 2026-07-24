"""Per-role planners + rollout runner for DDPO reward evaluation.

``RolloutRunner`` owns the simulation: it groups a batch of ``GeneratedScenes``
into numpy ``SimScene`` objects, partitions each scene's agents into three
roles, and steps the scenes while firing externally supplied metric hooks (the
reward layer decides WHAT to measure; see ``ddpo.reward``):

  * ``sut`` -- the system under test: the ego, local agent 0 of every scene;
  * ``adv`` -- THE generated adversary (``scenes.adv_local_idx``), if any;
  * ``env`` -- every other controlled agent (real background traffic).

Each role is driven by its own ``Planner``, selected per role in
``cfgs/ddpo/<flow>.yaml`` (``planner.sut`` / ``planner.env`` / ``planner.adv``,
each composed from ``cfgs/planner/<name>.yaml``). The only planner today is
``bad_driver``, so all three roles default to it.

A ``Planner`` is a two-phase policy over an agent subset: ``plan`` decides
actions from the current world state WITHOUT mutating it, ``apply`` integrates
them. The runner plans every role first and applies afterwards, so no role
observes another role's same-step movement -- with identical role configs this
reproduces the old single-planner batched semantics exactly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from ..interfaces import GeneratedScenes
from ..pufferdrive_sim import TYPE_CYCLIST, TYPE_VEHICLE, SimScene, load_sim_config
from ..reward_hooks import GenInvalidCheck, RolloutContext, adv_local_indices

ROLES = ("sut", "env", "adv")

# Per-agent conditioning obs slots of the frozen PufferDrive policy family
# (trailing ego features; see SimScene). The values are policy inputs -- how
# defensively THIS role's policy drives -- so they live in each role's planner
# yaml (``conditioning:``), not in the shared rollout dynamics.
CONDITIONING_FIELDS = ("collision_factor", "offroad_factor", "lane_width")


def _parse_conditioning(role: str, cfg):
    """Validate one role's ``conditioning`` node.

    Returns ``None`` (``conditioning: null``: the policy ignores the
    conditioning obs, its agents' slots stay 0) or a dict mapping each field to
    a float (fixed value) or a (lo, hi) pair (per-agent uniform sample).
    Strict: exactly the CONDITIONING_FIELDS keys.
    """
    if cfg is None:
        return None
    m = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
    unknown = sorted(set(m) - set(CONDITIONING_FIELDS))
    missing = sorted(set(CONDITIONING_FIELDS) - set(m))
    if unknown or missing:
        raise ValueError(
            f"planner.{role}.conditioning: unknown keys {unknown}, missing keys {missing}; "
            f"expected exactly {list(CONDITIONING_FIELDS)}"
        )
    out = {}
    for key in CONDITIONING_FIELDS:
        v = m[key]
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(
                    f"planner.{role}.conditioning.{key}: a range must be [lo, hi], got {v!r}"
                )
            out[key] = (float(v[0]), float(v[1]))
        else:
            out[key] = float(v)
    return out

# One plan/apply work unit: the scene sim + the agent ids this role currently
# drives in it (a subset of ``sim.controlled``).
PlanItem = tuple[SimScene, np.ndarray]


def to_puffer_agent_types(agent_types) -> np.ndarray:
    """Convert dataset/model ids (0 veh, 1 ped, 2 cyc) to PufferDrive ids.

    ``GeneratedScenes.agent_types`` deliberately keeps the model-side convention.
    PufferDrive observations and collision logic use entity ids 1..3, so planners
    convert at the boundary before constructing a sim scene or planner metrics.
    """
    return (
        np.asarray(agent_types, dtype=np.int64) + 1
    ).clip(TYPE_VEHICLE, TYPE_CYCLIST)


@dataclass
class SimulatorConfig:
    """Rollout / metric-measurement parameters shared by every planner.

    These are the knobs the rollout needs while stepping to produce the metric
    set the reward consumes (``cfgs/ddpo/<flow>.yaml`` ``simulator:`` section).
    Reward weights (ttc_tau, penalties, ...) stay in ``ddpo.reward`` and never
    reach the planner. All fields are required so a missing config key fails
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
    # When set, the conditionally-generated adversary is rejected (hard -1) if its
    # realized labels violate the requested condition; see RewardHookGenAgentInvalid.
    # None keeps the plain parked-adv gate (RewardHookGenAgentParking) only.
    # Injected by the caller (thresholds come from the dataset config), not yaml.
    gen_invalid: GenInvalidCheck | None = None
    # Ego off-road proxy: the ego counts as off-road on a step when its centre is
    # farther than this from every lane centerline (the maps carry no ROAD_EDGE,
    # so the real off-road check never fires). Diagnostic only; has a default so
    # existing SimulatorConfig call sites keep working.
    ego_offroad_threshold: float = 2.75


@dataclass
class RolloutResult:
    """Output of a runner rollout consumed by the reward."""

    metrics: dict[str, np.ndarray]
    trajectories: list[dict[str, Any]] | None = None


class Planner(ABC):
    """Two-phase action policy for one role's agents.

    ``plan`` must not mutate world state (planner-internal state such as an
    LSTM carry is fine): the runner calls every role's ``plan`` from the same
    pre-step state before any ``apply`` runs.
    """

    def __init__(self, planner_cfg, params: SimulatorConfig, *, role: str, device: str | None = None):
        self.params = params
        self.role = role

    @abstractmethod
    def plan(self, items: Sequence[PlanItem]) -> list:
        """Decide actions for every ``(sim, agent_ids)`` item; return them aligned."""

    @abstractmethod
    def apply(self, items: Sequence[PlanItem], plans: list) -> None:
        """Integrate the actions previously returned by ``plan`` (in place)."""


class RolloutRunner:
    """Advance generated scenes with per-role planners and return metrics.

    The SimScene step loop, role partition and trajectory recording are fixed
    here; the metric hooks are injected per rollout by the caller (the reward
    layer owns the metric set).
    """

    def __init__(self, planner_cfg, params: SimulatorConfig, *, device: str | None = None):
        self.params = params
        # Simulation dynamics / conditioning shared by the whole rollout
        # (the cfgs/rollout group, composed at ddpo.planner.sim) -- a property
        # of the sim, not of any single role's policy. Strict: must be present
        # and complete.
        self.sim_cfg = load_sim_config(planner_cfg["sim"])
        self.rng = np.random.default_rng(int(params.seed))
        self.planners: dict[str, Planner] = {}
        self.conditioning: dict[str, dict | None] = {}
        for role in ROLES:
            role_cfg = planner_cfg.get(role)
            if role_cfg is None:
                raise ValueError(
                    f"planner config must define '{role}' (per-role planners: {ROLES}); "
                    "compose it via planner@ddpo.planner.<role>: <name> in the entry config"
                )
            if "conditioning" not in role_cfg:
                raise ValueError(
                    f"planner config for role '{role}' must define 'conditioning' "
                    "(set conditioning: null if the policy ignores the conditioning obs)"
                )
            self.conditioning[role] = _parse_conditioning(role, role_cfg.get("conditioning"))
            self.planners[role] = build_planner(role_cfg, params, role=role, device=device)

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

        sims = []
        for s in range(scenes.num_scenes):
            sims.append(
                SimScene(
                    states[a_idx == s],
                    ptypes[a_idx == s],
                    lanes[l_idx == s],
                    sim_cfg=self.sim_cfg,
                )
            )
        return sims

    def _apply_conditioning(
        self, sims: list[SimScene], role_ids: list[dict[str, np.ndarray]]
    ) -> None:
        """Fill each agent's conditioning obs slots from its role's planner cfg.

        The conditioned frozen policy reads the factors from each agent's own
        observation slot, so per-role values change only that role's driving
        style; rule-based planners ignore the slots (conditioning: null). E.g.
        the adv role runs the reckless bad_driver variant (collision_factor 0)
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
            if not active:
                break

            # Two-phase advance: every role plans from the same pre-step state,
            # then all actions are applied (no role sees another role's
            # same-step movement).
            staged = []
            for role, planner in self.planners.items():
                items: list[PlanItem] = []
                for s in active:
                    ids = np.intersect1d(role_ids[s][role], sims[s].controlled)
                    if len(ids):
                        items.append((sims[s], ids))
                if items:
                    staged.append((planner, items, planner.plan(items)))
            for planner, items, plans in staged:
                planner.apply(items, plans)

            for s in active:
                sims[s].update_metrics()
                # Collision response: freeze the ego + any car it drove into so
                # boxes cannot pass through each other (kills the "rear-end and
                # re-emerge in front" TTC exploit at its source).
                sims[s].latch_ego_crash()
                ego_reached, _ = sims[s].goal_step()
                # goal_behavior='continue' lets non-ego agents drive past their
                # goal; retire them once their centre leaves the map square.
                sims[s].remove_out_of_bounds()
                for hook in hooks:
                    hook.after_step_scene(ctx, s, sims[s], ego_reached=ego_reached)

        for hook in hooks:
            hook.after_rollout(ctx)

        return RolloutResult(metrics=ctx.metrics, trajectories=ctx.trajectories)


# ---------------------------------------------------------------- factory
def build_planner(
    planner_cfg, params: SimulatorConfig, *, role: str, device: str | None = None
) -> Planner:
    """Instantiate the planner named by ``planner_cfg['name']`` for one role.

    ``planner_cfg`` is any mapping with a ``name`` key (an OmegaConf node from
    ``cfgs/planner/<name>.yaml`` or a plain dict). ``bad_driver`` is the only
    planner; the explicit check keeps stale configs failing loudly.
    """
    from .bad_driver import BadDriverPlanner

    name = planner_cfg.get("name")
    if str(name) != "bad_driver":
        raise ValueError(f"unknown planner {name!r}; only 'bad_driver' is available")
    return BadDriverPlanner(planner_cfg, params, role=role, device=device)
