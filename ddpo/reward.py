"""DDPO reward: roll generated scenes out with a planner, score the ego.

The rollout itself is delegated to a pluggable ``RolloutPlanner`` (see
``ddpo.planners``); this module only assembles the scalar reward from the
per-scene metrics the planner returns:

  * only the ego (scene agent 0) is scored;
  * the reward is assembled in three ordered branches -- park / invalid / valid:
      - park    (parked adversary): hard -1, not a critical scene;
      - invalid (adversary interpenetrates a neighbour at spawn): no criticality
        is credited, reward = -R_constraint <= 0;
      - valid:  ``clip(R_criticality - R_constraint, -1, 1)`` with
        R_criticality = risk_coef * noisy_OR(R_ttc, R_approach), where
        R_ttc = clip(1 - min_TTC/tau, 0, 1) (dense near-miss gradient) and
        R_approach only fires when the adversary both gets close AND actually
        closed in over the rollout (d0 - dmin), so spawning it next to the ego
        is not rewarded;
      - R_constraint = continuous lane-distance penalty + a graded init-overlap
        penalty ``init_overlap_penalty * frac`` (frac = intersection / adv area);
  * the criticality validity gate is hard (an invalid init earns zero
    criticality), but the overlap fraction still feeds the graded R_constraint
    penalty on every branch, so the policy keeps a smooth "separate the boxes"
    gradient;
  * collision does NOT enter the reward (a contact still surfaces implicitly via
    the dense TTC term); ``r_collision`` / ``c_trivial`` are logged diagnostics;
  * ``ego_offroad`` is kept for interface compatibility but is always 0: the
    generated maps carry no road edges (see pufferdrive_sim docstring);
  * a scene stops being stepped/scored once its ego reaches its goal;
  * ``evaluate`` also returns each reward component for diagnostics.

Which rollout policy runs (rule-based ``dummy``, frozen ``selfplay_drive`` net,
or PufferDrive's ``puffer_drive`` C env) is selected by ``planner_cfg`` /
``cfgs/planner/<name>.yaml``.
"""

from __future__ import annotations

import numpy as np
import torch

from .interfaces import GeneratedScenes
from .planners import RolloutParams, build_planner

# Legacy ``backend=`` values map onto planner names so existing callers keep
# working without a planner config.
_BACKEND_TO_PLANNER = {"numpy": "selfplay_drive", "puffer": "puffer_drive"}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable enough for the bounded inputs used here
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Hermite smoothstep: 0 at/below ``lo``, 1 at/above ``hi``, smooth between."""
    t = np.clip((np.asarray(x, dtype=np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


class PufferDriveReward:
    """Evaluate generated scenes by rolling them out with the configured planner.

    The reward object converts batched ``GeneratedScenes`` into rollout metrics
    via the planner and returns the collision reward plus goal validity
    penalties used by DDPO training.
    """

    def __init__(
        self,
        *,
        planner_cfg=None,
        sim_steps: int = 91,
        deterministic: bool | None = None,
        ttc_tau: float = 3.0,
        init_overlap_margin: float = 0.0,
        init_overlap_penalty: float = 0.5,
        init_overlap_gate_lo: float = 0.02,
        init_overlap_gate_hi: float = 0.20,
        goal_offlane_threshold: float = 3.0,
        goal_onroad_threshold: float = 2.0,
        goal_offlane_penalty: float = 0.25,
        parking_mismatch_penalty: float = 0.5,
        min_dist_coef: float = 0.0,
        min_dist_dmax: float = 20.0,
        gen_agent_parking_penalty: float = 0.0,
        risk_coef: float = 1.0,
        approach_d_safe: float = 6.0,
        approach_d_scale: float = 2.0,
        approach_close_delta: float = 2.0,
        approach_close_scale: float = 1.0,
        approach_warmup_time: float = 0.5,
        lane_soft: float = 0.5,
        collision_enabled: bool = False,
        collision_coef: float = 0.0,
        collision_warmup: float = 0.75,
        collision_window: float = 0.5,
        trivial_collision_t: float = 0.75,
        trivial_collision_penalty: float = 0.5,
        seed: int = 0,
        backend: str = "numpy",
        pufferdrive_root: str | None = None,
        gen_invalid=None,
    ):
        """Initialize planner-backed reward evaluation.

        Args:
            planner_cfg: Mapping (OmegaConf node or dict) selecting the rollout
                planner, e.g. ``{"name": "dummy"}``. When ``None``, a planner is
                synthesised from the legacy ``backend`` / ``deterministic`` /
                ``pufferdrive_root`` arguments (``numpy`` -> ``selfplay_drive``,
                ``puffer`` -> ``puffer_drive``).
            sim_steps: Maximum number of simulator steps per scene.
            deterministic: Whether planner actions should be deterministic. If
                ``None``, use the planner config default. Ignored when
                ``planner_cfg`` is provided (set it there instead).
            ttc_tau: Time-to-collision horizon (seconds) normalising the dense
                criticality reward ``clip(1 - min_TTC/ttc_tau, 0, 1)``.
            init_overlap_margin: Box-inflation margin (metres) for the t=0
                vehicle-overlap (init_invalid) check; 0 rejects only true overlap
                and allows bumper-to-bumper traffic-jam spawns.
            goal_offlane_threshold: Lane-centerline distance in meters above
                which a moving car's goal is considered off-lane.
            goal_onroad_threshold: Lane-centerline distance in meters above which
                a moving car's *spawn* is considered off-lane (folded into the
                same off-lane penalty as the goal; no longer an exemption gate).
            goal_offlane_penalty: Penalty scale applied to the off-lane
                fraction (moving cars off-lane at spawn or goal) for each scene.
            parking_mismatch_penalty: Penalty scale applied when generated
                parking/static state disagrees with ``meta["gt_parking_mask"]``.
            seed: RNG seed passed into the planner / simulator scenes.
            backend: Legacy rollout backend selector used only when
                ``planner_cfg`` is ``None``.
            pufferdrive_root: Optional path to the PufferDrive checkout for the
                ``puffer_drive`` planner. Defaults to ``<repo>/PufferDrive``.
        """
        self.ttc_tau = float(ttc_tau)
        self.goal_offlane_penalty = float(goal_offlane_penalty)
        # Init-overlap shaping. The overlap fraction (intersection / adv area)
        # feeds a graded penalty: constraint += init_overlap_penalty * frac. An
        # *invalid* init (adversary interpenetrating a neighbour, flagged by the
        # init_invalid metric) additionally hard-gates out criticality, so overlap
        # can never manufacture a fake near-miss / collision reward.
        self.init_overlap_penalty = float(init_overlap_penalty)
        # DEPRECATED: the smoothstep criticality gate was replaced by the hard
        # init_invalid gate; these are kept for config back-compat but unused.
        self.init_overlap_gate_lo = float(init_overlap_gate_lo)
        self.init_overlap_gate_hi = float(init_overlap_gate_hi)
        # DEPRECATED: parking is now a hard -1 (a parked adversary is rejected
        # outright), so the gt-mismatch fraction penalty is no longer applied.
        self.parking_mismatch_penalty = float(parking_mismatch_penalty)
        # DEPRECATED (Phase 2): the old linear proximity bonus
        # ``min_dist_coef * clip(1 - ego_adv_min_dist/min_dist_dmax, 0, 1)`` is
        # superseded by the gated approach bonus below. Still accepted so legacy
        # callers/configs do not break, but no longer drives the reward.
        self.min_dist_coef = float(min_dist_coef)
        self.min_dist_dmax = max(float(min_dist_dmax), 1e-6)
        # Penalise a generated adversary that is parked (goal within
        # MIN_DISTANCE_TO_GOAL of spawn) to force it to drive.
        self.gen_agent_parking_penalty = float(gen_agent_parking_penalty)

        # --- Phase 2 shaping params ------------------------------------------
        # Risk = noisy-OR of dense TTC and a *gated approach* bonus. The approach
        # bonus only fires when the adversary both gets close (dmin < d_safe) AND
        # actually closed in during the rollout (d0 - dmin > close_delta), so the
        # policy cannot farm it by spawning the adversary next to the ego.
        self.risk_coef = float(risk_coef)
        self.approach_d_safe = float(approach_d_safe)
        self.approach_d_scale = max(float(approach_d_scale), 1e-6)
        self.approach_close_delta = float(approach_close_delta)
        self.approach_close_scale = max(float(approach_close_scale), 1e-6)
        # Continuous lane penalty ramps from lane_soft to goal_offlane_threshold
        # (replaces the binary off-lane fraction, which is {0,1} for one agent).
        self.lane_soft = float(lane_soft)
        self.goal_offlane_threshold = float(goal_offlane_threshold)
        # Collision is an extra *bonus* gated on collision time, not an override:
        # late collisions ramp up, trivial early collisions (< trivial_collision_t)
        # incur a penalty instead. Disabled by default (collision_enabled=False).
        self.collision_enabled = bool(collision_enabled)
        self.collision_coef = float(collision_coef)
        self.collision_warmup = float(collision_warmup)
        self.collision_window = max(float(collision_window), 1e-6)
        self.trivial_collision_t = float(trivial_collision_t)
        self.trivial_collision_penalty = float(trivial_collision_penalty)

        if planner_cfg is None:
            name = _BACKEND_TO_PLANNER.get(backend, backend)
            planner_cfg = {
                "name": name,
                "deterministic": deterministic,
                "pufferdrive_root": pufferdrive_root,
            }
        params = RolloutParams(
            sim_steps=int(sim_steps),
            seed=int(seed),
            init_overlap_margin=float(init_overlap_margin),
            goal_offlane_threshold=float(goal_offlane_threshold),
            goal_onroad_threshold=float(goal_onroad_threshold),
            pufferdrive_root=pufferdrive_root,
            collision_enabled=bool(collision_enabled),
            approach_warmup_time=float(approach_warmup_time),
            gen_invalid=gen_invalid,
        )
        self.planner = build_planner(planner_cfg, params)
        # Back-compat: expose the native C backend for callers that poke at it
        # directly (e.g. scripts/benchmark_ddpo_rollout_backends.py).
        self.native_backend = getattr(self.planner, "backend", None)

    def _assemble_reward(self, m: dict) -> tuple[np.ndarray, dict]:
        """Compose the per-scene reward from rollout metrics.

        Three mutually exclusive branches, checked in order:

          * park    (gen_agent_is_parked > 0): hard -1 -- a parked adversary
            is not a critical scene.
          * invalid (init_invalid: adversary interpenetrates a neighbour at
            spawn): no criticality is credited, so reward = -R_constraint <= 0 --
            an overlapping init can never manufacture a fake near-miss.
          * valid:  ``reward = clip(R_criticality - R_constraint, -1, 1)``, with
              R_criticality = risk_coef * [1 - (1-R_ttc)(1-R_approach)]  in [0, 1]
              R_constraint  = goal_offlane_penalty * c_lane
                              + init_overlap_penalty * init_overlap_frac

        The overlap fraction still feeds the graded ``R_constraint`` penalty on
        every branch, so the policy keeps a smooth "separate the boxes" gradient
        even though the criticality validity gate is now hard. Collision does not
        enter the reward (a contact still surfaces via the dense TTC term). All
        components are returned so the training loop can log / normalise them.
        """
        n = len(m["init_invalid"])
        zeros = np.zeros(n, dtype=np.float32)

        # --- criticality ----------------------------------------------------
        r_ttc = np.clip(1.0 - m["ego_min_ttc"] / self.ttc_tau, 0.0, 1.0).astype(np.float32)

        d0 = m.get("ego_adv_init_dist")
        dmin = m.get("ego_adv_min_dist_warmup")
        if dmin is None:
            dmin = m.get("ego_adv_min_dist")
        if d0 is not None and dmin is not None:
            prox = _sigmoid((self.approach_d_safe - dmin) / self.approach_d_scale)
            closing = _sigmoid(
                (d0 - dmin - self.approach_close_delta) / self.approach_close_scale
            )
            r_approach = (prox * closing).astype(np.float32)
            finite = np.isfinite(d0) & np.isfinite(dmin)
            r_approach = np.where(finite, r_approach, 0.0).astype(np.float32)
        else:
            r_approach = zeros.copy()

        r_risk = (1.0 - (1.0 - r_ttc) * (1.0 - r_approach)).astype(np.float32)
        # Raw criticality (positive term, [0, risk_coef]); hard-gated to 0 for
        # parked / invalid scenes when the reward is assembled below.
        criticality_raw = (self.risk_coef * r_risk).astype(np.float32)

        # Collision is not rewarded; r_collision / c_trivial are kept purely as
        # logged diagnostics (a contact still surfaces via the dense TTC term).
        collision = m["ego_collision"].astype(np.float32)
        ctime = m.get("ego_collision_time")
        if ctime is None:
            ctime = np.full(n, np.inf, dtype=np.float32)
        r_collision = (
            collision
            * _smoothstep(ctime, self.collision_warmup, self.collision_warmup + self.collision_window)
        ).astype(np.float32)
        c_trivial = (collision * (ctime < self.trivial_collision_t)).astype(np.float32)

        # --- constraints (>= 0, subtracted on every branch) -----------------
        # Continuous adversary overlap fraction (intersection / adversary area)
        # vs neighbours at spawn feeds the graded overlap penalty.
        init_overlap_frac = np.asarray(m.get("init_overlap_frac", zeros), dtype=np.float32)
        c_overlap = init_overlap_frac
        
        spawn_lane_d = np.asarray(m.get("spawn_lane_dist", zeros), dtype=np.float32)
        c_spawn_lane = _smoothstep(spawn_lane_d, self.lane_soft, self.goal_offlane_threshold)
        
        goal_lane_d = np.asarray(m.get("goal_lane_dist", zeros), dtype=np.float32)
        c_goal_lane = _smoothstep(goal_lane_d, self.lane_soft, self.goal_offlane_threshold)
        
        constraint = (
            self.goal_offlane_penalty * c_spawn_lane
            + self.goal_offlane_penalty * c_goal_lane
            + self.init_overlap_penalty * c_overlap
        ).astype(np.float32)

        # --- assemble: reject -> init_invalid -> valid ----------------------
        # reject:       a parked / condition-violating adversary is not a critical
        #               scene -> hard -1. The condition-violation metric
        #               (``gen_agent_is_invalid``, from RewardHookGenAgentInvalid)
        #               supersedes the plain parked-adv gate when present: it
        #               already rejects a parked adversary whenever the target is
        #               motion=moving, and correctly ACCEPTS one when the target is
        #               motion=parked (which the raw parking gate would misflag).
        # init_invalid: an interpenetrating init can never earn criticality, so the
        #               reward is just the (negative) constraint (always <= 0).
        # valid:        reward = clip(criticality - constraint, -1, 1).
        c_parking = np.asarray(m.get("gen_agent_is_parked", zeros), dtype=np.float32)
        c_invalid = m.get("gen_agent_is_invalid")
        if c_invalid is not None:
            c_reject = np.asarray(c_invalid, dtype=np.float32)
        else:
            c_reject = c_parking
        reject = c_reject > 0
        init_invalid = constraint > 0.0
        valid = ~(reject | init_invalid)

        # Hard validity gate: criticality is only credited on valid scenes.
        criticality = np.where(valid, criticality_raw, 0.0).astype(np.float32)
        total = np.select(
            [reject, init_invalid],
            [
                np.full(n, -1.0, dtype=np.float32),
                np.clip(-constraint, -1.0, 0.0),
            ],
            default=np.clip(criticality_raw, 0.0, 1.0),
        ).astype(np.float32)

        components = {
            "r_ttc": r_ttc,
            "r_approach": r_approach,
            "r_risk": r_risk,
            "r_collision": r_collision,
            "criticality": criticality,
            "c_lane": c_spawn_lane + c_goal_lane,
            "c_spawn_lane": c_spawn_lane,
            "c_goal_lane": c_goal_lane,
            "c_parking": c_parking,
            "c_invalid": c_reject,
            "c_invalid_reason": m.get("gen_agent_invalid_reason", np.full(n, "", dtype=object)),
            "c_trivial": c_trivial,
            "c_overlap": c_overlap,
            "constraint": constraint,
        }
        return total, components

    # --------------------------------------------------------------- evaluate
    @torch.no_grad()
    def evaluate(
        self, scenes: GeneratedScenes, record_trajectories: bool = False
    ) -> dict:
        """Roll out generated scenes and return reward metrics.

        Args:
            scenes: Batched generated scenes containing agent states, types,
                agent-to-scene indices, lane polylines, and lane-to-scene
                metadata.
            record_trajectories: If true, include per-scene trajectory arrays
                suitable for rollout visualization.

        Returns:
            Dictionary of per-scene numpy arrays for reward, collision, offroad,
            initial invalid state, goal completion, goal off-lane fraction, and
            parking mismatch fraction. When ``record_trajectories`` is true, the
            dictionary also contains a ``trajectories`` list.
        """
        result = self.planner.rollout(scenes, record_trajectories=record_trajectories)
        metrics = result.metrics
        trajectories = result.trajectories

        rewards, components = self._assemble_reward(metrics)

        n = scenes.num_scenes
        inf = np.full(n, np.inf, dtype=np.float32)
        zeros = np.zeros(n, dtype=np.float32)
        out = {
            "reward": rewards,
            "ego_collision": metrics["ego_collision"],
            "ego_fault_collision": metrics.get("ego_fault_collision", zeros.copy()),
            "ego_collision_time": metrics.get("ego_collision_time", inf.copy()),
            "ego_min_ttc": metrics["ego_min_ttc"],
            "ego_offroad": metrics["ego_offroad"],
            "init_invalid": metrics["init_invalid"],
            "init_overlap_frac": metrics.get("init_overlap_frac", zeros.copy()),
            "reached_goal": metrics["reached_goal"],
            "goal_offlane_frac": metrics["goal_offlane_frac"],
            "goal_lane_dist": metrics.get("goal_lane_dist", zeros.copy()),
            "spawn_lane_dist": metrics.get("spawn_lane_dist", zeros.copy()),
            "parking_mismatch_frac": metrics["parking_mismatch_frac"],
            "ego_adv_min_dist": metrics.get("ego_adv_min_dist", inf.copy()),
            "ego_adv_init_dist": metrics.get("ego_adv_init_dist", inf.copy()),
            "ego_adv_min_dist_warmup": metrics.get("ego_adv_min_dist_warmup", inf.copy()),
            "gen_agent_is_parked": metrics.get("gen_agent_is_parked", zeros.copy()),
            "gen_agent_is_invalid": metrics.get("gen_agent_is_invalid", zeros.copy()),
        }
        # Per-component reward arrays (r_ttc, r_approach, r_risk, criticality,
        # c_lane, constraint, ...) for diagnostics / future per-component
        # advantage normalisation.
        out.update(components)
        if trajectories is not None:
            out["trajectories"] = trajectories
        return out
