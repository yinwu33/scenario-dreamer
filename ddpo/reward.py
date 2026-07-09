"""DDPO reward: roll generated scenes out with a planner, score the ego.

``PufferSimulator`` is configured by three separate configs, each owning one
concern (see ``cfgs/ddpo/ldm_adv.yaml``):

  * ``planner_cfg``    -- WHICH policy drives the agents (``cfgs/planner/<name>.yaml``:
    name, checkpoint, net arch, determinism, sim dynamics). Swapping planner
    weights or models means editing/adding a planner yaml, nothing here.
  * ``simulator_cfg``  -- HOW the rollout measures metrics while stepping
    (``ddpo.planners.SimulatorConfig``: sim_steps, seed, overlap margin, lane
    thresholds, approach warmup, optional condition-violation check).
  * ``reward_cfg``     -- HOW the scalar reward is assembled from those metrics
    (``RewardConfig`` below: weights and ramps only).

Configs are strict: every field is required and unknown keys raise, so a typo
or a stale yaml fails at construction instead of silently using a default.

The rollout is delegated to the pluggable ``RolloutPlanner`` selected by
``planner_cfg`` (see ``ddpo.planners``); this module only assembles the scalar
reward from the per-scene metrics the planner returns:

  * only the ego (scene agent 0) is scored;
  * the reward is assembled in three ordered branches -- reject / init_invalid /
    valid:
      - reject  (adversary violates its condition, or is parked when no
        condition check is configured): not a critical scene; graded
        -(base + (1-base) * gap/scale) down to -1 when ``invalid_grade_scale``
        is set (flat -1 otherwise, but a flat cliff gives GRPO no
        within-group contrast once a group is mostly rejected);
      - init_invalid (adversary interpenetrates a neighbour at spawn): no
        criticality is credited, reward = -R_constraint <= 0;
      - valid:  ``clip(R_criticality - R_constraint, -1, 1) + R_bonus`` with
        R_criticality = risk_coef * noisy_OR(R_ttc, w_app * R_approach), where
        R_ttc = clip(1 - min_TTC/tau, 0, 1) (dense near-miss gradient) and
        R_approach only fires when the adversary both gets close AND actually
        closed in over the rollout (d0 - dmin), so spawning it next to the ego
        is not rewarded. ``w_app`` is the annealable approach weight (see
        ``approach_coef*``): the approach term is a bootstrap gradient that
        would otherwise substitute for the sparse TTC/collision signal forever;
      - R_constraint = continuous lane-distance penalty + a graded init-overlap
        penalty ``init_overlap_penalty * frac`` (frac = intersection / adv area),
        subtracted softly on the valid branch (an off-lane spawn loses reward
        but keeps its criticality gradient);
  * only true spawn interpenetration (overlap frac > 0) hard-gates criticality
    to zero; the overlap fraction also feeds the graded R_constraint penalty on
    every branch, so the policy keeps a smooth "separate the boxes" gradient;
  * collision enters the reward only through the opt-in ``R_bonus``:
    ``r_collision * (collision_bonus + ego_fault_bonus * ego_fault)``, where
    ``r_collision`` is time-ramped (a trivial early contact earns nothing) and
    the bonus is only paid on the valid branch. Sized >> the GRPO within-group
    reward std, it makes a real collision decisively win its group where the
    dense-TTC-only reward left it ~0.1 above a deep near-miss (invisible under
    per-group whitening);
  * ``ego_offroad`` is kept for interface compatibility but is always 0: the
    generated maps carry no road edges (see pufferdrive_sim docstring);
  * a scene stops being stepped/scored once its ego reaches its goal;
  * ``evaluate`` also returns each reward component for diagnostics.

Metric access is strict: a planner must emit every metric the reward consumes
(the numpy ``SimScene`` planners always do, via the hooks in
``ddpo.reward_hooks``); a missing key raises instead of silently zeroing a term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .interfaces import GeneratedScenes
from .planners import SimulatorConfig, build_planner


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable enough for the bounded inputs used here
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Hermite smoothstep: 0 at/below ``lo``, 1 at/above ``hi``, smooth between."""
    t = np.clip((np.asarray(x, dtype=np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


@dataclass
class RewardConfig:
    """Scalar-assembly weights (``reward:`` section of the ddpo config).

    Pure post-rollout knobs: nothing here reaches the planner or the metric
    hooks. All fields are required -- a missing yaml key raises at construction.
    """

    # --- criticality ------------------------------------------------------
    # Time-to-collision horizon (s) normalising the dense criticality reward
    # ``clip(1 - min_TTC/ttc_tau, 0, 1)``.
    ttc_tau: float
    # Weight on the noisy-OR risk term (the positive branch of the reward).
    risk_coef: float
    # Gated approach bonus: fires when the adversary gets close (dmin below
    # ``approach_d_safe``, ramp sharpness ``approach_d_scale``) AND closed in
    # over the rollout (d0 - dmin above ``approach_close_delta``, sharpness
    # ``approach_close_scale``), so spawning next to the ego scores nothing.
    approach_d_safe: float
    approach_d_scale: float
    approach_close_delta: float
    approach_close_scale: float
    # --- constraints (>= 0, subtracted on every branch) ---------------------
    # Continuous lane penalty: smoothstep ramp of the adversary's spawn/goal
    # lane-centerline distance from ``lane_soft`` (m) to ``lane_hard`` (m),
    # each weighted by ``lane_penalty``.
    lane_soft: float
    lane_hard: float
    lane_penalty: float
    # Weight on the graded adversary spawn-overlap fraction (intersection area /
    # adversary area vs its neighbours).
    init_overlap_penalty: float
    # --- collision bonus (valid branch only) ---------------------------------
    # r_collision ramps up over [collision_warmup, collision_warmup +
    # collision_window] (s); c_trivial flags collisions earlier than
    # ``trivial_collision_t`` (s). With collision_bonus/ego_fault_bonus at 0
    # these stay logged diagnostics; otherwise they gate the bonus below.
    collision_warmup: float
    collision_window: float
    trivial_collision_t: float
    # --- graded condition-violation (reject) penalty -------------------------
    # A rejected adversary is penalised on a ramp instead of a flat -1:
    #   reward = -(invalid_penalty_base
    #              + (1 - invalid_penalty_base) * clip(gap / invalid_grade_scale, 0, 1))
    # where ``gap`` is the metric distance (m) to the nearest valid bucket
    # (``gen_agent_invalid_gap``; inf for a categorical type violation). A flat
    # cliff starves GRPO's within-group whitening of contrast once most of a
    # group is rejected (uniform -1 -> zero std -> zero gradient), so nothing
    # pulls the policy back over the boundary; the ramp keeps a "how far past
    # the boundary" ordering all the way down. invalid_grade_scale is the
    # distance (m) at which the penalty saturates at the full -1; 0 disables
    # grading (legacy flat -1). Defaulted (unlike the fields above) so configs
    # for flows without the condition-violation gate keep working unchanged.
    invalid_penalty_base: float = 0.5
    invalid_grade_scale: float = 0.0
    # --- explicit collision / ego-fault bonus (0 = legacy, no bonus) ---------
    # Paid on the valid branch only, scaled by the time-ramped ``r_collision``
    # (so a spawn-on-top contact inside the warmup earns nothing):
    #   R_bonus = r_collision * (collision_bonus + ego_fault_bonus * ego_fault)
    # Size against GRPO group noise: with within-group reward std ~0.2, a bonus
    # of 0.5 puts a collision ~2.5 sigma above its near-miss siblings; the dense
    # terms alone leave that margin at ~0.1 (0.5 sigma, i.e. invisible).
    collision_bonus: float = 0.0
    ego_fault_bonus: float = 0.0
    # --- approach-weight annealing (bootstrap -> sparse signal hand-off) -----
    # Effective risk term: noisy_OR(r_ttc, w_app * r_approach) with w_app moving
    # linearly from ``approach_coef`` to ``approach_coef_final`` over train
    # iterations [approach_anneal_begin, approach_anneal_end] (end <= begin
    # disables annealing; w_app then stays at approach_coef). The approach term
    # is dense and trivially farmable (spawn far, drive to ~7 m, planner keeps
    # separation), so left at full weight it absorbs the whole gradient; fully
    # removing it would leave only near-zero-support TTC/collision signal.
    approach_coef: float = 1.0
    approach_coef_final: float = 1.0
    approach_anneal_begin: int = 0
    approach_anneal_end: int = 0

    def __post_init__(self):
        self.approach_d_scale = max(float(self.approach_d_scale), 1e-6)
        self.approach_close_scale = max(float(self.approach_close_scale), 1e-6)
        self.collision_window = max(float(self.collision_window), 1e-6)


class PufferSimulator:
    """Evaluate generated scenes: planner rollout + scalar reward assembly.

    ``evaluate`` converts batched ``GeneratedScenes`` into per-scene rollout
    metrics via the configured planner and assembles the DDPO training reward
    plus per-component diagnostics.
    """

    def __init__(
        self,
        planner_cfg,
        simulator_cfg: SimulatorConfig,
        reward_cfg: RewardConfig,
    ):
        """Args:
            planner_cfg: Mapping (OmegaConf node or dict) selecting the rollout
                planner by ``name`` plus its own settings, from
                ``cfgs/planner/<name>.yaml``.
            simulator_cfg: Rollout / metric-measurement parameters shared by
                every planner.
            reward_cfg: Scalar-assembly weights (this module only).
        """
        self.cfg = reward_cfg
        # Reject gate source: with a condition-violation check configured, the
        # realized-vs-target metric supersedes the plain parked-adv gate (it
        # already rejects a parked adversary whenever the target is
        # motion=moving, and correctly ACCEPTS one when the target is
        # motion=parked, which the raw parking gate would misflag).
        self.gen_invalid_enabled = simulator_cfg.gen_invalid is not None
        self.planner = build_planner(planner_cfg, simulator_cfg)
        self._approach_coef = float(reward_cfg.approach_coef)

    @property
    def approach_coef(self) -> float:
        """Current (possibly annealed) weight on the approach term."""
        return self._approach_coef

    def set_train_iteration(self, it: int) -> None:
        """Advance the approach-weight annealing schedule to iteration ``it``.

        Called by the training loop once per iteration (eval calls between
        iterations reuse the latest weight, so train and eval rewards stay
        comparable). A no-op unless the config enables annealing.
        """
        cfg = self.cfg
        if cfg.approach_anneal_end <= cfg.approach_anneal_begin:
            self._approach_coef = float(cfg.approach_coef)
            return
        t = (it - cfg.approach_anneal_begin) / (
            cfg.approach_anneal_end - cfg.approach_anneal_begin
        )
        t = min(max(t, 0.0), 1.0)
        self._approach_coef = float(
            cfg.approach_coef + t * (cfg.approach_coef_final - cfg.approach_coef)
        )

    def _assemble_reward(self, m: dict) -> tuple[np.ndarray, dict]:
        """Compose the per-scene reward from rollout metrics.

        Three mutually exclusive branches, checked in order:

          * reject  (condition violation, or parked adversary when no condition
            check is configured): not a critical scene; graded from
            -invalid_penalty_base at the bucket boundary to -1 at
            gap >= invalid_grade_scale (flat -1 when grading is disabled).
          * init_invalid (adversary truly interpenetrates a neighbour at spawn,
            overlap frac > 0): no criticality is credited, so reward =
            -R_constraint <= 0 -- an overlapping init can never manufacture a
            fake near-miss.
          * valid:  ``reward = clip(R_criticality - R_constraint, -1, 1) + R_bonus``:
              R_criticality = risk_coef * [1 - (1-R_ttc)(1-w_app*R_approach)]
              R_constraint  = lane_penalty * (c_spawn_lane + c_goal_lane)
                              + init_overlap_penalty * init_overlap_frac
              R_bonus       = r_collision * (collision_bonus
                                             + ego_fault_bonus * ego_fault)
            with w_app the annealed approach weight (``set_train_iteration``).

        Only true interpenetration hard-gates criticality; lane penalties are
        soft-subtracted on the valid branch (gating them zeroed the reward of
        every off-lane spawn geometry -- cut-ins, merges -- and taught the
        policy to stay conservatively on-lane). The time-ramped collision bonus
        is paid on the valid branch only, so a rejected or overlapping init
        cannot buy reward with a crash. All components are returned so the
        training loop can log / normalise them.
        """
        cfg = self.cfg
        n = len(m["init_invalid"])

        # --- criticality ----------------------------------------------------
        r_ttc = np.clip(1.0 - m["ego_min_ttc"] / cfg.ttc_tau, 0.0, 1.0).astype(np.float32)

        # Approach bonus on the warmup-filtered min distance, so an adversary
        # that only starts close (d0 - dmin == 0) scores low. Non-finite
        # distances (no adversary in the scene) contribute nothing.
        d0 = m["ego_adv_init_dist"]
        dmin = m["ego_adv_min_dist_warmup"]
        prox = _sigmoid((cfg.approach_d_safe - dmin) / cfg.approach_d_scale)
        closing = _sigmoid((d0 - dmin - cfg.approach_close_delta) / cfg.approach_close_scale)
        finite = np.isfinite(d0) & np.isfinite(dmin)
        r_approach = np.where(finite, prox * closing, 0.0).astype(np.float32)

        w_app = self._approach_coef
        r_risk = (1.0 - (1.0 - r_ttc) * (1.0 - w_app * r_approach)).astype(np.float32)
        # Raw criticality (positive term, [0, risk_coef]); hard-gated to 0 for
        # rejected / invalid scenes when the reward is assembled below.
        criticality_raw = (cfg.risk_coef * r_risk).astype(np.float32)

        # Time-ramped collision indicator: 0 inside the warmup (a spawn-on-top
        # contact earns nothing), 1 past warmup+window. Feeds the explicit
        # collision / ego-fault bonus below; with both bonus weights at 0 it is
        # a logged diagnostic only.
        collision = m["ego_collision"].astype(np.float32)
        ctime = m["ego_collision_time"]
        r_collision = (
            collision
            * _smoothstep(ctime, cfg.collision_warmup, cfg.collision_warmup + cfg.collision_window)
        ).astype(np.float32)
        c_trivial = (collision * (ctime < cfg.trivial_collision_t)).astype(np.float32)
        ego_fault = np.asarray(m["ego_fault_collision"], dtype=np.float32)
        r_bonus = (
            r_collision * (cfg.collision_bonus + cfg.ego_fault_bonus * ego_fault)
        ).astype(np.float32)

        # --- constraints (>= 0, subtracted on every branch) -----------------
        # Continuous adversary overlap fraction (intersection / adversary area)
        # vs neighbours at spawn feeds the graded overlap penalty.
        c_overlap = np.asarray(m["init_overlap_frac"], dtype=np.float32)
        c_spawn_lane = _smoothstep(m["spawn_lane_dist"], cfg.lane_soft, cfg.lane_hard)
        c_goal_lane = _smoothstep(m["goal_lane_dist"], cfg.lane_soft, cfg.lane_hard)
        constraint = (
            cfg.lane_penalty * (c_spawn_lane + c_goal_lane)
            + cfg.init_overlap_penalty * c_overlap
        ).astype(np.float32)

        # --- assemble: reject -> init_invalid -> valid ----------------------
        c_parking = np.asarray(m["gen_agent_is_parked"], dtype=np.float32)
        if self.gen_invalid_enabled:
            c_reject = np.asarray(m["gen_agent_is_invalid"], dtype=np.float32)
            reject_reason = m["gen_agent_invalid_reason"]
        else:
            c_reject = c_parking
            reject_reason = np.full(n, "", dtype=object)
        reject = c_reject > 0
        # Hard gate on true spawn interpenetration only. Gating on ANY nonzero
        # constraint (as before) zeroed criticality for every off-lane spawn as
        # well, adding a ~risk_coef-high reward cliff at lane_soft and pruning
        # exactly the aggressive geometries (cut-ins, merges) a critical
        # adversary needs; those now keep their gradient and pay the soft
        # lane penalty instead.
        init_invalid = c_overlap > 0.0
        valid = ~(reject | init_invalid)

        # Graded reject penalty: ramp from -invalid_penalty_base at the bucket
        # boundary down to -1 at gap >= invalid_grade_scale, so rejected
        # samples keep a within-group ordering ("less invalid is better")
        # instead of a contrast-free flat -1. Grading needs the gap metric,
        # which only the condition-violation check emits.
        if self.gen_invalid_enabled and cfg.invalid_grade_scale > 0.0:
            gap = np.asarray(m["gen_agent_invalid_gap"], dtype=np.float32)
            c_invalid_sev = np.clip(gap / cfg.invalid_grade_scale, 0.0, 1.0).astype(np.float32)
            base_pen = float(np.clip(cfg.invalid_penalty_base, 0.0, 1.0))
            r_reject = -(base_pen + (1.0 - base_pen) * c_invalid_sev)
        else:
            c_invalid_sev = c_reject
            r_reject = np.full(n, -1.0, dtype=np.float32)

        # Hard validity gate: criticality is only credited on valid scenes.
        criticality = np.where(valid, criticality_raw, 0.0).astype(np.float32)
        # Valid branch: soft-subtract constraints, then pay the collision bonus
        # OUTSIDE the clip so it can never be absorbed by an already-saturated
        # dense reward (max valid reward = 1 + collision_bonus + ego_fault_bonus).
        total = np.select(
            [reject, init_invalid],
            [
                r_reject.astype(np.float32),
                np.clip(-constraint, -1.0, 0.0),
            ],
            default=np.clip(criticality_raw - constraint, -1.0, 1.0) + r_bonus,
        ).astype(np.float32)

        components = {
            "r_ttc": r_ttc,
            "r_approach": r_approach,
            "r_risk": r_risk,
            "r_collision": r_collision,
            "r_bonus": np.where(valid, r_bonus, 0.0).astype(np.float32),
            "criticality": criticality,
            "c_lane": c_spawn_lane + c_goal_lane,
            "c_spawn_lane": c_spawn_lane,
            "c_goal_lane": c_goal_lane,
            "c_parking": c_parking,
            "c_invalid": c_reject,
            "c_invalid_sev": c_invalid_sev,
            "c_invalid_reason": reject_reason,
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
            Dictionary of per-scene numpy arrays: the assembled ``reward``, the
            raw rollout metrics, and each reward component (r_ttc, r_approach,
            criticality, c_lane, constraint, ...) for diagnostics / logging.
            When ``record_trajectories`` is true, the dictionary also contains
            a ``trajectories`` list.
        """
        result = self.planner.rollout(scenes, record_trajectories=record_trajectories)
        metrics = result.metrics

        rewards, components = self._assemble_reward(metrics)

        out = {
            "reward": rewards,
            "ego_collision": metrics["ego_collision"],
            "ego_fault_collision": metrics["ego_fault_collision"],
            "ego_collision_time": metrics["ego_collision_time"],
            "ego_min_ttc": metrics["ego_min_ttc"],
            "ego_offroad": metrics["ego_offroad"],
            # Off-road proxy (numpy planners only; the legacy C backend does not
            # measure it, so these fall back to NaN instead of raising).
            "ego_offroad_proxy": metrics.get(
                "ego_offroad_proxy", np.full(len(rewards), np.nan, dtype=np.float32)
            ),
            "ego_offroad_frac": metrics.get(
                "ego_offroad_frac", np.full(len(rewards), np.nan, dtype=np.float32)
            ),
            "ego_lane_dist_max": metrics.get(
                "ego_lane_dist_max", np.full(len(rewards), np.nan, dtype=np.float32)
            ),
            "init_invalid": metrics["init_invalid"],
            "init_overlap_frac": metrics["init_overlap_frac"],
            "reached_goal": metrics["reached_goal"],
            "goal_offlane_frac": metrics["goal_offlane_frac"],
            "goal_lane_dist": metrics["goal_lane_dist"],
            "spawn_lane_dist": metrics["spawn_lane_dist"],
            "parking_mismatch_frac": metrics["parking_mismatch_frac"],
            "ego_adv_min_dist": metrics["ego_adv_min_dist"],
            "ego_adv_init_dist": metrics["ego_adv_init_dist"],
            "ego_adv_min_dist_warmup": metrics["ego_adv_min_dist_warmup"],
            "gen_agent_is_parked": metrics["gen_agent_is_parked"],
        }
        if self.gen_invalid_enabled:
            out["gen_agent_is_invalid"] = metrics["gen_agent_is_invalid"]
        out.update(components)
        if result.trajectories is not None:
            out["trajectories"] = result.trajectories
        return out
