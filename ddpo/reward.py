"""DDPO reward: roll generated scenes out with a planner, score the ego.

The rollout itself is delegated to a pluggable ``RolloutPlanner`` (see
``ddpo.planners``); this module only assembles the scalar reward from the
per-scene metrics the planner returns:

  * only the ego (scene agent 0) is scored;
  * the base reward is a DENSE criticality term ``clip(1 - min_TTC/tau, 0, 1)``
    over the ego's min time-to-collision along the rollout, so near-misses give
    gradient even without an actual crash; an ego collision caps it at 1;
  * ``init_invalid`` flags scenes with overlapping vehicles at t=0 (degenerate
    init / reward hacking) -> strong negative reward;
  * ``ego_offroad`` is kept for interface compatibility but is always 0: the
    generated maps carry no road edges (see pufferdrive_sim docstring);
  * a scene stops being stepped/scored once its ego reaches its goal.

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
        goal_offlane_threshold: float = 3.0,
        goal_onroad_threshold: float = 2.0,
        goal_offlane_penalty: float = 0.5,
        parking_mismatch_penalty: float = 0.5,
        min_dist_coef: float = 0.0,
        min_dist_dmax: float = 20.0,
        controlled_parking_penalty: float = 0.0,
        seed: int = 0,
        backend: str = "numpy",
        pufferdrive_root: str | None = None,
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
        self.parking_mismatch_penalty = float(parking_mismatch_penalty)
        # Dense shaping toward criticality: bonus = min_dist_coef * clip(1 -
        # ego_adv_min_dist / min_dist_dmax, 0, 1), where ego_adv_min_dist is the
        # smallest same-step centre distance between the ego and a controlled
        # adversary over the rollout (numpy planners only; +inf -> no bonus).
        self.min_dist_coef = float(min_dist_coef)
        self.min_dist_dmax = max(float(min_dist_dmax), 1e-6)
        # Penalise a controlled adversary generated parked (goal within
        # MIN_DISTANCE_TO_GOAL of spawn) to force it to drive.
        self.controlled_parking_penalty = float(controlled_parking_penalty)

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
        )
        self.planner = build_planner(planner_cfg, params)
        # Back-compat: expose the native C backend for callers that poke at it
        # directly (e.g. scripts/benchmark_ddpo_rollout_backends.py).
        self.native_backend = getattr(self.planner, "backend", None)

    def _reward(self, ego_collision, ego_min_ttc, init_invalid):
        """Dense critical-scene reward.

        Base term is continuous time-to-collision criticality in [0, 1]
        (``clip(1 - min_TTC/tau, 0, 1)``; ``min_TTC=inf`` -> 0). An actual ego
        collision caps it at 1; degenerate init (overlapping vehicles) -> -1.
        """
        ttc_term = np.clip(1.0 - ego_min_ttc / self.ttc_tau, 0.0, 1.0)
        r = np.where(ego_collision > 0, 1.0, ttc_term)
        r = np.where(init_invalid > 0, -1.0, r)
        return r.astype(np.float32)

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

        rewards = self._reward(
            metrics["ego_collision"],
            metrics["ego_min_ttc"],
            metrics["init_invalid"],
        )
        # Dense shaping: the closer the controlled adversary gets to the ego at any
        # single timestep, the larger the bonus. Gated off on degenerate t=0
        # overlap (init_invalid) so it cannot cancel the -1 floor - an adversary
        # spawned on top of the ego (min_dist=0) must not be rewarded.
        ego_adv_min_dist = metrics.get("ego_adv_min_dist")
        if self.min_dist_coef > 0.0 and ego_adv_min_dist is not None:
            dist_term = np.clip(1.0 - ego_adv_min_dist / self.min_dist_dmax, 0.0, 1.0)
            dist_term = np.where(metrics["init_invalid"] > 0, 0.0, dist_term).astype(np.float32)
            rewards = rewards + self.min_dist_coef * dist_term
        # All penalty terms are per-scene FRACTIONS in [0, 1] (count-normalised),
        # so scenes with many agents are not penalised more than sparse ones; the
        # coefficients set the scale relative to the +/-1 collision reward.
        rewards = rewards - self.goal_offlane_penalty * metrics["goal_offlane_frac"]
        rewards = (
            rewards - self.parking_mismatch_penalty * metrics["parking_mismatch_frac"]
        )
        # Penalise a controlled adversary that is generated parked (forces it to drive).
        rewards = rewards - self.controlled_parking_penalty * metrics.get(
            "controlled_parking_frac", 0.0
        )
        out = {
            "reward": rewards,
            "ego_collision": metrics["ego_collision"],
            "ego_min_ttc": metrics["ego_min_ttc"],
            "ego_offroad": metrics["ego_offroad"],
            "init_invalid": metrics["init_invalid"],
            "reached_goal": metrics["reached_goal"],
            "goal_offlane_frac": metrics["goal_offlane_frac"],
            "parking_mismatch_frac": metrics["parking_mismatch_frac"],
            "ego_adv_min_dist": metrics.get(
                "ego_adv_min_dist", np.full(scenes.num_scenes, np.inf, dtype=np.float32)
            ),
            "controlled_parking_frac": metrics.get(
                "controlled_parking_frac", np.zeros(scenes.num_scenes, dtype=np.float32)
            ),
        }
        if trajectories is not None:
            out["trajectories"] = trajectories
        return out
