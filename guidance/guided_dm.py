"""SceneControl-style guided sampling on top of the data-space diffusion model.

Control comes entirely from *sampling*: the model itself (``dm_goal``) is a plain
unconditional scene generator with no conditioning labels and no adversary stream.
At every denoising step we take the model's predicted clean scene ``x̂₀``, push one
designated agent down the gradient of :class:`~guidance.costs.CriticalSceneCost`, and
feed the modified ``x̂₀`` into the DDPM posterior.

Why gradients on ``x̂₀`` rather than on ``x_t``
-----------------------------------------------
The textbook classifier guidance term is ``∇_{x_t} J(x̂₀(x_t))``, which needs a full
backward pass through the 345M-parameter DiT at each of the 100 steps. The
reconstruction form used here (``x̂₀ <- x̂₀ - α ∇_{x̂₀} J``) differentiates only the
analytic cost, so guidance costs a few milliseconds per step instead of a second.
This is what CTG / Diffuser / SafeSim do in practice. ``mode="score"`` implements
the expensive variant for comparison.

What is allowed to move
-----------------------
The cost object declares its own ``adv_columns`` / ``ego_columns``; everything else
is frozen. Two rules hold for every objective:

* the **lanes** never receive gradient. Left free, the cheapest way to manufacture a
  conflict is to bend the road, which yields a critical-looking scene on an
  impossible map.
* vehicle ``length``/``width`` and the type logits never move, or the optimiser
  stretches the adversary toward the 22 m maximum to force an overlap.
"""

from __future__ import annotations

from typing import Optional

import torch

from guidance.costs import CriticalSceneCost, ProximityGoalCost, first_index_per_scene
from nn_modules.dm_goal import DMGoal

#: Selectable objectives. ``proximity`` is the baseline: three geometric constraints
#: and no dynamics assumption. ``criticality`` is the richer variant that extrapolates
#: trajectories -- useful as an ablation, but it bakes in a motion model of our own.
COST_TYPES = {"proximity": ProximityGoalCost, "criticality": CriticalSceneCost}


def default_adv_index(agent_batch: torch.Tensor, batch_size: int) -> torch.Tensor:
    """First non-ego agent of every scene, i.e. local index 1 -> global index.

    Deterministic on purpose. Which agent is "the adversary" is a *sampling-time*
    choice for a guidance baseline -- the model is symmetric in its agents -- so
    there is nothing to randomise. Scenes need >= 2 agents, which the layout prior
    guarantees (``--min-num-agents 2``).
    """
    return first_index_per_scene(agent_batch, batch_size) + 1


class GuidedDMGoal(DMGoal):
    """``DMGoal`` whose reverse process is steered by a differentiable scene cost.

    Only :meth:`p_mean_variance` is overridden, so the entire sampling loop, the
    conditioning modes and the type/lane decoding are inherited unchanged -- a guided
    run and an unguided run differ by exactly this one hook, which is what makes them
    comparable.
    """

    def __init__(self, cfg, guidance_cfg=None):
        super().__init__(cfg)
        self.gcfg = guidance_cfg
        self.cost_fn = None
        self.adv_idx: Optional[torch.Tensor] = None
        self.ego_idx: Optional[torch.Tensor] = None
        self.guidance_log: list = []
        if guidance_cfg is not None:
            cost_type = guidance_cfg.get("cost_type", "proximity")
            if cost_type not in COST_TYPES:
                raise ValueError(
                    f"Unknown guidance cost_type {cost_type!r}; expected one of {sorted(COST_TYPES)}"
                )
            self.cost_fn = COST_TYPES[cost_type](guidance_cfg.cost, self.cfg_dataset)

    # ------------------------------------------------------------------ setup --
    def enable_guidance(self, adv_idx: torch.Tensor, ego_idx: torch.Tensor) -> None:
        """Arm guidance for the batch about to be sampled. Both are ``[B]`` global indices."""
        self.adv_idx = adv_idx
        self.ego_idx = ego_idx
        self.guidance_log = []

    def disable_guidance(self) -> None:
        self.adv_idx = None
        self.ego_idx = None

    # ------------------------------------------------------------------- hook --
    def p_mean_variance(self, x_agent, x_lane, data, t_agent, t_lane):
        epsilon_agent, epsilon_lane = self.epsilon(x_agent, x_lane, data, t_agent, t_lane)

        t_agent = t_agent.detach().to(torch.int64)
        t_lane = t_lane.detach().to(torch.int64)
        x_agent_recon = self.predict_start_from_noise(x_agent, t=t_agent, noise=epsilon_agent)
        x_lane_recon = self.predict_start_from_noise(x_lane, t=t_lane, noise=epsilon_lane)

        if self.cost_fn is not None and self.adv_idx is not None:
            x_agent_recon = self._guide(x_agent_recon, x_lane_recon, data, t_agent)

        model_mean_agent, posterior_log_variance_agent = self.q_posterior(
            x_agent_recon, x_agent, t_agent
        )
        model_mean_lane, posterior_log_variance_lane = self.q_posterior(
            x_lane_recon, x_lane, t_lane
        )
        return (
            model_mean_agent,
            posterior_log_variance_agent,
            model_mean_lane,
            posterior_log_variance_lane,
        )

    # --------------------------------------------------------------- guidance --
    def _masks(self, x_agent_recon: torch.Tensor) -> torch.Tensor:
        """``[N, 1, D]`` multiplicative mask, driven by the cost's declared columns."""
        mask = torch.zeros_like(x_agent_recon)
        for rows, cols in (
            (self.adv_idx, self.cost_fn.adv_columns),
            (self.ego_idx, self.cost_fn.ego_columns),
        ):
            if not cols:
                continue
            col_idx = torch.tensor(cols, device=mask.device, dtype=torch.long)
            mask[rows[:, None], :, col_idx] = 1.0
        return mask

    def _guide(self, x_agent_recon, x_lane_recon, data, t_agent):
        # x̂₀ is meaningless at high noise -- guiding there just injects noise into
        # the trajectory. Skip until the chain has committed to a scene layout.
        step_t = int(t_agent[self.adv_idx[0]].item())
        if step_t > self.gcfg.t_start:
            return x_agent_recon

        num_points = self.cfg_model.num_points_per_lane
        lane_attr = self.cfg_model.lane_attr
        # Lanes are context only: detached so no gradient can reach the road.
        lane_states = x_lane_recon.detach()[:, 0, :].reshape(-1, num_points, lane_attr)

        reference = x_agent_recon.detach()
        mask = self._masks(x_agent_recon)
        x = x_agent_recon.detach()

        for _ in range(self.gcfg.num_grad_steps):
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                cost, terms = self.cost_fn(
                    x[:, 0, :9],
                    lane_states,
                    data["agent"].batch,
                    data["lane"].batch,
                    self.adv_idx,
                    data.batch_size,
                    reference_norm=reference[:, 0, :9],
                )
                grad = torch.autograd.grad(cost, x)[0]

            grad = grad * mask
            # Per-scene gradient normalisation over every row the cost may move
            # (adversary, and the ego's goal when the objective constrains it). The
            # hinges are squared metres and go to exactly zero once satisfied, so the
            # raw magnitude swings over orders of magnitude along the chain.
            # Normalising makes `step_size` a fixed displacement in the model's
            # [-1, 1] state space (0.02 ~ 1% of the full range) rather than something
            # that has to be retuned whenever a weight changes.
            rows = torch.stack([self.adv_idx, self.ego_idx], dim=0)      # [2, B]
            g_rows = grad[rows]                                          # [2, B, 1, D]
            norm = torch.linalg.norm(
                g_rows.permute(1, 0, 2, 3).reshape(rows.shape[1], -1), dim=-1
            )
            scale = self.gcfg.step_size / norm.clamp_min(1e-12)
            grad = grad.index_put((rows[0],), g_rows[0] * scale.view(-1, 1, 1))
            grad = grad.index_put((rows[1],), g_rows[1] * scale.view(-1, 1, 1))
            x = x - grad
            # Keep the guided columns inside the model's [-1, 1] range. Without this
            # the optimiser walks `speed` below -1, which decodes to a NEGATIVE speed
            # -- a degenerate way to shrink the ego-adversary gap (a stationary
            # adversary lets the ego close on it) that also produces an invalid scene.
            # The unnormalize deliberately does NOT clip, so gradients still point
            # back inward from the boundary; the clamp belongs here, on the state.
            x = torch.where(mask > 0, x.clamp(-1.0, 1.0), x)

        self.guidance_log.append({"t": step_t, **terms})
        return x.detach()
