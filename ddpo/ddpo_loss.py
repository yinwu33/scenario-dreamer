"""DDPO loss for scene-init models.

Model-agnostic: operates purely on per-scene / per-step log-probs returned by a
``SceneInitModel`` and per-scene advantages derived from the PufferDrive reward.

Two estimators (Black et al. 2023, "Training Diffusion Models with Reinforcement
Learning"):
  * "sf"  – score function / REINFORCE: grad = E[ A * grad logpi ]
  * "is"  – importance sampling / PPO-clip: enables >1 inner epoch per rollout

A KL-to-base penalty (DPOK-style) keeps generated scenes on the pretrained
manifold and is the main guard against reward hacking (e.g. degenerate scenes
that collide at t=0). It is the closed-form KL between the policy's and the
frozen reference's per-step reverse Gaussians, which share the fixed DDPM
posterior variance, so KL = sum_d (mu_theta - mu_ref)^2 / (2 var) >= 0 and is
differentiated through the policy mean. (The earlier sampled estimator
E[log pi_theta - log pi_ref], differentiated at fixed samples, has no opposing
term and instead drives log pi_theta down without bound -> divergence.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DDPOConfig:
    estimator: str = "is"          # "sf" | "is"
    clip_range: float = 1e-4       # PPO clip epsilon (diffusion ratios are tiny per step)
    kl_coef: float = 0.0           # weight of KL-to-base penalty (set >0 to enable)
    adv_clip: float = 5.0          # clip normalised advantages to +/- this
    adv_eps: float = 1e-6
    logratio_clip: float = 20.0    # clamp before exp() to avoid inf * 0 -> nan


def compute_advantages(rewards: torch.Tensor, cfg: DDPOConfig) -> torch.Tensor:
    """Per-scene whitened advantages from per-scene rewards. Shape [num_scenes]."""
    r = torch.nan_to_num(rewards.float(), nan=0.0, posinf=0.0, neginf=0.0)
    adv = (r - r.mean()) / (r.std(unbiased=False) + cfg.adv_eps)
    return adv.clamp(-cfg.adv_clip, cfg.adv_clip)


def ddpo_loss(
    new_logprob: torch.Tensor,   # [num_scenes, k]  (requires grad)
    old_logprob: torch.Tensor,   # [num_scenes, k]  (detached, from rollout)
    advantages: torch.Tensor,    # [num_scenes]
    cfg: DDPOConfig,
    kl_term: torch.Tensor | None = None,  # [num_scenes, k] analytic KL(policy||ref) >= 0
) -> tuple[torch.Tensor, dict]:
    new_logprob = torch.nan_to_num(new_logprob, nan=0.0, posinf=0.0, neginf=0.0)
    old_logprob = torch.nan_to_num(old_logprob, nan=0.0, posinf=0.0, neginf=0.0)
    adv = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(1)

    if cfg.estimator == "sf":
        pg = -(adv * new_logprob).mean()
        ratio_mean = 1.0
    elif cfg.estimator == "is":
        logratio = new_logprob - old_logprob
        logratio = torch.nan_to_num(logratio, nan=0.0, posinf=0.0, neginf=0.0)
        logratio = logratio.clamp(-cfg.logratio_clip, cfg.logratio_clip)
        ratio = logratio.exp()
        unclipped = ratio * adv
        clipped = ratio.clamp(1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv
        pg = -torch.min(unclipped, clipped).mean()
        ratio_mean = ratio.mean().item()
    else:
        raise ValueError(f"unknown estimator {cfg.estimator}")

    metrics = {"pg_loss": pg.item(), "ratio_mean": ratio_mean}
    loss = pg

    if cfg.kl_coef > 0.0 and kl_term is not None:
        # kl_term is the closed-form per-step KL(policy || reference) over the
        # differentiated reverse-diffusion steps (>= 0; differentiable through
        # the policy mean). Penalising its mean is a proper trust region that
        # pulls the policy back toward the pretrained manifold.
        kl_term = torch.nan_to_num(kl_term, nan=0.0, posinf=0.0, neginf=0.0)
        kl = kl_term.mean()
        loss = loss + cfg.kl_coef * kl
        metrics["kl_to_base"] = kl.item()

    metrics["loss"] = loss.item()
    return loss, metrics
