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
    adv_mode: str = "zscore"       # "zscore" | "rank" advantage transform
    per_context: bool = True       # whiten advantages within each context group
    group_skip_std: float = 1e-4   # zero out groups whose reward std is below this


def _rank_advantage(r: torch.Tensor) -> torch.Tensor:
    """Evenly-spaced rank advantages in [-1, 1] (ties broken by sort order).

    More robust than z-scoring to the heavy-tailed / multi-modal reward
    distributions DDPO produces: a single huge-reward outlier cannot dominate
    the update, and the per-group scale is fixed regardless of reward variance.
    """
    n = r.numel()
    if n <= 1:
        return torch.zeros_like(r)
    order = torch.argsort(r)
    ranks = torch.empty_like(r)
    ranks[order] = torch.arange(n, device=r.device, dtype=r.dtype)
    return 2.0 * ranks / (n - 1) - 1.0


def _whiten(r: torch.Tensor, cfg: DDPOConfig) -> torch.Tensor:
    """Transform one (sub)group of rewards to advantages.

    Degenerate groups (std below ``group_skip_std``, e.g. every sample invalid)
    return all-zero advantages so they contribute no policy gradient instead of
    amplifying float noise into a spurious signal.
    """
    std = r.std(unbiased=False)
    if std < cfg.group_skip_std:
        return torch.zeros_like(r)
    if cfg.adv_mode == "rank":
        return _rank_advantage(r)
    return (r - r.mean()) / (std + cfg.adv_eps)


def compute_advantages(
    rewards: torch.Tensor,
    cfg: DDPOConfig,
    group_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-scene advantages from per-scene rewards. Shape [num_scenes].

    When ``group_ids`` is provided and ``cfg.per_context`` is set, rewards are
    whitened WITHIN each context group (GRPO / per-context DDPO): this isolates
    "which generation is more critical in this map/ego context" from "which
    contexts are intrinsically easier", which a global z-score conflates.
    Otherwise the whole batch is whitened together (legacy behaviour).
    """
    r = torch.nan_to_num(rewards.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if group_ids is None or not cfg.per_context:
        return _whiten(r, cfg).clamp(-cfg.adv_clip, cfg.adv_clip)
    group_ids = group_ids.to(r.device)
    adv = torch.zeros_like(r)
    for g in torch.unique(group_ids):
        m = group_ids == g
        adv[m] = _whiten(r[m], cfg)
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
