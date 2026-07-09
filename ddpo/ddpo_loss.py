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

import math
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
    group_skip_std: float = 1e-4   # a group whose reward std is below this is degenerate
    # Degenerate-group handling. "skip" zeroes the group's advantages (legacy):
    # no gradient at all, so a fully-collapsed group (e.g. every sample rejected
    # at the same reward) exerts no restoring force and becomes a stable
    # attractor. "global" instead whitens those samples against the WHOLE
    # batch's reward statistics, so a uniformly-bad group is still pushed away
    # from as long as any other group in the batch does better.
    degenerate_group: str = "skip"  # "skip" | "global"
    # Per-sample-step trust region: drop (zero-weight) sample-steps whose
    # |log ratio| exceeds this before exp(). With inner_epochs == 1 the update
    # is on-policy and the true log-ratio is ~0; anything large is numerical
    # junk (low-noise steps divide by tiny posterior variances), and the PPO
    # min() keeps exactly those exploded ratios when the advantage is negative,
    # turning noise into an unbounded repulsion term. 0 disables (legacy).
    ratio_drop: float = 0.0
    # Adaptive KL-to-base controller (PPO-style multiplicative feedback) --
    # keeps the measured KL near kl_target instead of trusting a fixed coef:
    # coef *= kl_adapt_rate while KL > kl_target * kl_adapt_band and
    # coef /= kl_adapt_rate while KL < kl_target / kl_adapt_band, clamped to
    # [kl_coef_min, kl_coef_max]. kl_target == 0 disables (fixed kl_coef).
    kl_target: float = 0.0
    kl_adapt_band: float = 1.5
    kl_adapt_rate: float = 1.5
    kl_coef_min: float = 1e-3
    kl_coef_max: float = 100.0
    # Clip the pg and KL gradients separately (each to the training loop's
    # grad_clip) instead of one global norm over their sum: with a single clip
    # an exploding pg term rescales the KL pullback to nothing exactly when the
    # trust region is needed most.
    decouple_kl_grad: bool = False


class AdaptiveKLController:
    """Multiplicative feedback on the KL-to-base coefficient.

    ``update`` nudges ``coef`` after every optimiser step from the measured
    KL(policy || base) (see DDPOConfig.kl_target/kl_adapt_*). With
    ``kl_target == 0`` the controller is inert and ``coef`` stays at the fixed
    ``cfg.kl_coef``. ``coef`` is checkpointed/restored by the training loop so
    a resume does not reset the trust region.
    """

    def __init__(self, cfg: DDPOConfig, coef: float | None = None):
        self.cfg = cfg
        self.coef = float(cfg.kl_coef if coef is None else coef)

    @property
    def enabled(self) -> bool:
        return self.cfg.kl_target > 0.0

    def update(self, measured_kl: float) -> float:
        if not self.enabled or not math.isfinite(measured_kl):
            return self.coef
        if measured_kl > self.cfg.kl_target * self.cfg.kl_adapt_band:
            self.coef *= self.cfg.kl_adapt_rate
        elif measured_kl < self.cfg.kl_target / self.cfg.kl_adapt_band:
            self.coef /= self.cfg.kl_adapt_rate
        self.coef = float(min(max(self.coef, self.cfg.kl_coef_min), self.cfg.kl_coef_max))
        return self.coef


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

    Degenerate groups (within-group std below ``group_skip_std``) follow
    ``cfg.degenerate_group``: "skip" zeroes them (legacy, no gradient), while
    "global" z-scores those samples against the whole batch so a uniformly-bad
    group still gets a repulsive signal relative to better groups (whatever the
    per-group ``adv_mode``, the fallback is a z-score: ranks are meaningless
    within an all-tie group).
    """
    r = torch.nan_to_num(rewards.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if group_ids is None or not cfg.per_context:
        return _whiten(r, cfg).clamp(-cfg.adv_clip, cfg.adv_clip)
    group_ids = group_ids.to(r.device)
    adv = torch.zeros_like(r)
    degenerate_masks = []
    for g in torch.unique(group_ids):
        m = group_ids == g
        if r[m].std(unbiased=False) < cfg.group_skip_std:
            degenerate_masks.append(m)
            continue
        adv[m] = _whiten(r[m], cfg)
    if degenerate_masks and cfg.degenerate_group == "global":
        batch_std = r.std(unbiased=False)
        if batch_std >= cfg.group_skip_std:
            batch_mean = r.mean()
            for m in degenerate_masks:
                adv[m] = (r[m] - batch_mean) / (batch_std + cfg.adv_eps)
    return adv.clamp(-cfg.adv_clip, cfg.adv_clip)


def ddpo_loss(
    new_logprob: torch.Tensor,   # [num_scenes, k]  (requires grad)
    old_logprob: torch.Tensor,   # [num_scenes, k]  (detached, from rollout)
    advantages: torch.Tensor,    # [num_scenes]
    cfg: DDPOConfig,
    kl_term: torch.Tensor | None = None,  # [num_scenes, k] analytic KL(policy||ref) >= 0
    kl_coef: float | None = None,  # override cfg.kl_coef (adaptive controller)
) -> tuple[torch.Tensor, dict, dict]:
    """Returns ``(loss, metrics, parts)``.

    ``parts`` carries the differentiable components separately -- ``{"pg": pg,
    "kl": kl-or-None}`` -- so the training loop can clip their gradients
    independently (``cfg.decouple_kl_grad``); ``loss = pg + kl_coef * kl`` is
    the combined objective for the legacy single-backward path.
    """
    new_logprob = torch.nan_to_num(new_logprob, nan=0.0, posinf=0.0, neginf=0.0)
    old_logprob = torch.nan_to_num(old_logprob, nan=0.0, posinf=0.0, neginf=0.0)
    adv = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(1)

    if cfg.estimator == "sf":
        pg = -(adv * new_logprob).mean()
        ratio_mean = 1.0
        dropped_frac = 0.0
    elif cfg.estimator == "is":
        logratio = new_logprob - old_logprob
        logratio = torch.nan_to_num(logratio, nan=0.0, posinf=0.0, neginf=0.0)
        # Trust-region drop BEFORE exp(): an exploded |log ratio| is numerical
        # junk on an on-policy update, and min() below would keep it whenever
        # the advantage is negative (unbounded repulsion from its own samples).
        if cfg.ratio_drop > 0.0:
            keep = (logratio.abs() <= cfg.ratio_drop).float()
        else:
            keep = torch.ones_like(logratio)
        logratio = logratio.clamp(-cfg.logratio_clip, cfg.logratio_clip)
        ratio = logratio.exp()
        unclipped = ratio * adv
        clipped = ratio.clamp(1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv
        kept = keep.sum().clamp(min=1.0)
        pg = -(torch.min(unclipped, clipped) * keep).sum() / kept
        ratio_mean = float((ratio * keep).sum() / kept)
        dropped_frac = float(1.0 - keep.mean())
    else:
        raise ValueError(f"unknown estimator {cfg.estimator}")

    metrics = {"pg_loss": pg.item(), "ratio_mean": ratio_mean, "ratio_dropped_frac": dropped_frac}
    loss = pg
    parts: dict = {"pg": pg, "kl": None}

    coef = float(cfg.kl_coef if kl_coef is None else kl_coef)
    if coef > 0.0 and kl_term is not None:
        # kl_term is the closed-form per-step KL(policy || reference) over the
        # differentiated reverse-diffusion steps (>= 0; differentiable through
        # the policy mean). Penalising its mean is a proper trust region that
        # pulls the policy back toward the pretrained manifold.
        kl_term = torch.nan_to_num(kl_term, nan=0.0, posinf=0.0, neginf=0.0)
        kl = kl_term.mean()
        loss = loss + coef * kl
        parts["kl"] = kl
        metrics["kl_to_base"] = kl.item()
        metrics["kl_coef"] = coef

    metrics["loss"] = loss.item()
    return loss, metrics, parts
