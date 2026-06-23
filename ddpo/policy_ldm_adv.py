"""ldm_adv latent-diffusion model as a DDPO policy (single-adversary, init_adv).

The latent-space analogue of the map-conditioned dm flow: the base scene (one
real ego + the real lanes) is held fixed as conditioning and the policy denoises
ONLY the single adversary latent. A frozen goal autoencoder decodes the
``ego + adv`` scene for the reward, which scores the ego against the (sole)
generated adversary.

  * horizon  H = n_diffusion_timesteps (DDPM) or ddim_steps (DDIM)
  * state    s_t = (noisy adv latent x_adv_t, fixed ego/lane latents, graph)
  * action   a_t = sampled next adv latent
  * policy   pi(a_t | s_t) = N(transition_mean_theta(x_adv_t), Sigma_t)

DDPO log-prob / KL are accumulated over the adv latent only (one adv node per
scene). The base agent/lane streams are frozen (LDMAdv ``adv_only``), so they
carry no policy gradient.
"""

from __future__ import annotations

import copy
import math

import torch
from torch_geometric.data import Batch

from nn_modules.autoencoder import AutoEncoder
from nn_modules.ldm_adv import LDMAdv
from utils.data_container import ScenarioDreamerData
from utils.data_helpers import unnormalize_latents, unnormalize_scene

from .interfaces import GeneratedScenes, SamplingTrajectory

_LOG_2PI = math.log(2.0 * math.pi)


def _gaussian_logprob(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-node diagonal Gaussian log-density, summed over feature dims."""
    var = logvar.exp()
    per_elem = -0.5 * (((x - mean) ** 2) / var + logvar + _LOG_2PI)
    return per_elem.flatten(1).sum(dim=1)


def _gaussian_kl(mean: torch.Tensor, mean_ref: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-node KL between two diagonal Gaussians that share ``logvar`` (the fixed
    reverse-step variance), reducing to ``sum_d (mean - mean_ref)^2 / (2 var)``."""
    var = logvar.exp()
    per_elem = 0.5 * (mean - mean_ref) ** 2 / var
    return per_elem.flatten(1).sum(dim=1)


class LDMAdvDDPOPolicy:
    def __init__(
        self,
        ldm_cfg,
        ae_cfg,
        *,
        ldm_ckpt: str | None,
        ae_ckpt: str | None,
        device: str = "cuda",
        use_ema_weights: bool = True,
        sampler: str = "ddpm",
        ddim_steps: int | None = None,
        ddim_eta: float = 1.0,
        force_adv_vehicle: bool = True,
    ):
        self.cfg = ldm_cfg
        self.cfg_model = ldm_cfg.model
        self.cfg_dataset = ldm_cfg.dataset
        self.device = device
        self.force_adv_vehicle = bool(force_adv_vehicle)
        self.sampler = str(sampler).lower()
        if self.sampler not in ("ddpm", "ddim"):
            raise ValueError(f"sampler must be one of ('ddpm', 'ddim'), got {sampler!r}")

        self.net = LDMAdv(ldm_cfg).to(device)
        if ldm_ckpt is not None:
            self._load_ldm_checkpoint(ldm_ckpt, use_ema_weights)
        # DDPO only ever optimises the adversary LATENT denoiser; freeze the base
        # scene streams. The adv conditioning embedders are also frozen: the
        # condition is fixed to a constant target (see LDMAdvConditioningPool), so
        # the embedders just supply the supervised conditional prior and carry no
        # policy gradient.
        self.net.model.freeze_non_adv_parameters(freeze_cond_embedders=True)
        self.net.eval()

        self.ref = copy.deepcopy(self.net).to(device).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)

        self.ae = AutoEncoder(ae_cfg.model).to(device).eval()
        if ae_ckpt is not None:
            self._load_ae_checkpoint(ae_ckpt)
        for p in self.ae.parameters():
            p.requires_grad_(False)

        self._H = int(self.net.n_timesteps)
        self.agent_latent_dim = int(self.cfg_model.agent_latent_dim)
        self.diffusion_clip = self.cfg_model.diffusion_clip
        self.num_agent_types = int(self.cfg_dataset.num_agent_types)

        if ddim_steps is None:
            ddim_steps = self._H
        self.ddim_steps = int(ddim_steps)
        self.ddim_eta = float(ddim_eta)
        if self.sampler == "ddim":
            if self.ddim_steps < 2 or self.ddim_steps > self._H:
                raise ValueError(f"ddim_steps must be in [2, {self._H}], got {self.ddim_steps}")
            if self.ddim_eta < 0:
                raise ValueError(f"ddim_eta must be non-negative, got {self.ddim_eta}")
        self.step_pairs = self._build_step_pairs()

        self.agent_latents_mean = self.cfg_dataset.agent_latents_mean
        self.agent_latents_std = self.cfg_dataset.agent_latents_std
        self.lane_latents_mean = self.cfg_dataset.lane_latents_mean
        self.lane_latents_std = self.cfg_dataset.lane_latents_std

    # ------------------------------------------------------------------ load
    def _load_ldm_checkpoint(self, ckpt_path: str, use_ema_weights: bool) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        net_sd = {k[len("diff_model.") :]: v for k, v in sd.items() if k.startswith("diff_model.")}
        missing, unexpected = self.net.load_state_dict(net_sd, strict=False)
        if missing:
            print(f"[ddpo ldm_adv] {len(missing)} missing keys on LDMAdv load (e.g. {missing[:3]})")
        if unexpected:
            print(f"[ddpo ldm_adv] {len(unexpected)} unexpected keys on LDMAdv load (e.g. {unexpected[:3]})")
        # The conditioning embedders only exist on a checkpoint trained with
        # use_adv_conditioning=true. If they are missing the base model was NOT
        # conditioned, so feeding it a fixed adv_cond_target injects an UNTRAINED
        # constant bias into the adv stream (degrades generation) -- warn loudly.
        if getattr(self.net.model, "use_adv_conditioning", False):
            cond_missing = [k for k in missing if "_embedder" in k and "adv_" in k]
            if cond_missing:
                print(
                    f"[ddpo ldm_adv] WARNING: {len(cond_missing)} adv-conditioning embedder "
                    f"weights missing from checkpoint -- the base model was not trained with "
                    f"conditioning; the fixed adv_cond_target will inject an untrained bias."
                )
        if use_ema_weights:
            shadow = ckpt.get("ema_state_dict", {}).get("shadow_params", [])
            params = list(self.net.parameters())
            if len(shadow) == len(params) and len(shadow) > 0:
                with torch.no_grad():
                    for p, s in zip(params, shadow):
                        p.copy_(s.to(p.device))
                print(f"[ddpo ldm_adv] loaded {len(shadow)} EMA shadow params")
            else:
                print(
                    f"[ddpo ldm_adv] EMA shadow mismatch "
                    f"(shadow={len(shadow)}, params={len(params)}); raw LDMAdv weights"
                )

    def _load_ae_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        ae_sd = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
        missing, unexpected = self.ae.load_state_dict(ae_sd, strict=False)
        if missing:
            print(f"[ddpo ldm_adv] {len(missing)} missing keys on AE load (e.g. {missing[:3]})")
        if unexpected:
            print(f"[ddpo ldm_adv] {len(unexpected)} unexpected keys on AE load (e.g. {unexpected[:3]})")

    # ------------------------------------------------------------ properties
    @property
    def num_sampling_steps(self) -> int:
        return len(self.step_pairs)

    def trainable_parameters(self):
        return (p for p in self.net.parameters() if p.requires_grad)

    def state_dict(self) -> dict:
        return self.net.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.net.load_state_dict(sd)

    # ------------------------------------------------------- sampler schedule
    def _build_step_pairs(self) -> list[tuple[int, int]]:
        """Reverse diffusion schedule as ``(t, prev_t)`` pairs (``prev_t == -1`` is
        the final jump to clean x0). DDPM keeps the adjacent chain; DDIM uses a
        rounded linspace subset."""
        if self.sampler == "ddpm":
            timesteps = list(reversed(range(self._H)))
        else:
            ts = torch.linspace(0, self._H - 1, self.ddim_steps).round().to(torch.long)
            timesteps = sorted({int(t) for t in ts.tolist()}, reverse=True)
            if timesteps[0] != self._H - 1 or timesteps[-1] != 0:
                raise RuntimeError(f"invalid DDIM timestep schedule: {timesteps}")
            if len(timesteps) != self.ddim_steps:
                raise RuntimeError(
                    f"DDIM timestep schedule collapsed to {len(timesteps)} unique steps "
                    f"from requested {self.ddim_steps}"
                )
        return [
            (t, timesteps[i + 1] if i + 1 < len(timesteps) else -1)
            for i, t in enumerate(timesteps)
        ]

    def _ddim_sigma2(self, net, t: int, prev_t: int, device) -> torch.Tensor:
        alpha_t = net.alphas_cumprod[t].to(device)
        alpha_prev = (
            net.alphas_cumprod[prev_t].to(device)
            if prev_t >= 0
            else torch.ones((), device=device, dtype=alpha_t.dtype)
        )
        sigma2 = (self.ddim_eta ** 2) * (1.0 - alpha_prev) / (1.0 - alpha_t) * (
            1.0 - alpha_t / alpha_prev
        )
        return torch.clamp(sigma2, min=0.0)

    def _is_stochastic_transition(self, t: int, prev_t: int) -> bool:
        if self.sampler == "ddpm":
            return t > 0
        if self.ddim_eta <= 0:
            return False
        return bool((self._ddim_sigma2(self.net, t, prev_t, self.device) > 1e-20).item())

    def stochastic_step_indices(self, min_diffusion_t: int = 5) -> torch.Tensor:
        """Step indices safe to differentiate for DDPO (true diffusion timestep
        ``t >= min_diffusion_t`` and a stochastic transition)."""
        idx = [
            i
            for i, (t, prev_t) in enumerate(self.step_pairs)
            if t >= int(min_diffusion_t) and self._is_stochastic_transition(t, prev_t)
        ]
        return torch.as_tensor(idx, dtype=torch.long)

    @staticmethod
    def _alpha_at(net, t, shape, device):
        if int(t) < 0:
            return torch.ones(
                (shape[0],) + (1,) * (len(shape) - 1), device=device, dtype=net.alphas_cumprod.dtype
            )
        tt = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
        return net.alphas_cumprod.gather(0, tt).reshape(shape[0], *((1,) * (len(shape) - 1)))

    # ----------------------------------------------------------- conditioning
    def _agent_latents(self, data: Batch) -> torch.Tensor:
        return data["agent"].latents.float().to(self.device).unsqueeze(1)

    def _lane_latents(self, data: Batch) -> torch.Tensor:
        return data["lane"].latents.float().to(self.device).unsqueeze(1)

    def _adv_batch(self, data: Batch) -> torch.Tensor:
        if "batch" in data["adv"]:
            return data["adv"].batch
        return torch.arange(int(data.batch_size), device=self.device, dtype=torch.long)

    # ----------------------------------------------------------- transition
    def _adv_mean_logvar(self, net, x_adv, x_agent, x_lane, data, t: int, prev_t: int):
        """Reverse-step Gaussian (mean, logvar) for the adv latent at step ``t``.

        The base (ego + lane) streams are held at the clean conditioning latents
        (``t = 0``); only the adv stream is at noise level ``t``. The model
        computes all three epsilons but only the adv one drives the transition.
        """
        n_agent = x_agent.shape[0]
        n_lane = x_lane.shape[0]
        n_adv = x_adv.shape[0]
        adv_batch = self._adv_batch(data)
        t_agent = torch.zeros(n_agent, device=self.device, dtype=torch.long)
        t_lane = torch.zeros(n_lane, device=self.device, dtype=torch.long)
        t_adv = torch.full((n_adv,), int(t), device=self.device, dtype=torch.long)

        _, _, eps_adv = net.model(x_lane, x_agent, x_adv, data, t_agent, t_lane, t_adv)
        x0_adv = net.predict_start_from_noise(x_adv, t=t_adv, noise=eps_adv)

        if self.sampler == "ddpm":
            mean, logvar = net.q_posterior(x0_adv, x_adv, t_adv)
            return mean, logvar

        alpha_t = self._alpha_at(net, t, x_adv.shape, x_adv.device)
        alpha_prev = self._alpha_at(net, prev_t, x_adv.shape, x_adv.device)
        sigma2 = (self.ddim_eta ** 2) * (1.0 - alpha_prev) / (1.0 - alpha_t) * (
            1.0 - alpha_t / alpha_prev
        )
        sigma2 = torch.clamp(sigma2, min=0.0)
        dir_coef = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma2, min=0.0))
        mean = torch.sqrt(alpha_prev) * x0_adv + dir_coef * eps_adv
        logvar = torch.log(torch.clamp(sigma2, min=1e-20))
        return mean, logvar

    # ---------------------------------------------------------------- sample
    @torch.no_grad()
    def sample(
        self,
        conditioning: Batch,
        *,
        use_reference: bool = False,
    ) -> tuple[GeneratedScenes, SamplingTrajectory]:
        data = conditioning.to(self.device)
        net = self.ref if use_reference else self.net
        adv_batch = self._adv_batch(data)
        num_scenes = int(data.batch_size)

        x_agent = self._agent_latents(data)  # fixed ego base latent
        x_lane = self._lane_latents(data)    # fixed lane latents
        x_adv = torch.randn((adv_batch.shape[0], 1, self.agent_latent_dim), device=self.device)

        records = []
        n_steps = len(self.step_pairs)
        old_lp = torch.zeros((num_scenes, n_steps), device=self.device)

        for j, (t, prev_t) in enumerate(self.step_pairs):
            mean_a, logvar_a = self._adv_mean_logvar(net, x_adv, x_agent, x_lane, data, t, prev_t)
            stochastic = self._is_stochastic_transition(t, prev_t)
            if stochastic:
                x_next = mean_a + (0.5 * logvar_a).exp() * torch.randn_like(x_adv)
            else:
                x_next = mean_a
            x_next = torch.clip(x_next, -self.diffusion_clip, self.diffusion_clip)

            if stochastic:
                node_lp = _gaussian_logprob(x_next, mean_a, logvar_a)
                old_lp[:, j].index_add_(0, adv_batch, node_lp)

            records.append((x_adv.detach(), x_next.detach(), t, prev_t))
            x_adv = x_next

        scenes = self._decode(x_agent, x_lane, x_adv, data)
        traj = SamplingTrajectory(
            records={"steps": records, "adv_batch": adv_batch},
            old_logprob=old_lp.detach(),
            num_steps=n_steps,
            num_scenes=num_scenes,
        )
        return scenes, traj

    @torch.no_grad()
    def conditioning_scenes(self, conditioning: Batch) -> GeneratedScenes:
        """Decode the real (ground-truth) ego + adv latents for reference viz."""
        data = conditioning.to(self.device)
        x_agent = self._agent_latents(data)
        x_lane = self._lane_latents(data)
        x_adv = data["adv"].latents.float().to(self.device).unsqueeze(1)
        return self._decode(x_agent, x_lane, x_adv, data)

    # ----------------------------------------------------------- scoring
    def trajectory_logprob(
        self,
        trajectory: SamplingTrajectory,
        conditioning: Batch,
        step_indices: torch.Tensor,
        *,
        use_reference: bool = False,
        with_kl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Per-step log-prob of the recorded adv trajectory (optionally with KL).

        Both returned tensors are ``[num_scenes, len(step_indices)]``. The base
        ego/lane latents are recomputed from the conditioning (held fixed).
        """
        data = conditioning.to(self.device)
        net = self.ref if use_reference else self.net
        steps = trajectory.records["steps"]
        adv_batch = trajectory.records["adv_batch"]
        num_scenes = trajectory.num_scenes
        x_agent = self._agent_latents(data)
        x_lane = self._lane_latents(data)

        out = torch.zeros((num_scenes, len(step_indices)), device=self.device)
        kl = torch.zeros((num_scenes, len(step_indices)), device=self.device) if with_kl else None
        ctx = torch.no_grad() if use_reference else torch.enable_grad()
        with ctx:
            for col, s in enumerate(step_indices.tolist()):
                x_t, x_tm1, t, prev_t = steps[s]
                if not self._is_stochastic_transition(t, prev_t):
                    continue
                mean_a, logvar_a = self._adv_mean_logvar(net, x_t, x_agent, x_lane, data, t, prev_t)
                node_lp = _gaussian_logprob(x_tm1, mean_a, logvar_a)
                out[:, col] = out[:, col].index_add(0, adv_batch, node_lp)
                if with_kl:
                    with torch.no_grad():
                        mean_ref, _ = self._adv_mean_logvar(
                            self.ref, x_t, x_agent, x_lane, data, t, prev_t
                        )
                    node_kl = _gaussian_kl(mean_a, mean_ref, logvar_a)
                    kl[:, col] = kl[:, col].index_add(0, adv_batch, node_kl)
        return out, kl

    # ----------------------------------------------------------- decode
    @torch.no_grad()
    def _decode(self, x_agent, x_lane, x_adv, data) -> GeneratedScenes:
        """Decode the ``ego + adv`` scene: re-insert the adv into the base agent
        set, decode the full set in one pass (permutation-equivariant set decoder),
        then expose every scene as ``[ego, adv]`` with ``controlled_mask`` on the
        adv. Mirrors ``ScenarioDreamerLDMAdv._decode_scene_and_adv``."""
        agent_latents, lane_latents = unnormalize_latents(
            x_agent[:, 0],
            x_lane[:, 0],
            self.agent_latents_mean,
            self.agent_latents_std,
            self.lane_latents_mean,
            self.lane_latents_std,
        )
        # the adversary shares the agent latent statistics
        adv_latents = x_adv[:, 0] * self.agent_latents_std + self.agent_latents_mean

        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        adv_batch = self._adv_batch(data)
        num_base = agent_latents.shape[0]

        combined_latents = torch.cat([agent_latents, adv_latents], dim=0)
        combined_batch = torch.cat([agent_batch, adv_batch], dim=0)

        same_scene = combined_batch.unsqueeze(0) == combined_batch.unsqueeze(1)
        a2a_edge_index = same_scene.nonzero(as_tuple=False).t().contiguous()
        l2a_src, l2a_dst = (lane_batch.unsqueeze(1) == combined_batch.unsqueeze(0)).nonzero(as_tuple=True)
        l2a_edge_index = torch.stack([l2a_src, l2a_dst], dim=0)

        dec_data = ScenarioDreamerData()
        dec_data["lane"].x = lane_latents
        dec_data["lane", "to", "lane"].edge_index = data["lane", "to", "lane"].edge_index
        dec_data["agent", "to", "agent"].edge_index = a2a_edge_index
        dec_data["lane", "to", "agent"].edge_index = l2a_edge_index

        agent_states, lane_states, agent_types, _, _ = self.ae.forward_decoder(
            combined_latents, lane_latents, dec_data
        )
        agent_states, _ = unnormalize_scene(
            agent_states.clone(),
            lane_states.clone(),
            fov=self.cfg_dataset.fov,
            min_speed=self.cfg_dataset.min_speed,
            max_speed=self.cfg_dataset.max_speed,
            min_length=self.cfg_dataset.min_length,
            max_length=self.cfg_dataset.max_length,
            min_width=self.cfg_dataset.min_width,
            max_width=self.cfg_dataset.max_width,
            min_lane_x=self.cfg_dataset.min_lane_x,
            max_lane_x=self.cfg_dataset.max_lane_x,
            min_lane_y=self.cfg_dataset.min_lane_y,
            max_lane_y=self.cfg_dataset.max_lane_y,
        )
        agent_types = agent_types.clone()

        num_scenes = int(data.batch_size)
        controlled = torch.zeros(combined_latents.shape[0], dtype=torch.bool, device=self.device)
        controlled[num_base:] = True  # the appended adv rows
        if self.force_adv_vehicle:
            agent_states, agent_types = self._project_adv_vehicle(
                agent_states, agent_types, combined_batch, num_scenes, controlled
            )

        meta = {"lane_scene_idx": lane_batch, "controlled_mask": controlled}
        lane_edge_store = data["lane", "to", "lane"]
        if "edge_index" in lane_edge_store:
            meta["lane_edge_index"] = lane_edge_store.edge_index
        if "type" in lane_edge_store:
            meta["lane_edge_type"] = lane_edge_store.type
        return GeneratedScenes(
            agent_states=agent_states,
            agent_types=agent_types,
            agent_scene_idx=combined_batch,
            lane_polylines=data["lane"].road_points,
            num_scenes=num_scenes,
            meta=meta,
        )

    def _project_adv_vehicle(self, agent_states, agent_types, agent_batch, num_scenes, controlled):
        """Force the adversary to be an ego-sized vehicle (type + length/width),
        matching the dm flow's vehicle projection of controlled agents."""
        if not controlled.any():
            return agent_states, agent_types
        counts = torch.bincount(agent_batch, minlength=num_scenes)
        ego_idx = torch.cumsum(counts, 0) - counts  # first (ego) row per scene
        ego_size = agent_states[ego_idx, 5:7]
        agent_states[controlled, 5:7] = ego_size[agent_batch[controlled]]
        agent_types[controlled] = 0  # vehicle type id
        return agent_states, agent_types
