import numpy as np
import torch
from torch import nn

from nn_modules.dit_ex import DiT
from utils.diffusion_helpers import cosine_beta_schedule, extract
from utils.losses import GeometricLosses


class LDMAdv(nn.Module):
    """Latent diffusion over the goal-autoencoder latents with normal agents plus
    one adversarial agent per scene.

    The latent-space analogue of :class:`~nn_modules.dm_adv.DMAdv`: it reuses the
    exact adversary-aware ``dit_ex`` DiT (lane / agent / adv streams with the
    ``la2adv`` cross-attention) but diffuses the autoencoder latents instead of
    raw geometry. Because the latent already encodes the agent's full state
    (including its goal) and the autoencoder decoder produces the discrete
    attributes (types, lane connectivity), there is **no** type/connectivity loss
    here -- only the three latent epsilon losses.

    Generation modes:
      init_scene: generate lane, normal-agent, and adv latents.
      init_agent: condition on lane latents, generate normal-agent and adv latents.
      init_adv:   condition on lane and normal-agent latents, generate adv only.
    """

    def __init__(self, cfg):
        super(LDMAdv, self).__init__()
        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_dataset = self.cfg.dataset
        self.model = DiT(cfg)

        n_timesteps = self.cfg_model.n_diffusion_timesteps
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.lane_sampling_temperature = self.cfg_model.lane_sampling_temperature
        self.adv_weight = float(self.cfg.train.get("adv_weight", 1.0))
        self.train_mode = self.cfg.train.get("train_mode", "all")
        if bool(self.cfg.train.get("adv_only", False)):
            self.train_mode = "adv_only"
        if self.train_mode not in ("all", "adv_only"):
            raise ValueError(f"Unsupported ldm_adv train_mode: {self.train_mode!r}")

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

        loss_type = self.cfg.train.loss_type
        self.agent_loss_fn = GeometricLosses[loss_type]((1, 2))
        self.lane_loss_fn = GeometricLosses[loss_type]((1, 2))
        self.adv_loss_fn = GeometricLosses[loss_type]((1, 2))

        if self.train_mode == "adv_only":
            self.model.freeze_non_adv_parameters()

    def _agent_latent(self, data):
        return data["agent"].latents.float().unsqueeze(1)

    def _lane_latent(self, data):
        return data["lane"].latents.float().unsqueeze(1)

    def _adv_latent(self, data):
        return data["adv"].latents.float().unsqueeze(1)

    def _adv_batch(self, data):
        if "batch" in data["adv"]:
            return data["adv"].batch
        return torch.arange(data.batch_size, device=data["adv"].latents.device, dtype=torch.long)

    def _normalize_mode(self, mode):
        aliases = {
            "initial_scene": "init_scene",
            "train": "init_scene",
            "lane_conditioned": "init_agent",
            "adv_conditioned": "init_adv",
        }
        mode = aliases.get(mode, mode)
        if mode not in ("init_scene", "init_agent", "init_adv"):
            raise ValueError(f"Unsupported LDMAdv generation mode: {mode!r}")
        return mode

    def _zero_scene_loss(self, data, dtype, device):
        return torch.zeros(int(data.batch_size), device=device, dtype=dtype)

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ) * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x_agent, x_lane, x_adv, data, t_agent, t_lane, t_adv):
        epsilon_agent, epsilon_lane, epsilon_adv = self.model(
            x_lane,
            x_agent,
            x_adv,
            data,
            t_agent,
            t_lane,
            t_adv,
        )

        t_agent = t_agent.detach().to(torch.int64)
        t_lane = t_lane.detach().to(torch.int64)
        t_adv = t_adv.detach().to(torch.int64)
        x_agent_recon = self.predict_start_from_noise(x_agent, t=t_agent, noise=epsilon_agent)
        x_lane_recon = self.predict_start_from_noise(x_lane, t=t_lane, noise=epsilon_lane)
        x_adv_recon = self.predict_start_from_noise(x_adv, t=t_adv, noise=epsilon_adv)

        model_mean_agent, posterior_log_variance_agent = self.q_posterior(x_agent_recon, x_agent, t_agent)
        model_mean_lane, posterior_log_variance_lane = self.q_posterior(x_lane_recon, x_lane, t_lane)
        model_mean_adv, posterior_log_variance_adv = self.q_posterior(x_adv_recon, x_adv, t_adv)
        return (
            model_mean_agent,
            posterior_log_variance_agent,
            model_mean_lane,
            posterior_log_variance_lane,
            model_mean_adv,
            posterior_log_variance_adv,
        )

    @torch.no_grad()
    def p_sample(self, x_agent, x_lane, x_adv, data, t_agent, t_lane, t_adv):
        b_agent = t_agent.shape[0]
        b_lane = t_lane.shape[0]
        b_adv = t_adv.shape[0]
        (
            model_mean_agent,
            model_log_variance_agent,
            model_mean_lane,
            model_log_variance_lane,
            model_mean_adv,
            model_log_variance_adv,
        ) = self.p_mean_variance(x_agent, x_lane, x_adv, data, t_agent, t_lane, t_adv)

        noise_agent = torch.randn_like(x_agent)
        noise_lane = torch.randn_like(x_lane)
        noise_adv = torch.randn_like(x_adv)
        nonzero_mask_agent = (1 - (t_agent == 0).float()).reshape(b_agent, *((1,) * (len(x_agent.shape) - 1)))
        nonzero_mask_lane = (1 - (t_lane == 0).float()).reshape(b_lane, *((1,) * (len(x_lane.shape) - 1)))
        nonzero_mask_adv = (1 - (t_adv == 0).float()).reshape(b_adv, *((1,) * (len(x_adv.shape) - 1)))

        next_x_agent = model_mean_agent + nonzero_mask_agent * model_log_variance_agent.exp().sqrt() * noise_agent
        next_x_lane = model_mean_lane + nonzero_mask_lane * model_log_variance_lane.exp().sqrt() * noise_lane * self.lane_sampling_temperature
        next_x_adv = model_mean_adv + nonzero_mask_adv * model_log_variance_adv.exp().sqrt() * noise_adv
        return next_x_agent, next_x_lane, next_x_adv

    @torch.no_grad()
    def p_sample_loop(self, agent_shape, lane_shape, adv_shape, data, device="cuda", mode="initial_scene"):
        mode = self._normalize_mode(mode)
        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        adv_batch = self._adv_batch(data)
        batch_size = data.batch_size

        x_agent = torch.randn(agent_shape, device=device)
        x_lane = torch.randn(lane_shape, device=device) * self.lane_sampling_temperature
        x_adv = torch.randn(adv_shape, device=device)

        if mode in ("init_agent", "init_adv"):
            x_lane = self._lane_latent(data).to(device)
        if mode == "init_adv":
            x_agent = self._agent_latent(data).to(device)

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_agent = torch.zeros_like(timesteps[agent_batch]) if mode == "init_adv" else timesteps[agent_batch]
            t_lane = torch.zeros_like(timesteps[lane_batch]) if mode in ("init_agent", "init_adv") else timesteps[lane_batch]
            t_adv = timesteps[adv_batch]
            x_agent, x_lane, x_adv = self.p_sample(x_agent, x_lane, x_adv, data, t_agent, t_lane, t_adv)
            x_adv = torch.clip(x_adv, -self.cfg_model.diffusion_clip, self.cfg_model.diffusion_clip)
            if mode == "init_scene":
                x_lane = torch.clip(x_lane, -self.cfg_model.diffusion_clip, self.cfg_model.diffusion_clip)
                x_agent = torch.clip(x_agent, -self.cfg_model.diffusion_clip, self.cfg_model.diffusion_clip)
            elif mode == "init_agent":
                x_lane = self._lane_latent(data).to(device)
                x_agent = torch.clip(x_agent, -self.cfg_model.diffusion_clip, self.cfg_model.diffusion_clip)
            else:
                x_lane = self._lane_latent(data).to(device)
                x_agent = self._agent_latent(data).to(device)

        return x_agent[:, 0], x_lane[:, 0], x_adv[:, 0]

    @torch.no_grad()
    def forward(self, data, mode="initial_scene"):
        agent_shape = (data["agent"].latents.shape[0], 1, self.cfg_model.agent_latent_dim)
        lane_shape = (data["lane"].latents.shape[0], 1, self.cfg_model.lane_latent_dim)
        adv_shape = (data["adv"].latents.shape[0], 1, self.cfg_model.agent_latent_dim)
        x_agent, x_lane, x_adv = self.p_sample_loop(
            agent_shape,
            lane_shape,
            adv_shape,
            data,
            device=data["agent"].latents.device,
            mode=mode,
        )
        return x_agent, x_lane, x_adv

    def q_sample(self, x_start, t, noise=None):
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def p_losses(self, x_agent, x_lane, x_adv, data, t_agent, t_lane, t_adv):
        agent_noise = torch.randn_like(x_agent)
        lane_noise = torch.randn_like(x_lane)
        adv_noise = torch.randn_like(x_adv)
        if self.train_mode == "adv_only":
            x_agent_noisy = x_agent
            x_lane_noisy = x_lane
            agent_noise = torch.zeros_like(agent_noise)
            lane_noise = torch.zeros_like(lane_noise)
        else:
            x_agent_noisy = self.q_sample(x_start=x_agent, t=t_agent, noise=agent_noise)
            x_lane_noisy = self.q_sample(x_start=x_lane, t=t_lane, noise=lane_noise)
        x_adv_noisy = self.q_sample(x_start=x_adv, t=t_adv, noise=adv_noise)

        agent_noise_pred, lane_noise_pred, adv_noise_pred = self.model(
            x_lane_noisy,
            x_agent_noisy,
            x_adv_noisy,
            data,
            t_agent,
            t_lane,
            t_adv,
        )

        adv_batch = self._adv_batch(data)
        adv_loss = self.adv_loss_fn(adv_noise_pred, adv_noise, adv_batch)

        if self.train_mode == "adv_only":
            agent_loss = self._zero_scene_loss(data, adv_loss.dtype, adv_loss.device)
            lane_loss = self._zero_scene_loss(data, adv_loss.dtype, adv_loss.device)
            loss = self.adv_weight * adv_loss
        else:
            agent_loss = self.agent_loss_fn(agent_noise_pred, agent_noise, data["agent"].batch)
            lane_loss = self.lane_loss_fn(lane_noise_pred, lane_noise, data["lane"].batch)
            loss = agent_loss + self.adv_weight * adv_loss + self.cfg.train.lane_weight * lane_loss
        return loss, agent_loss, lane_loss, adv_loss

    def loss(self, data):
        x_agent = self._agent_latent(data)
        x_lane = self._lane_latent(data)
        x_adv = self._adv_latent(data)

        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        adv_batch = self._adv_batch(data)
        batch_size = data.batch_size
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_agent.device).long()
        if self.train_mode == "adv_only":
            t_agent = torch.zeros_like(t[agent_batch])
            t_lane = torch.zeros_like(t[lane_batch])
        else:
            t_agent = t[agent_batch]
            t_lane = t[lane_batch]
        t_adv = t[adv_batch]

        loss, agent_loss, lane_loss, adv_loss = self.p_losses(x_agent, x_lane, x_adv, data, t_agent, t_lane, t_adv)
        return {
            "loss": loss.mean(),
            "agent_loss": agent_loss.mean().detach(),
            "lane_loss": lane_loss.mean().detach(),
            "adv_loss": adv_loss.mean().detach(),
        }
