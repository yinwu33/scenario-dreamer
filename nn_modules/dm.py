import numpy as np
import torch
from torch import nn

from cfgs.config import BEFORE_PARTITION
from nn_modules.dit import DiT
from utils.diffusion_helpers import cosine_beta_schedule, extract
from utils.losses import GeometricLosses


class DM(nn.Module):
    def __init__(self, cfg):
        super(DM, self).__init__()
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
        self.agent_type_loss_fn = GeometricLosses["cross_entropy"](apply_mean=False)

    def _agent_target(self, data):
        return torch.cat([data["agent"].x.float(), data["agent"].type.float()], dim=-1).unsqueeze(1)

    def _lane_target(self, data):
        return data["lane"].x.float().reshape(data["lane"].x.shape[0], 1, -1)

    def _split_agent(self, x_agent):
        state_dim = self.cfg_model.state_dim
        return x_agent[:, :state_dim], x_agent[:, state_dim:]

    def _reshape_lane(self, x_lane):
        return x_lane.reshape(x_lane.shape[0], self.cfg_model.num_points_per_lane, self.cfg_model.lane_attr)

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

    def p_mean_variance(self, x_agent, x_lane, data, t_agent, t_lane):
        conditional_epsilon_agent, conditional_epsilon_lane = self.model(
            x_agent, x_lane, data, t_agent, t_lane, unconditional=False
        )
        unconditional_epsilon_agent, unconditional_epsilon_lane = self.model(
            x_agent, x_lane, data, t_agent, t_lane, unconditional=True
        )
        epsilon_agent = unconditional_epsilon_agent + self.cfg.train.guidance_scale * (
            conditional_epsilon_agent - unconditional_epsilon_agent
        )
        epsilon_lane = unconditional_epsilon_lane + self.cfg.train.guidance_scale * (
            conditional_epsilon_lane - unconditional_epsilon_lane
        )

        t_agent = t_agent.detach().to(torch.int64)
        t_lane = t_lane.detach().to(torch.int64)
        x_agent_recon = self.predict_start_from_noise(x_agent, t=t_agent, noise=epsilon_agent)
        x_lane_recon = self.predict_start_from_noise(x_lane, t=t_lane, noise=epsilon_lane)

        model_mean_agent, posterior_log_variance_agent = self.q_posterior(x_agent_recon, x_agent, t_agent)
        model_mean_lane, posterior_log_variance_lane = self.q_posterior(x_lane_recon, x_lane, t_lane)
        return model_mean_agent, posterior_log_variance_agent, model_mean_lane, posterior_log_variance_lane

    @torch.no_grad()
    def p_sample(self, x_agent, x_lane, data, t_agent, t_lane):
        b_agent = t_agent.shape[0]
        b_lane = t_lane.shape[0]
        model_mean_agent, model_log_variance_agent, model_mean_lane, model_log_variance_lane = self.p_mean_variance(
            x_agent, x_lane, data, t_agent, t_lane
        )

        noise_agent = torch.randn_like(x_agent)
        noise_lane = torch.randn_like(x_lane)
        nonzero_mask_agent = (1 - (t_agent == 0).float()).reshape(b_agent, *((1,) * (len(x_agent.shape) - 1)))
        nonzero_mask_lane = (1 - (t_lane == 0).float()).reshape(b_lane, *((1,) * (len(x_lane.shape) - 1)))

        next_x_agent = model_mean_agent + nonzero_mask_agent * model_log_variance_agent.exp().sqrt() * noise_agent
        next_x_lane = model_mean_lane + nonzero_mask_lane * model_log_variance_lane.exp().sqrt() * noise_lane
        return next_x_agent, next_x_lane

    @torch.no_grad()
    def p_sample_loop(self, agent_shape, lane_shape, data, device="cuda", mode="initial_scene"):
        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        batch_size = data.batch_size

        x_agent = torch.randn(agent_shape, device=device)
        x_lane = torch.randn(lane_shape, device=device) * self.lane_sampling_temperature

        if mode == "lane_conditioned":
            x_lane = self._lane_target(data).to(device)

        if mode in ("train", "inpainting"):
            target_agent = self._agent_target(data).to(device)
            target_lane = self._lane_target(data).to(device)
            if "partition_mask" in data["agent"]:
                agent_mask = data["agent"].partition_mask == BEFORE_PARTITION
                x_agent[agent_mask] = target_agent[agent_mask]
            if "partition_mask" in data["lane"]:
                lane_mask = data["lane"].partition_mask == BEFORE_PARTITION
                x_lane[lane_mask] = target_lane[lane_mask]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_agent = timesteps[agent_batch]
            t_lane = timesteps[lane_batch]
            x_agent, x_lane = self.p_sample(x_agent, x_lane, data, t_agent, t_lane)
            x_agent = torch.clip(x_agent, -self.cfg_model.diffusion_clip, self.cfg_model.diffusion_clip)
            x_lane = torch.clip(x_lane, -self.cfg_model.diffusion_clip, self.cfg_model.diffusion_clip)

            if mode == "lane_conditioned":
                x_lane = self._lane_target(data).to(device)
            if mode in ("train", "inpainting"):
                if "partition_mask" in data["agent"]:
                    agent_mask = data["agent"].partition_mask == BEFORE_PARTITION
                    x_agent[agent_mask] = target_agent[agent_mask]
                if "partition_mask" in data["lane"]:
                    lane_mask = data["lane"].partition_mask == BEFORE_PARTITION
                    x_lane[lane_mask] = target_lane[lane_mask]

        return x_agent[:, 0], x_lane[:, 0]

    @torch.no_grad()
    def forward(self, data, mode="initial_scene"):
        agent_shape = (data["agent"].x.shape[0], 1, self.cfg_model.agent_latent_dim)
        lane_shape = (data["lane"].x.shape[0], 1, self.cfg_model.lane_latent_dim)
        x_agent, x_lane = self.p_sample_loop(
            agent_shape,
            lane_shape,
            data,
            device=data["agent"].x.device,
            mode=mode,
        )
        return self.decode_outputs(x_agent, x_lane, data)

    def q_sample(self, x_start, t, noise=None):
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def p_losses(self, x_agent, x_lane, data, t_agent, t_lane):
        agent_noise = torch.randn_like(x_agent)
        lane_noise = torch.randn_like(x_lane)
        x_agent_noisy = self.q_sample(x_start=x_agent, t=t_agent, noise=agent_noise)
        x_lane_noisy = self.q_sample(x_start=x_lane, t=t_lane, noise=lane_noise)

        agent_mask = data["agent"].partition_mask == BEFORE_PARTITION
        lane_mask = data["lane"].partition_mask == BEFORE_PARTITION
        x_agent_noisy[agent_mask] = x_agent[agent_mask]
        x_lane_noisy[lane_mask] = x_lane[lane_mask]

        agent_noise_pred, lane_noise_pred = self.model(
            x_agent_noisy,
            x_lane_noisy,
            data,
            t_agent,
            t_lane,
        )

        agent_noise[agent_mask] = 0.0
        lane_noise[lane_mask] = 0.0
        agent_loss = self.agent_loss_fn(agent_noise_pred, agent_noise, data["agent"].batch)
        lane_loss = self.lane_loss_fn(lane_noise_pred, lane_noise, data["lane"].batch)

        x_agent_recon = self.predict_start_from_noise(x_agent_noisy, t_agent.to(torch.int64), agent_noise_pred)[:, 0]
        _, agent_type_logits = self._split_agent(x_agent_recon)
        agent_type_loss = self.agent_type_loss_fn(agent_type_logits, data["agent"].type.float(), data["agent"].batch)

        loss = (
            agent_loss
            + self.cfg.train.lane_weight * lane_loss
            + self.cfg.train.agent_type_weight * agent_type_loss
        )
        return loss, agent_loss, lane_loss, agent_type_loss

    def loss(self, data):
        x_agent = self._agent_target(data)
        x_lane = self._lane_target(data)

        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        batch_size = data.batch_size
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_agent.device).long()
        t_agent = t[agent_batch]
        t_lane = t[lane_batch]

        loss, agent_loss, lane_loss, agent_type_loss = self.p_losses(
            x_agent, x_lane, data, t_agent, t_lane
        )
        return {
            "loss": loss.mean(),
            "agent_loss": agent_loss.mean().detach(),
            "lane_loss": lane_loss.mean().detach(),
            "agent_type_loss": agent_type_loss.mean().detach(),
        }

    @torch.no_grad()
    def decode_outputs(self, x_agent, x_lane, data):
        agent_states, agent_type_logits = self._split_agent(x_agent)
        agent_types = torch.argmax(agent_type_logits, dim=1)
        lane_states = self._reshape_lane(x_lane)

        lane_conn_pred = torch.zeros(
            data["lane", "to", "lane"].edge_index.shape[1],
            self.cfg_dataset.num_lane_connection_types,
            device=x_agent.device,
            dtype=torch.float32,
        )
        lane_conn_pred[:, 0] = 1.0
        return agent_states, lane_states, agent_types, None, lane_conn_pred
