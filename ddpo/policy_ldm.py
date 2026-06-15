"""ldm_goal latent-diffusion model as a DDPO policy.

The policy acts in autoencoder-latent space:
  * LDM denoises agent latents while lane latents stay fixed as conditioning.
  * A frozen goal autoencoder decoder turns sampled latents into physical scenes.
  * DDPO log-prob is accumulated over agent-latent denoising steps only.
"""

from __future__ import annotations

import copy
import math

import torch
from torch_geometric.data import Batch

from nn_modules.autoencoder import AutoEncoder
from nn_modules.ldm import LDM
from utils.data_helpers import unnormalize_latents, unnormalize_scene

from .interfaces import GeneratedScenes, SamplingTrajectory

_LOG_2PI = math.log(2.0 * math.pi)


def _gaussian_logprob(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-node diagonal Gaussian log-density, summed over feature dims."""
    var = logvar.exp()
    per_elem = -0.5 * (((x - mean) ** 2) / var + logvar + _LOG_2PI)
    return per_elem.flatten(1).sum(dim=1)


class LDMGoalDDPOPolicy:
    def __init__(
        self,
        ldm_cfg,
        ae_cfg,
        *,
        ldm_ckpt: str | None,
        ae_ckpt: str | None,
        device: str = "cuda",
        use_ema_weights: bool = True,
    ):
        self.cfg = ldm_cfg
        self.cfg_model = ldm_cfg.model
        self.cfg_dataset = ldm_cfg.dataset
        self.device = device

        self.net = LDM(ldm_cfg).to(device)
        if ldm_ckpt is not None:
            self._load_ldm_checkpoint(ldm_ckpt, use_ema_weights)
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
            print(f"[ddpo ldm_goal] {len(missing)} missing keys on LDM load (e.g. {missing[:3]})")
        if unexpected:
            print(f"[ddpo ldm_goal] {len(unexpected)} unexpected keys on LDM load (e.g. {unexpected[:3]})")
        if use_ema_weights:
            shadow = ckpt.get("ema_state_dict", {}).get("shadow_params", [])
            params = list(self.net.parameters())
            if len(shadow) == len(params) and len(shadow) > 0:
                with torch.no_grad():
                    for p, s in zip(params, shadow):
                        p.copy_(s.to(p.device))
                print(f"[ddpo ldm_goal] loaded {len(shadow)} EMA shadow params")
            else:
                print(
                    f"[ddpo ldm_goal] EMA shadow mismatch "
                    f"(shadow={len(shadow)}, params={len(params)}); raw LDM weights"
                )

    def _load_ae_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        ae_sd = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
        missing, unexpected = self.ae.load_state_dict(ae_sd, strict=False)
        if missing:
            print(f"[ddpo ldm_goal] {len(missing)} missing keys on AE load (e.g. {missing[:3]})")
        if unexpected:
            print(f"[ddpo ldm_goal] {len(unexpected)} unexpected keys on AE load (e.g. {unexpected[:3]})")

    # ------------------------------------------------------------ properties
    @property
    def num_sampling_steps(self) -> int:
        return self._H

    def trainable_parameters(self):
        return (p for p in self.net.parameters() if p.requires_grad)

    def state_dict(self) -> dict:
        return self.net.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.net.load_state_dict(sd)

    # ----------------------------------------------------------- conditioning
    def _lane_latents(self, data: Batch) -> torch.Tensor:
        return data["lane"].latents.float().to(self.device).unsqueeze(1)

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
        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        num_scenes = int(data.batch_size)

        n_agent = data["agent"].x.shape[0]
        x_agent = torch.randn((n_agent, 1, self.agent_latent_dim), device=self.device)
        x_lane = self._lane_latents(data)
        target_lane = x_lane

        records = []
        old_lp = torch.zeros((num_scenes, self._H), device=self.device)

        for j, i in enumerate(reversed(range(self._H))):
            t = torch.full((num_scenes,), i, device=self.device, dtype=torch.long)
            t_agent = t[agent_batch]
            t_lane = t[lane_batch]
            mean_a, logvar_a, _, _ = net.p_mean_variance(x_agent, x_lane, data, t_agent, t_lane)

            x_t = x_agent
            if i > 0:
                x_next = mean_a + (0.5 * logvar_a).exp() * torch.randn_like(x_agent)
            else:
                x_next = mean_a
            x_next = torch.clip(x_next, -self.diffusion_clip, self.diffusion_clip)

            if i > 0:
                node_lp = _gaussian_logprob(x_next, mean_a, logvar_a)
                old_lp[:, j].index_add_(0, agent_batch, node_lp)

            records.append((x_t.detach(), x_next.detach(), i))
            x_agent = x_next
            x_lane = target_lane

        scenes = self._decode(x_agent, x_lane, data)
        traj = SamplingTrajectory(
            records={"steps": records, "agent_batch": agent_batch, "lane_batch": lane_batch},
            old_logprob=old_lp.detach(),
            num_steps=self._H,
            num_scenes=num_scenes,
        )
        return scenes, traj

    @torch.no_grad()
    def conditioning_scenes(self, conditioning: Batch) -> GeneratedScenes:
        data = conditioning.to(self.device)
        x_agent = data["agent"].latents.float().to(self.device).unsqueeze(1)
        x_lane = self._lane_latents(data)
        return self._decode(x_agent, x_lane, data)

    # ----------------------------------------------------------- scoring
    def trajectory_logprob(
        self,
        trajectory: SamplingTrajectory,
        conditioning: Batch,
        step_indices: torch.Tensor,
        *,
        use_reference: bool = False,
    ) -> torch.Tensor:
        data = conditioning.to(self.device)
        net = self.ref if use_reference else self.net
        steps = trajectory.records["steps"]
        agent_batch = trajectory.records["agent_batch"]
        lane_batch = trajectory.records["lane_batch"]
        num_scenes = trajectory.num_scenes
        target_lane = self._lane_latents(data)

        out = torch.zeros((num_scenes, len(step_indices)), device=self.device)
        ctx = torch.no_grad() if use_reference else torch.enable_grad()
        with ctx:
            for col, s in enumerate(step_indices.tolist()):
                x_t, x_tm1, i = steps[s]
                if i == 0:
                    continue
                t = torch.full((num_scenes,), i, device=self.device, dtype=torch.long)
                mean_a, logvar_a, _, _ = net.p_mean_variance(
                    x_t, target_lane, data, t[agent_batch], t[lane_batch]
                )
                node_lp = _gaussian_logprob(x_tm1, mean_a, logvar_a)
                out[:, col] = out[:, col].index_add(0, agent_batch, node_lp)
        return out

    # ----------------------------------------------------------- decode
    @torch.no_grad()
    def _decode(self, x_agent, x_lane, data) -> GeneratedScenes:
        agent_latents, lane_latents = unnormalize_latents(
            x_agent[:, 0],
            x_lane[:, 0],
            self.agent_latents_mean,
            self.agent_latents_std,
            self.lane_latents_mean,
            self.lane_latents_std,
        )
        agent_states, lane_states, agent_types, _, _ = self.ae.forward_decoder(
            agent_latents, lane_latents, data
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
        return GeneratedScenes(
            agent_states=agent_states,
            agent_types=agent_types,
            agent_scene_idx=data["agent"].batch,
            lane_polylines=data["lane"].road_points,
            num_scenes=int(data.batch_size),
            meta={"lane_scene_idx": data["lane"].batch},
        )
