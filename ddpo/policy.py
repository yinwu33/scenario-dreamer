"""dm_goal diffusion model as a DDPO policy, with three training modes.

Wraps the NATIVE scenario-dreamer ``DMGoal`` network (``nn_modules.dm_goal``) -
no vendored copy - and exposes the DDPM ancestral-sampling chain as a stochastic
policy:

  * horizon  H = n_diffusion_timesteps
  * state    s_t = (noisy chains x_t, fixed conditioning graph)
  * action   a_t = sampled x_{t-1} over the FREE dimensions of the mode
  * policy   pi_theta(a_t | s_t) = N(guided_posterior_mean_theta, Sigma_t)

Agent chain layout per node: [x, y, speed, cos, sin, length, width,
goal_x, goal_y, type_onehot(3)]  (12 dims). Lane chain: 20*2 = 40 dims per node.

Modes (``cfg.ddpo.mode``):
  * "goal"      - map fixed, agent init states fixed (inpainted from the real
                  conditioning scene), only the per-agent goal dims [7:9] are
                  generated/trained. Agent types are fixed to GT too.
  * "init_goal" - map fixed, all agent dims (init states + goals + types) are
                  generated/trained. This is the previous scene_init_ddpo
                  lane-conditioned behaviour.
  * "all"       - agent dims AND the lane chain are generated/trained (only the
                  graph structure - node counts, edges, lg_type - stays real).

Fixed agent dims are inpainted at every step with the forward-diffused ground
truth ``q_sample(x0, t)`` (standard replacement inpainting), or with the clean
``x0`` if ``inpaint_noised=False``. Log-probs are accumulated over free dims
only, so fixed dims never contribute policy gradient.
"""

from __future__ import annotations

import copy
import math

import torch
from torch_geometric.data import Batch

from models.scenario_dreamer_dm_goal import unnormalize_scene_with_goal
from nn_modules.dm_goal import DMGoal

from .interfaces import GeneratedScenes, SamplingTrajectory

_LOG_2PI = math.log(2.0 * math.pi)
MODES = ("goal", "init_goal", "all")
GOAL_DIMS = (7, 8)


def _masked_gaussian_logprob(x, mean, logvar, free_mask):
    """Diagonal-Gaussian log-density over free dims only; [N,1,D] -> [N].

    ``logvar`` is the per-node posterior log-variance (broadcast over dims);
    ``free_mask`` is a [D] bool tensor.
    """
    var = logvar.exp()
    per_elem = -0.5 * (((x - mean) ** 2) / var + logvar + _LOG_2PI)
    return per_elem[..., free_mask].flatten(1).sum(dim=1)


class DMGoalDDPOPolicy:
    def __init__(
        self,
        cfg,                      # cfg.dm_goal node (model / dataset / train children)
        ckpt_path: str | None,
        *,
        mode: str = "init_goal",
        device: str = "cuda",
        use_ema_weights: bool = True,
        inpaint_noised: bool = True,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.cfg = cfg
        self.cfg_model = cfg.model
        self.cfg_dataset = cfg.dataset
        self.mode = mode
        self.device = device
        self.inpaint_noised = inpaint_noised

        self.net = DMGoal(cfg).to(device)
        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path, use_ema_weights)
        # eval() disables dropout/label-dropout so recomputed log-probs are
        # deterministic (IS ratio == 1 on epoch 0); gradients still flow.
        self.net.eval()

        self.ref = copy.deepcopy(self.net).to(device).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)

        self._H = int(self.net.n_timesteps)
        self.agent_latent_dim = int(self.cfg_model.agent_latent_dim)
        self.diffusion_clip = self.cfg_model.diffusion_clip
        self.lane_free = mode == "all"

        free = torch.zeros(self.agent_latent_dim, dtype=torch.bool)
        if mode == "goal":
            free[list(GOAL_DIMS)] = True
        else:
            free[:] = True
        self.agent_free_mask = free.to(device)
        self.agent_fixed = bool((~free).any())

    # ------------------------------------------------------------------ load
    def _load_checkpoint(self, ckpt_path: str, use_ema_weights: bool) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        net_sd = {k[len("diff_model."):]: v for k, v in sd.items() if k.startswith("diff_model.")}
        missing, unexpected = self.net.load_state_dict(net_sd, strict=False)
        if missing:
            print(f"[ddpo] {len(missing)} missing keys on load (e.g. {missing[:3]})")
        if use_ema_weights:
            shadow = ckpt.get("ema_state_dict", {}).get("shadow_params", [])
            params = list(self.net.parameters())
            if len(shadow) == len(params) and len(shadow) > 0:
                with torch.no_grad():
                    for p, s in zip(params, shadow):
                        p.copy_(s.to(p.device))
                print(f"[ddpo] loaded {len(shadow)} EMA shadow params")
            else:
                print(f"[ddpo] EMA shadow mismatch (shadow={len(shadow)}, params={len(params)}); raw weights")

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

    # -------------------------------------------------------------- inpainting
    def _inpaint_agent(self, x_agent, target_agent, t_agent):
        """Overwrite fixed agent dims at the noise level of step t (mode=goal)."""
        if not self.agent_fixed:
            return x_agent
        if self.inpaint_noised:
            noise = torch.randn_like(target_agent)
            x_fixed = self.net.q_sample(target_agent, t_agent, noise=noise)
        else:
            x_fixed = target_agent
        return torch.where(self.agent_free_mask.view(1, 1, -1), x_agent, x_fixed)

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
        target_agent = net._agent_target(data).to(self.device) if self.agent_fixed else None
        target_lane = net._lane_target(data).to(self.device)

        if self.lane_free:
            x_lane = torch.randn_like(target_lane) * net.lane_sampling_temperature
        else:
            x_lane = target_lane

        records = []
        old_lp = torch.zeros((num_scenes, self._H), device=self.device)

        for j, i in enumerate(reversed(range(self._H))):
            t = torch.full((num_scenes,), i, device=self.device, dtype=torch.long)
            t_agent = t[agent_batch]
            t_lane = t[lane_batch]

            x_agent = self._inpaint_agent(x_agent, target_agent, t_agent)
            mean_a, logvar_a, mean_l, logvar_l = net.p_mean_variance(x_agent, x_lane, data, t_agent, t_lane)

            x_t_agent = x_agent
            x_t_lane = x_lane if self.lane_free else None
            if i > 0:
                x_next_a = mean_a + (0.5 * logvar_a).exp() * torch.randn_like(x_agent)
            else:
                x_next_a = mean_a
            x_next_a = torch.clip(x_next_a, -self.diffusion_clip, self.diffusion_clip)

            if self.lane_free:
                if i > 0:
                    x_next_l = mean_l + (0.5 * logvar_l).exp() * torch.randn_like(x_lane)
                else:
                    x_next_l = mean_l
                x_next_l = torch.clip(x_next_l, -self.diffusion_clip, self.diffusion_clip)
            else:
                x_next_l = target_lane

            if i > 0:
                node_lp = _masked_gaussian_logprob(x_next_a, mean_a, logvar_a, self.agent_free_mask)
                old_lp[:, j].index_add_(0, agent_batch, node_lp)
                if self.lane_free:
                    lane_mask = torch.ones(x_lane.shape[-1], dtype=torch.bool, device=self.device)
                    lane_lp = _masked_gaussian_logprob(x_next_l, mean_l, logvar_l, lane_mask)
                    old_lp[:, j].index_add_(0, lane_batch, lane_lp)

            records.append((
                x_t_agent.detach(),
                x_next_a.detach(),
                x_t_lane.detach() if x_t_lane is not None else None,
                x_next_l.detach() if self.lane_free else None,
                i,
            ))
            x_agent = x_next_a
            x_lane = x_next_l

        # decode with fixed dims restored to the exact clean targets
        if self.agent_fixed:
            x_agent = torch.where(self.agent_free_mask.view(1, 1, -1), x_agent, target_agent)
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
        """Decode the unmodified conditioning graph as simulator-ready scenes."""
        data = conditioning.to(self.device)
        x_agent = self.net._agent_target(data).to(self.device)
        x_lane = self.net._lane_target(data).to(self.device)
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
        target_lane = self.net._lane_target(data).to(self.device)

        out = torch.zeros((num_scenes, len(step_indices)), device=self.device)
        ctx = torch.no_grad() if use_reference else torch.enable_grad()
        with ctx:
            for col, s in enumerate(step_indices.tolist()):
                x_t_a, x_tm1_a, x_t_l, x_tm1_l, i = steps[s]
                if i == 0:
                    continue  # deterministic step carries no policy gradient
                t = torch.full((num_scenes,), i, device=self.device, dtype=torch.long)
                x_lane_in = x_t_l if self.lane_free else target_lane
                mean_a, logvar_a, mean_l, logvar_l = net.p_mean_variance(
                    x_t_a, x_lane_in, data, t[agent_batch], t[lane_batch]
                )
                node_lp = _masked_gaussian_logprob(x_tm1_a, mean_a, logvar_a, self.agent_free_mask)
                out[:, col] = out[:, col].index_add(0, agent_batch, node_lp)
                if self.lane_free:
                    lane_mask = torch.ones(x_lane_in.shape[-1], dtype=torch.bool, device=self.device)
                    lane_lp = _masked_gaussian_logprob(x_tm1_l, mean_l, logvar_l, lane_mask)
                    out[:, col] = out[:, col].index_add(0, lane_batch, lane_lp)
        return out

    # ----------------------------------------------------------- decode
    @torch.no_grad()
    def _decode(self, x_agent, x_lane, data) -> GeneratedScenes:
        agent_states, lane_states, agent_types, _, _ = self.net.decode_outputs(
            x_agent[:, 0], x_lane, data
        )
        # decode_outputs returns VIEWS into x_agent / x_lane (and x_lane may itself
        # be a view of data['lane'].x via _lane_target). unnormalize_* writes in
        # place, which would corrupt the conditioning graph and the recorded
        # trajectory -> clone before unnormalising.
        agent_states = agent_states.clone()
        lane_states = lane_states.clone()
        if self.mode == "goal":
            # types are not generated in goal mode; use the conditioning types
            agent_types = torch.argmax(data["agent"].type, dim=-1)
        agent_states, lane_states = unnormalize_scene_with_goal(agent_states, lane_states, self.cfg_dataset)
        meta = {"lane_scene_idx": data["lane"].batch}
        if self.mode == "goal":
            # GT parking state per agent (goal within reach radius of spawn). Only
            # meaningful in goal mode, where generated agents correspond 1:1 to the
            # conditioning agents (init states are inpainted from GT). The reward
            # penalises generated scenes whose parking state diverges from this.
            fov = float(self.cfg_dataset.fov)
            gt = data["agent"].x.float()
            gt_init = (gt[:, 0:2] + 1) / 2 * fov - fov / 2
            gt_goal = (gt[:, 7:9] + 1) / 2 * fov - fov / 2
            gt_dist = torch.linalg.norm(gt_goal - gt_init, dim=-1)
            meta["gt_parking_mask"] = gt_dist < 2.0  # MIN_DISTANCE_TO_GOAL
        return GeneratedScenes(
            agent_states=agent_states,
            agent_types=agent_types,
            agent_scene_idx=data["agent"].batch,
            lane_polylines=lane_states,
            num_scenes=int(data.batch_size),
            meta=meta,
        )
