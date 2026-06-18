"""dm_goal diffusion model as a DDPO policy, with three training modes.

Wraps the NATIVE scenario-dreamer ``DMGoal`` network (``nn_modules.dm_goal``) -
no vendored copy - and exposes either the default DDPM ancestral chain or a
stochastic DDIM sub-sampled chain (``eta > 0``) as a stochastic policy:

  * horizon  H = n_diffusion_timesteps (DDPM) or ddim_steps (DDIM)
  * state    s_t = (noisy chains x_t, fixed conditioning graph)
  * action   a_t = sampled next chain value over the FREE dimensions of the mode
  * policy   pi_theta(a_t | s_t) = N(guided_transition_mean_theta, Sigma_t)

Agent chain layout per node: [x, y, speed, cos, sin, length, width,
goal_x, goal_y, type_onehot(3)]  (12 dims). Lane chain: 20*2 = 40 dims per node.
For DDPO, controlled agents are projected to vehicle type with ego-sized
footprints; length, width, and type logits are not policy-gradient dimensions.

Modes (``cfg.ddpo.mode``) - select WHICH dims are free:
  * "goal_only"  - map fixed, agent init states fixed (inpainted from the real
                   conditioning scene), only the per-agent goal dims [7:9] are
                   generated/trained. Controlled agents still use the vehicle
                   projection for type and footprint.
  * "agent_only" - map fixed, agent kinematics and goals are generated/trained;
                   length, width, and type are fixed by the vehicle projection.
  * "full"       - agent kinematics/goals AND the lane chain are generated/trained (only the
                   graph structure - node counts, edges, lg_type - stays real).

Two knobs select WHICH agent nodes are free (ego is always local index 0; the
rest follow the dataset's deterministic ordering):
  * ``control_ego``       - if False the ego node is fixed to GT (generate only
                            other agents around a real ego).
  * ``control_agent_num`` - number of non-ego agents to generate (the first ``k``
                            in node order); -1 means all non-ego agents.
Restricting the free nodes shrinks the action space, which sharpens credit
assignment for the (mostly ego-centric) reward and improves DDPO stability.

NOTE: ``ConditioningPool.prune_agents`` already trims the conditioning graph to
``ego + control_agent_num`` agents (the rest are removed, not inpainted to GT),
so the denoiser and reward sim see only the controlled agents. The node mask
below is then redundant-but-consistent (it marks every remaining non-ego node),
and still does the right thing if an un-pruned graph is ever passed in.

Fixed agent (node, dim) entries are inpainted at every step with the forward-
diffused ground truth ``q_sample(x0, t)`` (standard replacement inpainting), or
with the clean ``x0`` if ``inpaint_noised=False``. Log-probs (and the KL trust
region) are accumulated over free dims of controlled nodes only, so fixed
entries never contribute policy gradient.
"""

from __future__ import annotations

import copy
import math

import torch
from torch_geometric.data import Batch

from models.scenario_dreamer_dm_goal import unnormalize_scene_with_goal
from nn_modules.dm_goal import DMGoal

from .goal_schema import (
    GOAL_DIMS,
    GOAL_SLICE,
    MIN_DISTANCE_TO_GOAL,
    SIZE_DIMS,
    TYPE_DIMS,
    VEHICLE_TYPE_ID,
    fov_unnormalize,
)
from .interfaces import GeneratedScenes, SamplingTrajectory

_LOG_2PI = math.log(2.0 * math.pi)
MODES = ("full", "agent_only", "goal_only")


def _masked_gaussian_logprob(x, mean, logvar, free_mask):
    """Diagonal-Gaussian log-density over free dims only; [N,1,D] -> [N].

    ``logvar`` is the per-node transition log-variance (broadcast over dims);
    ``free_mask`` is a [D] bool tensor.
    """
    var = logvar.exp()
    per_elem = -0.5 * (((x - mean) ** 2) / var + logvar + _LOG_2PI)
    return per_elem[..., free_mask].flatten(1).sum(dim=1)


def _masked_gaussian_kl(mean, mean_ref, logvar, free_mask):
    """KL between two diagonal Gaussians sharing ``logvar``, over free dims only.

    DDPM/DDIM reverse-step variance is fixed by the sampler schedule, so the
    policy's and reference's per-step Gaussians differ only in mean and KL reduces to the
    closed form ``sum_d (mean - mean_ref)^2 / (2 var)`` over the free dims
    (always >= 0); differentiable through ``mean``.
    """
    var = logvar.exp()
    per_elem = 0.5 * (mean - mean_ref) ** 2 / var
    return per_elem[..., free_mask].flatten(1).sum(dim=1)


class DMGoalDDPOPolicy:
    def __init__(
        self,
        cfg,                      # cfg.dm_goal node (model / dataset / train children)
        ckpt_path: str | None,
        *,
        mode: str = "agent_only",
        device: str = "cuda",
        use_ema_weights: bool = True,
        inpaint_noised: bool = True,
        control_ego: bool = True,
        control_agent_num: int = -1,
        sampler: str = "ddpm",
        ddim_steps: int | None = None,
        ddim_eta: float = 1.0,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if int(control_agent_num) < -1:
            raise ValueError(f"control_agent_num must be >= -1, got {control_agent_num}")
        if not control_ego and int(control_agent_num) == 0:
            raise ValueError("control_ego=False and control_agent_num=0 leaves nothing to generate")
        self.cfg = cfg
        self.cfg_model = cfg.model
        self.cfg_dataset = cfg.dataset
        self.mode = mode
        self.device = device
        self.inpaint_noised = inpaint_noised
        self.control_ego = bool(control_ego)
        self.control_agent_num = int(control_agent_num)
        self.sampler = str(sampler).lower()
        if self.sampler not in ("ddpm", "ddim"):
            raise ValueError(f"sampler must be one of ('ddpm', 'ddim'), got {sampler!r}")

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
        if ddim_steps is None:
            ddim_steps = self._H
        self.ddim_steps = int(ddim_steps)
        self.ddim_eta = float(ddim_eta)
        if self.sampler == "ddim":
            if self.ddim_steps < 2 or self.ddim_steps > self._H:
                raise ValueError(
                    f"ddim_steps must be in [2, {self._H}], got {self.ddim_steps}"
                )
            if self.ddim_eta < 0:
                raise ValueError(f"ddim_eta must be non-negative, got {self.ddim_eta}")
        self.step_pairs = self._build_step_pairs()
        self.agent_latent_dim = int(self.cfg_model.agent_latent_dim)
        self.diffusion_clip = self.cfg_model.diffusion_clip
        self.lane_free = mode == "full"

        free = torch.zeros(self.agent_latent_dim, dtype=torch.bool)
        if mode == "goal_only":
            free[list(GOAL_DIMS)] = True
        else:
            free[:] = True
            free[list(SIZE_DIMS + TYPE_DIMS)] = False
        self.agent_free_mask = free.to(device)
        # Inpainting is needed if some dims are fixed (goal_only) OR some nodes
        # are fixed (control_ego False / control_agent_num >= 0).
        self.agent_dim_fixed = bool((~free).any())
        self.node_control = (not self.control_ego) or (self.control_agent_num >= 0)
        self.needs_inpaint = self.agent_dim_fixed or self.node_control

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
        return len(self.step_pairs)

    def _build_step_pairs(self) -> list[tuple[int, int]]:
        """Reverse diffusion schedule as ``(t, prev_t)`` pairs.

        ``prev_t == -1`` is the final denoising jump to clean x0. DDPM keeps the
        original adjacent 100-step chain; DDIM uses a rounded linspace subset.
        """
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
        sigma2 = self._ddim_sigma2(self.net, t, prev_t, self.device)
        return bool((sigma2 > 1e-20).item())

    def stochastic_step_indices(self, min_diffusion_t: int = 5) -> torch.Tensor:
        """Step indices safe to differentiate for DDPO.

        The threshold is applied to the true diffusion timestep ``t`` rather than
        the sampler index, so DDIM sub-sampling still skips very low-noise jumps.
        """
        idx = [
            i
            for i, (t, prev_t) in enumerate(self.step_pairs)
            if t >= int(min_diffusion_t) and self._is_stochastic_transition(t, prev_t)
        ]
        return torch.as_tensor(idx, dtype=torch.long)

    def trainable_parameters(self):
        return (p for p in self.net.parameters() if p.requires_grad)

    def state_dict(self) -> dict:
        return self.net.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.net.load_state_dict(sd)

    # ------------------------------------------------------- node control
    def _controlled_node_mask(self, agent_batch, num_scenes):
        """[N_agents] bool: which agent nodes are generated (vs fixed to GT).

        Ego is local index 0 of each scene; the rest follow the dataset's
        deterministic ordering. ``control_agent_num`` keeps the first k non-ego
        nodes per scene (-1 = all); ``control_ego`` toggles the ego node.
        """
        counts = torch.bincount(agent_batch, minlength=num_scenes)
        offsets = torch.cumsum(counts, 0) - counts
        local_idx = (
            torch.arange(agent_batch.shape[0], device=agent_batch.device)
            - offsets[agent_batch]
        )
        is_ego = local_idx == 0
        if self.control_agent_num < 0:
            nonego = local_idx >= 1
        else:
            nonego = (local_idx >= 1) & ((local_idx - 1) < self.control_agent_num)
        return nonego | (is_ego & self.control_ego)

    def _agent_free_mask(self, controlled):
        """[N_agents, 1, D] bool: free (node, dim) entries = controlled & free-dim."""
        return controlled.view(-1, 1, 1) & self.agent_free_mask.view(1, 1, -1)

    def _project_target_vehicle_footprint(
        self, target_agent, agent_batch, num_scenes, controlled
    ):
        """Fix controlled agents to vehicle type and ego length/width in normalized space."""
        if target_agent is None or not controlled.any():
            return target_agent
        target_agent = target_agent.clone()
        counts = torch.bincount(agent_batch, minlength=num_scenes)
        ego_idx = torch.cumsum(counts, 0) - counts
        size_slice = slice(SIZE_DIMS[0], SIZE_DIMS[-1] + 1)
        type_slice = slice(TYPE_DIMS[0], TYPE_DIMS[-1] + 1)
        target_agent[controlled, :, size_slice] = target_agent[
            ego_idx[agent_batch[controlled]], :, size_slice
        ]
        target_agent[controlled, :, type_slice] = 0.0
        target_agent[controlled, :, TYPE_DIMS[VEHICLE_TYPE_ID]] = 1.0
        return target_agent

    def _project_decoded_vehicle_footprint(
        self, agent_states, agent_types, agent_batch, num_scenes, controlled
    ):
        """Fix controlled agents to vehicle type and ego length/width in physical units."""
        if not controlled.any():
            return agent_states, agent_types
        counts = torch.bincount(agent_batch, minlength=num_scenes)
        ego_idx = torch.cumsum(counts, 0) - counts
        ego_size = agent_states[ego_idx, 5:7]
        agent_states[controlled, 5:7] = ego_size[agent_batch[controlled]]
        agent_types[controlled] = VEHICLE_TYPE_ID
        return agent_states, agent_types

    # -------------------------------------------------------------- inpainting
    def _inpaint_agent(self, x_agent, target_agent, t_agent, free_mask):
        """Overwrite fixed agent (node, dim) entries at the noise level of step t.

        ``free_mask`` is a [N_agents, 1, D] bool of generated entries; everything
        else is replaced by the forward-diffused GT (or clean x0).
        """
        if target_agent is None:
            return x_agent
        if self.inpaint_noised:
            noise = torch.randn_like(target_agent)
            x_fixed = self.net.q_sample(target_agent, t_agent, noise=noise)
        else:
            x_fixed = target_agent
        return torch.where(free_mask, x_agent, x_fixed)

    # -------------------------------------------------------------- transition
    def _guided_eps_and_x0(self, net, x_agent, x_lane, data, t_agent, t_lane):
        """Classifier-free guided epsilon prediction plus reconstructed x0."""
        conditional_epsilon_agent, conditional_epsilon_lane = net.model(
            x_agent, x_lane, data, t_agent, t_lane, unconditional=False
        )
        unconditional_epsilon_agent, unconditional_epsilon_lane = net.model(
            x_agent, x_lane, data, t_agent, t_lane, unconditional=True
        )
        guidance = self.cfg.train.guidance_scale
        epsilon_agent = unconditional_epsilon_agent + guidance * (
            conditional_epsilon_agent - unconditional_epsilon_agent
        )
        epsilon_lane = unconditional_epsilon_lane + guidance * (
            conditional_epsilon_lane - unconditional_epsilon_lane
        )

        t_agent = t_agent.detach().to(torch.int64)
        t_lane = t_lane.detach().to(torch.int64)
        x0_agent = net.predict_start_from_noise(x_agent, t=t_agent, noise=epsilon_agent)
        x0_lane = net.predict_start_from_noise(x_lane, t=t_lane, noise=epsilon_lane)
        return epsilon_agent, x0_agent, epsilon_lane, x0_lane

    @staticmethod
    def _alpha_at(net, t, shape, device):
        if int(t) < 0:
            return torch.ones(
                (shape[0],) + (1,) * (len(shape) - 1),
                device=device,
                dtype=net.alphas_cumprod.dtype,
            )
        tt = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
        return net.alphas_cumprod.gather(0, tt).reshape(
            shape[0], *((1,) * (len(shape) - 1))
        )

    def _ddim_mean_logvar(self, net, x_agent, x_lane, data, t_agent, t_lane, t: int, prev_t: int):
        eps_a, x0_a, eps_l, x0_l = self._guided_eps_and_x0(
            net, x_agent, x_lane, data, t_agent, t_lane
        )

        alpha_t_a = self._alpha_at(net, t, x_agent.shape, x_agent.device)
        alpha_prev_a = self._alpha_at(net, prev_t, x_agent.shape, x_agent.device)
        sigma2_a = (self.ddim_eta ** 2) * (1.0 - alpha_prev_a) / (1.0 - alpha_t_a) * (
            1.0 - alpha_t_a / alpha_prev_a
        )
        sigma2_a = torch.clamp(sigma2_a, min=0.0)
        dir_coef_a = torch.sqrt(torch.clamp(1.0 - alpha_prev_a - sigma2_a, min=0.0))
        mean_a = torch.sqrt(alpha_prev_a) * x0_a + dir_coef_a * eps_a
        logvar_a = torch.log(torch.clamp(sigma2_a, min=1e-20))

        alpha_t_l = self._alpha_at(net, t, x_lane.shape, x_lane.device)
        alpha_prev_l = self._alpha_at(net, prev_t, x_lane.shape, x_lane.device)
        sigma2_l = (self.ddim_eta ** 2) * (1.0 - alpha_prev_l) / (1.0 - alpha_t_l) * (
            1.0 - alpha_t_l / alpha_prev_l
        )
        sigma2_l = torch.clamp(sigma2_l, min=0.0)
        dir_coef_l = torch.sqrt(torch.clamp(1.0 - alpha_prev_l - sigma2_l, min=0.0))
        mean_l = torch.sqrt(alpha_prev_l) * x0_l + dir_coef_l * eps_l
        logvar_l = torch.log(torch.clamp(sigma2_l, min=1e-20))
        return mean_a, logvar_a, mean_l, logvar_l

    def _step_mean_logvar(self, net, x_agent, x_lane, data, t_agent, t_lane, t: int, prev_t: int):
        if self.sampler == "ddpm":
            return net.p_mean_variance(x_agent, x_lane, data, t_agent, t_lane)
        return self._ddim_mean_logvar(net, x_agent, x_lane, data, t_agent, t_lane, t, prev_t)

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

        controlled = self._controlled_node_mask(agent_batch, num_scenes)
        controlled_f = controlled.to(torch.float32)
        free_mask = self._agent_free_mask(controlled)

        n_agent = data["agent"].x.shape[0]
        x_agent = torch.randn((n_agent, 1, self.agent_latent_dim), device=self.device)
        target_agent = net._agent_target(data).to(self.device) if self.needs_inpaint else None
        target_agent = self._project_target_vehicle_footprint(
            target_agent, agent_batch, num_scenes, controlled
        )
        target_lane = net._lane_target(data).to(self.device)

        if self.lane_free:
            x_lane = torch.randn_like(target_lane) * net.lane_sampling_temperature
        else:
            x_lane = target_lane

        records = []
        old_lp = torch.zeros((num_scenes, self.num_sampling_steps), device=self.device)

        for j, (i, prev_i) in enumerate(self.step_pairs):
            t = torch.full((num_scenes,), i, device=self.device, dtype=torch.long)
            t_agent = t[agent_batch]
            t_lane = t[lane_batch]

            x_agent = self._inpaint_agent(x_agent, target_agent, t_agent, free_mask)
            mean_a, logvar_a, mean_l, logvar_l = self._step_mean_logvar(
                net, x_agent, x_lane, data, t_agent, t_lane, i, prev_i
            )

            x_t_agent = x_agent
            x_t_lane = x_lane if self.lane_free else None
            stochastic = self._is_stochastic_transition(i, prev_i)
            if stochastic:
                x_next_a = mean_a + (0.5 * logvar_a).exp() * torch.randn_like(x_agent)
            else:
                x_next_a = mean_a
            x_next_a = torch.clip(x_next_a, -self.diffusion_clip, self.diffusion_clip)

            if self.lane_free:
                if stochastic:
                    x_next_l = mean_l + (0.5 * logvar_l).exp() * torch.randn_like(x_lane)
                else:
                    x_next_l = mean_l
                x_next_l = torch.clip(x_next_l, -self.diffusion_clip, self.diffusion_clip)
            else:
                x_next_l = target_lane

            if stochastic:
                node_lp = _masked_gaussian_logprob(x_next_a, mean_a, logvar_a, self.agent_free_mask)
                old_lp[:, j].index_add_(0, agent_batch, node_lp * controlled_f)
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
                prev_i,
            ))
            x_agent = x_next_a
            x_lane = x_next_l

        # decode with fixed (node, dim) entries restored to the exact clean targets
        if self.needs_inpaint:
            x_agent = torch.where(free_mask, x_agent, target_agent)
        scenes = self._decode(x_agent, x_lane, data)
        traj = SamplingTrajectory(
            records={"steps": records, "agent_batch": agent_batch, "lane_batch": lane_batch},
            old_logprob=old_lp.detach(),
            num_steps=self.num_sampling_steps,
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
        with_kl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Per-step (masked) log-prob of the trajectory (optionally with KL).

        Returns ``(logprob, kl)``, both ``[num_scenes, len(step_indices)]``. When
        ``with_kl`` the closed-form KL to the frozen reference policy over the
        free dims is also returned (a proper, >= 0 trust-region penalty); else
        ``kl`` is ``None``. The reference mean is computed in the same loop, so
        this costs one extra (no-grad) forward per step.
        """
        data = conditioning.to(self.device)
        net = self.ref if use_reference else self.net
        steps = trajectory.records["steps"]
        agent_batch = trajectory.records["agent_batch"]
        lane_batch = trajectory.records["lane_batch"]
        num_scenes = trajectory.num_scenes
        target_lane = self.net._lane_target(data).to(self.device)
        controlled_f = self._controlled_node_mask(agent_batch, num_scenes).to(torch.float32)

        out = torch.zeros((num_scenes, len(step_indices)), device=self.device)
        kl = torch.zeros((num_scenes, len(step_indices)), device=self.device) if with_kl else None
        ctx = torch.no_grad() if use_reference else torch.enable_grad()
        with ctx:
            for col, s in enumerate(step_indices.tolist()):
                record = steps[s]
                if len(record) == 5:
                    x_t_a, x_tm1_a, x_t_l, x_tm1_l, i = record
                    prev_i = i - 1
                else:
                    x_t_a, x_tm1_a, x_t_l, x_tm1_l, i, prev_i = record
                if not self._is_stochastic_transition(i, prev_i):
                    continue  # deterministic step carries no policy gradient
                t = torch.full((num_scenes,), i, device=self.device, dtype=torch.long)
                x_lane_in = x_t_l if self.lane_free else target_lane
                mean_a, logvar_a, mean_l, logvar_l = self._step_mean_logvar(
                    net, x_t_a, x_lane_in, data, t[agent_batch], t[lane_batch], i, prev_i
                )
                node_lp = _masked_gaussian_logprob(x_tm1_a, mean_a, logvar_a, self.agent_free_mask)
                out[:, col] = out[:, col].index_add(0, agent_batch, node_lp * controlled_f)
                if self.lane_free:
                    lane_mask = torch.ones(x_lane_in.shape[-1], dtype=torch.bool, device=self.device)
                    lane_lp = _masked_gaussian_logprob(x_tm1_l, mean_l, logvar_l, lane_mask)
                    out[:, col] = out[:, col].index_add(0, lane_batch, lane_lp)
                if with_kl:
                    with torch.no_grad():
                        mean_a_ref, _, mean_l_ref, _ = self._step_mean_logvar(
                            self.ref, x_t_a, x_lane_in, data, t[agent_batch], t[lane_batch], i, prev_i
                        )
                    node_kl = _masked_gaussian_kl(mean_a, mean_a_ref, logvar_a, self.agent_free_mask)
                    kl[:, col] = kl[:, col].index_add(0, agent_batch, node_kl * controlled_f)
                    if self.lane_free:
                        lane_mask = torch.ones(x_lane_in.shape[-1], dtype=torch.bool, device=self.device)
                        lane_kl = _masked_gaussian_kl(mean_l, mean_l_ref, logvar_l, lane_mask)
                        kl[:, col] = kl[:, col].index_add(0, lane_batch, lane_kl)
        return out, kl

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
        if self.mode == "goal_only":
            # types are not generated in goal_only mode; use the conditioning types
            agent_types = torch.argmax(data["agent"].type, dim=-1)
        agent_states, lane_states = unnormalize_scene_with_goal(agent_states, lane_states, self.cfg_dataset)
        agent_types = agent_types.clone()
        agent_batch = data["agent"].batch
        num_scenes = int(data.batch_size)
        controlled = self._controlled_node_mask(agent_batch, num_scenes)
        agent_states, agent_types = self._project_decoded_vehicle_footprint(
            agent_states, agent_types, agent_batch, num_scenes, controlled
        )
        meta = {"lane_scene_idx": data["lane"].batch}
        lane_edge_store = data["lane", "to", "lane"]
        if "edge_index" in lane_edge_store:
            meta["lane_edge_index"] = lane_edge_store.edge_index
        if "type" in lane_edge_store:
            meta["lane_edge_type"] = lane_edge_store.type
        # per-agent flag for viz: which nodes this policy actually generates
        meta["controlled_mask"] = controlled
        if self.mode == "goal_only":
            # GT parking state per agent (goal within reach radius of spawn). Only
            # meaningful in goal_only mode, where generated agents correspond 1:1 to
            # the conditioning agents (init states are inpainted from GT). The reward
            # penalises generated scenes whose parking state diverges from this.
            fov = float(self.cfg_dataset.fov)
            gt = data["agent"].x.float()
            gt_init = fov_unnormalize(gt[:, 0:2], fov)
            gt_goal = fov_unnormalize(gt[:, GOAL_SLICE], fov)
            gt_dist = torch.linalg.norm(gt_goal - gt_init, dim=-1)
            meta["gt_parking_mask"] = gt_dist < MIN_DISTANCE_TO_GOAL
        return GeneratedScenes(
            agent_states=agent_states,
            agent_types=agent_types,
            agent_scene_idx=data["agent"].batch,
            lane_polylines=lane_states,
            num_scenes=int(data.batch_size),
            meta=meta,
        )
