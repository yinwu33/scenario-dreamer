"""Planner-driven rollout runner for DDPO reward evaluation."""

from __future__ import annotations

import numpy as np
import torch

from .interfaces import GeneratedScenes
from .pufferdrive_sim import SimScene
from .reward_hooks import RewardHook, RolloutContext


class PlannerRolloutRunner:
    """Roll out SimScene batches with a frozen planner and metric hooks."""

    def __init__(
        self,
        *,
        planner,
        device: str,
        deterministic: bool,
        sim_steps: int,
        hooks: list[RewardHook],
    ):
        self.planner = planner
        self.device = device
        self.deterministic = bool(deterministic)
        self.sim_steps = int(sim_steps)
        self.hooks = list(hooks)

    def rollout(
        self,
        scenes: GeneratedScenes,
        sims: list[SimScene],
        *,
        record_trajectories: bool = False,
    ) -> RolloutContext:
        """Execute rollout and return the populated hook context."""
        m = len(sims)
        ctx = RolloutContext(
            scenes=scenes,
            sims=sims,
            finished=np.zeros(m, dtype=bool),
            metrics={
                "ego_collision": np.zeros(m, dtype=np.float32),
                "ego_offroad": np.zeros(m, dtype=np.float32),
                "init_invalid": np.zeros(m, dtype=np.float32),
                "reached_goal": np.zeros(m, dtype=np.float32),
            },
            record_trajectories=record_trajectories,
        )

        for hook in self.hooks:
            hook.before_rollout(ctx)

        for t in range(self.sim_steps):
            ctx.t = t

            # Score current state before stepping, matching the original reward.
            for s, sim in enumerate(sims):
                if ctx.finished[s]:
                    continue
                for hook in self.hooks:
                    hook.before_step_scene(ctx, s, sim)

            active = [s for s in range(m) if not ctx.finished[s]]
            if not active:
                break

            obs_list = [sims[s].compute_obs() for s in active]
            obs = torch.as_tensor(np.concatenate(obs_list), device=self.device)
            actions = (
                self.planner.act(obs, deterministic=self.deterministic).cpu().numpy()
            )

            off = 0
            for s, ob in zip(active, obs_list):
                n_ctrl = ob.shape[0]
                sims[s].step_dynamics(actions[off : off + n_ctrl])
                sims[s].update_metrics()
                ego_reached, _ = sims[s].goal_step()
                off += n_ctrl
                for hook in self.hooks:
                    hook.after_step_scene(ctx, s, sims[s], ego_reached=ego_reached)

        for hook in self.hooks:
            hook.after_rollout(ctx)

        return ctx
