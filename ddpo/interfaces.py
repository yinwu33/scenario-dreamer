"""Data contracts between the DDPO policy (diffusion model) and the reward (sim).

Ported from PufferDrive/scene_init_ddpo/interfaces.py, trimmed to what the
scenario-dreamer-hosted trainer needs (plain dataclasses, no ABC: the only policy
today is ``ddpo.policy.DMGoalDDPOPolicy``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class GeneratedScenes:
    """Decoded, simulator-ready output of one ``sample`` call (a batch of scenes).

    All agent tensors are flattened across the batch (PyG style); ``agent_scene_idx``
    maps every agent row to its scene in ``[0, num_scenes)``. By convention the
    ego/SDC is local index 0 within each scene (the dm_goal dataset is SDC-centric).

    ``agent_states`` layout (physical units, from dm_goal decode):
        [x, y, speed, cos_theta, sin_theta, length, width, goal_x, goal_y]
    """

    agent_states: torch.Tensor      # [N_agents, 9]
    agent_types: torch.Tensor       # [N_agents] int class ids (0 veh, 1 ped, 2 cyc)
    agent_scene_idx: torch.Tensor   # [N_agents] -> scene id in [0, num_scenes)
    lane_polylines: Any             # [N_lanes, P, 2] map geometry (fixed or generated)
    num_scenes: int
    meta: dict = field(default_factory=dict)   # carries "lane_scene_idx"


@dataclass
class SamplingTrajectory:
    """Record of a diffusion sampling rollout, sufficient to recompute log-probs.

    ``old_logprob`` is the per-scene, per-step log-density evaluated under the
    behaviour parameters at sampling time (PPO/IS reference). ``records`` is
    policy-specific (per-step (x_t, x_{t-1}, t) tuples).
    """

    records: Any
    old_logprob: torch.Tensor       # [num_scenes, num_steps] detached
    num_steps: int
    num_scenes: int
