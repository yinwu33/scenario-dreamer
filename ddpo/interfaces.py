"""Data contracts between the DDPO policy (diffusion model) and the reward (sim).

Ported from PufferDrive/scene_init_ddpo/interfaces.py, trimmed to what the
scenario-dreamer-hosted trainer needs (plain dataclasses, no ABC: the only policy
today is ``ddpo.policy_ldm_adv.LDMAdvDDPOPolicy``).
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
    # Canonical single-adversary handle: per-scene sim-local index of THE generated
    # adversary (the sole generated non-ego agent), or -1 if the scene has none
    # (e.g. no-adv conditioning scenes). ``meta['gen_agent_mask']`` is the same
    # information as a per-node bool, kept only for viz/analysis scripts.
    adv_local_idx: torch.Tensor | None = None   # [num_scenes] long, -1 == no adversary
    # Free-form side channel. Conventional keys:
    #   "lane_scene_idx"  -- [N_lanes] -> scene id, the lane counterpart of agent_scene_idx
    #   "gen_agent_mask"  -- [N_agents] bool, per-node form of adv_local_idx (viz/analysis)
    #   "lane_graph"      -- list[dict], one per scene: {"succ": [2, E], "lateral": [2, E]}
    #                        arrays of SCENE-LOCAL lane indices. ``succ`` edges run in the
    #                        driving direction; ``lateral`` pairs left/right neighbours.
    #                        Optional: the lane GRAPH, as opposed to the lane geometry in
    #                        lane_polylines. Route-planning planners (ddpo.routes) need it;
    #                        the frozen neural planners do not, so most producers leave it
    #                        out and SimScene.lane_graph stays None.
    meta: dict = field(default_factory=dict)


def single_adv_local_idx(gen_agent_mask, agent_scene_idx, num_scenes):
    """Per-scene sim-local index of the single generated non-ego adversary.

    Returns a ``[num_scenes]`` long tensor; ``-1`` marks a scene with no generated
    adversary. The local index is the agent's *order of appearance* within its
    scene (ego is local 0), which is exactly how ``Planner._build_scenes``
    slices each ``SimScene`` -- so it doubles as the sim-local index downstream.

    Enforces the single-adversary contract: at most one generated non-ego agent
    per scene (raises otherwise). ``agent_scene_idx`` need not be contiguous /
    sorted (ldm_adv appends the adv after the whole base set).
    """
    device = agent_scene_idx.device
    out = torch.full((num_scenes,), -1, dtype=torch.long, device=device)
    if gen_agent_mask is None or agent_scene_idx.numel() == 0:
        return out

    n = agent_scene_idx.shape[0]
    order = torch.arange(n, device=device)
    scene = agent_scene_idx.to(torch.long)
    # Order-preserving within-scene rank: stable-sort by (scene, appearance).
    perm = torch.argsort(scene * n + order)
    counts = torch.bincount(scene, minlength=num_scenes)
    offsets = torch.cumsum(counts, 0) - counts
    local = torch.empty(n, dtype=torch.long, device=device)
    local[perm] = order - offsets[scene[perm]]

    sel = gen_agent_mask.to(torch.bool) & (local > 0)   # generated, non-ego
    sel_scene = scene[sel]
    if sel_scene.numel():
        if int(torch.bincount(sel_scene, minlength=num_scenes).max()) > 1:
            raise ValueError(
                "single_adv_local_idx: more than one generated non-ego agent in a "
                "scene -- the DDPO reward assumes a single adversary "
                "(check control_agent_num)."
            )
        out[sel_scene] = local[sel]
    return out


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
