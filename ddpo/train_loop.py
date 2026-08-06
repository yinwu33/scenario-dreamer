"""DDPO fine-tuning loop against a frozen PufferDrive planner.

This module holds the actual training loop. It is dispatched from train.py when
``model_name == 'ddpo'`` (e.g. ``python train.py --config-name
config_critical_scene_ldm_adv_ddpo``). Heavy / PufferDrive-specific imports live
at module top level here so they are only paid when the ddpo path is actually
taken (train.py imports this module lazily inside its dispatch branch).

Pipeline per iteration:
    LDMAdvConditioningPool.sample_batch(B)    # real conditioning graphs
    policy.sample(cond)                       # record denoising trajectory + logprob
    PufferSimulator.evaluate(scenes)          # numpy sim port + frozen planner
    compute_advantages -> ddpo_loss over k random denoising steps (+ optional KL)
"""

from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from omegaconf import OmegaConf

from ddpo.conditioning import LDMAdvConditioningPool
from ddpo.ddpo_loss import AdaptiveKLController, DDPOConfig, compute_advantages, ddpo_loss
from ddpo.policy_ldm_adv import LDMAdvDDPOPolicy
from sim.runner import SimulatorConfig
from ddpo.reward import PufferSimulator, RewardConfig
from sim.hooks import GenInvalidCheck
from sim.scenes import GeneratedScenes
from utils.train_helpers import cache_latent_stats, set_latent_stats


def _set_dataset_name(cfg_node, dataset_name: str) -> None:
    OmegaConf.set_struct(cfg_node, False)
    cfg_node.dataset_name = dataset_name
    OmegaConf.set_struct(cfg_node, True)


def _build_policy_and_pool(cfg_root, cfg, device: str):
    model_type = cfg.model_type
    dataset_name = cfg_root.dataset_name.name

    if model_type == "ldm_adv":
        _set_dataset_name(cfg_root.ldm_adv, dataset_name)
        _set_dataset_name(cfg_root.ae_goal, dataset_name)
        if not Path(cfg_root.ldm_adv.dataset.latent_stats_path).exists():
            cache_latent_stats(cfg_root.ldm_adv)
        ldm_cfg = set_latent_stats(cfg_root.ldm_adv)
        policy = LDMAdvDDPOPolicy(
            ldm_cfg,
            cfg_root.ae_goal,
            ldm_ckpt=cfg.ldm_adv_ckpt,
            ae_ckpt=cfg.ae_ckpt,
            device=device,
            use_ema_weights=cfg.use_ema_weights,
            sampler=cfg.sampler,
            ddim_steps=cfg.ddim_steps,
            ddim_eta=cfg.ddim_eta,
        )
        pool = LDMAdvConditioningPool(
            ldm_cfg.dataset,
            split_name=cfg.train_split,
            pool_size=cfg.pool_size,
            device=device,
            seed=cfg.seed,
            min_ego_drive=cfg.min_ego_drive,
            prune_base_to_ego=cfg.prune_base_to_ego,
            insert_adv_as_extra=cfg.insert_adv_as_extra,
            adv_cond_target=cfg.adv_cond_target,
        )
        eval_dataset_cfg = ldm_cfg.dataset
    else:
        raise ValueError(f"Unsupported ddpo.model_type: {model_type}")

    return model_type, policy, pool, eval_dataset_cfg


def _build_gen_invalid(cfg, dataset_cfg):
    """Assemble the condition-violation check for GenAgentInvalidHook.

    Bucket thresholds are sourced from the (ldm_adv) dataset config so they stay
    the single source of truth shared with the training-time discretization; the
    per-field toggles come from ``cfg.gen_agent_invalid``. Returns ``None`` when
    the block is absent or ``enabled=false`` (falls back to the parked-adv gate).
    """
    spec = cfg.gen_agent_invalid
    if not bool(spec.enabled):
        return None
    return GenInvalidCheck(
        goaldist_near=float(dataset_cfg.cond_goaldist_near_threshold),
        goaldist_far=float(dataset_cfg.cond_goaldist_far_threshold),
        egodist_near=float(dataset_cfg.cond_egodist_near_threshold),
        egodist_far=float(dataset_cfg.cond_egodist_far_threshold),
        check_type=bool(spec.check_type),
        check_motion=bool(spec.check_motion),
        check_goal_dist=bool(spec.check_goal_dist),
        check_ego_dist=bool(spec.check_ego_dist),
    )


def _default_run_name(cfg, model_type: str) -> str:
    if model_type == "ldm_adv":
        if cfg.sampler == "ddim":
            return f"ddpo_ldm_adv_ddim{cfg.ddim_steps}_eta{cfg.ddim_eta}"
        return "ddpo_ldm_adv"
    return f"ddpo_{model_type}"


def _checkpoint_prefix(cfg, model_type: str) -> str:
    # Checkpoint filename tracks the run name (experiment.run_name, surfaced as
    # cfg.wandb.run_name via the group config); fall back to the legacy derived
    # name for configs without an experiment.* block.
    return cfg.wandb.run_name or _default_run_name(cfg, model_type)


def _bf16_autocast(device: str, enabled: bool = True):
    if enabled and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _rng_snapshot(device: str) -> dict:
    """Global RNG state for a faithful DDPO resume (torch CPU + numpy + cuda)."""
    snap = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        snap["cuda"] = torch.cuda.get_rng_state_all()
    return snap


def _rng_restore(snap: dict, device: str) -> None:
    """Inverse of ``_rng_snapshot``. RNG byte tensors are forced back to CPU since
    ``torch.load(..., map_location=device)`` may have moved them onto the GPU."""
    if not snap:
        return
    if "torch" in snap:
        torch.set_rng_state(snap["torch"].cpu())
    if "numpy" in snap:
        np.random.set_state(snap["numpy"])
    if "cuda" in snap and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in snap["cuda"]])


# Per-agent raw numbers logged as a wandb.Table next to the rollout images, so the
# adversary's generated output can be read off directly when it is hard to see in
# the render (e.g. it spawns outside the 64x64 m FOV the viz clips to, or sits on
# top of the ego). Positions are physical metres in the scene frame (centre = 0),
# the same frame as the viz; agent_states layout is
# [x, y, speed, cos, sin, length, width, goal_x, goal_y].
_ADV_TABLE_COLUMNS = [
    "variant", "scene", "idx", "type", "in_fov",
    "x", "y", "heading_deg", "speed", "length", "width",
    "goal_x", "goal_y", "goal_dist", "dist_to_ego",
    "traj_x0", "traj_y0", "traj_xT", "traj_yT", "n_steps", "reward",
]

_REWARD_COMPONENT_KEYS = (
    "criticality", "r_ttc", "r_approach",
    "constraint", "c_spawn_lane", "c_goal_lane", "c_overlap", "c_parking", "c_invalid",
    "c_invalid_sev",
    "spawn_lane_dist", "goal_lane_dist", "init_overlap_frac",
    "ego_adv_init_dist", "ego_adv_min_dist_warmup",
)
_VIZ_COMPONENT_KEYS = (*_REWARD_COMPONENT_KEYS, "c_invalid_reason")

_GROUP_TABLE_COLUMNS = [
    "iter", "group", "sample", "rank", "selected_for_media",
    "reward", "advantage", "group_mean", "group_std", "group_min", "group_max",
    "group_pos_count", "group_skipped",
    *_REWARD_COMPONENT_KEYS,
]


def _adv_agent_rows(name, s, states_s, types_s, gen_agent_s, traj, reward_s):
    """One row per DDPO-generated non-ego adversary agent in scene ``s``.

    ``states_s`` / ``types_s`` / ``gen_agent_s`` are this scene's slices of the
    decoded agent_states / agent_types / gen_agent_mask; ``traj`` is the scene's rollout
    dict (per-step [T, n_agents] arrays in the same per-agent order). The endpoints
    are taken from the first episode so they line up with what the viz draws.
    """
    from ddpo.viz import FOV, _first_episode_end

    rows = []
    half = FOV / 2.0
    ego_xy = states_s[0, 0:2] if len(states_s) else np.zeros(2)
    end = _first_episode_end(traj.get("done")) if isinstance(traj, dict) else None
    for i in range(len(states_s)):
        if i == 0 or not bool(gen_agent_s[i]):
            continue  # ego or a fixed neighbour: not the generated adversary
        a = states_s[i]
        x, y = float(a[0]), float(a[1])
        heading = float(np.degrees(np.arctan2(a[4], a[3])))
        gx, gy = float(a[7]), float(a[8])
        tx0 = ty0 = txT = tyT = float("nan")
        n_steps = 0
        if isinstance(traj, dict) and traj["x"].ndim == 2 and i < traj["x"].shape[1]:
            txa, tya = traj["x"][:end, i], traj["y"][:end, i]
            n_steps = int(len(txa))
            if n_steps:
                tx0, ty0, txT, tyT = float(txa[0]), float(tya[0]), float(txa[-1]), float(tya[-1])
        rows.append([
            name, int(s), int(i), int(types_s[i]),
            bool(abs(x) <= half and abs(y) <= half),
            x, y, heading, float(a[2]), float(a[5]), float(a[6]),
            gx, gy, float(np.hypot(gx - x, gy - y)),
            float(np.hypot(x - ego_xy[0], y - ego_xy[1])),
            tx0, ty0, txT, tyT, n_steps, float(reward_s),
        ])
    return rows


def _reward_components(metrics: dict, s: int) -> dict:
    return {k: metrics[k][s] for k in _VIZ_COMPONENT_KEYS if k in metrics}


def _to_numpy_index(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _index_like(value, mask):
    if isinstance(value, torch.Tensor):
        return value[mask.to(value.device)]
    return np.asarray(value)[_to_numpy_index(mask)]


def _subset_scenes(scenes: GeneratedScenes, scene_ids: list[int]) -> GeneratedScenes:
    """Return a compact GeneratedScenes batch containing the selected scene ids."""
    device = scenes.agent_scene_idx.device
    selected = [int(s) for s in scene_ids]
    agent_idx = scenes.agent_scene_idx
    agent_mask = torch.zeros(agent_idx.shape[0], dtype=torch.bool, device=device)
    new_agent_idx = torch.empty_like(agent_idx)
    for new_s, old_s in enumerate(selected):
        m = agent_idx == old_s
        agent_mask |= m
        new_agent_idx[m] = new_s
    new_agent_idx = new_agent_idx[agent_mask]

    lane_scene_idx = scenes.meta["lane_scene_idx"]
    lane_device = lane_scene_idx.device if isinstance(lane_scene_idx, torch.Tensor) else device
    lane_idx_t = torch.as_tensor(lane_scene_idx, device=lane_device)
    lane_mask = torch.zeros(lane_idx_t.shape[0], dtype=torch.bool, device=lane_device)
    new_lane_idx = torch.empty_like(lane_idx_t)
    for new_s, old_s in enumerate(selected):
        m = lane_idx_t == old_s
        lane_mask |= m
        new_lane_idx[m] = new_s
    new_lane_idx = new_lane_idx[lane_mask]

    lanes = scenes.lane_polylines
    if isinstance(lanes, torch.Tensor):
        lane_polylines = lanes[lane_mask.to(lanes.device)]
    else:
        lane_polylines = np.asarray(lanes)[_to_numpy_index(lane_mask)]

    meta = {"lane_scene_idx": new_lane_idx}
    gen_agent_mask = scenes.meta.get("gen_agent_mask")
    if gen_agent_mask is not None:
        meta["gen_agent_mask"] = _index_like(gen_agent_mask, agent_mask)
    gt_parking_mask = scenes.meta.get("gt_parking_mask")
    if gt_parking_mask is not None and getattr(gt_parking_mask, "shape", [0])[0] == agent_idx.shape[0]:
        meta["gt_parking_mask"] = _index_like(gt_parking_mask, agent_mask)
    adv_cond = scenes.meta.get("adv_cond")
    if adv_cond is not None:
        scene_idx = torch.as_tensor(selected, device=device, dtype=torch.long)
        if isinstance(adv_cond, torch.Tensor):
            meta["adv_cond"] = adv_cond[scene_idx.to(adv_cond.device)]
        else:
            meta["adv_cond"] = np.asarray(adv_cond)[selected]

    adv_local_idx = None
    if scenes.adv_local_idx is not None:
        adv_local_idx = scenes.adv_local_idx[
            torch.as_tensor(selected, device=scenes.adv_local_idx.device, dtype=torch.long)
        ]

    return GeneratedScenes(
        agent_states=scenes.agent_states[agent_mask],
        agent_types=scenes.agent_types[agent_mask],
        agent_scene_idx=new_agent_idx,
        lane_polylines=lane_polylines,
        num_scenes=len(selected),
        adv_local_idx=adv_local_idx,
        meta=meta,
    )


def _group_reward_summary(rewards: torch.Tensor, group_ids: torch.Tensor, ddpo_cfg: DDPOConfig) -> dict:
    group_ids = group_ids.to(rewards.device)
    uniq = torch.unique(group_ids)
    if uniq.numel() == 0:
        return {}
    stds, ranges, pos_counts = [], [], []
    skipped = 0
    for g in uniq:
        r = rewards[group_ids == g].float()
        std = r.std(unbiased=False)
        stds.append(std)
        ranges.append(r.max() - r.min())
        pos_counts.append(r.gt(0).float().sum())
        skipped += int(std < ddpo_cfg.group_skip_std)
    stds = torch.stack(stds)
    ranges = torch.stack(ranges)
    pos_counts = torch.stack(pos_counts)
    return {
        "train/group_reward_std": float(stds.mean()),
        "train/group_reward_std_max": float(stds.max()),
        "train/group_reward_range_mean": float(ranges.mean()),
        "train/group_reward_range_max": float(ranges.max()),
        "train/group_pos_count_mean": float(pos_counts.mean()),
        "train/group_skip_frac": float(skipped / max(int(uniq.numel()), 1)),
    }


@torch.no_grad()
def _visualize_train_group_diversity(
    scenes: GeneratedScenes,
    metrics: dict,
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    group_ids: torch.Tensor | None,
    reward_model: PufferSimulator,
    cfg,
    ddpo_cfg: DDPOConfig,
    it: int,
    wandb,
) -> dict:
    """Log top/bottom samples from the most reward-diverse GRPO train groups."""
    if group_ids is None:
        return {}

    every = int(cfg.train_group_viz_every)
    if every <= 0 or (it + 1) % every != 0:
        return {}

    group_ids_cpu = group_ids.detach().cpu()
    rewards_cpu = rewards.detach().cpu().float()
    advantages_cpu = advantages.detach().cpu().float()
    uniq = torch.unique(group_ids_cpu)
    if uniq.numel() == 0:
        return {}

    group_rows = []
    group_summaries = []
    selected_lookup: set[int] = set()
    max_groups = max(int(cfg.train_group_viz_num_groups), 1)
    extremes = max(int(cfg.train_group_viz_extremes), 1)

    for g in uniq.tolist():
        mask = group_ids_cpu == int(g)
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        r = rewards_cpu[idx]
        std = float(r.std(unbiased=False))
        gmin = float(r.min())
        gmax = float(r.max())
        order = idx[torch.argsort(r)]
        media_idx = torch.cat([order[:extremes], order[-extremes:]]).unique(sorted=True)
        group_summaries.append((std, gmax - gmin, int(g), idx, media_idx))

    group_summaries.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected_groups = group_summaries[:max_groups]
    media_scene_ids = []
    for _, _, _, _, media_idx in selected_groups:
        for s in media_idx.tolist():
            if int(s) not in selected_lookup:
                selected_lookup.add(int(s))
                media_scene_ids.append(int(s))
    if not media_scene_ids:
        return {}

    for std, _, g, idx, media_idx in selected_groups:
        r = rewards_cpu[idx]
        mean = float(r.mean())
        gmin = float(r.min())
        gmax = float(r.max())
        pos_count = int(r.gt(0).sum())
        skipped = bool(std < ddpo_cfg.group_skip_std)
        order_desc = idx[torch.argsort(r, descending=True)]
        rank_by_sample = {int(s): int(rank) for rank, s in enumerate(order_desc.tolist())}
        for s in idx.tolist():
            row = [
                int(it), int(g), int(s), rank_by_sample[int(s)],
                bool(int(s) in selected_lookup),
                float(rewards_cpu[s]), float(advantages_cpu[s]),
                mean, std, gmin, gmax, pos_count, skipped,
            ]
            row.extend(float(metrics[k][s]) if k in metrics else float("nan") for k in _REWARD_COMPONENT_KEYS)
            group_rows.append(row)

    print(
        f"   [train_group it {it:04d}] visualizing {len(selected_groups)} group(s), "
        f"{len(media_scene_ids)} sample(s); max_std={selected_groups[0][0]:.4f}"
    )

    if wandb is None:
        return {}

    log_payload = {
        "train_group/reward_table": wandb.Table(
            columns=_GROUP_TABLE_COLUMNS,
            data=group_rows,
        )
    }
    subset = _subset_scenes(scenes, media_scene_ids)
    viz_metrics = reward_model.evaluate(subset, record_trajectories=True)

    import matplotlib.pyplot as plt
    from ddpo.viz import CONTROL_COLOR, render_rollout, render_rollout_frames, save_gif

    save_gif_mode = bool(cfg.save_gif)
    media_dir = Path(cfg.output_dir) / "eval_media" / "train_group"
    if save_gif_mode:
        media_dir.mkdir(parents=True, exist_ok=True)

    lanes = subset.lane_polylines
    if isinstance(lanes, torch.Tensor):
        lanes = lanes.detach().cpu().numpy()
    lane_scene_idx = subset.meta["lane_scene_idx"].detach().cpu().numpy()
    states = subset.agent_states.detach().cpu().numpy()
    types = subset.agent_types.detach().cpu().numpy()
    agent_scene_idx = subset.agent_scene_idx.detach().cpu().numpy()
    gen_agent_mask = subset.meta.get("gen_agent_mask")
    if isinstance(gen_agent_mask, torch.Tensor):
        gen_agent_mask = gen_agent_mask.detach().cpu().numpy()

    group_by_scene = {int(s): int(group_ids_cpu[s]) for s in media_scene_ids}
    media = []
    for local_s, original_s in enumerate(media_scene_ids):
        a_sel = agent_scene_idx == local_s
        agent_colors = None
        if gen_agent_mask is not None:
            gen_agent_s = gen_agent_mask[a_sel]
            agent_colors = [
                CONTROL_COLOR if (i > 0 and bool(gen_agent_s[i])) else None
                for i in range(len(gen_agent_s))
            ]
        components = _reward_components(viz_metrics, local_s)
        title = (
            f"train_group it{it} g{group_by_scene[original_s]} "
            f"sample{original_s} r={float(rewards_cpu[original_s]):.3f}"
        )
        kwargs = dict(
            agent_states=states[a_sel],
            agent_types=types[a_sel],
            agent_colors=agent_colors,
            reward=viz_metrics["reward"][local_s],
            ego_collision=viz_metrics["ego_collision"][local_s] > 0,
            ego_offroad=viz_metrics["ego_offroad"][local_s] > 0,
            init_invalid=viz_metrics["init_invalid"][local_s] > 0,
            ego_min_ttc=viz_metrics["ego_min_ttc"][local_s],
            goal_offlane_frac=viz_metrics["goal_offlane_frac"][local_s],
            parking_mismatch_frac=viz_metrics["parking_mismatch_frac"][local_s],
            components=components,
            title=title,
        )
        if save_gif_mode:
            frames = render_rollout_frames(
                viz_metrics["trajectories"][local_s],
                lanes[lane_scene_idx == local_s],
                max_frames=int(cfg.gif_max_frames),
                **kwargs,
            )
            path = str(
                media_dir
                / f"it_{it:05d}_g{group_by_scene[original_s]:03d}_s{original_s:03d}.gif"
            )
            save_gif(frames, path, fps=int(cfg.gif_fps))
            media.append(wandb.Video(path, format="gif"))
        else:
            fig = render_rollout(
                viz_metrics["trajectories"][local_s],
                lanes[lane_scene_idx == local_s],
                **kwargs,
            )
            media.append(wandb.Image(fig))
            plt.close(fig)

    log_payload["train_group/rollouts"] = media
    return log_payload


@torch.no_grad()
def evaluate_and_visualize(
    policy,
    eval_pool,
    reward,
    cfg,
    it,
    wandb,
    *,
    media_tag="eval",
    include_static_baselines: bool = True,
):
    """Roll out the first ``eval_num_scenes`` fixed scenes of ``eval_pool`` and
    build trajectory media. ``media_tag`` namespaces the on-disk GIF directory so
    callers that pass different pools (e.g. val vs. a train-scene subset) don't
    overwrite each other's frames.

    Metrics are computed over all ``eval_num_scenes`` scenes; GIF/figure media
    are only rendered for the first ``eval_media_scenes`` of them (default: all),
    so the metric sample size can grow without inflating render time or wandb
    media volume. With 8 scenes a ~2% collision rate is indistinguishable from
    0 (resolution 0.125); rates need >= 64 scenes to be readable."""
    import matplotlib.pyplot as plt

    from ddpo.viz import CONTROL_COLOR, render_rollout, render_rollout_frames, save_gif

    n = min(int(cfg.eval_num_scenes), len(eval_pool))
    media_n = min(n, int(cfg.eval_media_scenes))
    cond = eval_pool.batch_from_indices(list(range(n)))

    save_gif_mode = bool(cfg.save_gif)
    gif_dir = Path(cfg.output_dir) / "eval_media" / media_tag
    if save_gif_mode:
        gif_dir.mkdir(parents=True, exist_ok=True)

    # Fixed conditioning + fixed noise: after_rl / before_rl differ only by model
    # weights. no_adv decodes the conditioning graph without appending the DDPO
    # adversary, so it has no generated-agent highlight.
    variants = {}
    torch.manual_seed(int(cfg.seed))
    variants["after_rl"] = policy.sample(cond)[0]
    if (
        include_static_baselines
        and cfg.eval_visualize_before_rl
    ):
        torch.manual_seed(int(cfg.seed))
        variants["before_rl"] = policy.sample(cond, use_reference=True)[0]
    if (
        include_static_baselines
        and cfg.eval_visualize_no_adv
    ):
        variants["no_adv"] = policy.conditioning_scenes(cond)

    metrics_by_variant = {}
    media_by_variant = {}
    adv_rows = []  # raw adv-agent numbers across every variant/scene -> one wandb.Table
    for name, scenes in variants.items():
        metrics = reward.evaluate(scenes, record_trajectories=True)
        metrics_by_variant[name] = metrics
        media = []

        lanes = scenes.lane_polylines
        if isinstance(lanes, torch.Tensor):
            lanes = lanes.detach().cpu().numpy()
        lane_scene_idx = scenes.meta["lane_scene_idx"].detach().cpu().numpy()
        states = scenes.agent_states.detach().cpu().numpy()
        types = scenes.agent_types.detach().cpu().numpy()
        agent_scene_idx = scenes.agent_scene_idx.detach().cpu().numpy()
        gen_agent_mask = scenes.meta.get("gen_agent_mask")
        if isinstance(gen_agent_mask, torch.Tensor):
            gen_agent_mask = gen_agent_mask.detach().cpu().numpy()

        if wandb is None:
            media_by_variant[name] = media
            continue
        for s in range(scenes.num_scenes):
            a_sel = agent_scene_idx == s
            # Flag DDPO-generated non-ego agents green so it is clear
            # which agents to watch; ego (local index 0) keeps its red.
            agent_colors = None
            if gen_agent_mask is not None:
                gen_agent_s = gen_agent_mask[a_sel]
                agent_colors = [
                    CONTROL_COLOR if (i > 0 and bool(gen_agent_s[i])) else None
                    for i in range(len(gen_agent_s))
                ]
                adv_rows.extend(_adv_agent_rows(
                    name, s, states[a_sel], types[a_sel], gen_agent_s,
                    metrics["trajectories"][s], metrics["reward"][s],
                ))
            # Every scene contributes metrics + a table row; only the first
            # eval_media_scenes get the (expensive) GIF/figure rendering.
            if s >= media_n:
                continue
            # Full Phase 2 reward breakdown overlaid on the GIF/figure.
            components = {
                k: metrics[k][s]
                for k in (
                    "criticality", "r_ttc", "r_approach",
                    "constraint", "c_spawn_lane", "c_goal_lane", "c_overlap", "c_parking", "c_invalid",
                    "spawn_lane_dist", "goal_lane_dist", "init_overlap_frac",
                    "ego_adv_init_dist", "ego_adv_min_dist_warmup",
                    "c_invalid_reason",
                )
                if k in metrics
            }
            kwargs = dict(
                agent_states=states[a_sel],
                agent_types=types[a_sel],
                agent_colors=agent_colors,
                reward=metrics["reward"][s],
                ego_collision=metrics["ego_collision"][s] > 0,
                ego_offroad=metrics["ego_offroad"][s] > 0,
                init_invalid=metrics["init_invalid"][s] > 0,
                ego_min_ttc=metrics["ego_min_ttc"][s],
                goal_offlane_frac=metrics["goal_offlane_frac"][s],
                parking_mismatch_frac=metrics["parking_mismatch_frac"][s],
                components=components,
                title=f"{name} it{it} scene{s}",
            )
            if save_gif_mode:
                frames = render_rollout_frames(
                    metrics["trajectories"][s], lanes[lane_scene_idx == s],
                    max_frames=int(cfg.gif_max_frames), **kwargs,
                )
                path = str(gif_dir / f"{name}_scene_{s}.gif")
                save_gif(frames, path, fps=int(cfg.gif_fps))
                media.append(wandb.Video(path, format="gif"))
            else:
                fig = render_rollout(metrics["trajectories"][s], lanes[lane_scene_idx == s], **kwargs)
                media.append(wandb.Image(fig))
                plt.close(fig)
        media_by_variant[name] = media
    adv_table = (
        wandb.Table(columns=_ADV_TABLE_COLUMNS, data=adv_rows)
        if (wandb is not None and adv_rows)
        else None
    )
    return metrics_by_variant, media_by_variant, adv_table


def _eval_and_log(
    policy,
    pool,
    reward,
    cfg,
    it,
    wandb,
    *,
    prefix,
    include_static_baselines: bool = True,
):
    """Roll out a fixed pool, print a summary, and log metrics + rollout media
    under ``prefix``: ``'val'`` for the held-out val scenes, ``'train_viz'`` for a
    fixed subset of the train scenes (so the rising train reward can be inspected
    visually). Returns the per-variant metrics dict."""
    ev_by_variant, images_by_variant, adv_table = evaluate_and_visualize(
        policy,
        pool,
        reward,
        cfg,
        it,
        wandb,
        media_tag=prefix,
        include_static_baselines=include_static_baselines,
    )
    ev = ev_by_variant["after_rl"]
    ev_crit = float((ev["ego_collision"] > 0).mean())
    ev_inval = float((ev["init_invalid"] > 0).mean())
    print(
        f"   [{prefix} it {it:04d}] after_rl critical_rate={ev_crit:.3f} "
        f"init_invalid={ev_inval:.3f} goal_offlane={float(ev['goal_offlane_frac'].mean()):.3f}"
    )
    for name, cmp_ev in ev_by_variant.items():
        if name == "after_rl":
            continue
        print(
            f"      [{prefix}/{name}] reward={float(cmp_ev['reward'].mean()):.3f} "
            f"critical_rate={float((cmp_ev['ego_collision'] > 0).mean()):.3f} "
            f"goal_offlane={float(cmp_ev['goal_offlane_frac'].mean()):.3f} "
            f"parking_mismatch={float(cmp_ev['parking_mismatch_frac'].mean()):.3f}"
        )
    if wandb is not None:
        log_payload = {
            "ddpo_step": int(it),
        }
        if adv_table is not None:
            log_payload[f"{prefix}/adv_data"] = adv_table
        for name, cmp_ev in ev_by_variant.items():
            cmp_ttc = cmp_ev["ego_min_ttc"]
            cmp_ttc_finite = np.isfinite(cmp_ttc)
            log_payload.update(
                {
                    f"{prefix}/{name}/reward": float(cmp_ev["reward"].mean()),
                    f"{prefix}/{name}/critical_rate": float((cmp_ev["ego_collision"] > 0).mean()),
                    f"{prefix}/{name}/ego_fault_collision_rate": float(
                        (cmp_ev["ego_fault_collision"] > 0).mean()
                    ),
                    f"{prefix}/{name}/init_invalid": float(cmp_ev["init_invalid"].mean()),
                    f"{prefix}/{name}/ego_min_ttc": (
                        float(cmp_ttc[cmp_ttc_finite].mean())
                        if cmp_ttc_finite.any()
                        else float("nan")
                    ),
                    f"{prefix}/{name}/ego_offroad_rate": float(cmp_ev["ego_offroad"].mean()),
                    f"{prefix}/{name}/reached_goal_rate": float(cmp_ev["reached_goal"].mean()),
                    f"{prefix}/{name}/goal_offlane_frac": float(cmp_ev["goal_offlane_frac"].mean()),
                    f"{prefix}/{name}/parking_mismatch_frac": float(
                        cmp_ev["parking_mismatch_frac"].mean()
                    ),
                    f"{prefix}/{name}/rollouts": images_by_variant.get(name, []),
                }
            )
        wandb.log(log_payload)
    return ev_by_variant


def run_ddpo(cfg_root):
    cfg = cfg_root.ddpo
    device = cfg.device
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    model_type, policy, pool, eval_dataset_cfg = _build_policy_and_pool(cfg_root, cfg, device)
    # Strict three-config construction: every yaml key under simulator:/reward:
    # must match a dataclass field (missing or unknown keys raise a TypeError).
    reward = PufferSimulator(
        planner_cfg=cfg.planner,
        simulator_cfg=SimulatorConfig(
            seed=int(cfg.seed),
            gen_invalid=_build_gen_invalid(cfg, eval_dataset_cfg),
            **OmegaConf.to_container(cfg.simulator, resolve=True),
        ),
        reward_cfg=RewardConfig(**OmegaConf.to_container(cfg.reward, resolve=True)),
    )
    ddpo_cfg = DDPOConfig(**OmegaConf.to_container(cfg.algo, resolve=True))
    # Adaptive KL-to-base coefficient (inert unless ddpo.kl_target > 0); its
    # current coef is checkpointed so a resume keeps the adapted trust region.
    kl_ctrl = AdaptiveKLController(ddpo_cfg)
    trainable_params = list(policy.trainable_parameters())
    opt = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- optional resume from last.ckpt -----------------------------------
    # Faithful continuation: restore the trainable net weights, the AdamW state
    # (so resume doesn't suffer a cold-Adam transient), the global RNG and the
    # iteration counter. Crucially the KL reference (policy.ref) is NOT touched:
    # it stays anchored to the base checkpoint the policy was just built from, so
    # kl_to_base keeps measuring drift from the pretrained manifold across resumes
    # (loading a DDPO ckpt as the base would silently reset that anchor to ~0).
    start_it = 0
    resume_wandb_id = None
    resume_path = out_dir / "last.ckpt"
    if cfg.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        sd = ckpt["state_dict"]
        net_sd = {k[len("diff_model."):]: v for k, v in sd.items() if k.startswith("diff_model.")}
        policy.load_state_dict(net_sd)
        rs = ckpt.get("ddpo", {})
        if "opt" in rs:
            opt.load_state_dict(rs["opt"])
        _rng_restore(rs.get("rng", {}), device)
        start_it = int(rs.get("it", 0))
        resume_wandb_id = rs.get("wandb_id", None)
        kl_ctrl.coef = float(rs.get("kl_coef", kl_ctrl.coef))
        print(
            f"[ddpo] resumed from {resume_path} at it={start_it} "
            f"(opt={'yes' if 'opt' in rs else 'no'}, "
            f"rng={'yes' if rs.get('rng') else 'no'}, wandb_id={resume_wandb_id})"
        )

    wandb = None
    wandb_run_id = None
    if cfg.wandb.enabled:
        import wandb as _wandb

        init_kwargs = dict(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name or _default_run_name(cfg, model_type),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        # Continue the same wandb run (same id) so the curves don't split.
        if resume_wandb_id is not None:
            init_kwargs["id"] = resume_wandb_id
            init_kwargs["resume"] = "allow"
        _wandb.init(**init_kwargs)
        # W&B's internal step can be ahead of the local checkpoint after an
        # interrupted run (history was logged, last.ckpt was not). Use an explicit
        # DDPO x-axis instead of passing `step=it`, so resumed logs are not dropped
        # for being lower than the run's current internal step.
        _wandb.define_metric("ddpo_step")
        _wandb.define_metric("*", step_metric="ddpo_step")
        wandb = _wandb
        wandb_run_id = _wandb.run.id

    eval_every = int(cfg.eval_every)
    eval_pool = None
    if eval_every > 0:
        eval_pool = LDMAdvConditioningPool(
            eval_dataset_cfg,
            split_name=cfg.eval_split,
            pool_size=cfg.eval_num_scenes,
            device=device,
            seed=cfg.seed,
            min_ego_drive=cfg.min_ego_drive,
            prune_base_to_ego=cfg.prune_base_to_ego,
            insert_adv_as_extra=cfg.insert_adv_as_extra,
            adv_cond_target=cfg.adv_cond_target,
        )

    min_diffusion_t = int(cfg.min_diffusion_t)  # TODO: understand the priciple
    if min_diffusion_t < 1 or min_diffusion_t >= policy.net.n_timesteps:
        raise ValueError(
            f"min_diffusion_t must be in [1, {policy.net.n_timesteps - 1}], got {min_diffusion_t}"
        )
    # Records are in reverse diffusion order. Skip very low-noise transitions by
    # true diffusion timestep, not sampler index, so DDIM sub-sampling works.
    stochastic_steps = policy.stochastic_step_indices(min_diffusion_t)
    if len(stochastic_steps) == 0:
        raise ValueError(
            f"no stochastic diffusion steps for sampler={policy.sampler}, "
            f"ddim_eta={getattr(policy, 'ddim_eta', None)}, min_diffusion_t={min_diffusion_t}"
        )
    # Per-context (GRPO) grouping: each batch is num_groups distinct contexts,
    # each replicated group_size times, and advantages are whitened within group
    # (see compute_advantages). group_size=1 -> legacy global whitening.
    group_size = int(ddpo_cfg.group_size)
    if group_size > 1 and cfg.batch_size % group_size != 0:
        raise ValueError(
            f"batch_size ({cfg.batch_size}) must be divisible by group_size ({group_size})"
        )
    static_baselines_once = bool(cfg.eval_static_baselines_once)
    static_baseline_prefixes_logged: set[str] = set()

    for it in range(start_it, cfg.num_iterations):
        # ---- rollout / collect -------------------------------------------------
        if group_size > 1:
            num_groups = cfg.batch_size // group_size
            cond, group_ids = pool.sample_group_batch(num_groups, group_size)
            group_ids = group_ids.to(device)
        else:
            cond = pool.sample_batch(cfg.batch_size)
            group_ids = None
        scenes, traj = policy.sample(cond)
        # Advance the approach-weight annealing schedule (no-op unless the
        # reward config enables it); the periodic evals below reuse the same
        # weight so train and val rewards stay comparable.
        reward.set_train_iteration(it)
        metrics = reward.evaluate(scenes)

        rewards = torch.as_tensor(metrics["reward"], device=device)
        advantages = compute_advantages(rewards, ddpo_cfg, group_ids)

        # ---- DDPO update over random-k steps ----------------------------------
        skipped_updates = 0
        log = {"loss": float("nan")}
        for _ in range(cfg.inner_epochs):
            k_idx = stochastic_steps[torch.randperm(len(stochastic_steps))[: cfg.k_steps]]
            # Log-density ratios are much more numerically sensitive than the
            # denoiser forward used for sampling. Keep them in fp32 by default:
            # bf16 rounding at low-noise diffusion steps can make the immediate
            # on-policy ratio drift far from 1 even with inner_epochs == 1.
            with _bf16_autocast(device, enabled=cfg.logprob_bf16):
                new_lp, kl_term = policy.trajectory_logprob(
                    traj, cond, k_idx,
                    with_kl=kl_ctrl.coef > 0 or ddpo_cfg.kl_target > 0,
                )
                old_lp = traj.old_logprob[:, k_idx]
                kl_term = kl_term.float() if kl_term is not None else None
                loss, log, parts = ddpo_loss(
                    new_lp.float(), old_lp.float(), advantages.float(), ddpo_cfg,
                    kl_term, kl_coef=kl_ctrl.coef,
                )
            loss = loss.float()
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped_updates += 1
                continue
            if ddpo_cfg.decouple_kl_grad and parts["kl"] is not None:
                # Clip the pg and KL gradients to grad_clip SEPARATELY: with a
                # single global clip, an exploding pg term rescales the KL
                # pullback to nothing exactly when the trust region is needed
                # most. Each term gets its own norm budget; the step uses the
                # sum of the two clipped directions.
                parts["pg"].backward(retain_graph=True)
                pg_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, cfg.grad_clip, error_if_nonfinite=False
                )
                if not torch.isfinite(pg_norm):
                    opt.zero_grad(set_to_none=True)
                    skipped_updates += 1
                    continue
                pg_grads = [
                    None if p.grad is None else p.grad.detach().clone()
                    for p in trainable_params
                ]
                opt.zero_grad(set_to_none=True)
                (kl_ctrl.coef * parts["kl"]).backward()
                kl_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, cfg.grad_clip, error_if_nonfinite=False
                )
                if not torch.isfinite(kl_norm):
                    opt.zero_grad(set_to_none=True)
                    skipped_updates += 1
                    continue
                for p, g in zip(trainable_params, pg_grads):
                    if g is None:
                        continue
                    if p.grad is None:
                        p.grad = g
                    else:
                        p.grad.add_(g)
                log["pg_grad_norm"] = float(pg_norm)
                log["kl_grad_norm"] = float(kl_norm)
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, cfg.grad_clip, error_if_nonfinite=False
                )
                if not torch.isfinite(grad_norm):
                    opt.zero_grad(set_to_none=True)
                    skipped_updates += 1
                    continue
            opt.step()
        # One controller nudge per iteration, from the last measured KL (with
        # inner_epochs == 1 this is THE update's KL).
        kl_ctrl.update(log.get("kl_to_base", float("nan")))

        crit = float(rewards.gt(0).float().mean())
        inval = float(metrics["init_invalid"].mean())
        mttc = metrics["ego_min_ttc"]
        finite_ttc = np.isfinite(mttc)
        mean_ttc = float(mttc[finite_ttc].mean()) if finite_ttc.any() else float("nan")
        adv_dist = metrics["ego_adv_min_dist"]
        finite_dist = np.isfinite(adv_dist)
        mean_adv_dist = float(adv_dist[finite_dist].mean()) if finite_dist.any() else float("nan")
        parked = float(metrics["gen_agent_is_parked"].mean())
        invalid_cond = float(metrics.get("gen_agent_is_invalid", np.zeros(1)).mean())
        # Within-group reward spread: this is the only signal per-context
        # advantage normalisation can use. If it collapses to ~0, the group has
        # no learnable contrast (every sample equally (in)valid) and is skipped.
        group_log = {}
        if group_ids is not None:
            group_log = _group_reward_summary(rewards, group_ids, ddpo_cfg)
            grp_std = group_log.get("train/group_reward_std", float("nan"))
        else:
            grp_std = float("nan")
        near_miss = float((metrics["r_risk"] > 0.5).mean())
        coll_rate = float(metrics["ego_collision"].mean())
        fault_rate = float(metrics["ego_fault_collision"].mean())
        print(
            f"[it {it:04d}] reward={rewards.mean():.3f} pos_reward_rate={crit:.3f} "
            f"near_miss={near_miss:.3f} coll={coll_rate:.3f} fault={fault_rate:.3f} "
            f"risk={float(metrics['r_risk'].mean()):.3f} "
            f"approach={float(metrics['r_approach'].mean()):.3f} "
            f"w_app={reward.approach_coef:.2f} min_ttc={mean_ttc:.2f} "
            f"adv_dist={mean_adv_dist:.2f} parked={parked:.3f} cond_invalid={invalid_cond:.3f} "
            f"init_invalid={inval:.3f} loss={log['loss']:.4f} grp_std={grp_std:.3f} "
            f"ratio={log.get('ratio_mean', 1.0):.3f} kl={log.get('kl_to_base', 0.0):.4f} "
            f"kl_coef={kl_ctrl.coef:.3g} drop={log.get('ratio_dropped_frac', 0.0):.3f} "
            f"skipped_updates={skipped_updates}"
        )

        group_viz_log = _visualize_train_group_diversity(
            scenes, metrics, rewards, advantages, group_ids, reward, cfg, ddpo_cfg, it, wandb
        )

        if wandb is not None:
            log_payload = {
                "ddpo_step": int(it),
                "train/reward": float(rewards.mean()),
                "train/pos_reward_rate": crit,
                "train/near_miss_rate": near_miss,
                "train/init_invalid": inval,
                "train/ego_min_ttc": mean_ttc,
                "train/ego_adv_min_dist": mean_adv_dist,
                "train/gen_agent_is_parked": parked,
                "train/gen_agent_is_invalid": invalid_cond,
                "train/ego_collision_rate": coll_rate,
                "train/ego_fault_collision_rate": fault_rate,
                "train/ego_offroad_rate": float(metrics["ego_offroad"].mean()),
                "train/reached_goal_rate": float(metrics["reached_goal"].mean()),
                "train/goal_offlane_frac": float(metrics["goal_offlane_frac"].mean()),
                "train/parking_mismatch_frac": float(metrics["parking_mismatch_frac"].mean()),
                # Phase 2 reward components (mean over batch).
                "train/r_ttc": float(metrics["r_ttc"].mean()),
                "train/r_approach": float(metrics["r_approach"].mean()),
                "train/approach_coef": float(reward.approach_coef),
                "train/r_collision": float(metrics["r_collision"].mean()),
                "train/r_bonus": float(metrics["r_bonus"].mean()),
                "train/r_risk": float(metrics["r_risk"].mean()),
                "train/criticality": float(metrics["criticality"].mean()),
                "train/c_lane": float(metrics["c_lane"].mean()),
                "train/c_trivial": float(metrics["c_trivial"].mean()),
                "train/constraint": float(metrics["constraint"].mean()),
                "train/group_reward_std": grp_std,
                "train/loss": log["loss"],
                "train/pg_loss": log.get("pg_loss", log["loss"]),
                "train/ratio_mean": log.get("ratio_mean", 1.0),
                "train/ratio_dropped_frac": log.get("ratio_dropped_frac", 0.0),
                "train/kl_to_base": log.get("kl_to_base", 0.0),
                "train/kl_coef": kl_ctrl.coef,
                "train/adv_std": float(advantages.std(unbiased=False)),
                "train/skipped_updates": skipped_updates,
            }
            if "pg_grad_norm" in log:
                log_payload["train/pg_grad_norm"] = log["pg_grad_norm"]
                log_payload["train/kl_grad_norm"] = log["kl_grad_norm"]
            log_payload.update(group_log)
            log_payload.update(group_viz_log)
            wandb.log(log_payload)

        # ---- periodic held-out eval + trajectory viz --------------------------
        if eval_pool is not None and (it + 1) % eval_every == 0:
            include_static = (
                not static_baselines_once
                or "val" not in static_baseline_prefixes_logged
            )
            _eval_and_log(
                policy,
                eval_pool,
                reward,
                cfg,
                it,
                wandb,
                prefix="val",
                include_static_baselines=include_static,
            )
            static_baseline_prefixes_logged.add("val")
            # Same eval/viz pass over a fixed subset of the *train* scenes, so the
            # rising train reward can be inspected visually. Reuses the already
            # loaded training pool's first eval_num_scenes slots (deterministic +
            # cached), logged under train_viz/* alongside the val/* media.
            if cfg.eval_visualize_train:
                include_static = (
                    not static_baselines_once
                    or "train_viz" not in static_baseline_prefixes_logged
                )
                _eval_and_log(
                    policy,
                    pool,
                    reward,
                    cfg,
                    it,
                    wandb,
                    prefix="train_viz",
                    include_static_baselines=include_static,
                )
                static_baseline_prefixes_logged.add("train_viz")

        if cfg.save_every and (it + 1) % cfg.save_every == 0:
            # Lightning-compatible layout (diff_model.* prefix) so the regular
            # scenario-dreamer eval/viz tooling can load DDPO checkpoints.
            sd = {f"diff_model.{k}": v for k, v in policy.state_dict().items()}
            torch.save({"state_dict": sd}, out_dir / f"{_checkpoint_prefix(cfg, model_type)}_{it + 1:05d}.ckpt")
            # Full resume snapshot (net + AdamW + RNG + iter + wandb id), written
            # atomically to last.ckpt so `resume` can continue this exact run. The
            # numbered checkpoints above stay net-only for the eval/viz tooling.
            resume_ckpt = {
                "state_dict": sd,
                "ddpo": {
                    "it": it + 1,
                    "opt": opt.state_dict(),
                    "rng": _rng_snapshot(device),
                    "wandb_id": wandb_run_id,
                    "kl_coef": kl_ctrl.coef,
                    "base_ckpt": cfg.ldm_adv_ckpt,
                },
            }
            tmp_path = out_dir / "last.ckpt.tmp"
            torch.save(resume_ckpt, tmp_path)
            tmp_path.replace(out_dir / "last.ckpt")

    if wandb is not None:
        wandb.finish()
