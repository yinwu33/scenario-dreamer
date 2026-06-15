"""DDPO fine-tuning loops against a frozen PufferDrive planner.

This module holds the actual training loop. It is dispatched from train.py
when ``model_name == 'ddpo'`` (i.e. ``python train.py --config-name config_ddpo``),
and also from the deprecated ``train_ddpo.py`` shim. Heavy / PufferDrive-specific
imports live at module top level here so they are only paid when the ddpo path is
actually taken (train.py imports this module lazily inside its dispatch branch).

Pipeline per iteration:
    ConditioningPool.sample_batch(B)          # real conditioning graphs
    policy.sample(cond)                       # record denoising trajectory + logprob
    PufferDriveReward.evaluate(scenes)        # numpy sim port + frozen planner
    compute_advantages -> ddpo_loss over k random denoising steps (+ optional KL)
"""

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from ddpo.conditioning import ConditioningPool, LDMGoalConditioningPool
from ddpo.ddpo_loss import DDPOConfig, compute_advantages, ddpo_loss
from ddpo.policy import DMGoalDDPOPolicy
from ddpo.policy_ldm import LDMGoalDDPOPolicy
from ddpo.reward import PufferDriveReward
from utils.train_helpers import cache_latent_stats, set_latent_stats


def _set_dataset_name(cfg_node, dataset_name: str) -> None:
    OmegaConf.set_struct(cfg_node, False)
    cfg_node.dataset_name = dataset_name
    OmegaConf.set_struct(cfg_node, True)


def _build_policy_and_pool(cfg_root, cfg, device: str):
    model_type = cfg.get("model_type", "dm_goal")
    dataset_name = cfg_root.dataset_name.name

    if model_type == "dm_goal":
        _set_dataset_name(cfg_root.dm_goal, dataset_name)
        policy = DMGoalDDPOPolicy(
            cfg_root.dm_goal,
            ckpt_path=cfg.model_ckpt,
            mode=cfg.mode,
            device=device,
            use_ema_weights=cfg.get("use_ema_weights", True),
            inpaint_noised=cfg.get("inpaint_noised", True),
        )
        pool = ConditioningPool(
            cfg_root.dm_goal.dataset,
            split_name=cfg.train_split,
            pool_size=cfg.pool_size,
            device=device,
            seed=cfg.seed,
        )
        eval_dataset_cfg = cfg_root.dm_goal.dataset
    elif model_type == "ldm_goal":
        _set_dataset_name(cfg_root.ldm_goal, dataset_name)
        _set_dataset_name(cfg_root.ae_goal, dataset_name)
        if not Path(cfg_root.ldm_goal.dataset.latent_stats_path).exists():
            cache_latent_stats(cfg_root.ldm_goal)
        ldm_cfg = set_latent_stats(cfg_root.ldm_goal)
        policy = LDMGoalDDPOPolicy(
            ldm_cfg,
            cfg_root.ae_goal,
            ldm_ckpt=cfg.ldm_ckpt,
            ae_ckpt=cfg.ae_ckpt,
            device=device,
            use_ema_weights=cfg.get("use_ema_weights", True),
        )
        pool = LDMGoalConditioningPool(
            ldm_cfg.dataset,
            split_name=cfg.train_split,
            pool_size=cfg.pool_size,
            device=device,
            seed=cfg.seed,
        )
        eval_dataset_cfg = ldm_cfg.dataset
    else:
        raise ValueError(f"Unsupported ddpo.model_type: {model_type}")

    return model_type, policy, pool, eval_dataset_cfg


def _default_run_name(cfg, model_type: str) -> str:
    if model_type == "dm_goal":
        return f"ddpo_dm_goal_{cfg.mode}"
    return f"ddpo_{model_type}"


def _checkpoint_prefix(cfg, model_type: str) -> str:
    if model_type == "dm_goal":
        return f"ddpo_dm_goal_{cfg.mode}"
    return f"ddpo_{model_type}"


@torch.no_grad()
def evaluate_and_visualize(policy, eval_pool, reward, cfg, it, wandb):
    """Roll out fixed held-out val scenes and build trajectory media."""
    import matplotlib.pyplot as plt

    from ddpo.viz import render_rollout, render_rollout_frames, save_gif

    n = min(int(cfg.eval_num_scenes), len(eval_pool))
    cond = eval_pool.batch_from_indices(list(range(n)))

    save_gif_mode = bool(cfg.get("save_gif", False))
    gif_dir = Path(cfg.output_dir) / "eval_media"
    if save_gif_mode:
        gif_dir.mkdir(parents=True, exist_ok=True)

    # Fixed conditioning + fixed noise: current/reference visuals differ only by
    # model weights, while conditioning uses the validation graph's original goals.
    variants = {}
    torch.manual_seed(int(cfg.seed))
    variants["current"] = policy.sample(cond)[0]
    if cfg.get("eval_visualize_reference", True):
        torch.manual_seed(int(cfg.seed))
        variants["reference"] = policy.sample(cond, use_reference=True)[0]
    if cfg.get("eval_visualize_conditioning", True):
        variants["conditioning"] = policy.conditioning_scenes(cond)

    metrics_by_variant = {}
    media_by_variant = {}
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

        if wandb is None:
            media_by_variant[name] = media
            continue
        for s in range(scenes.num_scenes):
            a_sel = agent_scene_idx == s
            kwargs = dict(
                agent_states=states[a_sel],
                agent_types=types[a_sel],
                reward=metrics["reward"][s],
                ego_collision=metrics["ego_collision"][s] > 0,
                ego_offroad=metrics["ego_offroad"][s] > 0,
                init_invalid=metrics["init_invalid"][s] > 0,
                title=f"{name} it{it} scene{s}",
            )
            if save_gif_mode:
                frames = render_rollout_frames(
                    metrics["trajectories"][s], lanes[lane_scene_idx == s],
                    max_frames=int(cfg.get("gif_max_frames", 50)), **kwargs,
                )
                path = str(gif_dir / f"{name}_scene_{s}.gif")
                save_gif(frames, path, fps=int(cfg.get("gif_fps", 10)))
                media.append(wandb.Video(path, format="gif"))
            else:
                fig = render_rollout(metrics["trajectories"][s], lanes[lane_scene_idx == s], **kwargs)
                media.append(wandb.Image(fig))
                plt.close(fig)
        media_by_variant[name] = media
    return metrics_by_variant, media_by_variant


def run_ddpo(cfg_root):
    cfg = cfg_root.ddpo
    device = cfg.device
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    model_type, policy, pool, eval_dataset_cfg = _build_policy_and_pool(cfg_root, cfg, device)
    reward = PufferDriveReward(
        sim_steps=cfg.sim_steps,
        deterministic=cfg.get("planner_deterministic", None),
        ttc_tau=cfg.get("ttc_tau", 3.0),
        init_overlap_margin=cfg.get("init_overlap_margin", 0.0),
        goal_offlane_threshold=cfg.get("goal_offlane_threshold", 3.0),
        goal_onroad_threshold=cfg.get("goal_onroad_threshold", 2.0),
        goal_offlane_penalty=cfg.get("goal_offlane_penalty", 0.5),
        parking_mismatch_penalty=cfg.get("parking_mismatch_penalty", 0.5),
        seed=cfg.seed,
    )
    ddpo_cfg = DDPOConfig(**OmegaConf.to_container(cfg.ddpo, resolve=True))
    trainable_params = list(policy.trainable_parameters())
    opt = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    wandb = None
    if cfg.get("wandb", {}).get("enabled", False):
        import wandb as _wandb

        _wandb.init(
            project=cfg.wandb.get("project", "scenario_dreamer_ddpo"),
            entity=cfg.wandb.get("entity", None),
            name=cfg.wandb.get("run_name", None) or _default_run_name(cfg, model_type),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb = _wandb

    eval_every = int(cfg.get("eval_every", 0))
    eval_pool = None
    if eval_every > 0:
        pool_cls = ConditioningPool if model_type == "dm_goal" else LDMGoalConditioningPool
        eval_pool = pool_cls(
            eval_dataset_cfg,
            split_name=cfg.eval_split,
            pool_size=cfg.eval_num_scenes,
            device=device,
            seed=cfg.seed,
        )

    H = policy.num_sampling_steps
    min_diffusion_t = int(cfg.get("min_diffusion_t", 5))  # TODO: understand the priciple
    if min_diffusion_t < 1 or min_diffusion_t >= H:
        raise ValueError(f"min_diffusion_t must be in [1, {H - 1}], got {min_diffusion_t}")
    # records are in reverse diffusion order: index 0 is t=H-1, index H-1 is the
    # deterministic t=0 step. Skip very low-t steps (tiny posterior variance ->
    # numerically brittle log-prob ratios).
    stochastic_steps = torch.arange(0, H - min_diffusion_t)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for it in range(cfg.num_iterations):
        # ---- rollout / collect -------------------------------------------------
        cond = pool.sample_batch(cfg.batch_size)
        scenes, traj = policy.sample(cond)
        metrics = reward.evaluate(scenes)

        rewards = torch.as_tensor(metrics["reward"], device=device)
        advantages = compute_advantages(rewards, ddpo_cfg)

        # ---- DDPO update over random-k steps ----------------------------------
        skipped_updates = 0
        log = {"loss": float("nan")}
        for _ in range(cfg.inner_epochs):
            k_idx = stochastic_steps[torch.randperm(len(stochastic_steps))[: cfg.k_steps]]
            new_lp = policy.trajectory_logprob(traj, cond, k_idx)
            old_lp = traj.old_logprob[:, k_idx]
            ref_lp = (
                policy.trajectory_logprob(traj, cond, k_idx, use_reference=True)
                if ddpo_cfg.kl_coef > 0
                else None
            )
            loss, log = ddpo_loss(new_lp, old_lp, advantages, ddpo_cfg, ref_lp)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped_updates += 1
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, cfg.grad_clip, error_if_nonfinite=False
            )
            if not torch.isfinite(grad_norm):
                opt.zero_grad(set_to_none=True)
                skipped_updates += 1
                continue
            opt.step()

        crit = float(rewards.gt(0).float().mean())
        inval = float(metrics["init_invalid"].mean())
        mttc = metrics["ego_min_ttc"]
        finite_ttc = np.isfinite(mttc)
        mean_ttc = float(mttc[finite_ttc].mean()) if finite_ttc.any() else float("nan")
        print(
            f"[it {it:04d}] reward={rewards.mean():.3f} critical_rate={crit:.3f} "
            f"collide={float(metrics['ego_collision'].mean()):.3f} min_ttc={mean_ttc:.2f} "
            f"init_invalid={inval:.3f} loss={log['loss']:.4f} "
            f"ratio={log.get('ratio_mean', 1.0):.3f} kl={log.get('kl_to_base', 0.0):.4f} "
            f"skipped_updates={skipped_updates}"
        )

        if wandb is not None:
            wandb.log(
                {
                    "train/reward": float(rewards.mean()),
                    "train/critical_rate": crit,
                    "train/init_invalid": inval,
                    "train/ego_min_ttc": mean_ttc,
                    "train/ego_collision_rate": float(metrics["ego_collision"].mean()),
                    "train/ego_offroad_rate": float(metrics["ego_offroad"].mean()),
                    "train/reached_goal_rate": float(metrics["reached_goal"].mean()),
                    "train/goal_offlane_frac": float(metrics["goal_offlane_frac"].mean()),
                    "train/parking_mismatch_frac": float(metrics["parking_mismatch_frac"].mean()),
                    "train/loss": log["loss"],
                    "train/pg_loss": log.get("pg_loss", log["loss"]),
                    "train/ratio_mean": log.get("ratio_mean", 1.0),
                    "train/kl_to_base": log.get("kl_to_base", 0.0),
                    "train/adv_std": float(advantages.std(unbiased=False)),
                    "train/skipped_updates": skipped_updates,
                },
                step=it,
            )

        # ---- periodic held-out eval + trajectory viz --------------------------
        if eval_pool is not None and (it + 1) % eval_every == 0:
            ev_by_variant, images_by_variant = evaluate_and_visualize(policy, eval_pool, reward, cfg, it, wandb)
            ev = ev_by_variant["current"]
            ev_crit = float((ev["ego_collision"] > 0).mean())
            ev_inval = float((ev["init_invalid"] > 0).mean())
            print(
                f"   [eval it {it:04d}] current critical_rate={ev_crit:.3f} "
                f"init_invalid={ev_inval:.3f} goal_offlane={float(ev['goal_offlane_frac'].mean()):.3f}"
            )
            for name, cmp_ev in ev_by_variant.items():
                if name == "current":
                    continue
                print(
                    f"      [{name}] reward={float(cmp_ev['reward'].mean()):.3f} "
                    f"critical_rate={float((cmp_ev['ego_collision'] > 0).mean()):.3f} "
                    f"goal_offlane={float(cmp_ev['goal_offlane_frac'].mean()):.3f} "
                    f"parking_mismatch={float(cmp_ev['parking_mismatch_frac'].mean()):.3f}"
                )
            if wandb is not None:
                ev_ttc = ev["ego_min_ttc"]
                ev_ttc_finite = np.isfinite(ev_ttc)
                log_payload = {
                    "val/critical_rate": ev_crit,
                    "val/init_invalid": ev_inval,
                    "val/ego_min_ttc": float(ev_ttc[ev_ttc_finite].mean()) if ev_ttc_finite.any() else float("nan"),
                    "val/ego_offroad_rate": float(ev["ego_offroad"].mean()),
                    "val/reached_goal_rate": float(ev["reached_goal"].mean()),
                    "val/goal_offlane_frac": float(ev["goal_offlane_frac"].mean()),
                    "val/parking_mismatch_frac": float(ev["parking_mismatch_frac"].mean()),
                    "val/rollouts": images_by_variant.get("current", []),
                }
                for name, cmp_ev in ev_by_variant.items():
                    log_payload.update(
                        {
                            f"val/{name}/reward": float(cmp_ev["reward"].mean()),
                            f"val/{name}/critical_rate": float((cmp_ev["ego_collision"] > 0).mean()),
                            f"val/{name}/init_invalid": float(cmp_ev["init_invalid"].mean()),
                            f"val/{name}/ego_offroad_rate": float(cmp_ev["ego_offroad"].mean()),
                            f"val/{name}/reached_goal_rate": float(cmp_ev["reached_goal"].mean()),
                            f"val/{name}/goal_offlane_frac": float(cmp_ev["goal_offlane_frac"].mean()),
                            f"val/{name}/rollouts": images_by_variant.get(name, []),
                        }
                    )
                wandb.log(log_payload, step=it)

        if cfg.save_every and (it + 1) % cfg.save_every == 0:
            # Lightning-compatible layout (diff_model.* prefix) so the regular
            # scenario-dreamer eval/viz tooling can load DDPO checkpoints.
            sd = {f"diff_model.{k}": v for k, v in policy.state_dict().items()}
            torch.save({"state_dict": sd}, out_dir / f"{_checkpoint_prefix(cfg, model_type)}_{it + 1:05d}.ckpt")

    if wandb is not None:
        wandb.finish()
