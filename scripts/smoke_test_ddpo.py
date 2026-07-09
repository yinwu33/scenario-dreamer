"""End-to-end DDPO smoke test: one tiny iteration per mode.

Run from the repo root after `source scripts/define_env_variables.sh`:
    .venv/bin/python scripts/smoke_test_ddpo.py [--modes full agent_only goal_only]
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from hydra import compose, initialize_config_dir

from cfgs.config import CONFIG_PATH
from ddpo.conditioning import ConditioningPool
from ddpo.ddpo_loss import DDPOConfig, compute_advantages, ddpo_loss
from ddpo.policy import DMGoalDDPOPolicy
from ddpo.reward import PufferDriveReward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["full", "agent_only", "goal_only"])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--control-agent-num", type=int, default=-1,
                    help="number of non-ego agents to generate (-1 = all)")
    ap.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    ap.add_argument("--ddim-steps", type=int, default=25)
    ap.add_argument("--ddim-eta", type=float, default=1.0)
    ap.add_argument("--no-control-ego", dest="control_ego", action="store_false",
                    help="fix ego to GT (generate only other agents)")
    ap.set_defaults(control_ego=True)
    args = ap.parse_args()

    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        cfg = compose(config_name="config_critical_scene_dm_goal_ddpm")

    pool = ConditioningPool(cfg.dm_goal.dataset, split_name="val", pool_size=8,
                            device=args.device, seed=0,
                            control_agent_num=args.control_agent_num)
    reward = PufferDriveReward(
        sim_steps=91,
        goal_offlane_threshold=cfg.ddpo.get("goal_offlane_threshold", 3.0),
        goal_offlane_penalty=cfg.ddpo.get("goal_offlane_penalty", 0.5),
        parking_mismatch_penalty=cfg.ddpo.get("parking_mismatch_penalty", 0.5),
        seed=0,
    )
    ddpo_cfg = DDPOConfig(estimator="is", kl_coef=0.01)

    for mode in args.modes:
        print(f"\n=== mode={mode} ===")
        t0 = time.time()
        policy = DMGoalDDPOPolicy(cfg.dm_goal, ckpt_path=cfg.ddpo.model_ckpt, mode=mode,
                                  device=args.device, control_ego=args.control_ego,
                                  control_agent_num=args.control_agent_num,
                                  sampler=args.sampler, ddim_steps=args.ddim_steps,
                                  ddim_eta=args.ddim_eta)
        cond = pool.sample_batch(args.batch_size)
        scenes, traj = policy.sample(cond)
        t_sample = time.time() - t0
        print(f"sampled {scenes.num_scenes} scenes, {scenes.agent_states.shape[0]} agents, "
              f"{t_sample:.1f}s; old_lp finite={torch.isfinite(traj.old_logprob).all().item()}")

        t0 = time.time()
        metrics = reward.evaluate(scenes, record_trajectories=False)
        print(f"rollout {time.time() - t0:.1f}s: reward={metrics['reward']}, "
              f"collision={metrics['ego_collision']}, init_invalid={metrics['init_invalid']}, "
              f"reached_goal={metrics['reached_goal']}")

        rewards = torch.as_tensor(metrics["reward"], device=args.device)
        adv = compute_advantages(rewards, ddpo_cfg)
        stochastic_steps = policy.stochastic_step_indices(min_diffusion_t=5)
        k_idx = stochastic_steps[torch.randperm(len(stochastic_steps))[:4]]
        new_lp, kl_term = policy.trajectory_logprob(
            traj, cond, k_idx, with_kl=ddpo_cfg.kl_coef > 0
        )
        old_lp = traj.old_logprob[:, k_idx]
        loss, log, _parts = ddpo_loss(new_lp, old_lp, adv, ddpo_cfg, kl_term)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(list(policy.trainable_parameters()), 1.0)
        # ratio should be ~1 on the same parameters; grads must be finite
        print(f"loss={log['loss']:.4f} ratio={log['ratio_mean']:.4f} "
              f"kl={log.get('kl_to_base', 0):.5f} grad_norm={float(gnorm):.3f}")
        assert torch.isfinite(loss), "non-finite loss"
        assert abs(log["ratio_mean"] - 1.0) < 1e-2, "IS ratio drifted on identical params"
        del policy
        torch.cuda.empty_cache()
    print("\nsmoke test OK")


if __name__ == "__main__":
    main()
