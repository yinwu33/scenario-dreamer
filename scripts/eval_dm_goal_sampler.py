"""Compare DDPM and stochastic DDIM sampling quality for the dm_goal base model.

This is intentionally inference-only: it does not use DDPO loss or update model
weights. The usual workflow is to validate DDIM25 first, fall back to DDIM50 if
needed, then enable the chosen sampler in DDPO training.

Example:
    .venv/bin/python scripts/eval_dm_goal_sampler.py \
        --config-name config_ddpo_dm_goal --num-scenes 256 --candidates 25 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cfgs.config import CONFIG_PATH
from ddpo.conditioning import ConditioningPool
from ddpo.policy import DMGoalDDPOPolicy
from ddpo.reward import PufferDriveReward


METRIC_KEYS = (
    "reward",
    "ego_collision",
    "init_invalid",
    "goal_offlane_frac",
    "parking_mismatch_frac",
    "controlled_parking_frac",
    "ego_min_ttc",
    "ego_adv_min_dist",
)


def _set_dataset_name(cfg_node, dataset_name: str) -> None:
    OmegaConf.set_struct(cfg_node, False)
    cfg_node.dataset_name = dataset_name
    OmegaConf.set_struct(cfg_node, True)


def _build_reward(cfg):
    return PufferDriveReward(
        sim_steps=cfg.sim_steps,
        deterministic=cfg.get("planner_deterministic", None),
        ttc_tau=cfg.get("ttc_tau", 3.0),
        init_overlap_margin=cfg.get("init_overlap_margin", 0.0),
        goal_offlane_threshold=cfg.get("goal_offlane_threshold", 3.0),
        goal_onroad_threshold=cfg.get("goal_onroad_threshold", 2.0),
        goal_offlane_penalty=cfg.get("goal_offlane_penalty", 0.5),
        parking_mismatch_penalty=cfg.get("parking_mismatch_penalty", 0.5),
        min_dist_coef=cfg.get("min_dist_coef", 0.0),
        min_dist_dmax=cfg.get("min_dist_dmax", 20.0),
        controlled_parking_penalty=cfg.get("controlled_parking_penalty", 0.0),
        seed=cfg.seed,
        backend=cfg.get("reward_backend", "numpy"),
        pufferdrive_root=cfg.get("pufferdrive_root", None),
    )


def _build_policy(cfg_root, cfg, device: str, *, sampler: str, ddim_steps: int | None):
    return DMGoalDDPOPolicy(
        cfg_root.dm_goal,
        ckpt_path=cfg.model_ckpt,
        mode=cfg.mode,
        device=device,
        use_ema_weights=cfg.get("use_ema_weights", True),
        inpaint_noised=cfg.get("inpaint_noised", True),
        control_ego=cfg.get("control_ego", True),
        control_agent_num=cfg.get("control_agent_num", -1),
        sampler=sampler,
        ddim_steps=ddim_steps,
        ddim_eta=cfg.get("ddim_eta", 1.0),
    )


def _concat_metrics(chunks: list[dict]) -> dict[str, np.ndarray]:
    out = {}
    for key in METRIC_KEYS:
        vals = [m[key] for m in chunks if key in m]
        if vals:
            out[key] = np.concatenate(vals, axis=0)
    return out


def _summarize(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    summary = {}
    for key, value in metrics.items():
        arr = np.asarray(value, dtype=np.float32)
        finite = np.isfinite(arr)
        summary[key] = float(arr[finite].mean()) if finite.any() else float("nan")
    return summary


@torch.no_grad()
def _evaluate_sampler(
    *,
    name: str,
    sampler: str,
    ddim_steps: int | None,
    cfg_root,
    pool: ConditioningPool,
    reward: PufferDriveReward,
    indices: list[int],
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[dict[str, float], float]:
    print(f"loading policy for {name}", flush=True)
    policy = _build_policy(cfg_root, cfg_root.ddpo, device, sampler=sampler, ddim_steps=ddim_steps)
    chunks = []
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    t0 = time.perf_counter()
    for start in range(0, len(indices), batch_size):
        print(f"{name}: batch {start // batch_size + 1}", flush=True)
        cond = pool.batch_from_indices(indices[start : start + batch_size])
        scenes, _ = policy.sample(cond)
        chunks.append(reward.evaluate(scenes, record_trajectories=False))
    elapsed = time.perf_counter() - t0
    del policy
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    summary = _summarize(_concat_metrics(chunks))
    summary["seconds"] = float(elapsed)
    summary["scenes_per_second"] = float(len(indices) / max(elapsed, 1e-6))
    print(
        f"{name:8s} reward={summary['reward']:+.3f} "
        f"init={summary['init_invalid']:.3f} goalOff={summary['goal_offlane_frac']:.3f} "
        f"parked={summary['controlled_parking_frac']:.3f} "
        f"advDist={summary['ego_adv_min_dist']:.2f} time={elapsed:.1f}s",
        flush=True,
    )
    return summary, elapsed


def _passes(base: dict[str, float], cand: dict[str, float], args) -> tuple[bool, list[str]]:
    failures = []
    checks = (
        ("init_invalid", args.max_init_invalid_delta),
        ("goal_offlane_frac", args.max_goal_offlane_delta),
        ("controlled_parking_frac", args.max_controlled_parking_delta),
        ("parking_mismatch_frac", args.max_parking_mismatch_delta),
    )
    for key, max_delta in checks:
        delta = cand[key] - base[key]
        if not np.isfinite(delta) or delta > max_delta:
            failures.append(f"{key} delta {delta:+.3f} > {max_delta:.3f}")
    return len(failures) == 0, failures


@torch.no_grad()
def _save_visuals(
    *,
    samplers: list[tuple[str, str, int | None]],
    cfg_root,
    pool: ConditioningPool,
    reward: PufferDriveReward,
    out_dir: Path,
    num_visuals: int,
    device: str,
    seed: int,
    save_gifs: bool,
) -> None:
    import matplotlib.pyplot as plt

    from ddpo.viz import CONTROL_COLOR, render_rollout, render_rollout_frames, save_gif

    out_dir.mkdir(parents=True, exist_ok=True)
    vis_indices = list(range(num_visuals))
    for name, sampler, ddim_steps in samplers:
        policy = _build_policy(cfg_root, cfg_root.ddpo, device, sampler=sampler, ddim_steps=ddim_steps)
        torch.manual_seed(seed)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        cond = pool.batch_from_indices(vis_indices)
        scenes, _ = policy.sample(cond)
        metrics = reward.evaluate(scenes, record_trajectories=True)

        lanes = scenes.lane_polylines
        if isinstance(lanes, torch.Tensor):
            lanes = lanes.detach().cpu().numpy()
        lane_scene_idx = scenes.meta["lane_scene_idx"].detach().cpu().numpy()
        states = scenes.agent_states.detach().cpu().numpy()
        types = scenes.agent_types.detach().cpu().numpy()
        agent_scene_idx = scenes.agent_scene_idx.detach().cpu().numpy()
        controlled = scenes.meta.get("controlled_mask")
        if isinstance(controlled, torch.Tensor):
            controlled = controlled.detach().cpu().numpy()

        for s in range(scenes.num_scenes):
            a_sel = agent_scene_idx == s
            agent_colors = None
            if controlled is not None:
                ctrl_s = controlled[a_sel]
                agent_colors = [
                    CONTROL_COLOR if (i > 0 and bool(ctrl_s[i])) else None
                    for i in range(len(ctrl_s))
                ]
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
                title=f"{name} scene{s}",
            )
            fig = render_rollout(
                metrics["trajectories"][s],
                lanes[lane_scene_idx == s],
                **kwargs,
            )
            fig.savefig(out_dir / f"{name}_scene{s}.png", dpi=160)
            plt.close(fig)
            if save_gifs:
                frames = render_rollout_frames(
                    metrics["trajectories"][s],
                    lanes[lane_scene_idx == s],
                    max_frames=90,
                    **kwargs,
                )
                save_gif(frames, str(out_dir / f"{name}_scene{s}.gif"), fps=10)
        del policy
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    print(f"saved visuals to {out_dir}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_ddpo_dm_goal")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-scenes", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pool-size", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidates", type=int, nargs="+", default=[25, 50])
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--max-init-invalid-delta", type=float, default=0.03)
    ap.add_argument("--max-goal-offlane-delta", type=float, default=0.05)
    ap.add_argument("--max-controlled-parking-delta", type=float, default=0.05)
    ap.add_argument("--max-parking-mismatch-delta", type=float, default=0.05)
    ap.add_argument("--out-dir", default="outputs/dm_goal_sampler_eval")
    ap.add_argument("--num-visuals", type=int, default=4)
    ap.add_argument("--save-visuals", action="store_true")
    ap.add_argument("--save-visuals-on-failure", action="store_true", default=True)
    ap.add_argument("--save-gifs", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    print("composing config", flush=True)
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        cfg_root = compose(config_name=args.config_name, overrides=args.overrides)
    _set_dataset_name(cfg_root.dm_goal, cfg_root.dataset_name.name)
    OmegaConf.set_struct(cfg_root.ddpo, False)
    cfg_root.ddpo.ddim_eta = args.eta
    OmegaConf.set_struct(cfg_root.ddpo, True)

    pool_size = args.pool_size or args.num_scenes
    print("building conditioning pool", flush=True)
    pool = ConditioningPool(
        cfg_root.dm_goal.dataset,
        split_name=args.split,
        pool_size=max(pool_size, args.num_scenes, args.num_visuals),
        device=args.device,
        seed=args.seed,
        control_agent_num=cfg_root.ddpo.get("control_agent_num", -1),
    )
    indices = list(range(min(args.num_scenes, len(pool))))
    print("building reward backend", flush=True)
    reward = _build_reward(cfg_root.ddpo)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"evaluating {len(indices)} scenes on split={args.split}, eta={args.eta}", flush=True)
    summaries = {}
    base, _ = _evaluate_sampler(
        name="ddpm100",
        sampler="ddpm",
        ddim_steps=None,
        cfg_root=cfg_root,
        pool=pool,
        reward=reward,
        indices=indices,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    summaries["ddpm100"] = base

    selected = None
    failures_by_name = {}
    evaluated_samplers = [("ddpm100", "ddpm", None)]
    for steps in args.candidates:
        name = f"ddim{steps}"
        cand, _ = _evaluate_sampler(
            name=name,
            sampler="ddim",
            ddim_steps=steps,
            cfg_root=cfg_root,
            pool=pool,
            reward=reward,
            indices=indices,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        )
        summaries[name] = cand
        evaluated_samplers.append((name, "ddim", steps))
        ok, failures = _passes(base, cand, args)
        if ok:
            selected = steps
            print(f"SELECTED ddim_steps={steps} eta={args.eta}", flush=True)
            break
        failures_by_name[name] = failures
        print(f"REJECTED {name}: " + "; ".join(failures), flush=True)

    payload = {
        "selected_ddim_steps": selected,
        "eta": args.eta,
        "num_scenes": len(indices),
        "summaries": summaries,
        "failures": failures_by_name,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    if args.save_visuals or (
        selected is None and args.save_visuals_on_failure and args.num_visuals > 0
    ):
        _save_visuals(
            samplers=evaluated_samplers,
            cfg_root=cfg_root,
            pool=pool,
            reward=reward,
            out_dir=out_dir / "visuals",
            num_visuals=min(args.num_visuals, len(pool)),
            device=args.device,
            seed=args.seed,
            save_gifs=args.save_gifs,
        )

    if selected is None:
        print("No DDIM candidate passed the configured thresholds.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
