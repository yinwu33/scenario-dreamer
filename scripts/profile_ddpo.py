#!/usr/bin/env python
"""Phase profiler for the ldm_adv DDPO training loop.

Answers "where does an iteration's wall clock actually go?" by replaying the
body of ``ddpo.train_loop.run_ddpo`` (conditioning draw -> policy.sample ->
reward.evaluate -> DDPO update) with the SAME construction path, and timing
each phase plus a sub-breakdown obtained by monkeypatching -- no production
code is modified.

Levels reported:

  cond      pool.sample_group_batch / sample_batch (dataset -> conditioning graph)
  sample    policy.sample: the denoising chain (net forward per step) + AE decode
  reward    RewardModel.evaluate: scene build, per-role planner plan/apply,
            hooks, SimScene bookkeeping
  update    trajectory_logprob (policy fwd + optional ref fwd for KL), backward,
            optimizer step

GPU sections are timed with an explicit ``torch.cuda.synchronize()`` on both
ends, so async kernel launches are attributed to the phase that issued them.

Usage:
    python scripts/profile_ddpo.py --config-name config_ldm_adv_ddpo_idm_idm
    python scripts/profile_ddpo.py --config-name config_ldm_adv_ddpo_ppo_ppo \
        --iters 3 --cprofile
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cfgs.config import CONFIG_PATH


# --------------------------------------------------------------------- timers
class Prof:
    """Accumulating wall-clock spans, keyed by the CALL PATH they occur on.

    Keys nest automatically: a span opened inside another is recorded under
    ``parent.child``. That matters here because the same method serves several
    phases -- ``_adv_mean_logvar`` is the denoiser forward for both the sampling
    chain and the log-prob recomputation -- and a flat key would merge the two.
    """

    def __init__(self, cuda: bool):
        self.cuda = cuda
        self.t: dict[str, float] = defaultdict(float)
        self.n: dict[str, int] = defaultdict(int)
        self.stack: list[str] = []
        self.enabled = True

    def reset(self) -> None:
        self.t = defaultdict(float)
        self.n = defaultdict(int)

    @contextmanager
    def span(self, key: str, sync: bool = False):
        if not self.enabled:
            yield
            return
        path = ".".join([*self.stack, key])
        self.stack.append(key)
        if sync and self.cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync and self.cuda:
                torch.cuda.synchronize()
            self.t[path] += time.perf_counter() - t0
            self.n[path] += 1
            self.stack.pop()


def wrap_obj(prof: Prof, obj, name: str, key: str, sync: bool = False):
    """Time an instance method in place (leaves the class untouched)."""
    orig = getattr(obj, name)

    def timed(*a, **kw):
        with prof.span(key, sync):
            return orig(*a, **kw)

    setattr(obj, name, timed)
    return orig


def wrap_cls(prof: Prof, cls, name: str, key: str, sync: bool = False):
    """Time a method for every instance of ``cls``."""
    orig = getattr(cls, name)

    def timed(self, *a, **kw):
        with prof.span(key, sync):
            return orig(self, *a, **kw)

    setattr(cls, name, timed)
    return orig


# ----------------------------------------------------------------- config
def _compose_cfg(args):
    overrides = [
        # Never touch a real run's output dir / wandb from the profiler.
        "experiment.output_dir=${scratch_root}/critical_scene/profile_ddpo",
        "ddpo.wandb.enabled=false",
        "ddpo.resume=false",
        *( [f"ddpo.batch_size={args.batch_size}"] if args.batch_size else [] ),
        *( [f"ddpo.pool_size={args.pool_size}"] if args.pool_size else [] ),
        *args.override,
    ]
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        return compose(config_name=args.config_name, overrides=overrides)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config-name", default="config_ldm_adv_ddpo_idm_idm",
                   help="hydra entry config (e.g. config_ldm_adv_ddpo_ppo_ppo)")
    p.add_argument("--iters", type=int, default=3, help="timed iterations")
    p.add_argument("--warmup", type=int, default=1, help="untimed warmup iterations")
    p.add_argument("--batch-size", type=int, default=None, help="override ddpo.batch_size")
    p.add_argument("--pool-size", type=int, default=2000,
                   help="override ddpo.pool_size (smaller = faster startup)")
    p.add_argument("--empty-cache", action="store_true",
                   help="torch.cuda.empty_cache() between iterations (needed when the GPU is "
                        "shared with another job; adds a small fixed cost per iteration)")
    p.add_argument("--min-pct", type=float, default=0.3,
                   help="hide tree rows below this %% of the iteration (JSON keeps everything)")
    p.add_argument("--cprofile", action="store_true",
                   help="also cProfile one reward.evaluate call (CPU hot functions)")
    p.add_argument("--out", default=None, help="write the phase table as JSON here")
    p.add_argument("--workers", type=int, default=None,
                   help="override ddpo.rollout_workers (0 = single-process rollout)")
    p.add_argument("--override", action="append", default=[],
                   help="extra hydra overrides (repeatable)")
    return p.parse_args()


# ------------------------------------------------------------------- report
def _report(prof: Prof, iters: int, top_keys: list[str], title: str, min_pct: float) -> list[dict]:
    """Print the span tree: total time, share of the iteration, and SELF time
    (total minus the children instrumented below it)."""
    total = sum(prof.t[k] for k in top_keys)
    children: dict[str, list[str]] = defaultdict(list)
    for key in prof.t:
        if "." in key:
            children[key.rsplit(".", 1)[0]].append(key)

    rows: list[dict] = []
    print(f"\n=== {title} ===")
    print(f"{'phase':<38}{'s/iter':>9}{'% iter':>8}{'self s':>9}{'calls/iter':>12}")
    print("-" * 76)

    def walk(key: str, depth: int) -> None:
        t = prof.t[key]
        share = 100.0 * t / total if total else float("nan")
        self_t = t - sum(prof.t[c] for c in children.get(key, []))
        label = ("  " * depth) + key.rsplit(".", 1)[-1]
        rows.append({
            "phase": key, "sec_per_iter": t / iters, "pct_total": share,
            "self_sec_per_iter": self_t / iters, "calls_per_iter": prof.n[key] / iters,
        })
        if share >= min_pct or depth == 0:
            print(
                f"{label:<38}{t / iters:>9.4f}{share:>7.1f}%{self_t / iters:>9.4f}"
                f"{prof.n[key] / iters:>12.1f}"
            )
        for c in sorted(children.get(key, []), key=lambda k: -prof.t[k]):
            walk(c, depth + 1)

    for key in sorted(top_keys, key=lambda k: -prof.t[k]):
        walk(key, 0)
    print("-" * 76)
    print(f"{'TOTAL (top-level phases)':<38}{total / iters:>9.4f}{100.0:>7.1f}%")
    print(f"(rows below {min_pct}% of the iteration are omitted from the print, kept in --out)")
    return rows


def main():
    args = _parse_args()
    cfg_root = _compose_cfg(args)
    cfg = cfg_root.ddpo
    device = cfg.device
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    prof = Prof(cuda)

    from ddpo.train_loop import _build_policy_and_pool, _build_gen_invalid, _bf16_autocast
    from ddpo.ddpo_loss import AdaptiveKLController, DDPOConfig, compute_advantages, ddpo_loss
    from ddpo.reward import RewardModel, build_reward_config
    from sim.runner import SimulatorConfig
    from sim.world import SimScene

    t0 = time.perf_counter()
    model_type, policy, pool, eval_dataset_cfg = _build_policy_and_pool(cfg_root, cfg, device)
    reward = RewardModel(
        planner_cfg=cfg.planner,
        simulator_cfg=SimulatorConfig(
            seed=int(cfg.seed),
            gen_invalid=_build_gen_invalid(cfg, eval_dataset_cfg),
            **OmegaConf.to_container(cfg.simulator, resolve=True),
        ),
        reward_cfg=build_reward_config(cfg.reward),
        num_workers=int(cfg.rollout_workers if args.workers is None else args.workers),
    )
    ddpo_cfg = DDPOConfig(**OmegaConf.to_container(cfg.algo, resolve=True))
    kl_ctrl = AdaptiveKLController(ddpo_cfg)
    trainable_params = list(policy.trainable_parameters())
    opt = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    setup_s = time.perf_counter() - t0

    stochastic_steps = policy.stochastic_step_indices(int(cfg.min_diffusion_t))
    group_size = int(ddpo_cfg.group_size)
    with_kl = kl_ctrl.coef > 0 or ddpo_cfg.kl_target > 0

    planner_names = {r: reward.runner.planners[r].name for r in ("sut", "env", "adv")}
    print(
        f"[profile] config={args.config_name} sampler={cfg.sampler} "
        f"steps={policy.num_sampling_steps} batch={cfg.batch_size} group_size={group_size} "
        f"k_steps={cfg.k_steps} inner_epochs={cfg.inner_epochs} with_kl={with_kl} "
        f"rollout_workers={reward.num_workers}\n"
        f"[profile] planners={planner_names} sim_steps={cfg.simulator.sim_steps} "
        f"stochastic_steps={len(stochastic_steps)} setup={setup_s:.1f}s"
    )

    # ------------------------------------------------------------ instrument
    # Keys nest under whichever top-level phase is on the stack, so the SAME
    # denoiser forward is attributed to sample.* and update.* separately.
    wrap_obj(prof, policy, "_adv_mean_logvar", "denoise_net", sync=True)
    wrap_obj(prof, policy, "_decode", "ae_decode", sync=True)
    # reward/sim internals. With rollout_workers > 0 all of this runs INSIDE the
    # workers, where the parent's timers cannot see it -- and the wrappers are
    # local closures, so wrapping the hooks would make them unpicklable and the
    # rollout could not be shipped at all. The sharded run therefore reports the
    # top-level phases only; use --workers 0 for the per-hook breakdown.
    if reward.num_workers == 0:
        wrap_obj(prof, reward.runner, "_build_scenes", "build_scenes", sync=True)
        wrap_obj(prof, reward.runner, "_assign_roles", "assign_roles")
        wrap_obj(prof, reward.runner, "_apply_conditioning", "apply_conditioning")
        for role, planner in reward.runner.planners.items():
            wrap_obj(prof, planner, "plan", f"plan_{role}")
            wrap_obj(prof, planner, "apply", f"apply_{role}")
        for meth in ("update_metrics", "latch_ego_crash", "goal_step", "remove_out_of_bounds",
                     "compute_obs", "step_dynamics"):
            if hasattr(SimScene, meth):
                wrap_cls(prof, SimScene, meth, f"sim_{meth}")
        # hooks are rebuilt per evaluate(): wrap them as they are handed out.
        orig_make_hooks = reward._make_hooks

        def make_hooks(*args, **kwargs):
            hooks = orig_make_hooks(*args, **kwargs)
            for h in hooks:
                for meth in ("before_rollout", "before_step_scene", "after_step_scene",
                             "after_rollout"):
                    wrap_obj(prof, h, meth, f"hook_{type(h).__name__}")
            return hooks

        reward._make_hooks = make_hooks
    else:
        # The parent's own share of a sharded rollout: the centrally batched
        # forwards, which is the part that does NOT parallelise.
        for role in reward.runner._remote_roles:
            wrap_obj(prof, reward.runner.planners[role], "forward",
                     f"central_forward_{role}", sync=True)
    # update: the policy forward (+ ref forward when KL is on) inside logprob.
    wrap_obj(prof, policy, "trajectory_logprob", "logprob_fwd", sync=True)

    # ---------------------------------------------------------------- loop
    def one_iteration(it: int):
        with prof.span("cond", sync=True):
            if group_size > 1:
                cond, group_ids = pool.sample_group_batch(cfg.batch_size // group_size, group_size)
                group_ids = group_ids.to(device)
            else:
                cond = pool.sample_batch(cfg.batch_size)
                group_ids = None
        with prof.span("sample", sync=True):
            scenes, traj = policy.sample(cond)
        reward.set_train_iteration(it)
        with prof.span("reward", sync=True):
            metrics = reward.evaluate(scenes)

        rewards = torch.as_tensor(metrics["reward"], device=device)
        advantages = compute_advantages(rewards, ddpo_cfg, group_ids)

        for _ in range(cfg.inner_epochs):
            with prof.span("update", sync=True):
                k_idx = stochastic_steps[torch.randperm(len(stochastic_steps))[: cfg.k_steps]]
                with _bf16_autocast(device, enabled=cfg.logprob_bf16):
                    new_lp, kl_term = policy.trajectory_logprob(traj, cond, k_idx, with_kl=with_kl)
                    old_lp = traj.old_logprob[:, k_idx]
                    kl_term = kl_term.float() if kl_term is not None else None
                    loss, log, parts = ddpo_loss(
                        new_lp.float(), old_lp.float(), advantages.float(), ddpo_cfg,
                        kl_term, kl_coef=kl_ctrl.coef,
                    )
                loss = loss.float()
                opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    continue
                with prof.span("backward", sync=True):
                    if ddpo_cfg.decouple_kl_grad and parts["kl"] is not None:
                        parts["pg"].backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params, cfg.grad_clip, error_if_nonfinite=False)
                        pg_grads = [None if p.grad is None else p.grad.detach().clone()
                                    for p in trainable_params]
                        opt.zero_grad(set_to_none=True)
                        (kl_ctrl.coef * parts["kl"]).backward()
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params, cfg.grad_clip, error_if_nonfinite=False)
                        for p, g in zip(trainable_params, pg_grads):
                            if g is None:
                                continue
                            p.grad = g if p.grad is None else p.grad.add_(g)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params, cfg.grad_clip, error_if_nonfinite=False)
                with prof.span("opt_step", sync=True):
                    opt.step()
        del traj, cond
        opt.zero_grad(set_to_none=True)
        if cuda and args.empty_cache:
            torch.cuda.empty_cache()
        return scenes, metrics

    for it in range(args.warmup):
        prof.enabled = False
        one_iteration(it)
        prof.enabled = True
        print(f"[profile] warmup {it + 1}/{args.warmup} done", flush=True)
    prof.reset()

    wall0 = time.perf_counter()
    for it in range(args.iters):
        t_it = time.perf_counter()
        scenes, metrics = one_iteration(args.warmup + it)
        print(f"[profile] iter {it + 1}/{args.iters}: {time.perf_counter() - t_it:.2f}s", flush=True)
    wall = time.perf_counter() - wall0

    n_agents = int(scenes.agent_states.shape[0])
    print(
        f"\n[profile] measured wall {wall / args.iters:.3f} s/iter over {args.iters} iters "
        f"({n_agents} agents / {scenes.num_scenes} scenes = "
        f"{n_agents / max(scenes.num_scenes, 1):.1f} agents per scene)"
    )
    rows = _report(
        prof, args.iters, ["cond", "sample", "reward", "update"],
        f"{args.config_name} ({cfg.sampler}-{policy.num_sampling_steps}, "
        f"sut={planner_names['sut']} env={planner_names['env']} adv={planner_names['adv']})",
        args.min_pct,
    )

    # ------------------------------------------------- optional CPU deep dive
    if args.cprofile:
        prof.enabled = False
        cond = pool.sample_group_batch(cfg.batch_size // group_size, group_size)[0] \
            if group_size > 1 else pool.sample_batch(cfg.batch_size)
        scenes, _ = policy.sample(cond)
        pr = cProfile.Profile()
        pr.enable()
        reward.evaluate(scenes)
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
        print("\n=== cProfile: one reward.evaluate call (cumulative) ===")
        print(s.getvalue())
        s2 = io.StringIO()
        pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(25)
        print("=== cProfile: one reward.evaluate call (tottime) ===")
        print(s2.getvalue())

    if args.out:
        out = {
            "config_name": args.config_name,
            "sampler": str(cfg.sampler),
            "sampling_steps": policy.num_sampling_steps,
            "batch_size": int(cfg.batch_size),
            "group_size": group_size,
            "k_steps": int(cfg.k_steps),
            "sim_steps": int(cfg.simulator.sim_steps),
            "planners": planner_names,
            "iters": args.iters,
            "wall_sec_per_iter": wall / args.iters,
            "phases": rows,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[profile] wrote {args.out}")


if __name__ == "__main__":
    main()
