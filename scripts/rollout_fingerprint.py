#!/usr/bin/env python
"""Bit-exactness harness for the rollout: score one FIXED batch of scenes and
dump every metric array, so two builds of the code can be diffed elementwise.

The scenes are sampled once and cached to disk, so the diffusion policy (whose
sampling no rollout change touches) never enters the comparison: the only thing
that varies between two runs of this script is how the scenes were rolled out
and scored.

Usage:
    # reference, on the current build
    python scripts/rollout_fingerprint.py --out /tmp/ref.npz

    # candidate, after a change (reuses the cached scenes)
    python scripts/rollout_fingerprint.py --out /tmp/new.npz --workers 16

    # compare
    python scripts/rollout_fingerprint.py --compare /tmp/ref.npz /tmp/new.npz
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from cfgs.config import CONFIG_PATH


def _parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config-name", default="config_ldm_adv_ddpo_idm_ppo")
    p.add_argument("--out", default=None, help="write the metric fingerprint here (.npz)")
    p.add_argument("--scenes", default=None, help="cached scene batch (created on first use)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pool-size", type=int, default=2000)
    p.add_argument("--workers", type=int, default=0,
                   help="0 = the single-process runner; >0 = sim.parallel with N workers")
    p.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"))
    p.add_argument("--time", type=int, default=0, metavar="N",
                   help="time reward.evaluate() over N repeats for each of --time-workers; "
                        "no instrumentation, so the numbers are apples-to-apples")
    p.add_argument("--time-workers", default="0,4,8,16",
                   help="comma-separated worker counts for --time")
    p.add_argument("--selfcheck", type=int, default=0, metavar="N",
                   help="run N DIFFERENT batches through both runners in one process and "
                        "assert they agree elementwise; this is the test that exercises "
                        "worker reuse, buffer reuse and cross-rollout state isolation")
    p.add_argument("--override", action="append", default=[])
    return p.parse_args()


def _differs(x: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """(rows that differ, max finite |delta|). NaN/inf count as equal to themselves:
    ego_min_ttc is legitimately inf when the ego never closes on anyone."""
    if x.dtype == object:
        # ragged per-scene payloads (e.g. recorded trajectories): compare the
        # serialised bytes, which is exactness in the strictest sense.
        n = sum(0 if pickle.dumps(u) == pickle.dumps(v) else 1 for u, v in zip(x, y))
        return n, 0.0
    if x.dtype.kind in "fc":
        same = np.isclose(x, y, rtol=0, atol=0, equal_nan=True)
        finite = np.isfinite(x) & np.isfinite(y)
        d = np.abs(np.where(finite, x - y, 0.0))
        return int((~same).sum()), (float(d.max()) if d.size else 0.0)
    same = x == y
    d = np.abs(x.astype(np.int64) - y.astype(np.int64)) if x.size else np.zeros(1)
    return int((~same).sum()), float(d.max())


def compare(path_a: str, path_b: str) -> int:
    a = np.load(path_a, allow_pickle=True)
    b = np.load(path_b, allow_pickle=True)
    keys = sorted(set(a.files) | set(b.files))
    total_differ, worst, structural = 0, 0.0, []
    print(f"{'metric':<34}{'max|delta|':>14}{'n_differ':>10}")
    print("-" * 58)
    for k in keys:
        if k not in a.files or k not in b.files:
            structural.append(f"{k}: present in only one fingerprint")
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            structural.append(f"{k}: shape {x.shape} vs {y.shape}")
            continue
        n, d = _differs(x, y)
        total_differ += n
        worst = max(worst, d)
        print(f"{k:<34}{d:>14.3e}{n:>10}{'' if n == 0 else '   <-- DIFFERS'}")
    print("-" * 58)
    for msg in structural:
        print("STRUCTURAL:", msg)
    ok = not structural and total_differ == 0
    print("RESULT:", "BIT-EXACT (every metric identical)" if ok
          else f"DIFFERS ({total_differ} elements, max|delta|={worst:.3e})")
    return 0 if ok else 1


def _build(cfg, workers: int):
    """RewardModel exactly as ddpo.train_loop builds it, plus the worker count."""
    from ddpo.reward import RewardModel, build_reward_config
    from ddpo.train_loop import _build_gen_invalid
    from sim.runner import SimulatorConfig

    cfg_ddpo = cfg.ddpo
    eval_dataset_cfg = cfg.ldm_adv.dataset
    kwargs = dict(
        planner_cfg=cfg_ddpo.planner,
        simulator_cfg=SimulatorConfig(
            seed=int(cfg_ddpo.seed),
            gen_invalid=_build_gen_invalid(cfg_ddpo, eval_dataset_cfg),
            **OmegaConf.to_container(cfg_ddpo.simulator, resolve=True),
        ),
        reward_cfg=build_reward_config(cfg_ddpo.reward),
    )
    if workers:
        # The pre-parallel build has no such parameter; this script must run on
        # both so it can produce the reference fingerprint.
        kwargs["num_workers"] = workers
        kwargs["train_batch_size"] = cfg.ddpo.batch_size
    return RewardModel(**kwargs)


def _sample_scenes(cfg, batch_size: int):
    from ddpo.train_loop import _build_policy_and_pool

    cfg_ddpo = cfg.ddpo
    _, policy, pool, _ = _build_policy_and_pool(cfg, cfg_ddpo, cfg_ddpo.device)
    torch.manual_seed(0)
    cond, _ = pool.sample_group_batch(batch_size // 8, 8)
    scenes, _ = policy.sample(cond)
    return scenes


def selfcheck(cfg, args) -> int:
    """Single-process vs sharded, N fresh batches, same process.

    Training calls evaluate() thousands of times against one worker pool, so the
    interesting failures are the ones that only appear on the SECOND rollout:
    recurrent carry surviving in a worker, a stale shared buffer, a barrier that
    drifted by a round. One batch cannot catch any of those.
    """
    from ddpo.train_loop import _build_policy_and_pool

    cfg_ddpo = cfg.ddpo
    _, policy, pool, _ = _build_policy_and_pool(cfg, cfg_ddpo, cfg_ddpo.device)
    serial = _build(cfg, 0)
    parallel = _build(cfg, args.workers)
    serial.set_train_iteration(0)
    parallel.set_train_iteration(0)

    failures = 0
    try:
        for it in range(args.selfcheck):
            torch.manual_seed(100 + it)
            cond, _ = pool.sample_group_batch(args.batch_size // 8, 8)
            scenes, _ = policy.sample(cond)
            a = serial.evaluate(scenes)
            b = parallel.evaluate(scenes)
            worst, differing = 0.0, []
            for key in a:
                x, y = np.asarray(a[key]), np.asarray(b[key])
                n, d = _differs(x, y)
                worst = max(worst, d)
                if n:
                    differing.append(f"{key}({n})")
            status = "BIT-EXACT" if not differing else "DIFFERS: " + ", ".join(differing)
            failures += bool(differing)
            print(f"[selfcheck] batch {it}: {scenes.num_scenes} scenes  "
                  f"reward={np.mean(a['reward']):+.8f}  max|delta|={worst:.3e}  {status}")
    finally:
        parallel.close()
    print("RESULT:", "BIT-EXACT over all batches" if not failures
          else f"{failures}/{args.selfcheck} batches DIFFER")
    return 1 if failures else 0


def timeit(cfg, scenes, args) -> int:
    """Wall-clock reward.evaluate() at several worker counts, same scenes.

    Deliberately runs with NO profiler instrumentation: the per-method wrappers
    the phase profiler installs only exist in the single-process path, so timing
    the two against each other under instrumentation flatters the sharded one.
    """
    import time

    counts = [int(x) for x in args.time_workers.split(",")]
    baseline = None
    print(f"{'workers':>8}{'best s':>10}{'median s':>11}{'speedup':>10}")
    print("-" * 39)
    for workers in counts:
        model = _build(cfg, workers)
        model.set_train_iteration(0)
        model.evaluate(scenes)          # warm the pool / lane-grid cache
        times = []
        for _ in range(args.time):
            t0 = time.perf_counter()
            model.evaluate(scenes)
            times.append(time.perf_counter() - t0)
        if workers:
            model.close()
        best, med = min(times), float(np.median(times))
        baseline = baseline if baseline is not None else best
        print(f"{workers:>8}{best:>10.3f}{med:>11.3f}{baseline / best:>9.2f}x")
    return 0


def main() -> int:
    args = _parse()
    if args.compare:
        return compare(*args.compare)

    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=list(args.override))
    with open_dict(cfg):
        cfg.ddpo.batch_size = args.batch_size
        cfg.ddpo.pool_size = args.pool_size

    torch.manual_seed(0)
    np.random.seed(0)

    if args.selfcheck:
        return selfcheck(cfg, args)

    cache = Path(args.scenes or f"/tmp/fingerprint_scenes_{args.config_name}_{args.batch_size}.pkl")
    if cache.exists():
        with open(cache, "rb") as fh:
            scenes = pickle.load(fh)
        print(f"[fingerprint] loaded cached scenes from {cache}")
    else:
        scenes = _sample_scenes(cfg, args.batch_size)
        with open(cache, "wb") as fh:
            pickle.dump(scenes, fh)
        print(f"[fingerprint] cached scenes to {cache}")

    if args.time:
        return timeit(cfg, scenes, args)

    reward = _build(cfg, args.workers)
    reward.set_train_iteration(0)
    out = reward.evaluate(scenes)
    try:
        arrays = {k: np.asarray(v) for k, v in out.items()
                  if isinstance(v, np.ndarray) or np.isscalar(v)}
        print(f"[fingerprint] {len(arrays)} metric arrays over {scenes.num_scenes} scenes "
              f"(workers={args.workers})")
        print(f"[fingerprint] reward mean={np.mean(arrays['reward']):.8f} "
              f"collision={np.mean(arrays['ego_collision']):.6f}")
        if args.out:
            np.savez(args.out, **arrays)
            print(f"[fingerprint] wrote {args.out}")
    finally:
        close = getattr(reward, "close", None)
        if close is not None:
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
