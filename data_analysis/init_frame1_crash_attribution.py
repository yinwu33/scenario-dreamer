#!/usr/bin/env python3
"""Frame-1 ego crash attribution in the real ldm_adv DDPO rollout.

Complements `init_ego_normal_overlap.py` (which measures the SPAWN-state overlap).
Here we run the actual `bad_driver` planner rollout for a few steps and record,
per scene, the first step at which the general ego-crash fires
(`SimScene.latch_ego_crash`, which freezes + stops the scene on ANY ego<->vehicle
overlap regardless of fault) and WHO the ego hit:

  * a NON-adversary normal vehicle  -> the user's reported case;
  * the generated adversary          -> the adversary's fault.

A crash latched on step 0 == "scene ends on frame 1". Because the ego actually
drives, this also catches a normal the ego drives INTO within the first step,
which the pure spawn-overlap test cannot see.

Run from the repo root:

    source scripts/define_env_variables.sh
    .venv/bin/python data_analysis/init_frame1_crash_attribution.py \
        --split train --scenes 1024 --max-steps 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _default_env() -> None:
    os.environ.setdefault("PROJECT_ROOT", str(REPO_ROOT))
    os.environ.setdefault("SCRATCH_ROOT", "data")
    os.environ.setdefault("DATASET_ROOT", os.environ["SCRATCH_ROOT"])
    os.environ.setdefault("CONFIG_PATH", str(REPO_ROOT / "cfgs"))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _load_cfg(config_name, overrides):
    _default_env()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_dir = os.environ.get("CONFIG_PATH", str(REPO_ROOT / "cfgs"))
    with initialize_config_dir(version_base=None, config_dir=str(Path(config_dir).resolve())):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def _first_crash_attribution(planner, scenes, max_steps: int) -> dict[str, np.ndarray]:
    """Run the planner rollout for `max_steps` and record, per scene, the first
    ego-crash step and whether the partner(s) were the adversary / a normal.

    Mirrors `NumpyPlanner.rollout` (advance -> update_metrics -> latch_ego_crash ->
    goal_step -> remove_out_of_bounds), but stops tracking a scene once its ego
    first crashes and classifies that step's contacted partners.
    """
    from ddpo.reward_hooks import adv_local_indices

    sims = planner._build_scenes(scenes)
    m = len(sims)
    adv_local = adv_local_indices(scenes, m)

    crash_step = np.full(m, -1, dtype=np.int64)      # first crash step (-1 = none)
    hit_adv = np.zeros(m, dtype=bool)
    hit_normal = np.zeros(m, dtype=bool)
    finished = np.zeros(m, dtype=bool)
    # Egos spawned at their goal / uncontrolled never crash-drive; skip them.
    for s, sim in enumerate(sims):
        if sim.n <= 1 or 0 not in sim.controlled:
            finished[s] = True

    for t in range(max_steps):
        active = [s for s in range(m) if not finished[s]]
        if not active:
            break
        planner._advance(sims, active)
        for s in active:
            sim = sims[s]
            sim.update_metrics()
            sim.latch_ego_crash()
            partners = sim.last_ego_collision_partners
            if partners.size:
                crash_step[s] = t
                a = int(adv_local[s])
                is_adv = a >= 0 and bool(np.any(partners == a))
                hit_adv[s] = is_adv
                hit_normal[s] = bool(np.any(partners != a)) if a >= 0 else bool(partners.size)
                finished[s] = True
                continue
            ego_reached, _ = sim.goal_step()
            sim.remove_out_of_bounds()
            if ego_reached:
                finished[s] = True
    return {"crash_step": crash_step, "hit_adv": hit_adv, "hit_normal": hit_normal}


def _summarize(att: dict[str, np.ndarray], max_steps: int) -> dict:
    cs = att["crash_step"]
    n = int(cs.size)
    frame1 = cs == 0                     # crashed on the very first advance
    crashed_any = cs >= 0
    f1 = int(frame1.sum())
    return {
        "n_scenes": n,
        "max_steps": max_steps,
        "P_crash_frame1": float(frame1.mean()),
        "P_crash_within_maxsteps": float(crashed_any.mean()),
        "n_crash_frame1": f1,
        # Of the frame-1 crashes, who did the ego hit?
        "frame1_hit_normal_rate": float(att["hit_normal"][frame1].mean()) if f1 else float("nan"),
        "frame1_hit_adv_rate": float(att["hit_adv"][frame1].mean()) if f1 else float("nan"),
        # Absolute over ALL scenes: a frame-1 crash caused by a normal / by the adv.
        "P_frame1_crash_by_normal": float((frame1 & att["hit_normal"]).mean()),
        "P_frame1_crash_by_adv": float((frame1 & att["hit_adv"]).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_critical_scene_ldm_adv_ddpo")
    ap.add_argument("--override", dest="overrides", action="append", default=[])
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--scenes", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--policy", choices=["base", "current"], default="base",
                    help="base = frozen reference (start of DDPO); current = trained net")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output-dir", default="data_analysis/init_ego_normal_overlap")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cfg_root = _load_cfg(args.config_name, args.overrides)
    cfg = cfg_root.ddpo
    OmegaConf.set_struct(cfg, False)
    cfg.device = device
    cfg.train_split = args.split
    OmegaConf.set_struct(cfg, True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from ddpo.train_loop import _build_policy_and_pool
    from data_analysis.analyze_ldm_adv_ddpo_support import _build_reward

    print(f"[build] policy + {args.split} pool + reward  device={device}")
    model_type, policy, pool, _ = _build_policy_and_pool(cfg_root, cfg, device)
    assert model_type == "ldm_adv"
    reward = _build_reward(cfg)
    planner = reward.planner

    n = min(args.scenes, len(pool))
    parts = []
    t0 = time.time()
    for start in range(0, n, args.chunk):
        chunk = list(range(start, min(start + args.chunk, n)))
        cond = pool.batch_from_indices(chunk)
        torch.manual_seed(args.seed + start)
        scenes, _ = policy.sample(cond, use_reference=(args.policy == "base"))
        parts.append(_first_crash_attribution(planner, scenes, args.max_steps))
        print(f"  {min(start + args.chunk, n)}/{n}  elapsed={time.time() - t0:.1f}s", flush=True)

    att = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    summary = _summarize(att, args.max_steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frame1_crash_attribution.json").write_text(
        json.dumps({"split": args.split, "policy": args.policy, "summary": summary}, indent=2),
        encoding="utf-8")

    print("\n============ FRAME-1 EGO CRASH ATTRIBUTION (real rollout) ============")
    r = summary
    print(f"  policy={args.policy}  n={r['n_scenes']}  max_steps={r['max_steps']}")
    print(f"  P(ego crashes on frame 1)         : {r['P_crash_frame1']*100:5.2f}%"
          f"   [{r['n_crash_frame1']}/{r['n_scenes']}]")
    print(f"  P(ego crashes within {r['max_steps']} steps)   : {r['P_crash_within_maxsteps']*100:5.2f}%")
    print(f"  of frame-1 crashes: hit a NORMAL   : {r['frame1_hit_normal_rate']*100:5.1f}%")
    print(f"  of frame-1 crashes: hit the ADV    : {r['frame1_hit_adv_rate']*100:5.1f}%")
    print(f"  => P(frame-1 crash BY NORMAL, all) : {r['P_frame1_crash_by_normal']*100:5.2f}%  <-- user's case")
    print(f"  => P(frame-1 crash BY ADV,    all) : {r['P_frame1_crash_by_adv']*100:5.2f}%")
    print(f"\n[done] wrote {out_dir/'frame1_crash_attribution.json'}")


if __name__ == "__main__":
    main()
