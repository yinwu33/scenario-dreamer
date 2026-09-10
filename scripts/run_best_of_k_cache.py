#!/usr/bin/env python
"""Best-of-$K$ sampling from the FROZEN base generator, written as scene caches.

The same baseline ``scripts/run_best_of_k.py`` builds, moved onto the cache
pipeline: ``generate_scene.py`` writes one cache per (checkpoint, init mode,
conditioning protocol), and ``eval_rollout.py`` scores it. A best-of-$K$ row is
just another cache, so it lands next to ``base`` / ``base_null`` / ``main_*`` and
inherits their columns -- ``ttc_lt_3s`` included -- with no scoring code of its
own.

The two scripts are NOT interchangeable. ``run_best_of_k.py`` draws its scenes
from val-scene conditioning graphs and reports rates on the driving-ego subset;
this one draws from the layout prior (``--mode init_scene``) and every scene is
in the denominator, which is what the paper's main table does.

Stage 1 is drawn ONCE per batch and stage 2 $K$ times, so the $K$ candidates
share a base scene and differ only in the adversary -- the same axis DDPO
fine-tunes. Draw 0 reuses ``generate_scene.py``'s adversary seed, so the
candidate set literally contains that script's ``base``/``base_null`` sample and
``<name>_bok1`` reproduces it.

Selection is by the DDPO reward, the objective a practitioner would optimize.
The draws are written to disk first and selected on what ``eval_rollout.py``
would load, so the rollout that chooses a draw and the rollout that scores it
see the same bytes.

    .venv/bin/python scripts/run_best_of_k_cache.py \
        --conditioning cond_adv_ego --name base_cond -k 16 \
        --out-root data/final/cache/scene_bok --workers 16 \
        --overrides ddpo/reward=hierarchical_v4 \
                    planner@ddpo.planner.sut=ppo_normal \
                    planner@ddpo.planner.env=ppo_normal \
                    planner@ddpo.planner.adv=ppo_normal
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from critical_scene.ldm_adv_eval import (
    _git_commit,
    _seed_all,
    benchmark_payload,
    build_policy,
    build_reward,
    compose_eval_cfg,
    make_generated_cond,
    prepare_ldm_cfg,
    set_generation_conditioning,
    write_json,
)
from eval_rollout import cache_payload
from generate_scene import (
    CONDITIONING,
    CONDITIONING_PROTOCOL,
    EGO_LOCAL_IDX,
    base_scene_latents,
    prior_data_list,
    write_chunk,
)
from models.scenario_dreamer_ldm_adv import ScenarioDreamerLDMAdv
from run_best_of_k import _curve, _draw_seed, _ladder

MODE = "init_scene"


def _parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config-name", default="config_ldm_adv_ddpo")
    p.add_argument("--overrides", nargs="*", default=[])
    p.add_argument("--conditioning", choices=CONDITIONING, required=True)
    p.add_argument("--name", required=True,
                   help="cache name prefix; budgets land in <out-root>/<name>_bok<k>/init_scene")
    p.add_argument("--out-root", required=True)
    p.add_argument("-k", "--num-draws", type=int, default=16)
    p.add_argument("--num-scenes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--benchmark-batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def draw_dir(out_root: Path, name: str, i: int) -> Path:
    """Candidate draws live beside the budget caches but out of their glob.

    ``eval_rollout.py --caches-root`` matches ``*/init_scene``, and these are
    ``<name>_draws/draw_XX``, so a scoring batch never picks up a candidate."""
    return out_root / f"{name}_draws" / f"draw_{i:02d}"


def base_manifest(args, cfg_root, ldm_cfg, base_ckpt: str) -> dict:
    """``generate_scene.py``'s manifest fields, minus the per-scene index.

    Same keys in the same order, so a bok cache and the ``base`` cache it
    branches from diff cleanly."""
    return {
        "mode": MODE,
        "conditioning": args.conditioning,
        "base_ckpt": base_ckpt,
        "ddpo_ckpt": None,
        "ae_ckpt": str(cfg_root.ddpo.ae_ckpt),
        "sampler": str(cfg_root.ddpo.sampler),
        "ddim_steps": int(cfg_root.ddpo.ddim_steps),
        "use_ema_weights": bool(cfg_root.ddpo.use_ema_weights),
        "num_scenes": args.num_scenes,
        "seed": args.seed,
        "val_index": None,
        "init_prob_matrix": str(ldm_cfg.eval.init_prob_matrix_path),
        "adv_cond_target": {k: v for k, v in cfg_root.ddpo.adv_cond_target.items()},
        "ego_local_idx": EGO_LOCAL_IDX,
    }


def generate_draws(args, cfg_root, ldm_cfg, base_ckpt: str) -> None:
    """Write the $K$ candidate caches, sharing one stage-1 scene per batch."""
    out_root = Path(args.out_root)
    k = int(args.num_draws)
    dirs = [draw_dir(out_root, args.name, i) for i in range(k)]
    if all((d / "manifest.json").exists() for d in dirs):
        print(f"[bok] draws: skip, {k} candidate caches exist")
        return

    lit = ScenarioDreamerLDMAdv.load_from_checkpoint(
        base_ckpt, cfg=ldm_cfg, cfg_ae=cfg_root.ae_goal
    ).to(args.device).eval()
    policy = build_policy(cfg_root, ldm_cfg, ckpt=base_ckpt, device=args.device)

    # The layout prior is drawn under this seed alone, so every cache built with
    # the same --seed covers the same (num_lanes, num_agents) scenes.
    torch.manual_seed(args.seed)
    data_list = prior_data_list(
        lit, cfg_root, args.num_scenes, args.batch_size, args.conditioning
    )
    batches = list(DataLoader(data_list, batch_size=args.batch_size, shuffle=False,
                              drop_last=False))
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[bok] {MODE} / {args.conditioning}: {args.num_scenes} scenes x {k} draws "
          f"in {len(batches)} batches")

    manifests: list[dict] = [{} for _ in range(k)]
    for bi, batch in enumerate(batches):
        data = set_generation_conditioning(
            batch.to(args.device), CONDITIONING_PROTOCOL[args.conditioning]
        )
        _seed_all(args.seed * 1_000_003 + 1000 + bi, args.device)
        with lit.ema.average_parameters():
            x_agent, x_lane = base_scene_latents(lit, data, MODE, args.device)
        gen = make_generated_cond(policy, data, x_agent, x_lane)
        for i in range(k):
            _seed_all(_draw_seed(int(args.seed), bi, i, k), args.device)
            # write_chunk writes the decode back onto the batch it is handed, so
            # each draw gets its own copy of the shared stage-1 conditioning.
            manifests[i].update(write_chunk(
                lit, policy, copy.deepcopy(gen), bi, args.batch_size,
                dirs[i], None, None, None,
            ))
        print(f"[bok]   batch {bi + 1}/{len(batches)}: {k} draws", flush=True)

    manifest = base_manifest(args, cfg_root, ldm_cfg, base_ckpt)
    for i, d in enumerate(dirs):
        (d / "manifest.json").write_text(json.dumps(
            {**manifest, "draw": i, "num_draws": k,
             "scene_val_idx": manifests[i],
             "git_commit": _git_commit(),
             "created": datetime.now().isoformat(timespec="seconds")},
            indent=1, default=str))
    del policy, lit
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()


def score_draws(args, cfg_root, ldm_cfg) -> Path:
    """Roll every candidate out and persist the scene x draw reward matrix."""
    out_root = Path(args.out_root)
    k = int(args.num_draws)
    path = out_root / f"{args.name}_select.npz"
    if path.exists():
        print(f"[bok] select: skip, {path} exists")
        return path

    reward = build_reward(cfg_root, ldm_cfg, num_workers=int(args.workers),
                          batch_size=int(args.benchmark_batch_size))
    stems, cols = None, {"reward": [], "ego_collision": []}
    for i in range(k):
        payload, s = cache_payload(draw_dir(out_root, args.name, i))
        if stems is None:
            stems = s
        elif s != stems:
            # A budget cache is assembled by copying one draw's pkl per scene, so
            # a draw that dropped a scene would leave a hole in it.
            raise SystemExit(f"draw {i} covers different scenes than draw 0")
        metrics = benchmark_payload(reward, payload,
                                    batch_size=int(args.benchmark_batch_size),
                                    label=f"{args.name} draw {i + 1}/{k}")
        for key in cols:
            cols[key].append(metrics[key])
    if args.workers:
        reward.close()   # rollout workers outlive the process otherwise

    np.savez_compressed(
        path, scenario=np.array(stems),
        **{key: np.stack(v, axis=1) for key, v in cols.items()},
    )
    print(f"[bok] wrote {path}")
    return path


def assemble(args, cfg_root, ldm_cfg, base_ckpt: str, select: Path) -> None:
    """Copy the reward-selected draw of each scene into one cache per budget."""
    out_root = Path(args.out_root)
    k = int(args.num_draws)
    blob = np.load(select, allow_pickle=True)
    stems = [str(s) for s in blob["scenario"]]
    R = blob["reward"]

    manifest = base_manifest(args, cfg_root, ldm_cfg, base_ckpt)
    for budget in _ladder(k):
        # Budget k is one honest run of "sample k, keep the best": the first k
        # draws, not the best k of K.
        best = R[:, :budget].argmax(axis=1)
        cache = out_root / f"{args.name}_bok{budget}" / MODE
        cache.mkdir(parents=True, exist_ok=True)
        for stem, b in zip(stems, best):
            shutil.copyfile(draw_dir(out_root, args.name, int(b)) / f"{stem}.pkl",
                            cache / f"{stem}.pkl")
        (cache / "manifest.json").write_text(json.dumps(
            {**manifest,
             "num_draws": budget,
             "selection": "max DDPO reward within the first k draws",
             "selection_overrides": list(args.overrides),
             "scene_val_idx": {s: None for s in stems},
             "selected_draw": {s: int(b) for s, b in zip(stems, best)},
             "git_commit": _git_commit(),
             "created": datetime.now().isoformat(timespec="seconds")},
            indent=1, default=str))
        print(f"[bok] wrote {cache} ({len(stems)} scenes)")

    keep = np.ones(R.shape[0], dtype=bool)
    write_json(out_root / f"{args.name}_bok{k}_curve.json", {
        "num_draws": k,
        "num_scenes": int(R.shape[0]),
        "selection": "max DDPO reward within the k-subset",
        "curve": _curve(R, blob["ego_collision"], keep),
    })


def main() -> int:
    args = _parse()
    cfg_root = compose_eval_cfg(args.config_name, args.overrides)
    ldm_cfg = prepare_ldm_cfg(cfg_root)
    base_ckpt = str(cfg_root.ddpo.ldm_adv_ckpt)
    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    generate_draws(args, cfg_root, ldm_cfg, base_ckpt)
    select = score_draws(args, cfg_root, ldm_cfg)
    assemble(args, cfg_root, ldm_cfg, base_ckpt, select)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
