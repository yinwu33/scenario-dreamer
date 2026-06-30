#!/usr/bin/env python3
"""How often does the DDPO ego overlap a NON-adversary (normal) agent at spawn?

The user observed that some ldm_adv DDPO rollouts terminate on frame 1 because
the ego collides immediately -- and that the culprit is often a *real normal*
agent that the AE decodes overlapping the ego, not the generated adversary. The
general collision response (`SimScene.latch_ego_crash`) freezes + stops the scene
on ANY ego<->vehicle overlap regardless of fault, so a normal-on-ego init kills
the scene at frame 1. None of the recorded reward metrics capture this: both
`init_invalid` and `ego_collision` are adversary-only.

This script measures it directly. It reproduces the DDPO conditioning + decode
(default `prune_base_to_ego=false`, so the full real normal scene is kept), then
at the spawn state (t=0) replays the EXACT collision test the sim uses
(`ddpo.geometry._corners` / `_sat_overlap`, the 15 m broad-phase gate, pedestrian
exclusion) for:

  * ego vs each NON-adversary vehicle  -> ego_overlaps_normal   (the question)
  * ego vs the generated adversary     -> ego_overlaps_adv      (adversary's fault)

Because the normals are fixed conditioning (their latents are held constant and
the adversary is excluded from the normal test), the normal-overlap rate does not
need the 1000-step diffusion sampler: it is read from `conditioning_scenes` (base
+ real adv decode) over a large scene count. A smaller `--sample` validation pass
runs the real DDPO sampler to confirm the generated adversary does not change the
normal-overlap rate.

Run from the repo root:

    source scripts/define_env_variables.sh
    .venv/bin/python data_analysis/init_ego_normal_overlap.py \
        --split train --cond-scenes 3000 --sample-scenes 256
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


def _load_cfg(config_name: str, overrides: list[str]):
    _default_env()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_dir = os.environ.get("CONFIG_PATH", str(REPO_ROOT / "cfgs"))
    with initialize_config_dir(version_base=None, config_dir=str(Path(config_dir).resolve())):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _spawn_breakdown(scenes) -> dict[str, np.ndarray]:
    """Per-scene spawn (t=0) overlap flags, mirroring SimScene.latch_ego_crash.

    Returns per-scene boolean / count arrays. The ego is local agent 0; the
    adversary is `scenes.adv_local_idx`; every other non-pedestrian vehicle is a
    "normal". A box is a collision candidate only within the 15 m gate, then the
    oriented-box SAT test decides overlap -- exactly the sim's two-phase check.
    """
    from ddpo.geometry import _corners, _sat_overlap
    from ddpo.pufferdrive_sim import COLLISION_DIST2_GATE, TYPE_PEDESTRIAN
    from ddpo.planners.type_utils import to_puffer_agent_types
    from ddpo.reward_hooks import adv_local_indices

    n_scenes = int(scenes.num_scenes)
    states = _to_numpy(scenes.agent_states).astype(np.float64)
    ptype = to_puffer_agent_types(_to_numpy(scenes.agent_types))
    scene_idx = _to_numpy(scenes.agent_scene_idx).astype(np.int64)
    adv_local = adv_local_indices(scenes, n_scenes)

    # Box params, mirroring SimScene.__init__ exactly.
    x, y = states[:, 0], states[:, 1]
    heading = np.arctan2(states[:, 4], states[:, 3])
    length = np.maximum(states[:, 5], 0.5)
    width = np.maximum(states[:, 6], 0.5)

    out = {
        k: np.zeros(n_scenes, dtype=np.int64 if "n_" in k else bool)
        for k in ("ego_olap_normal", "ego_olap_adv", "ego_olap_any",
                  "n_normals", "n_normals_olap", "has_adv")
    }

    def box(g):
        return _corners(x[g], y[g], heading[g], length[g], width[g])

    for s in range(n_scenes):
        gidx = np.nonzero(scene_idx == s)[0]          # increasing -> local 0 == ego
        if gidx.size == 0:
            continue
        ego = gidx[0]
        a_local = int(adv_local[s])
        adv_g = gidx[a_local] if 0 <= a_local < gidx.size else -1
        out["has_adv"][s] = adv_g >= 0

        # Normals: every non-ego, non-adv, non-pedestrian agent.
        normals = [g for li, g in enumerate(gidx)
                   if li != 0 and g != adv_g and ptype[g] != TYPE_PEDESTRIAN]
        out["n_normals"][s] = len(normals)

        ego_box = box(np.array([ego]))[0]

        if normals:
            normals = np.asarray(normals)
            d2 = (x[normals] - x[ego]) ** 2 + (y[normals] - y[ego]) ** 2
            gated = normals[d2 <= COLLISION_DIST2_GATE]
            if gated.size:
                ov = _sat_overlap(ego_box, box(gated))
                out["n_normals_olap"][s] = int(ov.sum())
                out["ego_olap_normal"][s] = bool(ov.any())

        if adv_g >= 0 and ptype[adv_g] != TYPE_PEDESTRIAN:
            d2 = (x[adv_g] - x[ego]) ** 2 + (y[adv_g] - y[ego]) ** 2
            if d2 <= COLLISION_DIST2_GATE:
                out["ego_olap_adv"][s] = bool(_sat_overlap(ego_box, box(np.array([adv_g]))).any())

        out["ego_olap_any"][s] = bool(out["ego_olap_normal"][s] or out["ego_olap_adv"][s])

    return out


def _summarize(b: dict[str, np.ndarray]) -> dict:
    n = int(b["ego_olap_normal"].size)
    olap_normal = b["ego_olap_normal"]
    olap_adv = b["ego_olap_adv"]
    olap_any = b["ego_olap_any"]
    any_n = int(olap_any.sum())
    return {
        "n_scenes": n,
        "frac_full_scene_has_normals": float((b["n_normals"] > 0).mean()),
        "mean_normals_per_scene": float(b["n_normals"].mean()),
        # THE answer: ego overlaps >=1 normal (non-adv) vehicle at spawn.
        "P_ego_overlaps_normal_at_spawn": float(olap_normal.mean()),
        "P_ego_overlaps_adv_at_spawn": float(olap_adv.mean()),
        "P_ego_overlaps_any_at_spawn": float(olap_any.mean()),
        "n_ego_overlaps_normal": int(olap_normal.sum()),
        "n_ego_overlaps_adv": int(olap_adv.sum()),
        "n_ego_overlaps_any": any_n,
        # Of scenes whose ego is overlapped at spawn, who is to blame?
        "among_overlap_normal_only": float(((olap_normal & ~olap_adv).sum() / any_n) if any_n else float("nan")),
        "among_overlap_adv_only": float(((~olap_normal & olap_adv).sum() / any_n) if any_n else float("nan")),
        "among_overlap_both": float(((olap_normal & olap_adv).sum() / any_n) if any_n else float("nan")),
        "mean_normals_overlapping_when_any": float(
            b["n_normals_olap"][olap_normal].mean()) if olap_normal.any() else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_critical_scene_ldm_adv_ddpo")
    ap.add_argument("--override", dest="overrides", action="append", default=[])
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--cond-scenes", type=int, default=3000,
                    help="scenes decoded via conditioning_scenes (no sampler); headline rate")
    ap.add_argument("--sample-scenes", type=int, default=256,
                    help="scenes run through the real DDPO sampler for validation (0 to skip)")
    ap.add_argument("--chunk", type=int, default=256)
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

    print(f"[build] policy + {args.split} pool (prune_base_to_ego="
          f"{cfg.get('prune_base_to_ego', False)})  device={device}")
    model_type, policy, pool, _ = _build_policy_and_pool(cfg_root, cfg, device)
    assert model_type == "ldm_adv", f"expected ldm_adv, got {model_type}"

    def gather(indices, sampler: str):
        parts = []
        t0 = time.time()
        for start in range(0, len(indices), args.chunk):
            chunk = indices[start:start + args.chunk]
            cond = pool.batch_from_indices(chunk)
            if sampler == "cond":
                scenes = policy.conditioning_scenes(cond)
            else:
                torch.manual_seed(args.seed + start)
                scenes, _ = policy.sample(cond, use_reference=True)
            parts.append(_spawn_breakdown(scenes))
            print(f"  [{sampler}] {min(start + args.chunk, len(indices))}/{len(indices)}"
                  f"  elapsed={time.time() - t0:.1f}s", flush=True)
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    pool_n = len(pool)
    results = {}

    n_cond = min(args.cond_scenes, pool_n)
    print(f"[cond] decoding {n_cond} base scenes (real adv, no sampler)")
    results["cond_real_adv"] = _summarize(gather(list(range(n_cond)), "cond"))

    if args.sample_scenes > 0:
        n_s = min(args.sample_scenes, pool_n)
        print(f"[sample] running the real DDPO sampler on {n_s} scenes")
        results["ddpo_sampled"] = _summarize(gather(list(range(n_s)), "sample"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config_name": args.config_name,
        "split": args.split,
        "prune_base_to_ego": bool(cfg.get("prune_base_to_ego", False)),
        "pool_size": pool_n,
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n================ SPAWN EGO<->NORMAL OVERLAP ================")
    for name, r in results.items():
        print(f"\n--- {name}  (n={r['n_scenes']}) ---")
        print(f"  scenes with any normal agent : {r['frac_full_scene_has_normals']*100:5.1f}%"
              f"   (mean {r['mean_normals_per_scene']:.1f} normals/scene)")
        print(f"  P(ego overlaps a NORMAL @t=0): {r['P_ego_overlaps_normal_at_spawn']*100:5.2f}%"
              f"   <-- the answer  [{r['n_ego_overlaps_normal']}/{r['n_scenes']}]")
        print(f"  P(ego overlaps the ADV  @t=0): {r['P_ego_overlaps_adv_at_spawn']*100:5.2f}%"
              f"               [{r['n_ego_overlaps_adv']}/{r['n_scenes']}]")
        print(f"  P(ego overlaps ANYTHING @t=0): {r['P_ego_overlaps_any_at_spawn']*100:5.2f}%"
              f"               [{r['n_ego_overlaps_any']}/{r['n_scenes']}]")
        print(f"  of overlapping scenes: normal-only {r['among_overlap_normal_only']*100:4.1f}% | "
              f"adv-only {r['among_overlap_adv_only']*100:4.1f}% | both {r['among_overlap_both']*100:4.1f}%")
    print(f"\n[done] wrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
