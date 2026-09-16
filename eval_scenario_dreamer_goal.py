#!/usr/bin/env python
"""Generate rollout-compatible ScenarioDreamer-Base scenes with goals.

The model is the standard ``ScenarioDreamerLDM``: it has no conditioning and no
dedicated adversary branch. Samples are drawn in ``initial_scene`` mode from the
goal-data joint lane/agent-count prior. For compatibility with the existing
generated-scene rollout loader, the nearest non-ego agent (vehicles first) is
designated as the adversary and moved to the final row of each scene.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir

from cfgs.config import CONFIG_PATH
from critical_scene.log_scenes import closest_agent_adv_idx
from model_registry import collapse_cfg
from models.scenario_dreamer_ldm import ScenarioDreamerLDM
from utils.train_helpers import set_latent_stats

ROOT = Path(__file__).resolve().parent
STATE_DIM = 9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_rollout_compatible(path: Path) -> int:
    with path.open("rb") as f:
        record = pickle.load(f)

    states = np.asarray(record["agent_states"])
    types = np.asarray(record["agent_types"])
    lanes = np.asarray(record["road_points"])
    connections = np.asarray(record["road_connection_types"])
    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError(f"{path}: expected agent_states [N, {STATE_DIM}], got {states.shape}")
    if len(states) != len(types) or len(states) != int(record["num_agents"]):
        raise ValueError(f"{path}: inconsistent agent counts")
    if len(lanes) != int(record["num_lanes"]):
        raise ValueError(f"{path}: inconsistent lane count")
    if len(connections) != len(lanes) * len(lanes):
        raise ValueError(f"{path}: lane connection rows do not match the dense lane graph")

    type_ids = types.argmax(axis=-1)
    scene_idx = np.zeros(len(states), dtype=np.int64)
    adv_idx = int(closest_agent_adv_idx(states, type_ids, scene_idx, 1)[0])
    if adv_idx > 0 and adv_idx != len(states) - 1:
        order = [i for i in range(len(states)) if i != adv_idx] + [adv_idx]
        record["agent_states"] = states[order]
        record["agent_types"] = types[order]

    record["ego_local_idx"] = 0
    record["adv_local_idx"] = len(states) - 1 if len(states) > 1 else -1
    with path.open("wb") as f:
        pickle.dump(record, f)
    return int(record["adv_local_idx"] >= 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_scenario_dreamer_goal_base_waymo")
    ap.add_argument("--ckpt", type=Path,
                    default=Path("data/final/scenario-dreamer/last.ckpt"))
    ap.add_argument("--ae-ckpt", type=Path,
                    default=Path("data/final/advscene_base_ae/last.ckpt"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/final/cache/scene/scenario-dreamer/init_scene"))
    ap.add_argument("--num-scenes", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    ae_ckpt = args.ae_ckpt.resolve()
    if not ckpt.is_file() or not ae_ckpt.is_file():
        raise FileNotFoundError(f"checkpoint missing: ckpt={ckpt}, ae_ckpt={ae_ckpt}")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty cache: {args.out}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    args.out.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        cfg_root = compose(config_name=args.config_name, overrides=list(args.overrides))
    spec, cfg, cfg_ae = collapse_cfg(cfg_root, "ldm")
    if spec.model_cls is not ScenarioDreamerLDM:
        raise TypeError(f"expected ScenarioDreamerLDM, got {spec.model_cls.__name__}")
    cfg.model.autoencoder_path = str(ae_ckpt)
    cfg = set_latent_stats(cfg)

    pl.seed_everything(args.seed, workers=True)
    model = ScenarioDreamerLDM.load_from_checkpoint(
        str(ckpt), cfg=cfg, cfg_ae=cfg_ae, map_location="cpu"
    ).to(args.device).eval()
    model.generate(
        mode="initial_scene",
        num_samples=args.num_scenes,
        batch_size=args.batch_size,
        cache_samples=True,
        visualize=False,
        conditioning_path=None,
        cache_dir=str(args.out),
        viz_dir=None,
        save_wandb=False,
        return_samples=False,
    )

    paths = sorted(args.out.glob("*.pkl"))
    if len(paths) != args.num_scenes:
        raise RuntimeError(f"generated {len(paths)} pickles, expected {args.num_scenes}")
    num_with_adv = sum(make_rollout_compatible(path) for path in paths)

    manifest = {
        "producer": Path(__file__).name,
        "model": "ScenarioDreamerLDM",
        "mode": "init_scene",
        "model_mode": "initial_scene",
        "conditioning": "none",
        "base_ckpt": str(ckpt),
        "ddpo_ckpt": None,
        "ae_ckpt": str(ae_ckpt),
        "ckpt_sha256": sha256(ckpt),
        "ae_ckpt_sha256": sha256(ae_ckpt),
        "sampler": "ddpm",
        "diffusion_steps": int(cfg.model.n_diffusion_timesteps),
        "use_ema_weights": True,
        "num_scenes": args.num_scenes,
        "num_scenes_with_designated_adv": num_with_adv,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "val_index": None,
        "init_prob_matrix": str(Path(cfg.eval.init_prob_matrix_path).resolve()),
        "state_dim": STATE_DIM,
        "ego_local_idx": 0,
        "adversary_designation": "nearest non-ego agent, vehicles first; reordered last",
        "scene_val_idx": {path.stem: None for path in paths},
        "git_commit": git_commit(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[scenario-dreamer] {len(paths)} scenes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
