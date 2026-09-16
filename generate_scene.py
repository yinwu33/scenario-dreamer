#!/usr/bin/env python
"""Write one scene cache: the generation half of the evaluation pipeline.

One cache is (checkpoint, init mode, conditioning protocol). ``eval_scene.py``
scores its distributions and ``eval_rollout.py`` rolls it out, so both read the
same objects and a table row is one cache.

The three init modes differ only in how much of the scene the model draws:

  init_scene   lane + agent + adv, from the (num_lanes, num_agents) layout prior.
               No real scene backs it, so rows built this way compare to the log
               at DISTRIBUTION level only.
  init_agent   lanes come from a val scene, agents and adv are generated.
  init_adv     lanes and agents come from a val scene, only the adv is generated.

``init_agent`` and ``init_adv`` read the val scenes named by ``--val-index``, so
every cache built from that index covers the SAME scenes and pairs element-wise
with the log. Build the index once with ``--write-index``; it is the first
``--num-scenes`` val scenes that yield a conditioning graph.

The adversary is always sampled by ``LDMAdvDDPOPolicy.sample`` (the ddim/30
sampler DDPO trains under), so a base cache and a DDPO cache differ only in the
adv-branch weights. Stage 1 runs under the base checkpoint's EMA weights in
every mode.

Conditioning protocol (``--conditioning``):
  null    every normal-agent label and all four adv labels are null tokens.
              The unconditional protocol, and the only one comparable with a
              model that has no adversary branch.
  cond_adv  the adversary is pinned to ``ddpo.adv_cond_target`` and no ego or
              normal-agent label is imposed. This is the protocol DDPO trains
              under, so a cache built this way asks what the fine-tuned policy
              does at its own operating point.
  cond_adv_ego   cond_adv plus the ego pinned to vehicle / moving / far. Fixing
              the ego's goal bucket removes the parked and short-goal egos that
              otherwise reach their goal for free, so every scene in the cache
              is one where the ego has somewhere to drive.

The two ``post_*`` names are the protocols ``set_generation_conditioning``
implements as ``dataset`` and ``ego_far``; the vocabulary here is the pipeline's,
the behaviour is that function's.

Note: ``init_agent`` is an untrained configuration. Every saved config in this
repo has ``mode_probs.init_agent = 0.0``, so the model has seen clean lanes only
next to clean agents (init_adv) and noisy agents only next to noisy lanes
(init_scene), never the combination. The interface is kept; the samples are
out-of-distribution until a base model is trained with that mode enabled.

Usage (env vars from scripts/define_env_variables.sh must be set)::

    .venv/bin/python generate_scene.py --write-index --num-scenes 1000
    .venv/bin/python generate_scene.py --mode init_scene --conditioning null \
        --out data/scenes/base_init_scene
    .venv/bin/python generate_scene.py --mode init_adv --conditioning cond_adv_ego \
        --ckpt data/final/advscene_rl_main/ppo-ppo_norm/last.ckpt \
        --out data/scenes/ddpo_ppo-ppo_norm_init_adv
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from critical_scene.ldm_adv_eval import (
    _git_commit,
    _seed_all,
    build_policy,
    build_pool,
    compose_eval_cfg,
    make_generated_cond,
    prepare_ldm_cfg,
    set_generation_conditioning,
)
from critical_scene.log_scenes import list_scene_files
from ddpo.conditioning import _ADV_COND_FIELDS, LDMAdvConditioningPool
from models.scenario_dreamer_ldm_adv import ScenarioDreamerLDMAdv
from utils.data_helpers import convert_batch_to_scenarios
from utils.viz import plot_scene

MODES = ("init_scene", "init_agent", "init_adv")
CONDITIONING = ("null", "cond_adv", "cond_adv_ego")
# The pipeline's vocabulary -> the protocol set_generation_conditioning implements.
CONDITIONING_PROTOCOL = {"null": "all_null", "cond_adv": "dataset", "cond_adv_ego": "ego_far"}
# set_generation_conditioning pins the ego to this triple; the prior path carries
# no agent labels at all, so cond_adv_ego has to materialise the tensor it edits.
NUM_AGENT_COND_FIELDS = 3
# sim.scenes and set_generation_conditioning both read local index 0 as the ego.
EGO_LOCAL_IDX = 0


# ------------------------------------------------------------------ val index
def write_val_index(pool: LDMAdvConditioningPool, num_scenes: int, path: Path) -> None:
    """The first ``num_scenes`` val scenes that build a conditioning graph.

    ``build_scene`` returns None for a scene with no non-ego agent, which cannot
    condition the model at all; nothing else is filtered, so the index is 'the
    first N val scenes' in dataset order and does not depend on any planner,
    checkpoint or ego-goal threshold."""
    kept: list[int] = []
    for scene_idx in range(len(pool.dataset)):
        if len(kept) == num_scenes:
            break
        if pool.build_scene(scene_idx, require_driving_ego=False) is not None:
            kept.append(scene_idx)
    if len(kept) < num_scenes:
        raise RuntimeError(f"only {len(kept)} usable val scenes, need {num_scenes}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"split": "val", "num_scenes": len(kept), "scene_idx": kept}, indent=1))
    print(f"[index] {len(kept)} val scenes -> {path}")


def read_val_index(path: Path, num_scenes: int) -> list[int]:
    idx = json.loads(path.read_text())["scene_idx"]
    if len(idx) < num_scenes:
        raise ValueError(f"{path} holds {len(idx)} scenes, need {num_scenes}")
    return idx[:num_scenes]


# ------------------------------------------------------------- conditioning
def prior_data_list(lit, cfg_root, num_scenes: int, batch_size: int, conditioning: str):
    """Layout-prior scenes with the labels the conditioning protocol needs.

    ``_initialize_pyg_dset`` samples (num_lanes, num_agents) from the goal layout
    prior and leaves every label absent. The adv labels are then set to the same
    per-scene draw the DDPO pool makes, and ``ego_far`` additionally needs an
    agent-label tensor to pin the ego in -- the prior path has none, so it is
    created here with the non-ego rows left at zero (set_generation_conditioning
    drops them). The other two protocols impose no normal-agent label, so they
    leave the prior's absent labels alone and the model uses its null tokens."""
    data_list, _ = lit._initialize_pyg_dset("init_scene", num_scenes, batch_size, None, False)

    targets = LDMAdvConditioningPool._parse_adv_cond_target(cfg_root.ddpo.adv_cond_target)
    if targets is None:
        raise ValueError("adv_cond_target is disabled; the DDPO runs were trained with it on.")
    seed = int(cfg_root.ddpo.seed)
    for i, d in enumerate(data_list):
        rng = np.random.default_rng((seed, i))
        d["adv"].cond = torch.tensor(
            [[int(rng.choice(targets[f])) for f in _ADV_COND_FIELDS]], dtype=torch.long
        )
        if conditioning == "cond_adv_ego":
            d["agent"].cond = torch.zeros(
                (d["num_agents"], NUM_AGENT_COND_FIELDS), dtype=torch.long
            )
    return data_list


def val_data_list(pool: LDMAdvConditioningPool, scene_idx: list[int]):
    """Conditioning graphs for named val scenes, in index order.

    ``build_scene`` is the same entry the DDPO pool's own sampling uses, so the
    model is conditioned exactly the way training conditions it -- but addressed
    by dataset index rather than by a random draw, which is what makes two caches
    cover the same scenes."""
    out = []
    for i in scene_idx:
        d = pool.build_scene(int(i), require_driving_ego=False)
        if d is None:
            raise RuntimeError(f"val scene {i} from the index no longer builds")
        out.append(d)
    return out


# -------------------------------------------------------------------- stages
@torch.no_grad()
def base_scene_latents(lit, data, mode: str, device: str):
    """Stage 1: the part of the scene the mode asks the base model to draw.

    ``init_adv`` draws nothing -- the val scene's own latents are returned, which
    is what makes it the "log + generated adversary" row."""
    if mode == "init_adv":
        return data["agent"].latents, data["lane"].latents

    net = lit.diff_model
    agent_dim = int(lit.cfg_model.agent_latent_dim)
    lane_dim = int(lit.cfg_model.lane_latent_dim)
    x_agent, x_lane, _ = net.p_sample_loop(
        (data["agent"].x.shape[0], 1, agent_dim),
        (data["lane"].x.shape[0], 1, lane_dim),
        (data["adv"].x.shape[0], 1, agent_dim),
        data,
        device=device,
        mode=mode,
    )
    return x_agent, x_lane


@torch.no_grad()
def write_chunk(lit, policy, data, batch_idx: int, batch_size: int, out_dir: Path,
                scene_ids: list[int] | None, val_files: list[str] | None,
                viz_dir: Path | None) -> dict[str, int | None]:
    """Sample the adversary, decode, and write one pkl per scene.

    The adversary is sampled by the policy but decoded by ``lit``: the policy's
    own decode drops the lane-connection matrix, which ``eval_scene.py`` needs
    for the lane-graph metrics.

    ``convert_batch_to_scenarios`` builds the record (and appends the adv as each
    scene's last agent); the adv and ego indices are then written into it so no
    consumer has to re-derive them from that ordering.

    For the val-conditioned modes the lane geometry is taken from the source
    scene rather than from the decode. The decoder returns polylines in its
    latents' order while the connection matrix it predicts comes out in the
    dataset's own order (``reorder_indices`` permutes one and not the other), and
    pairing the two puts a mean 29 m gap between a lane's end and its successor's
    start -- which corrupts the lane-graph metrics AND every rule-based planner's
    route. The source scene's two fields are consistent by construction, and the
    mode's definition is that its lanes ARE that scene's, so they are what the
    record should carry. Generated lanes have no such source and keep the decode,
    where both come out of one forward pass and agree."""
    _, traj = policy.sample(data)
    x_adv = traj.records["steps"][-1][1][:, 0]

    agent_s, lane_s, agent_t, _, lane_conn, adv_s, adv_t = lit._decode_scene_and_adv(
        data["agent"].latents, data["lane"].latents, x_adv, data
    )
    num_types = lit.cfg_dataset.num_agent_types
    data["agent"].x = agent_s
    data["lane"].x = lane_s
    data["agent"].type = torch.nn.functional.one_hot(agent_t, num_classes=num_types)
    data["lane", "to", "lane"].type = lane_conn
    data["adv"].x = adv_s
    data["adv"].type = torch.nn.functional.one_hot(adv_t, num_classes=num_types)

    records = convert_batch_to_scenarios(
        data, batch_size=batch_size, batch_idx=batch_idx,
        cache_dir=None, cache_samples=False, cache_lane_types=False, mode="init_scene",
    )

    manifest: dict[str, int | None] = {}
    for offset, (scenario_id, record) in enumerate(records.items()):
        record["adv_local_idx"] = int(record["num_agents"]) - 1
        record["ego_local_idx"] = EGO_LOCAL_IDX
        if scene_ids is not None:
            with open(val_files[int(scene_ids[offset])], "rb") as f:
                source = pickle.load(f)
            record["road_points"] = np.asarray(source["road_points"], dtype=np.float32)
            record["road_connection_types"] = np.asarray(source["road_connection_types"])
            record["num_lanes"] = len(record["road_points"])
        with open(out_dir / f"{scenario_id}.pkl", "wb") as f:
            pickle.dump(record, f)
        if viz_dir is not None:
            # The record's own arrays, so the png shows what the pkl holds -- with
            # the adversary split back off the tail so plot_scene draws it green.
            states = record["agent_states"]
            # plot_scene indexes agent_types as class ids; the record stores one-hot.
            types = np.argmax(record["agent_types"], axis=1)
            plot_scene(
                states[:-1], record["road_points"], types[:-1], None,
                name=f"{scenario_id}.png", save_dir=str(viz_dir),
                adv_states=states[-1:], adv_types=types[-1:],
            )
        manifest[scenario_id] = None if scene_ids is None else int(scene_ids[offset])
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_ldm_adv_ddpo")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--conditioning", choices=CONDITIONING)
    ap.add_argument("--ckpt", help="DDPO checkpoint; omit to sample the adversary from the base model")
    ap.add_argument("--out", help="cache directory to write")
    ap.add_argument("--num-scenes", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viz", action="store_true",
                    help="also render one png per scene into <out>/viz")
    ap.add_argument("--val-index", default="metadata/val1000.json")
    ap.add_argument("--write-index", action="store_true",
                    help="write --val-index from the first --num-scenes val scenes and exit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg_root = compose_eval_cfg(args.config_name, args.overrides)
    ldm_cfg = prepare_ldm_cfg(cfg_root)
    base_ckpt = str(cfg_root.ddpo.ldm_adv_ckpt)
    val_index_path = Path(args.val_index)

    if args.write_index:
        pool = build_pool(cfg_root, ldm_cfg, split="val", pool_size=args.num_scenes, device=args.device)
        write_val_index(pool, args.num_scenes, val_index_path)
        return 0

    for required in ("mode", "conditioning", "out"):
        if getattr(args, required) is None:
            ap.error(f"--{required.replace('_', '-')} is required unless --write-index")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz" if args.viz else None
    if viz_dir is not None:
        viz_dir.mkdir(exist_ok=True)

    lit = ScenarioDreamerLDMAdv.load_from_checkpoint(
        base_ckpt, cfg=ldm_cfg, cfg_ae=cfg_root.ae_goal
    ).to(args.device).eval()
    policy = build_policy(cfg_root, ldm_cfg, ckpt=args.ckpt or base_ckpt, device=args.device)

    torch.manual_seed(args.seed)
    if args.mode == "init_scene":
        data_list = prior_data_list(lit, cfg_root, args.num_scenes, args.batch_size, args.conditioning)
        scene_idx, val_files = None, None
    else:
        pool = build_pool(cfg_root, ldm_cfg, split="val", pool_size=args.num_scenes, device=args.device)
        scene_idx = read_val_index(val_index_path, args.num_scenes)
        data_list = val_data_list(pool, scene_idx)
        # The preprocessed scene, not the latent one build_scene reads: only the
        # former carries unnormalised polylines and a road_connection_types
        # matrix. The two directories are the same sorted file list, so the index
        # addresses the same scene in both.
        val_files = list_scene_files(ROOT / "data" / "advscene_preprocess_waymo", "val")
    batches = list(DataLoader(data_list, batch_size=args.batch_size, shuffle=False, drop_last=False))
    print(f"[generate] {args.mode} / {args.conditioning}: {args.num_scenes} scenes in {len(batches)} batches")

    manifest: dict[str, int | None] = {}
    for bi, batch in enumerate(batches):
        data = set_generation_conditioning(
            batch.to(args.device), CONDITIONING_PROTOCOL[args.conditioning]
        )
        _seed_all(args.seed * 1_000_003 + 1000 + bi, args.device)
        with lit.ema.average_parameters():
            x_agent, x_lane = base_scene_latents(lit, data, args.mode, args.device)
        if args.mode == "init_adv":
            gen = data
        else:
            gen = make_generated_cond(policy, data, x_agent, x_lane)

        _seed_all(args.seed * 1_000_003 + 2000 + bi, args.device)
        lo = bi * args.batch_size
        ids = None if scene_idx is None else scene_idx[lo:lo + int(batch.batch_size)]
        manifest.update(write_chunk(
            lit, policy, gen, bi, args.batch_size, out_dir, ids, val_files, viz_dir
        ))
        print(f"[generate]   batch {bi + 1}/{len(batches)}")

    (out_dir / "manifest.json").write_text(json.dumps({
        "mode": args.mode,
        "conditioning": args.conditioning,
        "base_ckpt": base_ckpt,
        "ddpo_ckpt": args.ckpt,
        "ae_ckpt": str(cfg_root.ddpo.ae_ckpt),
        "sampler": str(cfg_root.ddpo.sampler),
        "ddim_steps": int(cfg_root.ddpo.ddim_steps),
        "use_ema_weights": bool(cfg_root.ddpo.use_ema_weights),
        "num_scenes": args.num_scenes,
        "seed": args.seed,
        "val_index": None if scene_idx is None else str(val_index_path),
        "init_prob_matrix": str(ldm_cfg.eval.init_prob_matrix_path) if args.mode == "init_scene" else None,
        "adv_cond_target": {k: v for k, v in cfg_root.ddpo.adv_cond_target.items()},
        "ego_local_idx": EGO_LOCAL_IDX,
        "scene_val_idx": manifest,
        "git_commit": _git_commit(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }, indent=1, default=str))
    print(f"[generate] {len(manifest)} scenes -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
