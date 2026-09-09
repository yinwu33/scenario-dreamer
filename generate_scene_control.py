#!/usr/bin/env python
"""Write one SceneControl scene cache: the guided-sampling baseline's generation half.

The counterpart of ``generate_scene.py`` for the ``dm_goal`` baseline, writing the
SAME record format so ``eval_scene.py`` and ``eval_rollout.py`` read it without
knowing which generator produced it.

What this baseline is
---------------------
``dm_goal`` is a plain unconditional data-space scene generator: no conditioning
labels, no adversary branch. All control happens at SAMPLING time, by pushing one
designated agent down the gradient of :class:`~guidance.costs.ProximityGoalCost`
(``guidance/guided_dm.py``). That is SceneControl's *mechanism* applied to our
objective, not SceneControl's own guidance function -- the paper's guidance is for
controllable generation (density, spacing), not adversarial criticality. Say so
when the row is described.

Only the fixed-map mode is offered, and the mode name is ``init_agent`` to match
the ``generate_scene.py`` vocabulary it has to sit beside: lanes come from a val
scene, agents are generated. Concretely that is ``p_sample_loop``'s
``lane_conditioned``, which pins ``x_lane`` to the scene's ground-truth lanes at
every denoising step.

There is deliberately NO free-map mode here. ``dm_goal`` does not model lane
connectivity -- ``nn_modules.dm.DM.decode_outputs`` can only report the graph of
the scene it was conditioned on, and for a generated map there is none. Such a
cache decodes to an EDGELESS lane graph, which ``sim.scenes.lane_graph_edges``
refuses outright, and which would otherwise cut ``build_route`` coverage from
95.4% to 65.7% of driving agents (measured, 200 val scenes) without saying so.

Why the lanes are the decode's own, unlike ``generate_scene.py``
---------------------------------------------------------------
``generate_scene.py:write_chunk`` has to substitute the source scene's polylines
and connection matrix for the autoencoder's, because that decoder returns
polylines in its latents' order while predicting connections in the dataset's
order (~29 m mean gap between a lane's end and its successor's start). ``dm_goal``
has no such split: in ``lane_conditioned`` the decoded lanes ARE
``data['lane'].x`` and the connection matrix is that same graph object, so the
pair is consistent by construction. Checked on 8 scenes: decoded ``succ`` edge
count equals the raw records' exactly, and the end-to-start gap measured on the
DECODED geometry is 0.000 m.

Usage (env vars from scripts/define_env_variables.sh must be set)::

    .venv/bin/python generate_scene_control.py --viz-first 16 \\
        --out data/final/scenecontrol/init_agent
    .venv/bin/python generate_scene_control.py --guidance off \\
        --out data/final/scenecontrol_unguided/init_agent

Naming: ``eval_rollout.pair_from_name`` reads a planner pair out of the cache's
directory name and only parses one starting with ``base``/``main_``/``kl_`` or
naming a pair outright. ``scenecontrol`` parses as none of those, so scoring this
cache through ``--caches-root`` needs BOTH ``--sut`` and ``--env`` given
explicitly (each is consulted separately), or the ``--caches <path>`` form, which
never calls that function. Being a baseline it wants ``--all-pairs`` either way.
"""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from torch_geometric.loader import DataLoader

from cfgs.config import CONFIG_PATH
from critical_scene.ldm_adv_eval import _git_commit, _seed_all
from critical_scene.log_scenes import list_scene_files
from datasets.waymo.dataset_dm_goal_waymo import WaymoDatasetDMGoal
from guidance.costs import first_index_per_scene
from guidance.guided_dm import GuidedDMGoal, default_adv_index
from model_registry import collapse_cfg
from models.scenario_dreamer_dm_goal import (
    ScenarioDreamerDMGoal,
    unnormalize_scene_with_goal,
)
from utils.data_helpers import convert_batch_to_scenarios
from utils.viz import plot_scene

# sim.scenes reads local index 0 as the ego, and WaymoDatasetDMGoal sorts the ego
# first, so the two agree without any remapping.
EGO_LOCAL_IDX = 0
# The mode name generate_scene.py uses for "lanes from a val scene, agents
# generated"; p_sample_loop's own name for it is lane_conditioned.
CACHE_MODE = "init_agent"
SAMPLER_MODE = "lane_conditioned"


def load_ema_state(cfg, ckpt_path: str) -> dict:
    """The diffusion module's EMA weights.

    The EMA object wraps ``model.diff_model``'s parameter tensors, so the averaged
    values have to be read out BEFORE the weights are handed to the guided module;
    swapping first would silently ship the raw (non-EMA) weights.
    """
    lit = ScenarioDreamerDMGoal.load_from_checkpoint(ckpt_path, cfg=cfg, map_location="cpu")
    with lit.ema.average_parameters():
        return {k: v.clone() for k, v in lit.diff_model.state_dict().items()}


def read_val_index(path: Path, num_scenes: int) -> list[int]:
    """The same index ``generate_scene.py`` reads, addressing the same scenes.

    ``val1000.json`` indexes the sorted val file list, and
    ``WaymoDatasetDMGoal`` globs that same directory with the same sort, so an
    entry means the same scene in both pipelines and the caches pair element-wise.
    """
    idx = json.loads(Path(path).read_text())["scene_idx"]
    if len(idx) < num_scenes:
        raise ValueError(f"{path} holds {len(idx)} scenes, need {num_scenes}")
    return idx[:num_scenes]


def val_data_list(dset: WaymoDatasetDMGoal, files: list[str], scene_idx: list[int]):
    """Conditioning graphs for the named val scenes, in index order."""
    out = []
    for i in scene_idx:
        with open(files[int(i)], "rb") as f:
            record = pickle.load(f)
        d = dset.get_data(record, int(i), files[int(i)])
        if int(d["num_agents"]) < 2:
            # Guidance needs the ego plus one agent to push. The index is built
            # from scenes that carry a non-ego agent, so this cannot fire without
            # the index and the dataset having drifted apart.
            raise RuntimeError(f"val scene {i} has {int(d['num_agents'])} agents, need >= 2")
        out.append(d)
    return out


def move_adv_last(record: dict, adv_local: int) -> None:
    """Reorder a record's agents so the guided one is last, in place.

    ``sim.scenes`` and ``eval_rollout.cache_payload`` both take the LAST agent of
    a scene to be the adversary -- that is the only handle a cache carries -- so
    the agent guidance actually steered has to be moved there. The ego stays at
    row 0 because the permutation keeps it first.
    """
    n = int(record["num_agents"])
    order = [0] + [i for i in range(1, n) if i != adv_local] + [adv_local]
    record["agent_states"] = record["agent_states"][order]
    record["agent_types"] = record["agent_types"][order]
    record["adv_local_idx"] = n - 1
    record["ego_local_idx"] = EGO_LOCAL_IDX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_dm_goal")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--out", required=True, help="cache directory to write")
    ap.add_argument("--ckpt", default=None,
                    help="default: <train.save_dir>/<train.run_name>/last.ckpt")
    ap.add_argument("--num-scenes", type=int, default=1000)
    # Matches generate_scene.py's default so the two caches get identical
    # scenario ids ("<i>_<batch_idx>") and therefore identical name order.
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guidance", choices=("on", "off"), default="on",
                    help="off writes the unguided control from the same noise")
    ap.add_argument("--val-index", default="metadata/val1000.json")
    ap.add_argument("--viz-first", type=int, default=0, metavar="N",
                    help="render the first N scenes into <out>/viz (0 = none). The"
                         " full 1000 is never useful and costs ~500 MB.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        cfg_root = compose(config_name=args.config_name, overrides=list(args.overrides))
    _, cfg, _ = collapse_cfg(cfg_root, "dm_goal")

    ckpt = args.ckpt or os.path.join(cfg.train.save_dir, cfg.train.run_name, "last.ckpt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"no checkpoint at {ckpt}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz" if args.viz_first > 0 else None
    if viz_dir is not None:
        viz_dir.mkdir(exist_ok=True)

    print(f"[generate] loading EMA weights from {ckpt}")
    state = load_ema_state(cfg, ckpt)
    net = GuidedDMGoal(cfg, guidance_cfg=cfg.guidance).to(args.device)
    net.load_state_dict(state)
    net.eval()

    files = list_scene_files(cfg.dataset.preprocess_dir, "val")
    scene_idx = read_val_index(Path(args.val_index), args.num_scenes)
    dset = WaymoDatasetDMGoal(cfg.dataset, split_name="val", mode="eval")
    data_list = val_data_list(dset, files, scene_idx)
    batches = list(DataLoader(data_list, batch_size=args.batch_size, shuffle=False, drop_last=False))
    print(f"[generate] {CACHE_MODE} / guidance={args.guidance}: "
          f"{args.num_scenes} scenes in {len(batches)} batches")

    num_types = int(cfg.dataset.num_agent_types)
    manifest: dict[str, int] = {}
    for bi, batch in enumerate(batches):
        data = batch.to(args.device)
        ego_idx = first_index_per_scene(data["agent"].batch, data.batch_size)
        adv_idx = default_adv_index(data["agent"].batch, data.batch_size)
        if args.guidance == "on":
            net.enable_guidance(adv_idx, ego_idx)
        else:
            net.disable_guidance()

        _seed_all(args.seed * 1_000_003 + 1000 + bi, args.device)
        with torch.no_grad():
            agent_s, lane_s, agent_t, _, lane_conn = net.forward(data, mode=SAMPLER_MODE)
        agent_s, lane_s = unnormalize_scene_with_goal(agent_s.clone(), lane_s.clone(), cfg.dataset)

        # convert_batch_to_scenarios reads the record straight off these fields.
        data["agent"].x = agent_s
        data["agent"].type = F.one_hot(agent_t, num_classes=num_types)
        data["lane"].x = lane_s
        data["lane", "to", "lane"].type = lane_conn

        records = convert_batch_to_scenarios(
            data, batch_size=args.batch_size, batch_idx=bi, cache_dir=None,
            cache_samples=False, cache_lane_types=False, mode="initial_scene",
        )
        lo = bi * args.batch_size
        ids = scene_idx[lo:lo + int(batch.batch_size)]
        for offset, (scenario_id, record) in enumerate(records.items()):
            # Guidance steers local index 1 (default_adv_index); the cache
            # contract puts the adversary last.
            move_adv_last(record, 1)
            record["val_scene_idx"] = int(ids[offset])
            with open(out_dir / f"{scenario_id}.pkl", "wb") as f:
                pickle.dump(record, f)
            if viz_dir is not None and len(manifest) < args.viz_first:
                states = record["agent_states"]
                types = np.argmax(record["agent_types"], axis=1)
                plot_scene(
                    states[:-1], record["road_points"], types[:-1], None,
                    name=f"{scenario_id}.png", save_dir=str(viz_dir),
                    adv_states=states[-1:], adv_types=types[-1:],
                )
            manifest[scenario_id] = int(ids[offset])
        print(f"[generate]   batch {bi + 1}/{len(batches)}", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps({
        "mode": CACHE_MODE,
        "sampler_mode": SAMPLER_MODE,
        "conditioning": f"guidance:{args.guidance}",
        "generator": "dm_goal (SceneControl-style guided sampling)",
        "base_ckpt": ckpt,
        "ddpo_ckpt": None,
        "guidance": {
            "cost_type": str(cfg.guidance.cost_type),
            "t_start": int(cfg.guidance.t_start),
            "num_grad_steps": int(cfg.guidance.num_grad_steps),
            "step_size": float(cfg.guidance.step_size),
            "cost": {k: v for k, v in cfg.guidance.cost.items()},
            "enabled": args.guidance == "on",
        },
        "n_diffusion_timesteps": int(cfg.model.n_diffusion_timesteps),
        "num_scenes": args.num_scenes,
        "seed": args.seed,
        "val_index": str(args.val_index),
        "ego_local_idx": EGO_LOCAL_IDX,
        "scene_val_idx": manifest,
        "git_commit": _git_commit(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }, indent=1, default=str))
    print(f"[generate] {len(manifest)} scenes -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
