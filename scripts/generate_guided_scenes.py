"""Guided (SceneControl-style) generation from a trained dm_goal checkpoint.

Runs the SAME scenes twice from the SAME initial noise -- once unguided, once with
:class:`~guidance.guided_dm.GuidedDMGoal` -- so the two sets differ only by the
guidance hook. That paired design is the point: it isolates what guidance did from
what the generator would have produced anyway.

Usage::

    python scripts/generate_guided_scenes.py --config-name config_dm_goal \\
        dm_goal.eval.num_samples=32 dm_goal.eval.batch_size=16

Overrides work as usual, e.g. ``dm_goal.guidance.step_size=0.05``.
"""

import os
import pickle
import sys
from pathlib import Path

import hydra
import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfgs.config import CONFIG_PATH
from guidance.costs import first_index_per_scene
from guidance.guided_dm import COST_TYPES, GuidedDMGoal, default_adv_index
from model_registry import collapse_cfg
from models.scenario_dreamer_dm_goal import ScenarioDreamerDMGoal, unnormalize_scene_with_goal
from utils.viz import visualize_batch


def _load_ema_weights(cfg, ckpt_path):
    """Return the diffusion module's EMA state dict.

    The EMA object is built over ``model.diff_model``'s parameter tensors, so the
    swap has to happen *after* the averaged weights are read out -- otherwise the
    guided module would silently get the raw (non-EMA) weights.
    """
    model = ScenarioDreamerDMGoal.load_from_checkpoint(ckpt_path, cfg=cfg, map_location="cpu")
    with model.ema.average_parameters():
        return {k: v.clone() for k, v in model.diff_model.state_dict().items()}


def _sample(diff_model, data, seed, adv_idx=None, ego_idx=None):
    """One full reverse chain. ``adv_idx=None`` disables guidance (the control run)."""
    if adv_idx is None:
        diff_model.disable_guidance()
    else:
        diff_model.enable_guidance(adv_idx, ego_idx)
    torch.manual_seed(seed)  # identical initial noise for the paired runs
    return diff_model.forward(data, mode="initial_scene")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config_dm_goal")
def main(cfg):
    spec, cfg, _ = collapse_cfg(cfg, "dm_goal")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = os.path.join(cfg.eval.save_dir, cfg.eval.run_name)
    ckpt_path = os.path.join(save_dir, "last.ckpt")
    assert os.path.exists(ckpt_path), f"No checkpoint at {ckpt_path}"
    print(f"Loading {ckpt_path}")

    state_dict = _load_ema_weights(cfg, ckpt_path)
    diff_model = GuidedDMGoal(cfg, guidance_cfg=cfg.guidance).to(device)
    diff_model.load_state_dict(state_dict)
    diff_model.eval()

    # Reuse the Lightning module only to build the prior-sampled layouts.
    shell = ScenarioDreamerDMGoal(cfg)
    dset, _ = shell._initialize_pyg_dset("initial_scene", cfg.eval.num_samples)
    del shell

    # Score both runs with the ACTIVE objective so the table matches what was optimised.
    measure = COST_TYPES[cfg.guidance.get("cost_type", "proximity")](
        cfg.guidance.cost, cfg.dataset
    )
    out_dir = Path(cfg.eval.viz_dir + "_guided")
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"unguided": [], "guided": []}
    scenes = {}

    loader = DataLoader(dset, batch_size=cfg.eval.batch_size, shuffle=False, drop_last=False)
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        ego_idx = first_index_per_scene(data["agent"].batch, data.batch_size)
        adv_idx = default_adv_index(data["agent"].batch, data.batch_size)
        seed = 1000 + batch_idx

        for tag, idx in (("unguided", None), ("guided", adv_idx)):
            batch = data.clone()
            agents, lanes, agent_types, lane_types, lane_conn = _sample(
                diff_model, batch, seed, idx, ego_idx
            )

            with torch.no_grad():
                _, terms = measure(
                    agents[:, :9], lanes,
                    batch["agent"].batch, batch["lane"].batch,
                    adv_idx, batch.batch_size,
                )
            stats[tag].append(terms)

            agents_m, lanes_m = unnormalize_scene_with_goal(
                agents.clone(), lanes.clone(), cfg.dataset
            )
            # draw the guided agent in green on top of the scene
            adv_states = agents_m[adv_idx]
            visualize_batch(
                min(cfg.eval.num_samples, batch.batch_size),
                agents_m, lanes_m, agent_types, lane_types, lane_conn, batch,
                str(out_dir), epoch=0, batch_idx=batch_idx, save_wandb=False,
                tag=f"scene_{tag}",
                adv_samples=adv_states,
                adv_batch=torch.arange(batch.batch_size, device=device),
                adv_types=agent_types[adv_idx],
            )
            scenes[f"{tag}_{batch_idx}"] = {
                "agent_states": agents.detach().cpu().numpy(),
                "lane_states": lanes.detach().cpu().numpy(),
                "agent_batch": batch["agent"].batch.cpu().numpy(),
                "adv_idx": adv_idx.cpu().numpy(),
            }

    print("\n" + "=" * 78)
    keys = sorted({k for s in stats["unguided"] for k in s})
    print(f"{'term':<16}{'unguided':>14}{'guided':>14}{'delta':>14}")
    print("-" * 78)
    for k in keys:
        u = sum(s[k] for s in stats["unguided"]) / len(stats["unguided"])
        g = sum(s[k] for s in stats["guided"]) / len(stats["guided"])
        print(f"{k:<16}{u:>14.4f}{g:>14.4f}{g - u:>+14.4f}")
    print("=" * 78)
    print("dmin = closest ego-adversary gap (m) over the 5 s rollout; lower = more critical.")

    with open(out_dir / "guided_scenes.pkl", "wb") as f:
        pickle.dump(scenes, f)
    print(f"\nScenes + plots -> {out_dir}")


if __name__ == "__main__":
    main()
