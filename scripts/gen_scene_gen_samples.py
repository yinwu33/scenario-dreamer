"""Generate the sample sets behind the scene-generation-quality table.

Two-stage protocol, mirroring the ``base_gen`` / ``ddpo_gen`` pair in
``critical_scene/ldm_adv_eval.py`` so this table and the criticality table
describe the same objects:

  stage 1  the base model draws a scene from the layout prior with every
           conditioning label left at its trained null token, using the
           supervised DDPM chain. Run ONCE and shared by every row, so the
           lane-graph columns are bit-identical across rows (the adv stream is
           downstream-only in ``FactorizedDiTBlock``, and DDPO freezes
           everything else).
  stage 2  the adversary latent is re-sampled with the sampler DDPO trains
           under (ddim/30/eta=1.0, from cfgs/ddpo/ldm_adv.yaml) under the run's
           ``adv_cond_target``. Only the adv-branch weights differ between rows.

Samples are written as per-scene pickles in the layout ``metrics.py`` reads.

Usage (env vars from scripts/define_env_variables.sh must be set):
  .venv/bin/python scripts/gen_scene_gen_samples.py --num-scenes 1000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from critical_scene.ldm_adv_eval import (
    build_policy,
    compose_eval_cfg,
    prepare_ldm_cfg,
    _seed_all,
)
from ddpo.conditioning import _ADV_COND_FIELDS, LDMAdvConditioningPool
from models.scenario_dreamer_ldm_adv import ScenarioDreamerLDMAdv
from utils.data_helpers import convert_batch_to_scenarios

# The eight DDPO runs that back PROVENANCE.json's eight cells; each row of the
# table's AdvScene entry is one of these, aggregated as mean +- std.
DDPO_RUNS = (
    "idm-idm_hier_v2",
    "idm-ppo_norm_hier_v2",
    "idm-ppo_aggressive_hier_v2",
    "idm-ppo_caution_hier_v2",
    "ppo-idm_hier_v2",
    "ppo-ppo_norm_hier_v2",
    "ppo-ppo_aggressive_hier_v2",
    "ppo-ppo_caution_hier_v2",
)


def ddpo_ckpt(run: str) -> Path:
    d = ROOT / "data" / "critical_scene" / f"critical_scene_ddpo_ldm_adv_ddim_{run}"
    return d / f"critical_scene_ddpo_ldm_adv_ddim_{run}_03000.ckpt"


def build_prior_batches(lit, cfg_root, num_scenes: int, batch_size: int):
    """Prior-mode layouts with the DDPO adv conditioning target attached.

    ``_initialize_pyg_dset`` samples (num_lanes, num_agents) from the goal layout
    prior and leaves every label absent. The adv labels are then overwritten with
    the same per-scene draw ``LDMAdvConditioningPool._apply_target_cond`` makes,
    so the fine-tuned adv branch runs at the operating point it was trained at."""
    data_list, _ = lit._initialize_pyg_dset("init_scene", num_scenes, batch_size, None, False)

    targets = LDMAdvConditioningPool._parse_adv_cond_target(cfg_root.ddpo.adv_cond_target)
    if targets is None:
        raise ValueError("adv_cond_target is disabled; the DDPO runs were trained with it on.")
    seed = int(cfg_root.ddpo.seed)
    for i, d in enumerate(data_list):
        rng = np.random.default_rng((seed, i))
        labels = [int(rng.choice(targets[f])) for f in _ADV_COND_FIELDS]
        d["adv"].cond = torch.tensor([labels], dtype=torch.long)

    return list(DataLoader(data_list, batch_size=batch_size, shuffle=False, drop_last=False))


@torch.no_grad()
def stage_one(lit, batches, seed: int, device: str):
    """Base scene latents, under the base model's EMA weights.

    The decoded lane polylines are produced here too: ``policy.sample`` reads
    ``lane.road_points``, and the base scene is shared by every row, so they are
    decoded once rather than per row. The jointly denoised adv latent is
    discarded -- every row re-samples the adversary in stage 2."""
    out = []
    with lit.ema.average_parameters():
        for bi, data in enumerate(batches):
            data = data.to(device)
            _seed_all(seed * 1_000_003 + 1000 + bi, device)
            x_agent, x_lane, x_adv = lit.diff_model.forward(data, mode="init_scene")
            _, lane_states, *_ = lit._decode_scene_and_adv(x_agent, x_lane, x_adv, data)
            out.append((x_agent.cpu(), x_lane.cpu(), lane_states.cpu()))
    return out


@torch.no_grad()
def stage_two(lit, policy, batches, base_latents, *, seed: int, device: str,
              batch_size: int, cache_dir: Path):
    """Re-sample the adversary with ``policy`` and write the decoded scenes."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    num_types = lit.cfg_dataset.num_agent_types
    for bi, data in enumerate(batches):
        data = data.to(device)
        x_agent, x_lane, lane_states = base_latents[bi]
        data["agent"].latents = x_agent.to(device)
        data["lane"].latents = x_lane.to(device)
        data["lane"].road_points = lane_states.to(device)

        _seed_all(seed * 1_000_003 + 2000 + bi, device)
        _, traj = policy.sample(data)
        x_adv = traj.records["steps"][-1][1][:, 0]

        agent_s, lane_s, agent_t, _, lane_conn, adv_s, adv_t = lit._decode_scene_and_adv(
            data["agent"].latents, data["lane"].latents, x_adv, data
        )
        data["agent"].x = agent_s
        data["lane"].x = lane_s
        data["agent"].type = torch.nn.functional.one_hot(agent_t, num_classes=num_types)
        data["lane", "to", "lane"].type = lane_conn
        data["adv"].x = adv_s
        data["adv"].type = torch.nn.functional.one_hot(adv_t, num_classes=num_types)

        convert_batch_to_scenarios(
            data,
            batch_size=batch_size,
            batch_idx=bi,
            cache_dir=str(cache_dir),
            cache_samples=True,
            cache_lane_types=False,
            mode="init_scene",
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_ldm_adv_ddpo")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--num-scenes", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out-root", default="data/scene_gen_table")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows", nargs="+", default=["base", *DDPO_RUNS])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg_root = compose_eval_cfg(args.config_name, args.overrides)
    ldm_cfg = prepare_ldm_cfg(cfg_root)
    base_ckpt = str(cfg_root.ddpo.ldm_adv_ckpt)
    out_root = Path(args.out_root)

    print(f"[scene_gen] base ckpt: {base_ckpt}")
    lit = ScenarioDreamerLDMAdv.load_from_checkpoint(
        base_ckpt, cfg=ldm_cfg, cfg_ae=cfg_root.ae_goal
    ).to(args.device).eval()

    torch.manual_seed(args.seed)
    batches = build_prior_batches(lit, cfg_root, args.num_scenes, args.batch_size)
    print(f"[scene_gen] {args.num_scenes} prior layouts in {len(batches)} batches")

    print("[scene_gen] stage 1: base scene (shared by every row)")
    base_latents = stage_one(lit, batches, args.seed, args.device)

    for row in args.rows:
        cache_dir = out_root / row
        ckpt = base_ckpt if row == "base" else str(ddpo_ckpt(row))
        print(f"[scene_gen] stage 2: {row} <- {Path(ckpt).name}")
        policy = build_policy(cfg_root, ldm_cfg, ckpt=ckpt, device=args.device)
        stage_two(lit, policy, batches, base_latents, seed=args.seed,
                  device=args.device, batch_size=args.batch_size, cache_dir=cache_dir)
        del policy
        torch.cuda.empty_cache()
        print(f"[scene_gen]   -> {len(list(cache_dir.glob('*.pkl')))} pickles in {cache_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
