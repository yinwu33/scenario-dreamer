

import argparse
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.goal_runtime import prepare_scene

_CFG = None  # per-worker config for the v2 source, set by the pool initializer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate the (num_lanes, num_agents) layout prior for the goal chain."
    )
    parser.add_argument(
        "--source",
        choices=["latents", "v2"],
        default="latents",
        help=(
            "latents: count from the goal autoencoder latent cache (ldm_adv). "
            "v2: count from the v2 goal records themselves, running prepare_scene "
            "so the agent counts match what the dataset yields (dm_goal). Both "
            "produce the same prior when run on corresponding splits."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="split directory to scan; defaults per --source",
    )
    parser.add_argument(
        "--latent-dir",
        default=None,
        help="deprecated alias for --input-dir (kept so existing commands keep working)",
    )
    parser.add_argument(
        "--output",
        default="metadata/initial_prob_matrix_goal_waymo.pt",
    )
    parser.add_argument("--max-num-lanes", type=int, default=100)
    parser.add_argument("--max-num-agents", type=int, default=30)
    parser.add_argument(
        "--min-num-agents",
        type=int,
        default=2,
        help=(
            "Scenes with fewer agents are dropped from the prior. The default of 2 "
            "reflects that both adversarial generators need the ego plus at least "
            "one non-ego agent (ldm_adv splits one off as the adv stream; dm_goal "
            "guides one of them)."
        ),
    )
    parser.add_argument(
        "--offroad-threshold", type=float, default=1.5, help="--source v2 only (prepare_scene)"
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="only scan the first N files (debugging); -1 scans everything",
    )
    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = args.latent_dir
    if args.input_dir is None:
        args.input_dir = (
            "data/advscene_ae_goal_latents_waymo/train"
            if args.source == "latents"
            else "data/advscene_preprocess_waymo/train"
        )
    return args


def _init_worker(cfg):
    global _CFG
    _CFG = cfg


def _read_counts_latents(path):
    """Return ``(num_lanes, num_agents)`` for one latent-cache scene."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return (
        int(data["lane_mu"].shape[0]),
        int(data["agent_mu"].shape[0]),
    )


def _read_counts_v2(path):
    """Return ``(num_lanes, num_agents)`` for one v2 goal record.

    ``num_agents`` is counted **after** prepare_scene, i.e. after the valid-goal
    and off-road filters and the closest-N cap, so it is exactly the agent count
    WaymoDatasetDMGoal will produce for this scene. Reading ``record["num_agents"]``
    instead would count the unfiltered set and skew the prior.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    scene = prepare_scene(data, _CFG)
    return int(data["num_lanes"]), int(len(scene["agent_states"]))


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.pkl"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No pickles found in {input_dir}")

    counts = torch.zeros(
        (args.max_num_lanes + 1, args.max_num_agents + 1),
        dtype=torch.float64,
    )
    num_dropped = 0
    cfg = SimpleNamespace(
        offroad_threshold=args.offroad_threshold,
        max_num_agents=args.max_num_agents,
    )
    read_counts = _read_counts_latents if args.source == "latents" else _read_counts_v2

    with Pool(args.num_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for num_lanes, num_agents in tqdm(
            pool.imap_unordered(read_counts, files, chunksize=256),
            total=len(files),
            desc=f"Scanning {input_dir} ({args.source})",
        ):
            # Both adversarial generators need the ego plus at least one non-ego
            # agent, and the DiT positional embeddings bound lanes/agents above.
            if not (args.min_num_agents <= num_agents <= args.max_num_agents):
                num_dropped += 1
                continue
            if not (1 <= num_lanes <= args.max_num_lanes):
                num_dropped += 1
                continue
            counts[num_lanes, num_agents] += 1

    kept = int(counts.sum().item())
    if kept == 0:
        raise RuntimeError(
            "No scene satisfied the lane/agent bounds; nothing to normalize."
        )

    # Normalize the WHOLE matrix to a joint distribution over (num_lanes,
    # num_agents). The goal chain has no map_id dimension, so -- unlike the
    # baseline's (num_map_ids, num_lanes, num_agents) matrix, where each map_id
    # slice sums to 1 -- there is nothing to normalize per row here. Normalizing
    # per num_lanes row would give every lane count equal mass (a uniform
    # lane-count prior); both _initialize_pyg_dset implementations sample from
    # prior.reshape(-1), i.e. they read this as one joint distribution.
    probs = (counts / counts.sum()).float()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probs, output)

    print(f"Scanned {len(files)} scenes, kept {kept}, dropped {num_dropped}.")
    for num_lanes in range(args.max_num_lanes + 1):
        total = int(counts[num_lanes].sum().item())
        if total == 0:
            continue
        print(f"  num_lanes={num_lanes}: {total} scenes ({total / kept:.1%})")
    print(f"Saved {tuple(probs.shape)} prior to {output}")


if __name__ == "__main__":
    main()
