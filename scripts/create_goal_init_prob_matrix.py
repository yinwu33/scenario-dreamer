

import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent-dir",
        default="data/advscene_ae_goal_latents_waymo/train",
        help="split directory of the goal autoencoder latent cache to estimate the prior on",
    )
    parser.add_argument(
        "--output",
        default="metadata/initial_prob_matrix_goal_waymo.pt",
    )
    parser.add_argument("--max-num-lanes", type=int, default=100)
    parser.add_argument("--max-num-agents", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="only scan the first N files (debugging); -1 scans everything",
    )
    return parser.parse_args()


def _read_counts(path):
    """Return ``(num_lanes, num_agents)`` for one latent-cache scene."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return (
        int(data["lane_mu"].shape[0]),
        int(data["agent_mu"].shape[0]),
    )


def main():
    args = parse_args()
    latent_dir = Path(args.latent_dir)
    files = sorted(latent_dir.glob("*.pkl"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No latent pickles found in {latent_dir}")

    counts = torch.zeros(
        (args.max_num_lanes + 1, args.max_num_agents + 1),
        dtype=torch.float64,
    )
    num_dropped = 0
    with Pool(args.num_workers) as pool:
        for num_lanes, num_agents in tqdm(
            pool.imap_unordered(_read_counts, files, chunksize=256),
            total=len(files),
            desc=f"Scanning {latent_dir}",
        ):
            # LDM-Adv needs the ego plus at least one non-ego agent (the adversary),
            # and the DiT positional embeddings bound lanes/agents from above.
            if not (2 <= num_agents <= args.max_num_agents):
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
    # num_agents). ldm_adv has no map_id dimension, so -- unlike the baseline's
    # (num_map_ids, num_lanes, num_agents) matrix, where each map_id slice sums to
    # 1 -- there is nothing to normalize per row here. Normalizing per num_lanes
    # row would give every lane count equal mass (a uniform lane-count prior), and
    # ScenarioDreamerLDMAdv._initialize_pyg_dset samples from prior.reshape(-1),
    # i.e. it reads this as one joint distribution.
    probs = (counts / counts.sum()).float()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probs, output)

    print(f"Scanned {len(files)} scenes, kept {kept}, dropped {num_dropped}.")
    for num_lanes in range(args.max_num_lanes + 1):
        share = counts[num_lanes].sum().item() / kept
        print(
            f"  num_lanes={num_lanes}: {int(counts[num_lanes].sum().item())} scenes ({share:.1%})"
        )
    print(f"Saved {tuple(probs.shape)} prior to {output}")


if __name__ == "__main__":
    main()
