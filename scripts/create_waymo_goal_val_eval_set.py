"""Build the shared ground-truth reference set for the goal chain.

``metrics.Metrics`` reads ``eval.metrics.eval_set`` as ``{"files": [basename, ...]}``
and loads each name from ``eval.metrics.gt_test_dir`` (with ``gt_format: goal`` it runs
``utils.goal_runtime.prepare_scene`` on every record, exactly as the datasets do).

Both adversarial generators are scored against this ONE file so their numbers are
directly comparable:

* ``dm_goal``  -- data-space diffusion (the SceneControl-style guidance baseline)
* ``ldm_adv``  -- latent diffusion over the goal-autoencoder latents

Usage::

    python scripts/create_waymo_goal_val_eval_set.py \
        --input-dir data/advscene_preprocess_waymo/val \
        --output metadata/waymo_goal_val_eval_set.pkl
"""

import argparse
import pickle
import random
import sys
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.goal_runtime import prepare_scene

_CFG = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input-dir", default="data/advscene_preprocess_waymo/val")
    parser.add_argument("--output", default="metadata/waymo_goal_val_eval_set.pkl")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="cap on the reference set size after shuffling; -1 keeps everything",
    )
    parser.add_argument("--max-num-agents", type=int, default=30)
    parser.add_argument(
        "--min-num-agents",
        type=int,
        default=2,
        help=(
            "Drop reference scenes with fewer agents (counted AFTER prepare_scene). "
            "The default of 2 matches the layout prior both generators sample from "
            "(scripts/create_goal_init_prob_matrix.py --min-num-agents 2), so the "
            "real and generated pools cover the same scene sizes. Pass 1 to keep "
            "single-agent scenes."
        ),
    )
    parser.add_argument("--offroad-threshold", type=float, default=1.5)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _init_worker(cfg):
    global _CFG
    _CFG = cfg


def _num_agents(path):
    """Agent count for one v2 record, after the same filtering the datasets apply."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return Path(path).name, int(len(prepare_scene(data, _CFG)["agent_states"]))


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No pickles found in {input_dir}")

    cfg = SimpleNamespace(
        offroad_threshold=args.offroad_threshold,
        max_num_agents=args.max_num_agents,
    )
    kept, dropped = [], 0
    with Pool(args.num_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for name, num_agents in tqdm(
            pool.imap_unordered(_num_agents, [str(f) for f in files], chunksize=256),
            total=len(files),
            desc=f"Scanning {input_dir}",
        ):
            if args.min_num_agents <= num_agents <= args.max_num_agents:
                kept.append(name)
            else:
                dropped += 1

    if not kept:
        raise RuntimeError("No scene satisfied the agent-count bounds.")

    # Shuffle so that --num-samples / metrics.num_gt_samples take an unbiased subset
    # rather than a contiguous block of one tfrecord shard.
    kept.sort()
    random.Random(args.seed).shuffle(kept)
    if args.num_samples > 0:
        kept = kept[: args.num_samples]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump({"files": kept}, f)

    print(f"Scanned {len(files)} scenes, kept {len(kept)}, dropped {dropped}.")
    print(f"Saved reference set to {output}")


if __name__ == "__main__":
    main()
