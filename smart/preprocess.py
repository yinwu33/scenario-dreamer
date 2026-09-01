#!/usr/bin/env python
"""Cache the action label of every logged transition, once.

Labelling is closed-loop (``smart.actions.chase``): each action is the one that
best reaches the next logged pose FROM THE POSE THE INTEGRATOR IS ACTUALLY AT.
Measured on 150 val scenes, that tracks a logged trajectory to a median final
error of 0.023 m over 8.4 s, against 1.21 m for naive per-step labelling -- the
action table is expressive, greedy labelling just accumulates its residuals.
Two consequences worth knowing:

  * the model's targets are drift-CORRECTING, since a label is chosen from a pose
    that has already drifted a little. That is a feature: at rollout time nothing
    restores the truth, so correcting is exactly the behavior to imitate;
  * the pose a label was chosen from differs from the logged pose the
    observation is built at, by a median 2.3 cm. Far below the observation's
    resolution, so both are treated as the same state.

Chasing is a per-step Python loop and costs ~0.2 s per scene, which is too slow
to sit inside the training loop -- hence this cache. Output is one ``int8``
array per scene, ``[agents, steps]``, with -1 where no transition exists.

Usage:
    python smart/preprocess.py --split val --workers 32
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart.actions import NUM_ACTIONS, chase
from smart.records import load_scene, scene_paths

DT = 0.1  # cfgs/rollout/base.yaml
NO_LABEL = -1
DEFAULT_OUT = "data/smart_action_labels"


def contiguous_runs(valid: np.ndarray):
    """(start, stop) of every maximal True run, stop exclusive."""
    padded = np.concatenate(([False], valid, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def label_scene(scene: dict) -> np.ndarray:
    """[agents, steps] int8 action labels; -1 where there is no transition."""
    state, valid = scene["state"], scene["valid"]
    labels = np.full((state.shape[0], state.shape[1] - 1), NO_LABEL, dtype=np.int8)
    for a in range(state.shape[0]):
        for start, stop in contiguous_runs(valid[a]):
            if stop - start < 2:
                continue
            acts, _ = chase(state[a, start:stop], scene["length"][a], scene["width"][a], DT)
            labels[a, start : stop - 1] = acts.astype(np.int8)
    return labels


def _one(path: Path, out_dir: Path) -> int:
    target = out_dir / f"{path.stem}.npy"
    if target.exists():
        return 0
    labels = label_scene(load_scene(path))
    np.save(target, labels)
    return int((labels != NO_LABEL).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N scenes, for a smoke run")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    paths = scene_paths(args.split)
    if args.limit:
        paths = paths[: args.limit]
    out_dir = Path(args.out, args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    fn = partial(_one, out_dir=out_dir)
    if args.workers > 1:
        with Pool(args.workers) as pool:
            counts = pool.map(fn, paths, chunksize=16)
    else:
        counts = [fn(p) for p in paths]

    total = int(np.sum(counts))
    print(f"{args.split}: {len(paths)} scenes -> {out_dir}")
    print(f"labelled transitions: {total} ({total / max(len(paths), 1):.1f} per scene, "
          f"vocabulary {NUM_ACTIONS})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
