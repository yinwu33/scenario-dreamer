#!/usr/bin/env python
"""Can the shared 7x13 accel/steer table express how logged agents actually move?

This decides the model's output space. If the table reproduces logged motion
tightly, the traffic model emits an index into it and the simulator keeps ONE
integrator. If it does not, the fallback is a k-disks motion vocabulary
(``utils.k_disks_helpers``) and the paper has to declare a SECOND integrator
exception alongside ``ctrl_sim``.

Reports both errors, because they answer different questions:

  * teacher-forced -- per-step, true state restored each step. Optimistic.
  * replay drift -- the whole labelled sequence integrated open loop from the
    first state. This is the one that matters: at rollout time nothing restores
    the truth, so this is the floor on how far ANY model driving through this
    action space can stay from a logged trajectory.

Usage:
    python smart/check_action_table.py --scenes 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart.actions import chase, label_transitions, replay
from smart.records import load_scene, scene_paths, transitions

DT = 0.1  # cfgs/rollout/base.yaml


def _longest_run(valid: np.ndarray) -> tuple[int, int]:
    """(start, length) of the longest contiguous True run in a 1-D mask."""
    best_start = best_len = cur_start = cur_len = 0
    for i, v in enumerate(valid):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0
    return best_start, best_len


def _pct(x, qs=(50, 90, 99)):
    return " ".join(f"p{q}={np.percentile(x, q):.3f}" for q in qs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="val")
    ap.add_argument("--scenes", type=int, default=200)
    ap.add_argument("--min-run", type=int, default=20, help="steps a track needs to be replayed")
    args = ap.parse_args()

    paths = scene_paths(args.split)
    step = max(1, len(paths) // args.scenes)
    paths = paths[::step][: args.scenes]

    step_err = []
    ade, fde, run_lens, speeds = [], [], [], []
    chase_ade, chase_fde = [], []
    for path in paths:
        scene = load_scene(path)
        tr = transitions(scene)
        if len(tr["state"]):
            _, err = label_transitions(
                tr["state"], tr["next_state"], tr["length"], tr["width"], DT
            )
            step_err.append(err)
            speeds.append(np.abs(tr["state"][:, 3]))

        # open-loop replay of each track's longest valid run
        for a in range(scene["state"].shape[0]):
            start, n = _longest_run(scene["valid"][a])
            if n < args.min_run:
                continue
            seq = scene["state"][a, start : start + n]
            length = np.array([scene["length"][a]])
            width = np.array([scene["width"][a]])
            acts, _ = label_transitions(
                seq[:-1], seq[1:],
                np.repeat(length, n - 1), np.repeat(width, n - 1), DT,
            )
            out = replay(seq[:1], length, acts[None, :], DT)[0]
            d = np.hypot(out[:, 0] - seq[1:, 0], out[:, 1] - seq[1:, 1])
            ade.append(d.mean())
            fde.append(d[-1])
            run_lens.append(n)

            _, poses = chase(seq, scene["length"][a], scene["width"][a], DT)
            cd = np.hypot(poses[:, 0] - seq[1:, 0], poses[:, 1] - seq[1:, 1])
            chase_ade.append(cd.mean())
            chase_fde.append(cd[-1])

    step_err = np.concatenate(step_err)
    speeds = np.concatenate(speeds)
    ade, fde, run_lens = map(np.asarray, (ade, fde, run_lens))
    chase_ade, chase_fde = np.asarray(chase_ade), np.asarray(chase_fde)

    print(f"scenes={len(paths)}  transitions={len(step_err)}  replayed tracks={len(ade)}"
          f"  (median run {np.median(run_lens):.0f} steps = {np.median(run_lens) * DT:.1f} s)")
    print(f"median |speed| = {np.median(speeds):.2f} m/s "
          f"-> median step displacement ~ {np.median(speeds) * DT:.3f} m\n")

    print("teacher-forced per-step pose error (m, mean over box corners)")
    print(f"  mean={step_err.mean():.3f}  {_pct(step_err)}  max={step_err.max():.3f}")
    for th in (0.05, 0.10, 0.25, 0.50):
        print(f"  <= {th:.2f} m : {100 * np.mean(step_err <= th):5.1f}%")

    print("\nopen-loop replay of the labelled action sequence")
    print(f"  ADE mean={ade.mean():.3f}  {_pct(ade)}  max={ade.max():.3f}")
    print(f"  FDE mean={fde.mean():.3f}  {_pct(fde)}  max={fde.max():.3f}")
    for th in (0.5, 1.0, 2.0):
        print(f"  FDE <= {th:.1f} m : {100 * np.mean(fde <= th):5.1f}%")

    print("\nclosed-loop labelling (chase): what the ACTION SPACE can track")
    print(f"  ADE mean={chase_ade.mean():.3f}  {_pct(chase_ade)}  max={chase_ade.max():.3f}")
    print(f"  FDE mean={chase_fde.mean():.3f}  {_pct(chase_fde)}  max={chase_fde.max():.3f}")
    for th in (0.1, 0.25, 0.5):
        print(f"  FDE <= {th:.2f} m : {100 * np.mean(chase_fde <= th):5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
