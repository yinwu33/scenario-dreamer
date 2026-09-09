#!/usr/bin/env python
"""Designate the LOGGED agent closest to the ego as that scene's adversary.

`original.pt` carries `adv_local_idx = -1`, so every adversary-scoped metric
(`ego_collision` vs the adversary, `ego_fault_collision`, `ego_min_ttc`) is
structurally undefined for the Log row. This rewrites that one field so the Log
row measures the same thing every other row does: the ego against ONE designated
non-ego agent. Nothing else in the payload changes -- the scene, the agents and
their goals are the logged ones.

The designation is `critical_scene.log_scenes.closest_agent_adv_idx`, shared with
`eval_rollout.log_payload` so the two ways of producing this row cannot drift --
they used to, disagreeing on 91 of 1000 val scenes. See that function for the
rule (nearest vehicle, falling back to the nearest agent of any type) and why.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from critical_scene.log_scenes import closest_agent_adv_idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="original.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    blob = torch.load(args.src, map_location="cpu", weights_only=False)
    p = blob["payload"]
    states = np.asarray(p["agent_states"], dtype=np.float32)
    types = np.asarray(p["agent_types"], dtype=np.int64)
    scene_idx = np.asarray(p["agent_scene_idx"], dtype=np.int64)
    n_scenes = int(p["num_scenes"])

    adv = closest_agent_adv_idx(states, types, scene_idx, n_scenes)

    p["adv_local_idx"] = torch.from_numpy(adv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    got = int((adv >= 0).sum())
    print(f"[closest_adv] {got}/{n_scenes} scenes got an adversary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
