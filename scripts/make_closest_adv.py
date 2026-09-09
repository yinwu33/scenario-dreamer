#!/usr/bin/env python
"""Designate the LOGGED agent closest to the ego as that scene's adversary.

`original.pt` carries `adv_local_idx = -1`, so every adversary-scoped metric
(`ego_collision` vs the adversary, `ego_fault_collision`, `ego_min_ttc`) is
structurally undefined for the Log row. This rewrites that one field so the Log
row measures the same thing every other row does: the ego against ONE designated
non-ego agent. Nothing else in the payload changes -- the scene, the agents and
their goals are the logged ones.

Candidates are restricted to VEHICLES, matching the generated adversary's fixed
`adv_cond_target.type: vehicle`; a scene whose only neighbours are pedestrians
or cyclists keeps -1 and stays undefined.

Distance is spawn-to-spawn (columns 0,1 of the [N, 9] state), i.e. a property of
the initialisation, not of any rollout, so the choice is planner-independent and
one artifact serves every SUT.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


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

    # `_build_scenes` slices with `agent_scene_idx == s`, so a scene's local
    # order is its payload order and local 0 is the ego.
    adv = np.full(n_scenes, -1, dtype=np.int64)
    for s in range(n_scenes):
        rows = np.flatnonzero(scene_idx == s)
        if len(rows) < 2:
            continue
        xy = states[rows, :2]
        d = np.hypot(xy[1:, 0] - xy[0, 0], xy[1:, 1] - xy[0, 1])
        veh = types[rows[1:]] == 0  # dataset ids: 0 veh / 1 ped / 2 cyc
        if not veh.any():
            continue
        d = np.where(veh, d, np.inf)
        adv[s] = int(np.argmin(d)) + 1  # +1: local 0 is the ego

    p["adv_local_idx"] = torch.from_numpy(adv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    got = int((adv >= 0).sum())
    print(f"[closest_adv] {got}/{n_scenes} scenes got an adversary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
