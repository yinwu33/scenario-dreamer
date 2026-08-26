#!/usr/bin/env python
"""Build a context-prior manifest for prioritized DDPO context sampling.

Reads one or more headroom-probe JSONs (scripts/headroom_probe.py) and writes
the manifest ``LDMAdvConditioningPool`` consumes (``ddpo.context_prior.path``):
per-dataset-scene weights concentrating training draws on contexts where the
frozen base model demonstrably CAN produce critical adversaries. Rationale
(2026-08-09 idm-idm probe): collisions exist in only ~25% of contexts, so
uniform sampling spends ~75% of every batch on contexts with zero collision
gradient.

Weights (per context, from the probe's per-sample outcomes):

    weight = collision_weight * collision_hits + near_miss_weight * near_miss_hits

Contexts with weight 0 are dropped -- they stay reachable through the pool's
uniform (1 - focus_frac) share. The manifest is keyed by DATASET scene index,
so it is independent of pool_size / pool seed, but it IS planner-pair specific:
build one per pair, from a probe run with that pair.

    python scripts/headroom_probe.py --sut idm --env idm --adv idm \
        --num-contexts 1024 --samples-per-context 32
    python scripts/build_context_prior.py data/headroom_probe/idm-idm-idm.json \
        --out data/headroom_probe/context_prior_idm-idm.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("probes", nargs="+", help="headroom-probe JSON file(s) to merge")
    p.add_argument("--out", required=True, help="output manifest path")
    p.add_argument("--collision-weight", type=float, default=4.0)
    p.add_argument("--near-miss-weight", type=float, default=1.0)
    args = p.parse_args()

    # scene_idx -> weight; duplicates across probe files keep the max (the
    # probes may share contexts at different sample counts).
    weights: dict[int, float] = {}
    scanned = 0
    for path in args.probes:
        d = json.loads(Path(path).read_text())
        pc = d["per_context"]
        scanned += len(pc["scene_idx"])
        for ds, coll, near in zip(
            pc["scene_idx"], pc["collision_hits"], pc["near_miss_hits"]
        ):
            ds = int(ds)
            if ds < 0:
                continue
            w = args.collision_weight * float(coll) + args.near_miss_weight * float(near)
            if w > 0:
                weights[ds] = max(weights.get(ds, 0.0), w)

    if not weights:
        raise SystemExit("no context with positive weight -- nothing to prioritize")

    scene_idx = sorted(weights)
    weight = [weights[s] for s in scene_idx]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source_probes": [str(Path(p)) for p in args.probes],
        "collision_weight": args.collision_weight,
        "near_miss_weight": args.near_miss_weight,
        "num_contexts_scanned": scanned,
        "scene_idx": scene_idx,
        "weight": weight,
    }, indent=2))

    w = np.asarray(weight)
    print(f"wrote {out}: {len(scene_idx)} priority contexts "
          f"(of {scanned} scanned, {len(scene_idx) / max(scanned, 1):.1%})")
    print(f"weight: min={w.min():.1f} median={np.median(w):.1f} max={w.max():.1f}; "
          f"top-10 mass={np.sort(w)[::-1][:10].sum() / w.sum():.1%}")


if __name__ == "__main__":
    main()
