#!/usr/bin/env python
"""Build the ``proximity_adv`` scene source from an ``original`` artifact.

The results table's "Log + proximity adversary" row: the recorded scene with one
extra vehicle inserted next to the ego, following the proximity-based selection
rule of closed-loop adversarial methods. It isolates how much of AdvScene's gain
comes from inserting an agent at all, as opposed to inserting it in the right
place, so it must differ from ``base_gen`` / ``ddpo_gen`` in WHERE the agent goes
and in nothing else: same map, same real traffic, same +1 agent count, same
payload schema, same ``adv_local_idx`` handle for the adversary role.

Placement is the nearest legal point of the lane graph: among all lane vertices
of the scene, take the one closest to the ego that clears every existing agent,
head it along the lane tangent, and give it a goal by walking the successor graph
forward. No generative model is involved.

    python scripts/make_proximity_adv.py \
        --original data/critical_scene/table_main/idm-ppo_norm/artifacts/original.pt \
        --reference data/critical_scene/table_main/idm-ppo_norm/artifacts/ddpo_gen.pt \
        --out data/critical_scene/table_main/idm-ppo_norm/artifacts/proximity_adv.pt
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.routes import _arclength, _lane_tangent_at, _successors

# agent_states layout: [x, y, speed, cos, sin, length, width, goal_x, goal_y]
X, Y, SPEED, COS, SIN, LENGTH, WIDTH, GOAL_X, GOAL_Y = range(9)
VEHICLE = 0


def _parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--original", required=True, help="artifacts/original.pt")
    p.add_argument("--reference", required=True,
                   help="artifacts/ddpo_gen.pt -- only its adversaries' goal distance and "
                        "footprint medians are read, so the inserted agent is the same KIND "
                        "of vehicle as the generated one and only its placement differs")
    p.add_argument("--out", required=True, help="artifacts/proximity_adv.pt")
    p.add_argument("--clearance", type=float, default=8.0,
                   help="metres the spawn must keep from every existing agent centre. "
                        "8 m is the smallest value that adds no spawn overlap over the log "
                        "distribution: measured on 1000 val scenes, the log scenes overlap in "
                        "6.8%% of scenes and so does this source at 8 m, while 5 m inflates it "
                        "to 14.8%% and buys collisions with an artifact rather than a conflict. "
                        "The collision rate is flat from 8 m to 16 m (6.5 / 6.4 / 6.0%%), so the "
                        "baseline is not sensitive to the exact choice above that floor.")
    return p.parse_args()


def _reference_stats(path: Path) -> dict[str, float]:
    """Goal distance and footprint of the generated adversaries, as medians."""
    payload = torch.load(path, map_location="cpu", weights_only=False)["payload"]
    states = payload["agent_states"].numpy()
    scene_idx = payload["agent_scene_idx"].numpy()
    adv_local = payload["adv_local_idx"].numpy()

    # ``insert_adv_as_extra`` appends the generated adversaries after all base
    # agents, so agent_scene_idx is NOT monotonic. A stable argsort groups by
    # scene while preserving each scene's internal order -- the same ordering the
    # boolean mask in ``slice_payload`` produces, which is what ``adv_local_idx``
    # indexes into.
    num_scenes = int(payload["num_scenes"])
    order = np.argsort(scene_idx, kind="stable")
    starts = np.concatenate([[0], np.cumsum(np.bincount(scene_idx, minlength=num_scenes))[:-1]])
    rows = [order[starts[s] + int(local)] for s, local in enumerate(adv_local) if local >= 0]
    adv = states[np.asarray(rows, dtype=np.int64)]
    goal_dist = np.linalg.norm(adv[:, [GOAL_X, GOAL_Y]] - adv[:, [X, Y]], axis=-1)
    return {
        "goal_dist": float(np.median(goal_dist)),
        "length": float(np.median(adv[:, LENGTH])),
        "width": float(np.median(adv[:, WIDTH])),
        "speed": float(np.median(adv[:, SPEED])),
    }


def _walk_forward(lanes: np.ndarray, succ: list[list[int]], lane_id: int,
                  vertex: int, distance: float) -> np.ndarray:
    """Point ``distance`` metres ahead of a lane vertex along the successor graph.

    Follows the longest successor at each junction, which keeps the goal on a
    through-lane rather than on a short intersection connector. Runs out of graph
    by returning the furthest point it reached, so a goal always exists.
    """
    lane = lanes[lane_id]
    cum = _arclength(lane)
    s = float(cum[vertex])
    remaining = float(distance)
    visited = {lane_id}

    while True:
        target = s + remaining
        if target <= cum[-1]:
            return np.stack([np.interp(target, cum, lane[:, 0]),
                             np.interp(target, cum, lane[:, 1])]).astype(np.float32)
        remaining -= cum[-1] - s
        nexts = [n for n in succ[lane_id] if n not in visited]
        if not nexts:
            return lane[-1].astype(np.float32)
        lane_id = max(nexts, key=lambda n: _arclength(lanes[n])[-1])
        visited.add(lane_id)
        lane = lanes[lane_id]
        cum = _arclength(lane)
        s = 0.0


def _insert_one(states: np.ndarray, lanes: np.ndarray, lane_graph, stats, clearance: float):
    """The proximity adversary's 9-vector, or ``None`` if the scene has no room."""
    ego = states[0]
    ego_xy = ego[[X, Y]]

    # Candidate spawns: every lane vertex, nearest to the ego first.
    pts = lanes.reshape(-1, 2)
    lane_of = np.repeat(np.arange(len(lanes)), lanes.shape[1])
    order = np.argsort(np.linalg.norm(pts - ego_xy, axis=-1))

    others = states[:, [X, Y]]
    succ = _successors(lane_graph, len(lanes))
    for k in order:
        spawn = pts[k]
        if np.linalg.norm(others - spawn, axis=-1).min() < clearance:
            continue
        lane_id, vertex = int(lane_of[k]), int(k % lanes.shape[1])
        tangent = _lane_tangent_at(lanes[lane_id], spawn)
        if tangent is None:
            continue
        goal = _walk_forward(lanes, succ, lane_id, vertex, stats["goal_dist"])
        out = np.zeros(9, dtype=np.float32)
        out[[X, Y]] = spawn
        out[SPEED] = ego[SPEED] if ego[SPEED] > 0 else stats["speed"]
        out[[COS, SIN]] = tangent
        out[LENGTH], out[WIDTH] = stats["length"], stats["width"]
        out[[GOAL_X, GOAL_Y]] = goal
        return out
    return None


def main() -> int:
    args = _parse()
    blob = torch.load(args.original, map_location="cpu", weights_only=False)
    payload, metadata = blob["payload"], dict(blob["metadata"])
    stats = _reference_stats(Path(args.reference))
    print(f"[proximity] reference adversary: {stats}", flush=True)

    states = payload["agent_states"].numpy()
    types = payload["agent_types"].numpy()
    scene_idx = payload["agent_scene_idx"].numpy()
    lanes_all = payload["lane_polylines"].numpy()
    lane_scene = payload["lane_scene_idx"].numpy()
    num_scenes = int(payload["num_scenes"])

    new_states, new_types, new_scene_idx, adv_local, gen_mask = [], [], [], [], []
    skipped = 0
    for s in range(num_scenes):
        rows = states[scene_idx == s]
        lanes = lanes_all[lane_scene == s]
        adv = _insert_one(rows, lanes, payload["lane_graph"][s], stats, args.clearance)
        new_states.append(rows)
        new_types.append(types[scene_idx == s])
        keep = len(rows)
        if adv is None:
            skipped += 1
            adv_local.append(-1)
        else:
            new_states.append(adv[None])
            new_types.append(np.array([VEHICLE], dtype=types.dtype))
            adv_local.append(keep)
            keep += 1
        new_scene_idx.append(np.full(keep, s, dtype=np.int64))
        gen_mask.append(np.array([False] * len(rows) + ([] if adv is None else [True])))
        if (s + 1) % 200 == 0:
            print(f"[proximity] {s + 1}/{num_scenes}", flush=True)

    out_payload = dict(payload)
    out_payload["agent_states"] = torch.from_numpy(np.concatenate(new_states).astype(np.float32))
    out_payload["agent_types"] = torch.from_numpy(np.concatenate(new_types)).long()
    out_payload["agent_scene_idx"] = torch.from_numpy(np.concatenate(new_scene_idx)).long()
    out_payload["adv_local_idx"] = torch.tensor(adv_local, dtype=torch.long)
    out_payload["gen_agent_mask"] = torch.from_numpy(np.concatenate(gen_mask)).bool()

    metadata["source"] = "proximity_adv"
    metadata["derived_from"] = str(args.original)
    metadata["proximity"] = {"clearance": float(args.clearance),
                             "reference": str(args.reference), **stats,
                             "scenes_without_room": skipped}
    metadata["created"] = datetime.now().isoformat(timespec="seconds")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"payload": out_payload, "metadata": metadata}, args.out)
    print(f"[proximity] wrote {args.out}  (+1 agent in {num_scenes - skipped}/{num_scenes} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
