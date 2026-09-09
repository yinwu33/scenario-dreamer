#!/usr/bin/env python
"""Score the scene caches ``generate_scene.py`` writes, as three separate groups.

  map    lane-graph shape: route length, endpoint distance, and the four urban
         planning Frechet distances.
  agent  the six agent-attribute JSDs over the scene's NORMAL agents.
  adv    the same six JSDs over the GENERATED ADVERSARY alone.

Keeping ``agent`` and ``adv`` apart is not cosmetic. Pooling them lets the two
cancel: the adversary is about an eighth of the vehicles, so a badly dispersed
adversary can pull a pooled histogram TOWARDS the reference when the normal
agents are under-dispersed, and the pooled number then improves while the
adversary itself gets worse. Split, each group answers its own question.

``adv`` is scored against a CONDITIONING-MATCHED reference subset. The adversary
is generated under ``adv_cond_target`` (vehicle / moving / {middle,far}), so
roughly half the reference vehicles are in a bucket it is forbidden to occupy;
scoring it against all of them charges it for obeying its own condition. The
manifest's ``adv_cond_target`` selects the subset, so the two always agree.

Every group is printed next to a FLOOR: the same statistic between a random draw
from the reference, at this cache's own sample size, and the whole reference. A
JSD at or below its floor is sample-size noise and carries no information about
the model -- at n = 1000 that is normally true of the angular-deviation column.

For ``init_agent`` and ``init_adv`` caches the lanes (and, for ``init_adv``, the
normal agents) are copied from the val scene rather than generated, so those
groups measure autoencoder reconstruction error and nothing else. The report
labels them ``(copied)`` rather than dropping them, so the distinction is
visible instead of implied.

Usage (env vars from scripts/define_env_variables.sh must be set)::

    .venv/bin/python eval_scene.py --caches data/scenes/* --out data/scenes/scene_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm

from critical_scene.ldm_adv_eval import compose_eval_cfg, prepare_ldm_cfg
from utils import metrics_helpers as mh
from utils.goal_runtime import prepare_scene

# (clip_min, clip_max, bin_size, scale) exactly as compute_jsd_metrics applies them.
JSD_SPEC = {
    "nearest_dist": (0, 50, 1, 10),
    "lat_dev": (0, 1.5, 0.1, 10),
    "ang_dev": (-200, 200, 5, 100),
    "length": (0, 25, 0.1, 100),
    "width": (0, 5, 0.1, 100),
    "speed": (0, 50, 1, 100),
}
AGENT_KEYS = tuple(JSD_SPEC)
MAP_KEYS = ("route_length_mean", "endpoint_dist_mean", "frechet_connectivity",
            "frechet_density", "frechet_reach", "frechet_convenience")
# Columns 5, 6 and 2 of the unified vehicle layout, plus the goal columns.
LENGTH, WIDTH, SPEED, GOAL_X = 5, 6, 2, 7


def scene_stats(unified) -> tuple[dict, dict]:
    """The six per-vehicle statistics, plus the vehicle index each row came from.

    Same helpers ``mh._collect_agent_stats`` calls, but index-tracked so a subset
    (the adversary) or a filter (the goal-distance bucket) can be applied without
    touching a metric definition."""
    veh = unified["vehicles"]
    lanes = mh.resample_lanes(unified["lanes"], num_points=100)
    n = len(veh)
    road_dist = np.linalg.norm(veh[:, None, :2] - lanes.reshape(-1, 2)[None, :, :], axis=-1).min(1)
    onroad = np.where(road_dist <= 1.5)[0]  # get_onroad_vehicles, tol=1.5

    stats, index = {}, {}
    if n > 1:
        stats["nearest_dist"] = mh.get_nearest_dists(veh)
        index["nearest_dist"] = np.arange(n)
    if len(onroad) > 0:
        stats["lat_dev"] = mh.get_lateral_devs(veh[onroad], lanes)
        stats["ang_dev"] = mh.get_angular_devs(veh[onroad], lanes)
        index["lat_dev"] = index["ang_dev"] = onroad
    for key, col in (("length", LENGTH), ("width", WIDTH), ("speed", SPEED)):
        stats[key] = veh[:, col]
        index[key] = np.arange(n)
    return stats, index


def goal_dists(unified) -> np.ndarray:
    veh = unified["vehicles"]
    return np.linalg.norm(veh[:, GOAL_X:GOAL_X + 2] - veh[:, :2], axis=-1)


def jsd(sim: np.ndarray, ref: np.ndarray, key: str) -> float:
    lo, hi, bin_size, scale = JSD_SPEC[key]
    return float(mh.jsd(sim, ref, clip_min=lo, clip_max=hi, bin_size=bin_size) * scale)


def adv_goal_threshold(adv_cond_target, dataset_cfg) -> float:
    """The smallest spawn->goal distance the adversary's condition allows.

    ``goal_dist`` is bucketed near / middle / far on the two dataset thresholds,
    so a target of {middle, far} admits everything past the near threshold and a
    target of {far} everything past the far one. Deriving it from the manifest
    keeps the reference subset tied to what the cache was actually generated
    under."""
    buckets = adv_cond_target["goal_dist"]
    buckets = [buckets] if isinstance(buckets, str) else list(buckets)
    if "near" in buckets:
        return 0.0
    if "middle" in buckets or "mid" in buckets:
        return float(dataset_cfg.cond_goaldist_near_threshold)
    return float(dataset_cfg.cond_goaldist_far_threshold)


def load_reference(dataset_cfg, eval_set: Path, gt_dir: Path, num_gt: int) -> list:
    with open(eval_set, "rb") as f:
        filenames = pickle.load(f)["files"][:num_gt]
    out = []
    for name in tqdm(filenames, desc="reference"):
        with open(gt_dir / name, "rb") as f:
            raw = pickle.load(f)
        scene = prepare_scene(raw, dataset_cfg)
        raw = dict(raw)
        raw["agent_states"] = scene["agent_states"]
        raw["agent_types"] = scene["agent_types"]
        out.append(mh.convert_data_to_unified_format(raw, dataset_name="waymo_gt"))
    return out


def reference_stats(reference: list) -> tuple[dict, dict]:
    """Pooled reference statistics, and the goal distance aligned to each row."""
    acc = {k: [] for k in AGENT_KEYS}
    acc_goal = {k: [] for k in AGENT_KEYS}
    for unified in tqdm(reference, desc="reference stats"):
        if len(unified["vehicles"]) == 0:
            continue
        stats, index = scene_stats(unified)
        gd = goal_dists(unified)
        for key, values in stats.items():
            acc[key].append(values)
            acc_goal[key].append(gd[index[key]])
    return ({k: np.concatenate(v) for k, v in acc.items()},
            {k: np.concatenate(v) for k, v in acc_goal.items()})


def load_cache(cache_dir: Path) -> tuple[list, list, dict]:
    """Unified-format scenes plus, per scene, the adversary's index among vehicles.

    ``generate_scene.py`` records ``adv_local_idx`` over ALL agents;
    ``convert_data_to_unified_format`` keeps only vehicles, in order, so the
    adversary's position among them is the rank of its index within the vehicle
    mask -- and it is absent entirely when the sampled adversary is not a
    vehicle, which is a condition violation rather than a missing record."""
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    scenes, adv_pos = [], []
    for path in tqdm(sorted(p for p in cache_dir.iterdir() if p.suffix == ".pkl"),
                     desc=f"load {cache_dir.name}"):
        with open(path, "rb") as f:
            record = pickle.load(f)
        unified = mh.convert_data_to_unified_format(record, dataset_name="waymo")
        if len(unified["G"]) == 0 or len(unified["vehicles"]) == 0:
            continue
        is_vehicle = np.argmax(record["agent_types"], axis=1) == mh.NUPLAN_VEHICLE
        adv = int(record["adv_local_idx"])
        scenes.append(unified)
        adv_pos.append(int(is_vehicle[:adv].sum()) if is_vehicle[adv] else -1)
    return scenes, adv_pos, manifest


def group_stats(scenes: list, adv_pos: list) -> tuple[dict, dict]:
    """Split every scene's per-vehicle statistics into (normal agents, adversary)."""
    agent = {k: [] for k in AGENT_KEYS}
    adv = {k: [] for k in AGENT_KEYS}
    for unified, pos in zip(scenes, adv_pos):
        stats, index = scene_stats(unified)
        for key, values in stats.items():
            hit = np.where(index[key] == pos)[0] if pos >= 0 else np.array([], dtype=int)
            agent[key].append(np.delete(values, hit))
            if len(hit):
                adv[key].append(values[hit])
    return ({k: np.concatenate(v) if v else np.array([]) for k, v in agent.items()},
            {k: np.concatenate(v) if v else np.array([]) for k, v in adv.items()})


def jsd_group(sim: dict, ref: dict, rng, floor_reps: int) -> dict:
    """Every JSD in a group, each with the sampling floor at that group's own n."""
    out = {}
    for key in AGENT_KEYS:
        values, reference = sim[key], ref[key]
        out[key] = jsd(values, reference, key)
        draws = [jsd(reference[rng.integers(0, len(reference), len(values))], reference, key)
                 for _ in range(floor_reps)]
        out[f"{key}_floor"] = float(np.mean(draws))
    out["mean"] = float(np.mean([out[k] for k in AGENT_KEYS]))
    out["mean_floor"] = float(np.mean([out[f"{k}_floor"] for k in AGENT_KEYS]))
    out["n"] = int(len(sim["length"]))
    return out


def render(results: dict) -> str:
    lines = ["# Scene metrics", ""]
    for name, row in results.items():
        man = row["manifest"]
        copied = {"init_scene": (), "init_agent": ("map",), "init_adv": ("map", "agent")}[man["mode"]]
        lines += [f"## {name}", "",
                  f"mode: `{man['mode']}`  conditioning: `{man['conditioning']}`  "
                  f"ckpt: `{Path(man['ddpo_ckpt']).name if man['ddpo_ckpt'] else 'base'}`",
                  f"scenes: {row['num_scenes']}  adversaries: {row['adv']['n']}", ""]
        if copied:
            lines += [f"NOTE: the {' and '.join(copied)} group is COPIED from the val scene, "
                      "not generated; it measures autoencoder reconstruction only.", ""]
        lines += ["| group | " + " | ".join(AGENT_KEYS) + " | mean |",
                  "|---|" + "---:|" * (len(AGENT_KEYS) + 1)]
        for group in ("agent", "adv"):
            tag = f"{group} (copied)" if group in copied else group
            cells = [f"{row[group][k]:.3f}" for k in AGENT_KEYS] + [f"{row[group]['mean']:.3f}"]
            lines.append(f"| {tag} | " + " | ".join(cells) + " |")
            floors = [f"*{row[group][f'{k}_floor']:.3f}*" for k in AGENT_KEYS]
            floors.append(f"*{row[group]['mean_floor']:.3f}*")
            lines.append(f"| {tag} floor | " + " | ".join(floors) + " |")
        lines += ["", "| " + " | ".join(MAP_KEYS) + " |", "|" + "---:|" * len(MAP_KEYS),
                  "| " + " | ".join(f"{row['map'][k]:.3f}" for k in MAP_KEYS) + " |", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_ldm_adv_ddpo")
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--num-gt-samples", type=int, default=43658)
    ap.add_argument("--floor-reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/scenes/scene_metrics.json")
    args = ap.parse_args()

    cfg_root = compose_eval_cfg(args.config_name, [])
    dataset_cfg = prepare_ldm_cfg(cfg_root).dataset

    reference = load_reference(
        dataset_cfg,
        ROOT / "metadata" / "waymo_goal_val_eval_set.pkl",
        ROOT / "data" / "advscene_preprocess_waymo" / "val",
        args.num_gt_samples,
    )
    ref, ref_goal = reference_stats(reference)
    print(f"[score] reference: {len(reference)} scenes, {len(ref['length'])} vehicles")

    rng = np.random.default_rng(args.seed)
    results = {}
    for cache in args.caches:
        cache_dir = Path(cache)
        scenes, adv_pos, manifest = load_cache(cache_dir)
        agent, adv = group_stats(scenes, adv_pos)

        threshold = adv_goal_threshold(manifest["adv_cond_target"], dataset_cfg)
        ref_adv = {k: ref[k][ref_goal[k] >= threshold] for k in AGENT_KEYS}
        print(f"[score] {cache_dir.name}: adv reference is goal>={threshold:.0f}m "
              f"({100 * len(ref_adv['length']) / len(ref['length']):.1f}% of vehicles)")

        results[cache_dir.name] = {
            "manifest": manifest,
            "num_scenes": len(scenes),
            "adv_goal_threshold": threshold,
            "map": {k: float(v) for k, v in mh.compute_lane_metrics(
                samples=scenes, gt_samples=reference).items() if k in MAP_KEYS},
            "agent": jsd_group(agent, ref, rng, args.floor_reps),
            "adv": jsd_group(adv, ref_adv, rng, args.floor_reps),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"num_gt_samples": len(reference), "rows": results}, indent=1))
    out.with_suffix(".md").write_text(render(results))
    print(f"[score] wrote {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
