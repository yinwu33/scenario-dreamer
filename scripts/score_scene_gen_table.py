"""Score every scene-generation sample set against one shared reference set.

Runs the SAME metric code ``metrics.py`` runs (utils.metrics_helpers), but loads
and converts the ground-truth pool once for all rows instead of once per row.
The two ground-truth statistic collectors are memoised on the reference list's
identity, so the metric definitions themselves are untouched.

Usage (env vars from scripts/define_env_variables.sh must be set):
  .venv/bin/python scripts/score_scene_gen_table.py --num-samples 1000
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from critical_scene.ldm_adv_eval import compose_eval_cfg, prepare_ldm_cfg
from scripts.gen_scene_gen_samples import DDPO_RUNS
from utils import metrics_helpers as mh
from utils.goal_runtime import prepare_scene


def load_generated(samples_path: Path, num_samples: int) -> list:
    paths = sorted(p for p in samples_path.iterdir() if p.suffix == ".pkl")[:num_samples]
    if len(paths) < num_samples:
        raise ValueError(f"{samples_path}: {len(paths)} pickles, need {num_samples}")
    out = []
    for p in tqdm(paths, desc=f"load {samples_path.name}"):
        with open(p, "rb") as f:
            sample = mh.convert_data_to_unified_format(pickle.load(f), dataset_name="waymo")
        if len(sample["G"]) > 0:
            out.append(sample)
    return out


def load_reference(cfg_dataset, eval_set: Path, gt_dir: Path, num_gt: int) -> list:
    with open(eval_set, "rb") as f:
        filenames = pickle.load(f)["files"][:num_gt]
    out = []
    for name in tqdm(filenames, desc="load reference"):
        with open(gt_dir / name, "rb") as f:
            raw = pickle.load(f)
        # gt_format=goal: the same prepare_scene the goal dataset runs, so the
        # reference agent set matches what the models were trained on.
        scene = prepare_scene(raw, cfg_dataset)
        raw = dict(raw)
        raw["agent_states"] = scene["agent_states"]
        raw["agent_types"] = scene["agent_types"]
        out.append(mh.convert_data_to_unified_format(raw, dataset_name="waymo_gt"))
    return out


def memoise_reference_stats(reference: list) -> None:
    """Cache the two ground-truth collectors for THIS reference list only.

    Identity ('is') against a list held alive for the whole run, so a generated
    set can never hit the cache."""
    for fn_name in ("_collect_urban_planning_stats", "_collect_agent_stats"):
        original = getattr(mh, fn_name)
        cache = {}

        def wrapped(dataset, _original=original, _cache=cache):
            if dataset is reference:
                if "v" not in _cache:
                    _cache["v"] = _original(dataset)
                return _cache["v"]
            return _original(dataset)

        setattr(mh, fn_name, wrapped)


def score(samples: list, reference: list) -> dict:
    out = {}
    out.update(mh.compute_lane_metrics(samples=samples, gt_samples=reference))
    out.update(mh.compute_agent_metrics(samples=samples, gt_samples=reference))
    if mh.has_goals(samples[0]["vehicles"]):
        out.update(mh.compute_goal_metrics(samples=samples, gt_samples=reference))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_ldm_adv_ddpo")
    ap.add_argument("--gen-root", default="data/scene_gen_table")
    ap.add_argument("--sd-samples",
                    default="data/checkpoints/scenario_dreamer_ldm_large_waymo/"
                            "initial_scene_advscene_fair10k_samples")
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--num-gt-samples", type=int, default=43658)
    ap.add_argument("--out", default="data/scene_gen_table/metrics.json")
    args = ap.parse_args()

    cfg_root = compose_eval_cfg(args.config_name, [])
    ldm_cfg = prepare_ldm_cfg(cfg_root)

    reference = load_reference(
        ldm_cfg.dataset,
        ROOT / "metadata" / "waymo_goal_val_eval_set.pkl",
        ROOT / "data" / "advscene_preprocess_waymo" / "val",
        args.num_gt_samples,
    )
    print(f"[score] reference: {len(reference)} scenes")
    memoise_reference_stats(reference)

    rows = {"scenario_dreamer": Path(args.sd_samples)}
    rows["advscene_base"] = Path(args.gen_root) / "base"
    for run in DDPO_RUNS:
        rows[run] = Path(args.gen_root) / run

    results = {}
    for name, path in rows.items():
        print(f"\n[score] ===== {name} ({path})")
        samples = load_generated(path, args.num_samples)
        print(f"[score] usable: {len(samples)}")
        results[name] = {k: float(v) for k, v in score(samples, reference).items()}
        results[name]["n_scenes"] = len(samples)
        for k, v in results[name].items():
            print(f"    {k:26s} {v:.4f}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"num_samples": args.num_samples,
                       "num_gt_samples": len(reference),
                       "rows": results}, f, indent=2)

    print(f"\n[score] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
