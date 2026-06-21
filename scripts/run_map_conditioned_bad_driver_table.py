"""Generate fixed-map critical-scene artifacts and benchmark bad_driver.

This is a resumable, chunked runner for the map-conditioned dm_fixed_map_agent_goal
experiment:

  splits: train, val
  sources: original, base_diffusion_full, base_diffusion_one, ddpo_diffusion
  metrics: collision/offroad/reached-goal rates under bad_driver rollout

It avoids one giant diffusion/rollout batch, which is necessary for 2000-scene
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from critical_scene.benchmark import _build_reward, _compose_cfg, _mean_finite, _rate, _write_csv
from critical_scene.generate import (
    _dm_fixed_map_batch,
    _dm_fixed_map_policy,
    compose_cfg,
)
from critical_scene.schema import (
    SceneArtifactMetadata,
    artifact_payload,
    assert_same_map,
    batch_map_ids,
    batch_to_generated_scenes,
    lane_hashes,
)
from ddpo.interfaces import GeneratedScenes


SOURCES = ("original", "base_diffusion_full", "base_diffusion_one", "ddpo_diffusion")
METRIC_KEYS = (
    "reward",
    "criticality",
    "ego_collision",
    "ego_offroad",
    "init_invalid",
    "reached_goal",
    "ego_min_ttc",
    "goal_offlane_frac",
    "parking_mismatch_frac",
    "ego_adv_min_dist",
    "controlled_parking_frac",
)


def _chunks(values: list[int], size: int):
    for start in range(0, len(values), int(size)):
        yield start, values[start : start + int(size)]


def _as_cpu_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value).detach().cpu()


def _cat_scene_chunks(chunks: list[GeneratedScenes]) -> GeneratedScenes:
    agent_states = []
    agent_types = []
    agent_scene_idx = []
    lane_polylines = []
    lane_scene_idx = []
    controlled = []
    gt_parking = []
    scene_offset = 0
    has_controlled = any("controlled_mask" in c.meta for c in chunks)
    has_gt_parking = any("gt_parking_mask" in c.meta for c in chunks)

    for scenes in chunks:
        a_idx = _as_cpu_tensor(scenes.agent_scene_idx).long()
        l_idx = _as_cpu_tensor(scenes.meta["lane_scene_idx"]).long()
        agent_states.append(_as_cpu_tensor(scenes.agent_states))
        agent_types.append(_as_cpu_tensor(scenes.agent_types).long())
        agent_scene_idx.append(a_idx + scene_offset)
        lane_polylines.append(_as_cpu_tensor(scenes.lane_polylines))
        lane_scene_idx.append(l_idx + scene_offset)
        if has_controlled:
            controlled.append(
                _as_cpu_tensor(scenes.meta.get("controlled_mask", torch.zeros_like(a_idx, dtype=torch.bool))).bool()
            )
        if has_gt_parking:
            gt_parking.append(
                _as_cpu_tensor(scenes.meta.get("gt_parking_mask", torch.zeros_like(a_idx, dtype=torch.bool))).bool()
            )
        scene_offset += int(scenes.num_scenes)

    meta = {"lane_scene_idx": torch.cat(lane_scene_idx, dim=0)}
    if has_controlled:
        meta["controlled_mask"] = torch.cat(controlled, dim=0)
    if has_gt_parking:
        meta["gt_parking_mask"] = torch.cat(gt_parking, dim=0)

    return GeneratedScenes(
        agent_states=torch.cat(agent_states, dim=0),
        agent_types=torch.cat(agent_types, dim=0),
        agent_scene_idx=torch.cat(agent_scene_idx, dim=0),
        lane_polylines=torch.cat(lane_polylines, dim=0),
        num_scenes=scene_offset,
        meta=meta,
    )


def _slice_scenes(scenes: GeneratedScenes, start: int, end: int) -> GeneratedScenes:
    a_idx = _as_cpu_tensor(scenes.agent_scene_idx).long()
    l_idx = _as_cpu_tensor(scenes.meta["lane_scene_idx"]).long()
    a_sel = (a_idx >= start) & (a_idx < end)
    l_sel = (l_idx >= start) & (l_idx < end)

    meta = {"lane_scene_idx": l_idx[l_sel] - start}
    if "controlled_mask" in scenes.meta:
        meta["controlled_mask"] = _as_cpu_tensor(scenes.meta["controlled_mask"]).bool()[a_sel]
    if "gt_parking_mask" in scenes.meta:
        meta["gt_parking_mask"] = _as_cpu_tensor(scenes.meta["gt_parking_mask"]).bool()[a_sel]

    return GeneratedScenes(
        agent_states=_as_cpu_tensor(scenes.agent_states)[a_sel],
        agent_types=_as_cpu_tensor(scenes.agent_types).long()[a_sel],
        agent_scene_idx=a_idx[a_sel] - start,
        lane_polylines=_as_cpu_tensor(scenes.lane_polylines)[l_sel],
        num_scenes=end - start,
        meta=meta,
    )


def _save_artifact(path: Path, scenes: GeneratedScenes, metadata: SceneArtifactMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact_payload(scenes, metadata), path)


def _generate_source(
    *,
    cfg_root,
    split: str,
    source: str,
    scene_ids: list[int],
    out_path: Path,
    batch_size: int,
    device: str,
    seed: int,
    base_ckpt: str,
    ddpo_ckpt: str,
    base_one_control_agent_num: int,
    ddpo_control_agent_num: int,
    force: bool,
    config_name: str,
) -> Path:
    if out_path.exists() and not force:
        print(f"[generate] skip existing {out_path}", flush=True)
        return out_path

    cfg = cfg_root.ddpo
    chunks: list[GeneratedScenes] = []
    resolved_ckpt = None
    policy = None
    if source == "base_diffusion_full":
        resolved_ckpt = base_ckpt
        policy = _dm_fixed_map_policy(
            cfg_root,
            cfg,
            base_ckpt,
            device,
            control_agent_num=-1,
            control_ego=False,
        )
    elif source == "base_diffusion_one":
        resolved_ckpt = base_ckpt
        policy = _dm_fixed_map_policy(
            cfg_root,
            cfg,
            base_ckpt,
            device,
            control_agent_num=base_one_control_agent_num,
            control_ego=False,
        )
    elif source == "ddpo_diffusion":
        resolved_ckpt = ddpo_ckpt
        policy = _dm_fixed_map_policy(
            cfg_root,
            cfg,
            ddpo_ckpt,
            device,
            control_agent_num=ddpo_control_agent_num,
            control_ego=False,
        )

    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    scene_id_out: list[int] = []
    map_id_out: list[int] = []
    lane_hash_out: list[str] = []
    for chunk_id, (_, chunk_ids) in enumerate(_chunks(scene_ids, batch_size), start=1):
        print(
            f"[generate] split={split} source={source} chunk={chunk_id} "
            f"scenes={chunk_ids[0]}..{chunk_ids[-1]}",
            flush=True,
        )
        if source == "base_diffusion_one":
            batch, valid_scene_ids = _dm_fixed_map_batch(
                cfg_root, split, chunk_ids, control_agent_num=base_one_control_agent_num
            )
        elif source == "ddpo_diffusion":
            batch, valid_scene_ids = _dm_fixed_map_batch(
                cfg_root, split, chunk_ids, control_agent_num=ddpo_control_agent_num
            )
        else:
            batch, valid_scene_ids = _dm_fixed_map_batch(cfg_root, split, chunk_ids)

        if source == "original":
            scenes = batch_to_generated_scenes(batch, cfg_root.dm_fixed_map_agent_goal.dataset)
        else:
            scenes, _ = policy.sample(batch)

        chunks.append(scenes)
        scene_id_out.extend(valid_scene_ids)
        map_id_out.extend(batch_map_ids(batch))
        lane_hash_out.extend(lane_hashes(scenes))

    if policy is not None:
        del policy
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

    scenes_all = _cat_scene_chunks(chunks)
    metadata = SceneArtifactMetadata(
        source=source,
        scene_ids=scene_id_out,
        split=split,
        generator_ckpt=str(resolved_ckpt) if resolved_ckpt is not None else None,
        sampler="ddpm" if source != "original" else None,
        planner_target="bad_driver",
        seed=seed,
        same_map=True,
        map_ids=map_id_out,
        lane_hashes=lane_hash_out,
        config_name=config_name,
    )
    _save_artifact(out_path, scenes_all, metadata)
    print(f"[generate] wrote {out_path}", flush=True)
    return out_path


def _summarize(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "reward": _mean_finite(metrics["reward"]),
        "criticality": _mean_finite(metrics["criticality"]),
        "reached_goal_rate": _rate(metrics["reached_goal"]),
        "ego_collision_rate": _rate(metrics["ego_collision"]),
        "ego_offroad_rate": _rate(metrics["ego_offroad"]),
        "init_invalid_rate": _rate(metrics["init_invalid"]),
        "ego_min_ttc": _mean_finite(metrics["ego_min_ttc"]),
        "goal_offlane_frac": _mean_finite(metrics["goal_offlane_frac"]),
        "parking_mismatch_frac": _mean_finite(metrics["parking_mismatch_frac"]),
        "ego_adv_min_dist": _mean_finite(metrics.get("ego_adv_min_dist", [])),
        "controlled_parking_frac": _mean_finite(metrics.get("controlled_parking_frac", [])),
    }


def _concat_metric_chunks(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    out = {}
    for key in METRIC_KEYS:
        vals = [m[key] for m in chunks if key in m]
        if vals:
            out[key] = np.concatenate(vals, axis=0)
    return out


def _benchmark_artifact(
    *,
    cfg_root,
    artifact_path: Path,
    out_dir: Path,
    source_name: str,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    summary_path = out_dir / "summary.json"
    per_scene_path = out_dir / "per_scene.csv"
    if summary_path.exists() and per_scene_path.exists() and not force:
        print(f"[benchmark] skip existing {out_dir}", flush=True)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["source"] = source_name
        return payload

    artifact = torch.load(artifact_path, map_location="cpu")
    scenes: GeneratedScenes = artifact["scenes"]
    metadata = artifact["metadata"]
    reward = _build_reward(cfg_root)
    metric_chunks = []

    for batch_id, start in enumerate(range(0, int(scenes.num_scenes), batch_size), start=1):
        end = min(start + batch_size, int(scenes.num_scenes))
        print(
            f"[benchmark] split={metadata['split']} source={source_name} "
            f"batch={batch_id} scenes={start}..{end - 1}",
            flush=True,
        )
        metric_chunks.append(reward.evaluate(_slice_scenes(scenes, start, end)))

    metrics = _concat_metric_chunks(metric_chunks)
    rows = []
    for i, scene_id in enumerate(metadata["scene_ids"]):
        row = {
            "source": source_name,
            "split": metadata["split"],
            "scene_id": int(scene_id),
            "map_id": int(metadata["map_ids"][i]),
            "lane_hash": metadata["lane_hashes"][i],
        }
        for key in METRIC_KEYS:
            if key in metrics:
                value = float(metrics[key][i])
                row[key] = value if np.isfinite(value) else float("nan")
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(per_scene_path, rows)
    payload = {
        "artifact": str(artifact_path),
        "source": source_name,
        "split": metadata["split"],
        "num_scenes": int(scenes.num_scenes),
        "summary": _summarize(metrics),
    }
    counts = torch.bincount(_as_cpu_tensor(scenes.agent_scene_idx).long(), minlength=int(scenes.num_scenes))
    payload["summary"].update(
        {
            "num_agents_mean": float(counts.float().mean().item()),
            "num_agents_min": int(counts.min().item()),
            "num_agents_max": int(counts.max().item()),
        }
    )
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[benchmark] wrote {summary_path}", flush=True)
    return payload


def _write_table(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    rows = []
    for payload in summaries:
        summary = payload["summary"]
        rows.append(
            {
                "split": payload["split"],
                "setup": payload["source"],
                "num_scenes": payload["num_scenes"],
                "avg_agents": summary.get("num_agents_mean", float("nan")),
                "reward_mean": summary["reward"],
                "criticality_mean": summary.get("criticality", float("nan")),
                "collision_rate": summary["ego_collision_rate"],
                "offroad_rate": summary["ego_offroad_rate"],
                "reached_goal_rate": summary["reached_goal_rate"],
            }
        )

    table_csv = out_dir / "table.csv"
    with table_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "setup",
                "num_scenes",
                "avg_agents",
                "reward_mean",
                "criticality_mean",
                "collision_rate",
                "offroad_rate",
                "reached_goal_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| split | setup | num_scenes | avg_agents | reward_mean | criticality_mean | collision_rate | offroad_rate | reached_goal_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['setup']} | {row['num_scenes']} | "
            f"{row['avg_agents']:.2f} | {row['reward_mean']:.4f} | "
            f"{row['criticality_mean']:.4f} | "
            f"{row['collision_rate']:.4f} | {row['offroad_rate']:.4f} | "
            f"{row['reached_goal_rate']:.4f} |"
        )
    table_md = out_dir / "table.md"
    table_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[table] wrote {table_csv} and {table_md}", flush=True)


def _load_available_summaries(benchmark_dir: Path, splits: list[str]) -> list[dict[str, Any]]:
    summaries = []
    for split in splits:
        for source in SOURCES:
            path = benchmark_dir / split / source / "summary.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["source"] = source
                summaries.append(payload)
    return summaries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_critical_scene_map_conditioned_dm_goal")
    ap.add_argument("--out-dir", default="data/critical_scene/map_conditioned_bad_driver_2000")
    ap.add_argument("--num-scenes", type=int, default=2000)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    ap.add_argument("--generation-batch-size", type=int, default=16)
    ap.add_argument("--benchmark-batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--planner-device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-ckpt", default="data/checkpoints/scenario_dreamer_dm_fixed_map_agent_goal_train/last.ckpt")
    ap.add_argument(
        "--ddpo-ckpt",
        default=(
            "data/critical_scene/critical_scene_ddpo_dm_fixed_map_agent_goal_ddpm_bad_driver_agent_only/"
            "critical_scene_ddpo_dm_fixed_map_agent_goal_ddpm_bad_driver_agent_only_01700.ckpt"
        ),
    )
    ap.add_argument("--ddpo-control-agent-num", type=int, default=1)
    ap.add_argument("--base-one-control-agent-num", type=int, default=1)
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-generation", action="store_true")
    ap.add_argument("--force-benchmark", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    artifact_dir = out_dir / "artifacts"
    benchmark_dir = out_dir / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_root = compose_cfg(args.config_name, [])
    bench_cfg = _compose_cfg(args.config_name, "bad_driver", [])
    OmegaConf.set_struct(bench_cfg.ddpo.planner, False)
    bench_cfg.ddpo.planner.device = args.planner_device
    OmegaConf.set_struct(bench_cfg.ddpo.planner, True)

    scene_ids = list(range(int(args.num_scenes)))
    artifacts: dict[tuple[str, str], Path] = {}
    selected_sources = tuple(args.sources)
    for split in args.splits:
        for source in selected_sources:
            path = artifact_dir / split / f"{source}.pt"
            artifacts[(split, source)] = path
            if not args.skip_generation:
                _generate_source(
                    cfg_root=cfg_root,
                    split=split,
                    source=source,
                    scene_ids=scene_ids,
                    out_path=path,
                    batch_size=args.generation_batch_size,
                    device=args.device,
                    seed=args.seed,
                    base_ckpt=args.base_ckpt,
                    ddpo_ckpt=args.ddpo_ckpt,
                    base_one_control_agent_num=args.base_one_control_agent_num,
                    ddpo_control_agent_num=args.ddpo_control_agent_num,
                    force=args.force or args.force_generation,
                    config_name=args.config_name,
                )

    summaries = []
    if not args.skip_benchmark:
        for split in args.splits:
            if len(selected_sources) > 1:
                split_artifacts = [
                    torch.load(artifacts[(split, s)], map_location="cpu")
                    for s in selected_sources
                ]
                reference = split_artifacts[0]["metadata"]
                for artifact in split_artifacts[1:]:
                    assert_same_map(reference, artifact["metadata"])
                del split_artifacts
                gc.collect()

            for source in selected_sources:
                payload = _benchmark_artifact(
                    cfg_root=bench_cfg,
                    artifact_path=artifacts[(split, source)],
                    out_dir=benchmark_dir / split / source,
                    source_name=source,
                    batch_size=args.benchmark_batch_size,
                    force=args.force or args.force_benchmark,
                )
                summaries.append(payload)
        _write_table(out_dir, _load_available_summaries(benchmark_dir, list(args.splits)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
