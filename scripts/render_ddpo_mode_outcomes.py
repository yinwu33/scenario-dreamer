#!/usr/bin/env python
"""Render outcome-stratified DDPO samples for every planner cell and init mode.

Each selected generated scene is written as a rollout GIF plus a static PNG with
the complete trajectories.  Selection is resumable because accepted scenes are
saved before rendering; a killed run continues from the next candidate batch.
Only valid ``hierarchical_v4`` samples (``tier > 0``) are retained.  Both media
formats contain only the plot: no title, annotation, or outer margin.

The three outcome buckets are deliberately disjoint:

* ``succ``: reached the goal without colliding with the generated adversary;
* ``coll``: collided with the generated adversary, but the ego was not at fault;
* ``ego_fault_coll``: collided with the generated adversary and the current
  front-face/moving-ego predicate assigns fault to the ego.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from critical_scene.ldm_adv_eval import (
    build_reward,
    cat_payloads,
    compose_eval_cfg,
    payload_to_scenes,
    prepare_ldm_cfg,
    scenes_to_payload,
    slice_payload,
)
from critical_scene.metrics_common import ego_goal_dist
from ddpo.conditioning import LDMAdvConditioningPool
from utils.viz import CONTROL_COLOR, render_rollout, render_rollout_frames, save_gif
from models.scenario_dreamer_ldm_adv import ScenarioDreamerLDMAdv
from sim.scenes import GeneratedScenes, batched_lane_graphs, single_adv_local_idx


CELLS = (
    ("ppo-idm", "ppo_normal", "idm"),
    ("idm-idm", "idm", "idm"),
    ("pdm-idm", "pdm", "idm"),
    ("ppo-ppo_norm", "ppo_normal", "ppo_normal"),
    ("idm-ppo_norm", "idm", "ppo_normal"),
    ("pdm-ppo_norm", "pdm", "ppo_normal"),
    ("ppo-ppo_caution", "ppo_normal", "ppo_caution"),
    ("idm-ppo_caution", "idm", "ppo_caution"),
    ("pdm-ppo_caution", "pdm", "ppo_caution"),
    ("ppo-ppo_aggressive", "ppo_normal", "ppo_aggressive"),
    ("idm-ppo_aggressive", "idm", "ppo_aggressive"),
    ("pdm-ppo_aggressive", "pdm", "ppo_aggressive"),
)
MODES = ("init_scene", "init_agent", "init_adv")
OUTCOMES = ("succ", "coll", "ego_fault_coll")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out-dir",
        default="data/critical_scene/v4kl5_mode_outcomes",
    )
    p.add_argument("--cells", nargs="+", choices=[c[0] for c in CELLS], default=None)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-candidates", type=int, default=50000)
    p.add_argument("--pool-size", type=int, default=40000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--split", default="train")
    p.add_argument("--focus-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=90)
    p.add_argument("--config-name", default="config_ldm_adv_ddpo")
    return p.parse_args()


def _checkpoint(cell: str) -> Path:
    run = f"critical_scene_ddpo_ldm_adv_ddim_{cell}_v4kl5_hier_v4"
    return ROOT / "data" / "critical_scene" / run / f"{run}_00500.ckpt"


def _cell_cfg(args, cell: str, sut: str, env: str):
    prior = ROOT / "data" / "headroom_probe" / f"context_prior_{cell}.json"
    return compose_eval_cfg(
        args.config_name,
        [
            "ddpo/reward=hierarchical_v4",
            f"planner@ddpo.planner.sut={sut}",
            f"planner@ddpo.planner.env={env}",
            f"planner@ddpo.planner.adv={env}",
            f"experiment.planner_name={cell}_v4kl5",
            f"ddpo.seed={args.seed}",
            f"ddpo.context_prior.path={prior}",
            f"ddpo.context_prior.focus_frac={args.focus_frac}",
            "ddpo.simulator.path_conflict.skip_rollout=false",
        ],
    )


def _load_model(cfg_root, ldm_cfg, ckpt_path: Path, device: str):
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    model = ScenarioDreamerLDMAdv(ldm_cfg, cfg_root.ae_goal)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    diff_state = {
        key.removeprefix("diff_model."): value
        for key, value in state_dict.items()
        if key.startswith("diff_model.")
    }
    model.diff_model.load_state_dict(diff_state, strict=True)
    del checkpoint, state_dict, diff_state
    # EMA belongs to the pre-DDPO Lightning training path.  The numbered DDPO
    # checkpoint already is the policy and intentionally has no EMA shadow.
    del model.ema
    model.to(device).eval()
    print(f"[model] loaded {ckpt_path}", flush=True)
    return model


def _type_ids(types: torch.Tensor) -> torch.Tensor:
    return types.argmax(dim=-1).long() if types.ndim == 2 else types.long()


@torch.no_grad()
def _sample_mode(model, conditioning, mode: str) -> GeneratedScenes:
    data, _ = model.forward(conditioning, mode, 0, visualize=False)
    agent_batch = data["agent"].batch
    lane_batch = data["lane"].batch
    adv_batch = data["adv"].batch
    base_count = data["agent"].x.shape[0]
    states = torch.cat([data["agent"].x, data["adv"].x], dim=0)
    types = torch.cat(
        [_type_ids(data["agent"].type), _type_ids(data["adv"].type)], dim=0
    )
    scene_idx = torch.cat([agent_batch, adv_batch], dim=0)
    generated = torch.zeros(states.shape[0], dtype=torch.bool, device=states.device)
    generated[base_count:] = True
    num_scenes = int(data.batch_size)
    meta = {
        "lane_scene_idx": lane_batch,
        "gen_agent_mask": generated,
        "lane_graph": batched_lane_graphs(
            data["lane", "to", "lane"].edge_index,
            data["lane", "to", "lane"].type,
            lane_batch,
            num_scenes,
        ),
    }
    if "cond" in data["adv"]:
        adv_cond = torch.full(
            (num_scenes, data["adv"].cond.shape[1]), -1, dtype=torch.long
        )
        adv_cond[adv_batch.cpu()] = data["adv"].cond.cpu()
        meta["adv_cond"] = adv_cond
    return GeneratedScenes(
        agent_states=states,
        agent_types=types,
        agent_scene_idx=scene_idx,
        lane_polylines=data["lane"].x,
        num_scenes=num_scenes,
        adv_local_idx=single_adv_local_idx(generated, scene_idx, num_scenes),
        meta=meta,
    )


def _outcome(metrics: dict, index: int) -> str | None:
    if not metrics["tier"][index] > 0:
        return None
    collision = bool(metrics["ego_collision"][index] > 0)
    fault = bool(metrics["ego_fault_collision"][index] > 0)
    reached = bool(metrics["reached_goal"][index] > 0)
    if fault:
        return "ego_fault_coll"
    if collision:
        return "coll"
    if reached:
        return "succ"
    return None


def _numeric_metrics(metrics: dict, index: int) -> dict:
    out = {}
    for key, values in metrics.items():
        arr = np.asarray(values)
        if arr.ndim != 1 or arr.shape[0] <= index or arr.dtype.kind not in "fiub":
            continue
        value = float(arr[index])
        out[key] = value if np.isfinite(value) else None
    return out


def _atomic_torch_save(value, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _load_progress(path: Path, *, cell: str, mode: str, count: int) -> dict:
    if not path.exists():
        return {
            "version": 1,
            "cell": cell,
            "mode": mode,
            "count": count,
            "next_batch": 0,
            "screened": 0,
            "selected": {outcome: [] for outcome in OUTCOMES},
            "records": {outcome: [] for outcome in OUTCOMES},
        }
    progress = torch.load(path, map_location="cpu", weights_only=False)
    expected = (cell, mode, count)
    actual = (progress["cell"], progress["mode"], int(progress["count"]))
    if actual != expected:
        raise ValueError(f"progress mismatch at {path}: expected {expected}, got {actual}")
    return progress


def _complete(progress: dict, count: int) -> bool:
    return all(len(progress["selected"][outcome]) >= count for outcome in OUTCOMES)


def _seed_batch(seed: int, cell_index: int, mode_index: int, batch_index: int) -> int:
    value = int(seed + 1_000_003 * cell_index + 10_007 * mode_index + batch_index)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    return value


def _screen_mode(
    args,
    *,
    cell: str,
    cell_index: int,
    mode: str,
    mode_index: int,
    model,
    pool,
    reward,
    mode_dir: Path,
) -> None:
    mode_dir.mkdir(parents=True, exist_ok=True)
    progress_path = mode_dir / "selection.pt"
    progress = _load_progress(progress_path, cell=cell, mode=mode, count=args.count)
    if _complete(progress, args.count):
        print(f"[screen] {cell}/{mode} already complete", flush=True)
        return

    while not _complete(progress, args.count):
        if progress["screened"] >= args.max_candidates:
            counts = {k: len(v) for k, v in progress["selected"].items()}
            raise RuntimeError(
                f"{cell}/{mode}: exhausted {progress['screened']} candidates with {counts}"
            )
        batch_index = int(progress["next_batch"])
        batch_seed = _seed_batch(args.seed, cell_index, mode_index, batch_index)
        pool.rng = np.random.default_rng(batch_seed)
        slots = pool._draw_slots(args.batch_size)
        conditioning = pool.batch_from_indices(slots)
        scenes = _sample_mode(model, conditioning, mode)
        payload = scenes_to_payload(scenes)
        metrics = reward.evaluate(scenes)
        # The conditioning pool guarantees a driving real ego.  Generated modes
        # can move its goal slightly, so retain the paper's explicit 10 m filter.
        driving = ego_goal_dist(payload) >= 10.0
        for row in range(args.batch_size):
            if not driving[row]:
                continue
            outcome = _outcome(metrics, row)
            if outcome is None or len(progress["selected"][outcome]) >= args.count:
                continue
            one = scenes_to_payload(slice_payload(payload, row, row + 1))
            pool_slot = int(slots[row])
            record = {
                "rank": len(progress["selected"][outcome]),
                "outcome": outcome,
                "batch": batch_index,
                "batch_seed": batch_seed,
                "row": row,
                "pool_slot": pool_slot,
                "dataset_scene_idx": int(pool.resolved_scene_idx[pool_slot]),
                "metrics": _numeric_metrics(metrics, row),
            }
            progress["selected"][outcome].append(one)
            progress["records"][outcome].append(record)

        progress["next_batch"] = batch_index + 1
        progress["screened"] += args.batch_size
        _atomic_torch_save(progress, progress_path)
        counts = {k: len(v) for k, v in progress["selected"].items()}
        print(
            f"[screen] {cell}/{mode} candidates={progress['screened']} selected={counts}",
            flush=True,
        )
        del conditioning, scenes, payload, metrics
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()


def _render_mode(args, cfg_root, ldm_cfg, *, cell: str, mode: str, mode_dir: Path) -> list[dict]:
    complete_path = mode_dir / "COMPLETE"
    progress = _load_progress(
        mode_dir / "selection.pt", cell=cell, mode=mode, count=args.count
    )
    if not _complete(progress, args.count):
        raise RuntimeError(f"cannot render incomplete selection for {cell}/{mode}")
    if complete_path.exists():
        return json.loads((mode_dir / "manifest.json").read_text(encoding="utf-8"))["samples"]

    payloads = []
    labels = []
    for outcome in OUTCOMES:
        for rank in range(args.count):
            payloads.append(progress["selected"][outcome][rank])
            labels.append((outcome, rank, progress["records"][outcome][rank]))
    scenes = payload_to_scenes(cat_payloads(payloads))
    reward = build_reward(cfg_root, ldm_cfg, num_workers=0, batch_size=len(payloads))
    metrics = reward.evaluate(scenes, record_trajectories=True)

    states = scenes.agent_states.detach().cpu().numpy()
    types = scenes.agent_types.detach().cpu().numpy()
    agent_scene = scenes.agent_scene_idx.detach().cpu().numpy()
    lanes = np.asarray(scenes.lane_polylines)
    lane_scene = scenes.meta["lane_scene_idx"].detach().cpu().numpy()
    generated = scenes.meta["gen_agent_mask"].detach().cpu().numpy()
    rows = []
    for index, (expected_outcome, rank, selection) in enumerate(labels):
        actual_outcome = _outcome(metrics, index)
        if actual_outcome != expected_outcome:
            raise RuntimeError(
                f"{cell}/{mode} sample {index}: screened as {expected_outcome}, "
                f"re-rendered as {actual_outcome}"
            )
        outcome_dir = mode_dir / expected_outcome
        outcome_dir.mkdir(parents=True, exist_ok=True)
        dataset_id = int(selection["dataset_scene_idx"])
        stem = f"{rank:02d}_dataset{dataset_id}_batch{selection['batch']:04d}_row{selection['row']:03d}"
        gif_path = outcome_dir / f"{stem}.gif"
        png_path = outcome_dir / f"{stem}.png"
        scene_agents = agent_scene == index
        colors = [CONTROL_COLOR if flag else None for flag in generated[scene_agents]]
        components = {
            key: values[index]
            for key, values in metrics.items()
            if isinstance(values, np.ndarray)
            and values.ndim == 1
            and values.dtype.kind in "fiub"
        }
        kwargs = dict(
            agent_states=states[scene_agents],
            agent_types=types[scene_agents],
            agent_colors=colors,
            reward=float(metrics["reward"][index]),
            ego_collision=bool(metrics["ego_collision"][index] > 0),
            ego_offroad=bool(metrics["ego_offroad"][index] > 0),
            init_invalid=bool(metrics["init_invalid"][index] > 0),
            ego_min_ttc=float(metrics["ego_min_ttc"][index]),
            goal_offlane_frac=float(metrics["goal_offlane_frac"][index]),
            components=components,
            title="",
        )
        trajectory = metrics["trajectories"][index]
        scene_lanes = lanes[lane_scene == index]
        frames = render_rollout_frames(
            trajectory,
            scene_lanes,
            max_frames=args.max_frames,
            annotate=False,
            **kwargs,
        )
        save_gif(frames, str(gif_path), fps=args.fps)
        fig = render_rollout(
            trajectory,
            scene_lanes,
            final_boxes_only=True,
            annotate=False,
            **kwargs,
        )
        fig.savefig(png_path, dpi=300, pad_inches=0)
        plt.close(fig)

        row = {
            **selection,
            "cell": cell,
            "mode": mode,
            "outcome": expected_outcome,
            "gif": str(gif_path.relative_to(Path(args.out_dir))),
            "png": str(png_path.relative_to(Path(args.out_dir))),
            "metrics": _numeric_metrics(metrics, index),
        }
        (outcome_dir / f"{stem}.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        rows.append(row)
        print(f"[render] wrote {gif_path} + {png_path}", flush=True)

    manifest = {
        "cell": cell,
        "mode": mode,
        "count_per_outcome": args.count,
        "outcome_definitions": {
            "succ": "tier > 0 and reached_goal and not ego_collision",
            "coll": "tier > 0 and ego_collision and not ego_fault_collision",
            "ego_fault_coll": "tier > 0 and ego_fault_collision",
        },
        "samples": rows,
    }
    (mode_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    complete_path.write_text("complete\n", encoding="utf-8")
    reward.close()
    return rows


def _write_root_manifest(args, rows: list[dict]) -> None:
    out_dir = Path(args.out_dir)
    manifest = {
        "checkpoint_family": "*_v4kl5_hier_v4_00500.ckpt",
        "reward": "hierarchical_v4",
        "split": args.split,
        "context_prior_focus_frac": args.focus_frac,
        "count_per_cell_mode_outcome": args.count,
        "valid_definition": "hierarchical_v4 tier > 0",
        "rendering": "no title, annotation, or outer margin",
        "outcome_definitions": {
            "succ": "tier > 0 and reached_goal and not ego_collision",
            "coll": "tier > 0 and ego_collision and not ego_fault_collision",
            "ego_fault_coll": "tier > 0 and ego_fault_collision (front-face/moving-ego predicate)",
        },
        "samples": rows,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = _parse()
    if args.count <= 0 or args.batch_size <= 0 or args.max_candidates < args.batch_size:
        raise ValueError("require count > 0 and max_candidates >= batch_size > 0")
    if args.workers > 0 and int(np.ceil(args.batch_size / 8)) < args.workers:
        raise ValueError("ceil(batch_size / 8) must be >= workers")
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("this 12-cell diffusion render requires an available CUDA device")

    chosen = set(args.cells or [c[0] for c in CELLS])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for cell_index, (cell, sut, env) in enumerate(CELLS):
        if cell not in chosen:
            continue
        cfg_root = _cell_cfg(args, cell, sut, env)
        ldm_cfg = prepare_ldm_cfg(cfg_root)
        pending = [
            mode
            for mode in args.modes
            if not (out_dir / cell / mode / "selection.pt").exists()
            or not _complete(
                _load_progress(
                    out_dir / cell / mode / "selection.pt",
                    cell=cell,
                    mode=mode,
                    count=args.count,
                ),
                args.count,
            )
        ]
        if pending:
            model = _load_model(cfg_root, ldm_cfg, _checkpoint(cell), args.device)
            cfg = cfg_root.ddpo
            pool = LDMAdvConditioningPool(
                ldm_cfg.dataset,
                split_name=args.split,
                pool_size=args.pool_size,
                device=args.device,
                seed=args.seed,
                min_ego_drive=float(cfg.min_ego_drive),
                prune_base_to_ego=bool(cfg.prune_base_to_ego),
                insert_adv_as_extra=bool(cfg.insert_adv_as_extra),
                adv_cond_target=cfg.adv_cond_target,
                context_prior=cfg.context_prior,
            )
            reward = build_reward(
                cfg_root, ldm_cfg, num_workers=args.workers, batch_size=args.batch_size
            )
            for mode_index, mode in enumerate(MODES):
                if mode not in args.modes:
                    continue
                _screen_mode(
                    args,
                    cell=cell,
                    cell_index=cell_index,
                    mode=mode,
                    mode_index=mode_index,
                    model=model,
                    pool=pool,
                    reward=reward,
                    mode_dir=out_dir / cell / mode,
                )
            reward.close()
            del reward, pool, model
            gc.collect()
            torch.cuda.empty_cache()

        for mode in args.modes:
            all_rows.extend(
                _render_mode(
                    args,
                    cfg_root,
                    ldm_cfg,
                    cell=cell,
                    mode=mode,
                    mode_dir=out_dir / cell / mode,
                )
            )
        _write_root_manifest(args, all_rows)
        print(f"[cell] {cell} complete", flush=True)

    expected = len(chosen) * len(args.modes) * len(OUTCOMES) * args.count
    if len(all_rows) != expected:
        raise RuntimeError(f"wrote {len(all_rows)} samples, expected {expected}")
    print(f"[done] wrote {len(all_rows)} GIF/PNG pairs under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
