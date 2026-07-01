#!/usr/bin/env python3
"""Map ldm_adv DDPO KL spike iterations back to dataset ids.

The train log only prints batch-mean ``kl_to_base``. This helper parses local
W&B stdout logs for high-KL iterations, reconstructs the deterministic
``LDMAdvConditioningPool`` index mapping, and replays the pool RNG used by
``sample_group_batch`` to list the dataset ``--id`` values in each spike batch.

It does not run the diffusion model, so it is CPU-friendly. The output ids are
directly consumable by ``test_scripts/test_rollout_ldm_adv.py --split train --id``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfgs.config import CONFIG_PATH
from ddpo.conditioning import LDMAdvConditioningPool
from ddpo.train_loop import _set_dataset_name
from utils.train_helpers import cache_latent_stats, set_latent_stats


_IT_RE = re.compile(r"\[it\s+(\d+)\].*?\bkl=([0-9.eE+-]+)")


def _load_cfg(config_name: str, overrides: list[str]):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(Path(CONFIG_PATH).resolve())):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def _iter_log_rows(paths: Iterable[str]):
    for path in paths:
        with open(path, errors="ignore") as f:
            for line in f:
                m = _IT_RE.search(line)
                if not m:
                    continue
                yield {
                    "log_path": path,
                    "iter": int(m.group(1)),
                    "kl": float(m.group(2)),
                    "line": line.strip(),
                }


def _top_spikes(log_globs: list[str], threshold: float, top_n: int):
    paths: list[str] = []
    for g in log_globs:
        paths.extend(glob.glob(g))
    rows = [r for r in _iter_log_rows(paths) if r["kl"] >= threshold]
    rows.sort(key=lambda r: (r["kl"], r["iter"]), reverse=True)
    return rows[:top_n]


def _build_pool(cfg_root, device: str):
    cfg = cfg_root.ddpo
    dataset_name = cfg_root.dataset_name.name
    _set_dataset_name(cfg_root.ldm_adv, dataset_name)
    _set_dataset_name(cfg_root.ae_goal, dataset_name)
    if not Path(cfg_root.ldm_adv.dataset.latent_stats_path).exists():
        cache_latent_stats(cfg_root.ldm_adv)
    ldm_cfg = set_latent_stats(cfg_root.ldm_adv)
    return LDMAdvConditioningPool(
        ldm_cfg.dataset,
        split_name=cfg.train_split,
        pool_size=cfg.pool_size,
        device=device,
        seed=cfg.seed,
        min_ego_drive=cfg.get("min_ego_drive", 10.0),
        prune_base_to_ego=cfg.get("prune_base_to_ego", False),
        insert_adv_as_extra=cfg.get("insert_adv_as_extra", False),
        adv_cond_target=cfg.get("adv_cond_target", None),
    )


def _batch_pool_slots(pool_len: int, seed: int, iteration: int, batch_size: int, group_size: int):
    rng = np.random.default_rng(seed)
    if group_size > 1:
        if batch_size % group_size != 0:
            raise ValueError(f"batch_size {batch_size} must be divisible by group_size {group_size}")
        num_groups = batch_size // group_size
        replace = num_groups > pool_len
        slots = None
        for _ in range(iteration + 1):
            groups = rng.choice(pool_len, size=num_groups, replace=replace)
            slots = np.repeat(groups, group_size)
        assert slots is not None
        group_ids = np.repeat(np.arange(num_groups), group_size)
        return slots.astype(np.int64), group_ids.astype(np.int64)

    slots = None
    for _ in range(iteration + 1):
        slots = rng.integers(0, pool_len, size=batch_size)
    assert slots is not None
    return slots.astype(np.int64), np.full(batch_size, -1, dtype=np.int64)


def _resolve_dataset_ids(pool: LDMAdvConditioningPool, slots: np.ndarray):
    dataset_ids = []
    raw_dataset_ids = []
    valid = []
    for slot in slots.tolist():
        d = pool._get(int(slot))
        ds_id = int(d["idx"]) if "idx" in d else -1
        dataset_ids.append(ds_id)
        raw_dataset_ids.append(int(pool.pool_indices[int(slot)]))
        valid.append(ds_id >= 0)
    return np.asarray(dataset_ids), np.asarray(raw_dataset_ids), np.asarray(valid, dtype=bool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_critical_scene_ldm_adv_ddpo")
    ap.add_argument("--override", dest="overrides", action="append", default=[])
    ap.add_argument(
        "--log-glob",
        action="append",
        default=["wandb/run-20260630_*/files/output.log"],
        help="Glob for stdout logs; repeatable.",
    )
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--iters", type=int, nargs="*", default=None, help="Explicit iterations to map.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-dir", default="data_analysis/ldm_adv_kl_spike_ids")
    args = ap.parse_args()

    cfg_root = _load_cfg(args.config_name, args.overrides)
    cfg = cfg_root.ddpo
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU for pool reconstruction.")
        args.device = "cpu"

    pool = _build_pool(cfg_root, args.device)
    batch_size = int(cfg.batch_size)
    group_size = int(cfg.get("group_size", 1))
    seed = int(cfg.seed)

    if args.iters is not None:
        spikes = [
            {"log_path": "<manual>", "iter": int(it), "kl": float("nan"), "line": ""}
            for it in args.iters
        ]
    else:
        spikes = _top_spikes(args.log_glob, args.threshold, args.top_n)
    if not spikes:
        raise SystemExit("no KL spikes matched")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "spike_batch_ids.csv"
    summary_csv = out_dir / "spike_summary.csv"

    rows = []
    summary_rows = []
    for spike_rank, spike in enumerate(spikes):
        slots, group_ids = _batch_pool_slots(len(pool), seed, int(spike["iter"]), batch_size, group_size)
        dataset_ids, raw_pool_dataset_ids, valid = _resolve_dataset_ids(pool, slots)
        unique_ids = list(dict.fromkeys(int(x) for x in dataset_ids.tolist()))
        summary_rows.append(
            {
                "rank": spike_rank,
                "iter": int(spike["iter"]),
                "kl": spike["kl"],
                "log_path": spike["log_path"],
                "num_unique_dataset_ids": len(unique_ids),
                "dataset_ids": " ".join(str(x) for x in unique_ids),
                "line": spike["line"],
            }
        )
        for sample, (slot, group, ds_id, raw_ds_id) in enumerate(
            zip(slots.tolist(), group_ids.tolist(), dataset_ids.tolist(), raw_pool_dataset_ids.tolist())
        ):
            rows.append(
                {
                    "spike_rank": spike_rank,
                    "iter": int(spike["iter"]),
                    "kl": spike["kl"],
                    "sample": int(sample),
                    "group": int(group),
                    "pool_slot": int(slot),
                    "dataset_id": int(ds_id),
                    "initial_pool_dataset_id": int(raw_ds_id),
                    "log_path": spike["log_path"],
                }
            )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[done] wrote {summary_csv}")
    print(f"[done] wrote {out_csv}")
    print("[top unique dataset ids]")
    for r in summary_rows[: min(5, len(summary_rows))]:
        print(f"iter={r['iter']} kl={r['kl']} ids={r['dataset_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
