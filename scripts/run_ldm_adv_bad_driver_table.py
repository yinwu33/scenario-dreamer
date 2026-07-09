"""Four-source ldm_adv critical-scene evaluation against bad_driver.

Generates paired scene artifacts for the sources in critical_scene.ldm_adv_eval
(original / base_gen / ddpo_gen / original_ddpo_adv, all from the same template
pool slots), rolls each out under the frozen bad_driver planner, and writes:

  <out-dir>/artifacts/<source>/chunk_XXXXX.pt   resumable per-chunk payloads
  <out-dir>/artifacts/<source>.pt               merged artifact + metadata
  <out-dir>/benchmark/<source>/per_scene.csv    one row per scene
  <out-dir>/benchmark/<source>/summary.json     aggregated rates
  <out-dir>/table.csv / table.md                cross-source comparison table

Usage (env vars from scripts/define_env_variables.sh must be set):
  .venv/bin/python scripts/run_ldm_adv_bad_driver_table.py \
      --num-scenes 1000 --chunk-size 32 --out-dir data/critical_scene/ldm_adv_bad_driver_eval
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from critical_scene.ldm_adv_eval import (
    SOURCES,
    benchmark_payload,
    build_metadata,
    build_policy,
    build_pool,
    build_reward,
    cat_payloads,
    compose_eval_cfg,
    generate_chunk,
    prepare_ldm_cfg,
    summarize,
    write_json,
    write_per_scene_csv,
    write_table,
)


def _chunks(n: int, size: int) -> list[list[int]]:
    return [list(range(s, min(s + size, n))) for s in range(0, n, size)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_critical_scene_ldm_adv_ddpo")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--out-dir", default="data/critical_scene/ldm_adv_bad_driver_eval")
    ap.add_argument("--num-scenes", type=int, default=1000)
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--benchmark-batch-size", type=int, default=64)
    ap.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--base-ckpt",
        default=None,
        help="base ldm_adv checkpoint (default: ddpo.ldm_adv_ckpt from the config)",
    )
    ap.add_argument(
        "--ddpo-ckpt",
        default="data/critical_scene/critical_scene_ddpo_ldm_adv_ddpm_bad_driver/last.ckpt",
    )
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--force-benchmark", action="store_true")
    args = ap.parse_args()

    cfg_root = compose_eval_cfg(args.config_name, args.overrides)
    ldm_cfg = prepare_ldm_cfg(cfg_root)
    base_ckpt = args.base_ckpt or str(cfg_root.ddpo.ldm_adv_ckpt)
    sources = tuple(args.sources)

    out_dir = Path(args.out_dir)
    artifact_dir = out_dir / "artifacts"
    benchmark_dir = out_dir / "benchmark"
    chunk_slots = _chunks(int(args.num_scenes), int(args.chunk_size))

    # ------------------------------------------------------------- generate
    if not args.skip_generation:
        pool = build_pool(
            cfg_root, ldm_cfg, split=args.split, pool_size=int(args.num_scenes), device=args.device
        )
        base_policy = None
        ddpo_policy = None
        for chunk_id, slots in enumerate(chunk_slots):
            paths = {s: artifact_dir / s / f"chunk_{chunk_id:05d}.pt" for s in sources}
            if all(p.exists() for p in paths.values()):
                print(f"[generate] skip existing chunk {chunk_id}", flush=True)
                continue
            if base_policy is None:
                base_policy = build_policy(cfg_root, ldm_cfg, ckpt=base_ckpt, device=args.device)
                ddpo_policy = build_policy(cfg_root, ldm_cfg, ckpt=args.ddpo_ckpt, device=args.device)
            print(
                f"[generate] chunk {chunk_id + 1}/{len(chunk_slots)} slots {slots[0]}..{slots[-1]}",
                flush=True,
            )
            payloads = generate_chunk(
                base_policy=base_policy,
                ddpo_policy=ddpo_policy,
                pool=pool,
                ldm_cfg=ldm_cfg,
                slots=slots,
                seed=int(args.seed),
                chunk_id=chunk_id,
                device=args.device,
                sources=sources,
            )
            resolved = [int(pool.resolved_scene_idx[s]) for s in slots]
            for source, payload in payloads.items():
                paths[source].parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"payload": payload, "slots": slots, "dataset_scene_idx": resolved},
                    paths[source],
                )
            # Template graphs are cached per slot and never revisited.
            pool._cache.clear()
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

        del base_policy, ddpo_policy
        gc.collect()
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

        # -------------------------------------------------------------- merge
        for source in sources:
            merged_path = artifact_dir / f"{source}.pt"
            chunk_files = [artifact_dir / source / f"chunk_{i:05d}.pt" for i in range(len(chunk_slots))]
            missing = [p for p in chunk_files if not p.exists()]
            if missing:
                raise RuntimeError(f"missing chunk artifacts for {source}: {missing[:3]}")
            blobs = [torch.load(p, map_location="cpu", weights_only=False) for p in chunk_files]
            slots = [s for b in blobs for s in b["slots"]]
            resolved = [i for b in blobs for i in b["dataset_scene_idx"]]
            metadata = build_metadata(
                source=source,
                config_name=args.config_name,
                overrides=args.overrides,
                split=args.split,
                seed=int(args.seed),
                slots=slots,
                resolved_scene_idx=resolved,
                base_ckpt=base_ckpt,
                ddpo_ckpt=args.ddpo_ckpt,
                cfg_root=cfg_root,
            )
            torch.save(
                {"payload": cat_payloads([b["payload"] for b in blobs]), "metadata": metadata},
                merged_path,
            )
            print(f"[merge] wrote {merged_path}", flush=True)

    # ------------------------------------------------------------ benchmark
    if not args.skip_benchmark:
        reward = build_reward(cfg_root, ldm_cfg)
        min_ego_drive = float(cfg_root.ddpo.get("min_ego_drive", 10.0))
        summaries = {}
        for source in sources:
            summary_path = benchmark_dir / source / "summary.json"
            per_scene_path = benchmark_dir / source / "per_scene.csv"
            if summary_path.exists() and per_scene_path.exists() and not args.force_benchmark:
                import json

                print(f"[benchmark] skip existing {source}", flush=True)
                summaries[source] = json.loads(summary_path.read_text(encoding="utf-8"))["summary"]
                continue
            blob = torch.load(artifact_dir / f"{source}.pt", map_location="cpu", weights_only=False)
            metrics = benchmark_payload(
                reward, blob["payload"], batch_size=int(args.benchmark_batch_size), label=source
            )
            summaries[source] = summarize(metrics, min_ego_drive=min_ego_drive)
            write_per_scene_csv(
                per_scene_path, source=source, metadata=blob["metadata"], metrics=metrics
            )
            write_json(
                summary_path,
                {
                    "source": source,
                    "artifact": str(artifact_dir / f"{source}.pt"),
                    "summary": summaries[source],
                    "metadata": blob["metadata"],
                },
            )
            print(f"[benchmark] wrote {summary_path}", flush=True)
        write_table(out_dir, summaries)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
