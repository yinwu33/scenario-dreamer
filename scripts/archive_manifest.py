#!/usr/bin/env python
"""Manifest of the archived artifact set under ``data/final``.

``data`` is gitignored, so git carries no record of the 443 GB this project
produced. This writes the record that ARCHIVE.md quotes: for every kept
checkpoint a SHA256 and the provenance the file carries about itself, and for
every kept result directory a file count and byte total.

A DDPO checkpoint's ``ddpo`` block is the provenance, not a guess: it records the
iteration it stopped at, the KL coefficient in force, the base checkpoint it was
initialised from, and the wandb run id that holds its training curves. Lightning
checkpoints carry ``global_step``/``epoch`` instead. Nothing here is transcribed
by hand.

    .venv/bin/python scripts/archive_manifest.py
    .venv/bin/python scripts/archive_manifest.py --verify   # re-check the local copies
    .venv/bin/python scripts/archive_manifest.py --remote onedrive:/Projects/P04_AdvScene

``--remote`` is the one that still works after the local copies are deleted. It
streams each checkpoint back with ``rclone cat`` and hashes the stream, so it
needs no scratch space and, unlike ``rclone check``, it compares CONTENT -- this
OneDrive remote exposes no hash to rclone, so a plain check only sees size and
modification time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data" / "final"
# Result directories kept whole: too many small files to hash individually, and
# each already carries its own manifest.json / PROVENANCE.json.
RESULT_DIRS = ["cache", "scene_gen", "table_main", "table_main_valid", "scenecontrol",
               "planners", "test", "test_scenario",
               "backup_20260910_pre_skiprollout_fix", "backup_20260915_pre_validity"]


def sha256(path: Path, buf: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while (chunk := f.read(buf)):
            h.update(chunk)
    return h.hexdigest()


def sha256_remote(remote: str, rel: str, buf: int = 16 << 20) -> str:
    """Hash a remote file by streaming it, so no local copy is needed."""
    proc = subprocess.Popen(["rclone", "cat", f"{remote}/{rel}"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    h = hashlib.sha256()
    while (chunk := proc.stdout.read(buf)):
        h.update(chunk)
    proc.stdout.close()
    if proc.wait() != 0:
        raise SystemExit(f"rclone cat failed for {rel}")
    return h.hexdigest()


def provenance(path: Path) -> dict:
    """What the checkpoint says about its own training run."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "ddpo" in ck:
        d = ck["ddpo"]
        return {"kind": "ddpo", "iteration": d["it"], "kl_coef": d["kl_coef"],
                "base_ckpt": d["base_ckpt"], "wandb_id": d["wandb_id"]}
    return {"kind": "lightning",
            "global_step": ck.get("global_step"), "epoch": ck.get("epoch")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="re-hash the local copies against the existing manifest")
    ap.add_argument("--remote", metavar="RCLONE_PATH",
                    help="hash the copies under this rclone path instead, by streaming them")
    args = ap.parse_args()

    out_json, out_sha = FINAL / "MANIFEST.json", FINAL / "MANIFEST.sha256"
    if args.verify or args.remote:
        manifest = out_json if out_json.exists() else ROOT / "archive" / "MANIFEST.json"
        known = {e["path"]: e["sha256"] for e in json.loads(manifest.read_text())["checkpoints"]}
        bad = 0
        for rel, want in sorted(known.items()):
            got = sha256_remote(args.remote, rel) if args.remote else sha256(ROOT / rel)
            bad += got != want
            print(f"{'OK  ' if got == want else 'FAIL'}  {rel}", flush=True)
        where = args.remote if args.remote else "local"
        print(f"\n{len(known)} checked against {where}, {bad} mismatched")
        return 1 if bad else 0

    ckpts = []
    for p in sorted(FINAL.rglob("*.ckpt")):
        rel = p.relative_to(ROOT).as_posix()
        ckpts.append({"path": rel, "bytes": p.stat().st_size,
                      "sha256": sha256(p), **provenance(p)})
        print(f"[manifest] {rel}")

    dirs = []
    for name in RESULT_DIRS:
        d = FINAL / name
        if not d.exists():
            continue
        files = [f for f in d.rglob("*") if f.is_file()]
        dirs.append({"path": d.relative_to(ROOT).as_posix(),
                     "num_files": len(files), "bytes": sum(f.stat().st_size for f in files)})
        print(f"[manifest] {d.relative_to(ROOT)}  {len(files)} files")

    out_json.write_text(json.dumps({"checkpoints": ckpts, "result_dirs": dirs}, indent=1) + "\n")
    out_sha.write_text("".join(f"{c['sha256']}  {c['path']}\n" for c in ckpts))
    print(f"\n[manifest] {len(ckpts)} checkpoints, {len(dirs)} result dirs -> {out_json}, {out_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
