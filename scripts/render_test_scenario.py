#!/usr/bin/env python
"""Roll out every scene cache under a root and render each scene twice.

For each scene:
  <out>/<policy>/<mode>/<scene>.gif   the rollout, one frame per step
  <out>/<policy>/<mode>/<scene>.png   one still of the whole episode, every
                                      agent's full trajectory drawn behind its
                                      final box

The stems match the cache's own ``<i>_<batch>`` names, so a pkl, its scene png
from ``generate_scene.py --viz``, and these two share one identity.

The planner pair comes from the cache's directory name, because that is what the
checkpoint in it was fine-tuned against: ``main_pdm-ppo_aggressive`` rolls out
under pdm ego / ppo_aggressive traffic. ``base`` and the ``kl_*`` arms are all
ppo-ppo_norm runs. Pass ``--sut``/``--env`` to override and render every cache
under one pair instead.

Collision is ego vs the generated adversary, and the adversary is drawn in
CONTROL_COLOR, so what the picture highlights is what the reward scored.

Usage (env vars from scripts/define_env_variables.sh must be set)::

    .venv/bin/python scripts/render_test_scenario.py \
        --caches-root data/final/test --out data/final/test_scenario \
        --reward hierarchical_v4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from critical_scene.gen_scenes import list_gen_scene_files, load_gen_scenes
from critical_scene.ldm_adv_eval import (
    build_reward,
    compose_eval_cfg,
    payload_to_scenes,
    prepare_ldm_cfg,
    scenes_to_payload,
)
from eval_rollout import pair_from_name
from utils.viz import CONTROL_COLOR, render_rollout, render_rollout_frames, save_gif

def scene_components(metrics: dict, index: int) -> dict:
    """The per-scene reward components the rollout title prints."""
    return {
        key: values[index]
        for key, values in metrics.items()
        if isinstance(values, np.ndarray) and values.ndim == 1
        and values.dtype.kind in "fiub"
    }


def render_cache(cache_dir: Path, out_dir: Path, cfg_root, ldm_cfg,
                 *, max_frames: int, fps: int, dpi: int,
                 select: list[str] | None = None, prefix: str = "",
                 scene_dir: str | None = None, role: str | None = None) -> int:
    """``select`` names the scene stems to render; None renders the whole cache.

    Only the selected scenes are loaded and rolled out, so a query over a handful
    of scenes costs a handful of rollouts. Per-scene results do not depend on
    which other scenes share the batch -- the simulator steps each scene
    independently, which is the same property that makes sharding bit-exact."""
    files = list_gen_scene_files(cache_dir)
    if select is None:
        indices = range(len(files))
    else:
        want = set(select)
        indices = [i for i, f in enumerate(files) if Path(f).stem in want]
        if len(indices) != len(want):
            missing = want - {Path(files[i]).stem for i in indices}
            raise ValueError(f"{cache_dir}: selected scenes not in cache: {sorted(missing)}")
    scenes, kept = load_gen_scenes(cache_dir, indices, files=files)
    reward = build_reward(cfg_root, ldm_cfg, num_workers=0, batch_size=scenes.num_scenes)
    metrics = reward.evaluate(scenes, record_trajectories=True)

    states = scenes.agent_states.detach().cpu().numpy()
    types = scenes.agent_types.detach().cpu().numpy()
    agent_scene = scenes.agent_scene_idx.detach().cpu().numpy()
    adv_local = scenes.adv_local_idx.detach().cpu().numpy()
    lanes = np.asarray(scenes.lane_polylines)
    lane_scene = scenes.meta["lane_scene_idx"].detach().cpu().numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(int(scenes.num_scenes)):
        stem = Path(files[kept[index]]).stem
        # Paired layout: one directory per (scene, planner pair), holding this
        # cache's rollout under a role name. The two roles are NOT the same
        # scene -- init_scene decodes each cache's own -- so the directory is a
        # shared CONTEXT, not a shared scene. See the README this writes.
        if scene_dir is not None:
            target = out_dir / scene_dir.format(stem=stem)
            target.mkdir(parents=True, exist_ok=True)
            gif_path, png_path = target / f"{role}.gif", target / f"{role}.png"
        else:
            gif_path, png_path = out_dir / f"{prefix}{stem}.gif", out_dir / f"{prefix}{stem}.png"
        rows = agent_scene == index
        # Green marks the generated adversary and nothing else.
        colors = [CONTROL_COLOR if a == adv_local[index] else None
                  for a in range(int(rows.sum()))]
        kwargs = dict(
            agent_states=states[rows],
            agent_types=types[rows],
            agent_colors=colors,
            reward=float(metrics["reward"][index]),
            ego_collision=bool(metrics["ego_collision"][index] > 0),
            ego_offroad=bool(metrics["ego_offroad_proxy"][index] > 0),
            init_invalid=bool(metrics["init_invalid"][index] > 0),
            ego_min_ttc=float(metrics["ego_min_ttc"][index]),
            goal_offlane_frac=float(metrics["goal_offlane_frac"][index]),
            components=scene_components(metrics, index),
            title="",
        )
        trajectory = metrics["trajectories"][index]
        scene_lanes = lanes[lane_scene == index]

        frames = render_rollout_frames(
            trajectory, scene_lanes, max_frames=max_frames, annotate=False, **kwargs
        )
        save_gif(frames, str(gif_path), fps=fps)

        fig = render_rollout(
            trajectory, scene_lanes, final_boxes_only=True, annotate=False, **kwargs
        )
        fig.savefig(png_path, dpi=dpi, pad_inches=0)
        plt.close(fig)
    return int(scenes.num_scenes)


def write_pair_readme(out_root: Path, baseline: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "README.md").write_text(
        "# scenario_viz\n\n"
        "One directory per (scene id, planner pair), named `<id>__<sut>-<env>`:\n\n"
        "* `advscene.gif` / `.png` -- the DDPO checkpoint trained for that pair\n"
        f"* `base.gif` / `.png` -- the `{baseline}` cache, same scene id, same pair\n\n"
        "A directory exists only where the baseline's ego did NOT collide with its\n"
        "generated adversary and the checkpoint's ego DID, both rolled out under the\n"
        "planner pair in the directory name.\n\n"
        "The two are NOT the same scene. `init_scene` has each cache decode its own\n"
        "lanes and agents; the caches share the layout prior draw and the stage-1\n"
        "latents, so a directory is one shared CONTEXT rendered by two generators,\n"
        "not one scene rolled out twice. Measured spread between them on the\n"
        "non-adversary agents: median 2.8 mm, p99 39 cm, max 17 m.\n\n"
        "The png is the whole episode on one frame (full trajectories behind each\n"
        "agent's final box); the gif is one frame per step. The generated adversary\n"
        "is the only green vehicle.\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="config_ldm_adv_ddpo")
    ap.add_argument("--caches-root", default="data/final/test")
    ap.add_argument("--out", default="data/final/test_scenario")
    ap.add_argument("--reward", required=True,
                    help="cfgs/ddpo/reward/<name>.yaml to score with; it sets the "
                         "reward and tier the rollout title prints.")
    ap.add_argument("--sut", help="override the pair read from the cache's name")
    ap.add_argument("--env", help="override the pair read from the cache's name")
    ap.add_argument("--select",
                    help="json mapping policy -> [scene stem, ...]; render only those "
                         "scenes, and only the caches the file names")
    ap.add_argument("--flat", action="store_true",
                    help="write <out>/<policy>__<scene>.{gif,png} instead of "
                         "<out>/<policy>/<mode>/<scene>.{gif,png}")
    ap.add_argument("--compare-with",
                    help="also render each selected scene from this cache under the "
                         "same planner pair, into <out>/<scene>__<sut>-<env>/ as "
                         "base.{gif,png} beside advscene.{gif,png}")
    ap.add_argument("--max-frames", type=int, default=90)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    root, out_root = Path(args.caches_root), Path(args.out)
    caches = sorted(p for p in root.glob("*/*") if (p / "manifest.json").exists())
    select = json.loads(Path(args.select).read_text()) if args.select else None
    if select is not None:
        caches = [c for c in caches if c.parent.name in select]
    if not caches:
        raise SystemExit(f"no scene caches under {root}")
    total = sum(len(select[c.parent.name]) for c in caches) if select else None
    print(f"[render] {len(caches)} caches under {root}"
          + (f", {total} selected scenes" if total is not None else ""))

    if args.compare_with:
        if select is None or args.flat:
            raise SystemExit("--compare-with needs --select and is incompatible with --flat")
        caches = [c for c in caches if c.parent.name != args.compare_with]
        write_pair_readme(out_root, args.compare_with)

    for cache_dir in caches:
        policy, mode = cache_dir.parent.name, cache_dir.name
        sut = args.sut or pair_from_name(policy)[0]
        env = args.env or pair_from_name(policy)[1]
        want = select[policy] if select else None
        if want is not None and not want:
            # A selection can legitimately name a cache and pick nothing from it
            # (every candidate was screened out); rendering zero scenes is not an
            # error, but load_gen_scenes raises on an empty index list.
            print(f"[skip] {policy}/{mode} (selection empty)")
            continue

        # (cache to render, role name); the comparison cache contributes the same
        # scene ids under the same planner pair, so the two land side by side.
        jobs = [(cache_dir, "advscene")]
        if args.compare_with:
            jobs.append((cache_dir.parent.parent / args.compare_with / mode, "base"))

        for src, role in jobs:
            if args.compare_with:
                out_dir, prefix = out_root, ""
                scene_dir = "{stem}__" + f"{sut}-{env}"
                done = len(list(out_root.glob(f"*__{sut}-{env}/{role}.gif")))
            else:
                out_dir = out_root if args.flat else out_root / policy / mode
                prefix = f"{policy}__" if args.flat else ""
                scene_dir = None
                done = len(list(out_dir.glob(f"{prefix}*.gif"))) if out_dir.exists() else 0
            expected = len(want) if want is not None else len(list_gen_scene_files(src))
            if done and done == expected:
                print(f"[skip] {policy}/{mode} {role}")
                continue

            cfg_root = compose_eval_cfg(args.config_name, [
                f"ddpo/reward={args.reward}",
                f"planner@ddpo.planner.sut={sut}",
                f"planner@ddpo.planner.env={env}",
                f"planner@ddpo.planner.adv={env}",
            ])
            ldm_cfg = prepare_ldm_cfg(cfg_root)
            print(f"[render] {policy}/{mode} [{role}] sut={sut} env={env}", flush=True)
            n = render_cache(src, out_dir, cfg_root, ldm_cfg,
                             max_frames=args.max_frames, fps=args.fps, dpi=args.dpi,
                             select=want, prefix=prefix, scene_dir=scene_dir, role=role)
            print(f"[render]   {n} scenes -> {out_dir}", flush=True)

    print(f"[render] done -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
