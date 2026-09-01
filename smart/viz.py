#!/usr/bin/env python
"""Visualisation for the SMART traffic model, independent of the root repo.

``utils/viz.py`` draws scenes for the diffusion pipeline: generated agents, an
adversary, goals, conditioning overlays. None of that applies here. What matters
for a behavior model is a different picture entirely -- where the model DROVE
against where the logged traffic actually went -- so this package draws its own.

One panel per scene: centrelines underneath, then each agent's logged track
against the model's, so a failure is legible at a glance (drifting off a curve,
cutting a corner, driving into another car). Moving and parked agents are drawn
differently because they are different problems: keeping a parked car parked is
easy and the model already does it, so mixing the two hides which one broke.

Used standalone to inspect a checkpoint, and from ``smart/train.py`` to push a
rollout image to wandb at every validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LANE_COLOR = "#d9d9d9"
LOG_MOVING, MODEL_MOVING = "#2c7fb8", "#e6550d"
LOG_PARKED, MODEL_PARKED = "#bdbdbd", "#fdae6b"


def _panel(ax, scene: dict, run: dict, step_t0: int):
    for poly in scene["lanes"]:
        v = np.isfinite(poly).all(axis=1)
        if v.sum() >= 2:
            ax.plot(poly[v, 0], poly[v, 1], color=LANE_COLOR, lw=0.8, zorder=0)

    live, track = run["live"], run["track"]
    for a, row in enumerate(live):
        moving = bool(scene["moving"][row])
        ok = run["ok"][:, a]
        if not ok.any():
            continue
        ref = scene["state"][row, step_t0 + 1 : step_t0 + 1 + len(ok), :2]
        ax.plot(ref[ok, 0], ref[ok, 1], color=LOG_MOVING if moving else LOG_PARKED,
                lw=1.4, zorder=1)
        ax.plot(track[ok, a, 0], track[ok, a, 1],
                color=MODEL_MOVING if moving else MODEL_PARKED,
                lw=1.4, ls="--", zorder=2)
        # start marker, so a track that goes the wrong way is readable
        ax.plot(ref[0, 0], ref[0, 1], marker="o", ms=2.5,
                color=LOG_MOVING if moving else LOG_PARKED, zorder=3)

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    off = 100 * run["offroad"].mean()
    coll = 100 * run["collided"].mean()
    ax.set_title(f"{scene['scenario_id'][:10]}  off {off:.0f}%  coll {coll:.0f}%",
                 fontsize=7)


def rollout_figure(scenes: list[dict], runs: list[dict], step_t0: int, cols: int = 3):
    """Grid of scenes: solid = logged track, dashed = model track."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(scenes)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.1 * rows))
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    for ax, scene, run in zip(np.atleast_1d(axes).ravel(), scenes, runs):
        _panel(ax, scene, run, step_t0)
    fig.legend(handles=[
        Line2D([], [], color=LOG_MOVING, lw=1.6, label="log (moving)"),
        Line2D([], [], color=MODEL_MOVING, lw=1.6, ls="--", label="model (moving)"),
        Line2D([], [], color=LOG_PARKED, lw=1.6, label="log (parked)"),
        Line2D([], [], color=MODEL_PARKED, lw=1.6, ls="--", label="model (parked)"),
    ], loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def _boxes(xy_yaw, length, width):
    """[N, 4, 2] footprint corners, so a rollout shows vehicles rather than dots.
    Boxes are what makes a collision or a lane departure visible."""
    x, y, yaw = xy_yaw[:, 0], xy_yaw[:, 1], xy_yaw[:, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    ox = np.stack([-hl, -hl, hl, hl], axis=-1)
    oy = np.stack([-hw, hw, hw, -hw], axis=-1)
    return np.stack([x[:, None] + ox * c[:, None] - oy * s[:, None],
                     y[:, None] + ox * s[:, None] + oy * c[:, None]], axis=-1)


def rollout_gif(scene: dict, run: dict, path: str, step_t0: int,
                stride: int = 2, fps: int = 10):
    """Animate one scene: logged vehicles against the model-driven ones.

    Solid outline is the log, filled is the model. A step where the log track has
    ended is simply not drawn on the log side, which is honest -- past that point
    there is nothing to compare against.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import PolyCollection

    live, track, ok = run["live"], run["track"], run["ok"]
    steps = track.shape[0]
    length, width = scene["length"][live], scene["width"][live]
    moving = scene["moving"][live]
    ref = scene["state"][live, step_t0 + 1 : step_t0 + 1 + steps, :3].transpose(1, 0, 2)

    fig, ax = plt.subplots(figsize=(6, 6))
    for poly in scene["lanes"]:
        v = np.isfinite(poly).all(axis=1)
        if v.sum() >= 2:
            ax.plot(poly[v, 0], poly[v, 1], color=LANE_COLOR, lw=0.9, zorder=0)
    log_c = PolyCollection([], facecolors="none", edgecolors=LOG_MOVING, lw=1.3, zorder=2)
    mdl_c = PolyCollection([], facecolors=MODEL_MOVING, alpha=0.55,
                           edgecolors="none", zorder=3)
    ax.add_collection(log_c); ax.add_collection(mdl_c)
    pts = np.concatenate([scene["lanes"].reshape(-1, 2)[
        np.isfinite(scene["lanes"].reshape(-1, 2)).all(1)], track[:, :, :2].reshape(-1, 2)])
    ax.set_xlim(pts[:, 0].min() - 5, pts[:, 0].max() + 5)
    ax.set_ylim(pts[:, 1].min() - 5, pts[:, 1].max() + 5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    title = ax.set_title("", fontsize=9)

    def frame(k):
        alive = ok[k]
        log_c.set_verts(list(_boxes(ref[k][alive], length[alive], width[alive])))
        mdl_c.set_verts(list(_boxes(track[k], length, width)))
        mdl_c.set_facecolors([MODEL_MOVING if m else MODEL_PARKED for m in moving])
        title.set_text(f"{scene['scenario_id'][:12]}   t = {(k + 1) * 0.1:.1f} s   "
                       f"outline = log, filled = model")
        return log_c, mdl_c, title

    anim = FuncAnimation(fig, frame, frames=range(0, steps, stride), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def render(weights: str, split: str, num_scenes: int, out: str | None,
           prime: bool = False, net=None):
    """Roll out ``num_scenes`` and draw them. ``net`` overrides the checkpoint,
    which is how training visualises the model it is holding in memory."""
    from omegaconf import OmegaConf
    from sim.planners import build_planner
    from smart.evaluate import SCENE_T0, rollout, sim_config
    from smart.records import load_scene, scene_paths

    cfg = OmegaConf.load("cfgs/planner/smart_probe.yaml")
    cfg.device = "cpu"
    cfg.weights = "random" if net is not None else str(Path(weights).resolve())
    planner = build_planner(cfg, role="env", device="cpu")
    if net is not None:
        # The planner was built on cpu, but a net handed over mid-training lives
        # wherever training put it. Its device is authoritative, or forward()
        # feeds cpu tensors to a cuda module.
        planner.net = net
        planner.device = str(next(net.parameters()).device)

    paths = scene_paths(split)
    paths = paths[:: max(1, len(paths) // num_scenes)][:num_scenes]
    scenes = [load_scene(p) for p in paths]
    runs = rollout(scenes, planner, sim_config("continue"), prime)
    fig = rollout_figure(scenes, runs, SCENE_T0)
    if out:
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--prime", action="store_true")
    ap.add_argument("--out", default="smart_rollout.png")
    ap.add_argument("--gif-dir", default=None,
                    help="also write one animated rollout per scene here")
    ap.add_argument("--planner", default="smart_probe")
    args = ap.parse_args()
    if args.gif_dir:
        from smart.evaluate import SCENE_T0, rollout, sim_config
        from smart.records import load_scene, scene_paths
        from omegaconf import OmegaConf
        from sim.planners import build_planner
        cfg = OmegaConf.load("cfgs/planner/smart_probe.yaml")
        cfg.device = "cpu"; cfg.weights = str(Path(args.weights).resolve())
        planner = build_planner(cfg, role="env", device="cpu")
        paths = scene_paths(args.split)
        paths = paths[:: max(1, len(paths) // args.scenes)][: args.scenes]
        scenes = [load_scene(p) for p in paths]
        runs = rollout(scenes, planner, sim_config("remove_off_map"), False)
        out_dir = Path(args.gif_dir); out_dir.mkdir(parents=True, exist_ok=True)
        for sc, r in zip(scenes, runs):
            f = out_dir / f"{sc['scenario_id'][:12]}.gif"
            rollout_gif(sc, r, str(f), SCENE_T0)
            print(f"wrote {f}")
        return 0
    render(args.weights, args.split, args.scenes, args.out, args.prime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
