"""Matplotlib rendering of a planner rollout for wandb eval logging.

Style follows scenario-dreamer's ``utils/viz.py`` ``plot_scene``: lanes are drawn
twice (a thin dashed grey centerline plus a wide light-grey solid stroke that mimics
the road surface), agents are rounded boxes (ego red, vehicles blue, ...), and a
moving agent's goal is a dotted line + same-colour ``x`` marker. Parked/static
agents (goal within 2 m of spawn) instead get a bold black ``x`` at their centre.

Two output modes:
  * static (``render_rollout``): the whole first episode on one frame — each agent is
    a fading sequence of boxes (faint past -> solid present) so motion is visible.
  * gif (``render_rollout_frames`` + ``save_gif``): one frame per rollout step, the
    agents move and a light trail grows behind them.

The title carries the per-scene reward breakdown (total + collision / offroad /
init_invalid signals). The PufferDrive env auto-resets an episode when it ends, so we
cut the trajectory at the first reset (one clean episode, no teleport line).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

_EGO_COLOR = "#de5959"      # light red  (ego = local index 0)
_VEH_COLOR = "#87b3e6"      # light blue (other vehicles)
_PED_COLOR = "#bea9f5"      # light purple (pedestrians)
_CYC_COLOR = "#5fa55f"      # green (cyclists)
_JUMP_THRESH = 10.0         # metres/step above which motion is a teleport, not driving
_PARKING_DIST = 2.0         # MIN_DISTANCE_TO_GOAL: goal within this of spawn => parked/static


def _agent_color(is_ego: bool, type_id) -> str:
    if is_ego:
        return _EGO_COLOR
    return {0: _VEH_COLOR, 1: _PED_COLOR, 2: _CYC_COLOR}.get(int(type_id), "grey")


def _agent_draw_color(a: int, is_ego: bool, agent_types, agent_colors) -> str:
    if agent_colors is not None and a < len(agent_colors) and agent_colors[a] is not None:
        return agent_colors[a]
    return _agent_color(is_ego, agent_types[a] if agent_types is not None else 0)


def _first_episode_end(done) -> int | None:
    if done is None or len(done) == 0 or not np.any(done):
        return None
    return int(np.argmax(done)) + 1  # include the last pre-reset state


def _respawn_mask(traj, a, end=None):
    """Per-step bool mask (True once agent ``a`` is inactive) or None if unavailable.

    Respawned or removed agents should no longer be drawn as normal traffic after
    goal arrival. Sliced to ``[:end]`` to match the trajectory slice."""
    ra = traj.get("respawn") if isinstance(traj, dict) else None
    if ra is None or getattr(ra, "ndim", 0) != 2 or ra.shape[1] <= a:
        return None
    return ra[:end, a] if end is not None else ra[:, a]


def _break_on_jumps(x, y):
    if len(x) < 2:
        return x, y
    bad = np.where(np.hypot(np.diff(x), np.diff(y)) > _JUMP_THRESH)[0]
    if len(bad) == 0:
        return x, y
    return np.insert(x.astype(float), bad + 1, np.nan), np.insert(y.astype(float), bad + 1, np.nan)


def _status_text(reward, collided, offroad, init_invalid) -> tuple[str, str]:
    """Reward breakdown line + a colour keyed to the outcome."""
    r = 0.0 if reward is None else float(reward)
    txt = (f"R={r:+.2f}   collision={int(bool(collided))}  "
           f"offroad={int(bool(offroad))}  init_invalid={int(bool(init_invalid))}")
    if collided:
        color = _EGO_COLOR
    elif init_invalid:
        color = "#ff7f0e"
    else:
        color = "0.3"
    return txt, color


def _draw_agent_box(ax, x, y, heading, length, width, color, lw, alpha=1.0):
    """Rounded width(x)*length(y) box centred at (x,y), rotated by degrees(heading)-90."""
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(heading)):
        return  # blanked-out step (e.g. post-respawn): nothing to draw
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, y - length / 2), width, length,
        ec="black", fc=color, linewidth=lw, alpha=alpha,
        boxstyle=mpatches.BoxStyle("Round", pad=0.3), zorder=6,
    )
    tr = transforms.Affine2D().rotate_deg_around(x, y, np.degrees(heading) - 90) + ax.transData
    rect.set_transform(tr)
    ax.add_patch(rect)


def _view(lanes_arr, x, y, end, n_agents):
    """Square view window + scale-aware linewidths (plot_scene uses a 64m reference)."""
    xs, ys = [], []
    if lanes_arr is not None:
        flat = lanes_arr.reshape(-1, 2)
        flat = flat[np.isfinite(flat).all(1)]
        if len(flat):
            xs.append(flat[:, 0]); ys.append(flat[:, 1])
    if n_agents:
        xs.append(x[:end].ravel()); ys.append(y[:end].ravel())
    if xs:
        ax_min, ax_max = float(np.concatenate(xs).min()), float(np.concatenate(xs).max())
        ay_min, ay_max = float(np.concatenate(ys).min()), float(np.concatenate(ys).max())
    else:
        ax_min, ax_max, ay_min, ay_max = -32.0, 32.0, -32.0, 32.0
    cx, cy = (ax_min + ax_max) / 2, (ay_min + ay_max) / 2
    half = max((ax_max - ax_min), (ay_max - ay_min), 40.0) / 2 * 1.12
    scale = (2 * half) / 64.0
    return {
        "xlim": (cx - half, cx + half), "ylim": (cy - half, cy + half),
        "base_lw": 1.5 / scale, "road_w": 20.0 / scale, "scatter": 8.0 / (scale ** 2),
        "bbox_lw": 0.35 / scale, "goal_lw": 0.6 / scale, "goal_ms": 28.0 / (scale ** 2),
    }


def _draw_lanes(ax, lanes_arr, V):
    if lanes_arr is None:
        return
    for poly in lanes_arr:
        pts = poly[np.isfinite(poly[:, 0]) & np.isfinite(poly[:, 1])]
        if len(pts) < 2:
            continue
        ax.plot(pts[:, 0], pts[:, 1], color="lightgrey", linewidth=V["road_w"], linestyle="solid", zorder=1)
        ax.plot(pts[:, 0], pts[:, 1], color="grey", linewidth=V["base_lw"], linestyle="dashed", zorder=2)
        ax.scatter(pts[[0, -1], 0], pts[[0, -1], 1], color="grey", s=V["scatter"], zorder=3)


def _draw_goals(ax, agent_states, x0, y0, color, V):
    if agent_states is None or agent_states.shape[0] < 9:
        return
    gx, gy = float(agent_states[7]), float(agent_states[8])
    if not (np.isfinite(gx) and np.isfinite(gy)):
        return
    if np.hypot(gx - x0, gy - y0) < _PARKING_DIST:
        # parked/static agent (goal sits on its spawn): no travel goal to draw -
        # mark the agent centre with a bold black x instead of the same-colour one.
        ax.scatter(x0, y0, marker="x", color="black", s=V["goal_ms"],
                   linewidths=max(V["goal_lw"] * 2.0, 1.2), zorder=8)
        return
    ax.plot([x0, gx], [y0, gy], color=color, linestyle=":", alpha=0.7, linewidth=V["goal_lw"], zorder=3)
    ax.scatter(gx, gy, marker="x", color=color, s=V["goal_ms"], linewidths=max(V["goal_lw"], 0.5), zorder=7)


def _finish(ax, fig, V, title, status_txt, status_color):
    ax.set_xlim(*V["xlim"]); ax.set_ylim(*V["ylim"])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(f"{title}\n{status_txt}", fontsize=8.5, color=status_color)
    fig.tight_layout()


def render_rollout(traj, lanes, *, agent_states=None, agent_types=None, agent_colors=None,
                   reward=None, ego_collision=False, ego_offroad=False, init_invalid=False,
                   title="") -> "plt.Figure":
    """Static summary of the first episode: each agent a fading sequence of boxes."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    x, y, hd = traj["x"], traj["y"], traj["heading"]
    n_agents = x.shape[1] if (x.ndim == 2 and x.size) else 0
    end = _first_episode_end(traj.get("done"))
    lanes_arr = np.asarray(lanes) if (lanes is not None and len(lanes) > 0) else None
    V = _view(lanes_arr, x, y, end, n_agents)
    _draw_lanes(ax, lanes_arr, V)

    n_boxes = 5
    for a in range(n_agents):
        is_ego = a == 0
        color = _agent_draw_color(a, is_ego, agent_types, agent_colors)
        xa, ya, ha = x[:end, a], y[:end, a], hd[:end, a]
        if len(xa) == 0:
            continue
        # Blank out inactive steps so we don't draw the agent as normal traffic.
        mask = _respawn_mask(traj, a, end)
        if mask is not None and mask.any():
            xa = np.where(mask, np.nan, xa)
            ya = np.where(mask, np.nan, ya)
        length, width = float(traj["length"][a]), float(traj["width"][a])
        xb, yb = _break_on_jumps(xa, ya)
        ax.plot(xb, yb, color=color, linewidth=V["base_lw"] * (1.6 if is_ego else 1.1),
                alpha=0.95 if is_ego else 0.7, zorder=5 if is_ego else 4, solid_capstyle="round")
        ax.scatter(xa, ya, color=color, s=V["scatter"] * 0.35, alpha=0.9 if is_ego else 0.6,
                   zorder=5 if is_ego else 4, edgecolors="none")
        idxs = np.unique(np.linspace(0, len(xa) - 1, min(n_boxes, len(xa))).round().astype(int))
        for j, t in enumerate(idxs):
            frac = (j + 1) / len(idxs)
            _draw_agent_box(ax, xa[t], ya[t], ha[t], length, width, color,
                            V["bbox_lw"] * (1.4 if t == idxs[-1] else 1.0), alpha=0.22 + 0.78 * frac)
        _draw_goals(ax, agent_states[a] if agent_states is not None else None, xa[0], ya[0], color, V)

    txt, scol = _status_text(reward, ego_collision, ego_offroad, init_invalid)
    _finish(ax, fig, V, title, txt, scol)
    return fig


def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[..., :3].copy()


def render_rollout_frames(traj, lanes, *, agent_states=None, agent_types=None, agent_colors=None,
                          reward=None, ego_collision=False, ego_offroad=False, init_invalid=False,
                          title="", max_frames=50) -> np.ndarray:
    """One frame per rollout step (agents move, trail grows). Returns [T, H, W, 3] uint8.

    The view window and reward text are fixed across frames so the GIF is stable.
    """
    x, y, hd = traj["x"], traj["y"], traj["heading"]
    n_agents = x.shape[1] if (x.ndim == 2 and x.size) else 0
    end = _first_episode_end(traj.get("done"))
    T = end if end is not None else (x.shape[0] if n_agents else 0)
    T = max(T, 1)
    lanes_arr = np.asarray(lanes) if (lanes is not None and len(lanes) > 0) else None
    V = _view(lanes_arr, x, y, end, n_agents)
    txt, scol = _status_text(reward, ego_collision, ego_offroad, init_invalid)
    lengths = [float(traj["length"][a]) for a in range(n_agents)]
    widths = [float(traj["width"][a]) for a in range(n_agents)]

    frame_ts = np.unique(np.linspace(0, T - 1, min(max_frames, T)).round().astype(int))
    frames = []
    for t in frame_ts:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        _draw_lanes(ax, lanes_arr, V)
        for a in range(n_agents):
            is_ego = a == 0
            color = _agent_draw_color(a, is_ego, agent_types, agent_colors)
            mask = _respawn_mask(traj, a, t + 1)
            if mask is not None and bool(mask[-1]):
                continue  # agent is inactive by now: drop it instead of faking traffic
            xa, ya, ha = x[:t + 1, a], y[:t + 1, a], hd[:t + 1, a]
            if mask is not None and mask.any():
                xa = np.where(mask, np.nan, xa)
                ya = np.where(mask, np.nan, ya)
            xb, yb = _break_on_jumps(xa, ya)
            ax.plot(xb, yb, color=color, linewidth=V["base_lw"] * (1.3 if is_ego else 0.9),
                    alpha=0.5, zorder=4, solid_capstyle="round")  # trail so far
            _draw_agent_box(ax, x[t, a], y[t, a], hd[t, a], lengths[a], widths[a], color,
                            V["bbox_lw"] * (1.4 if is_ego else 1.0), alpha=1.0)  # current pose
            _draw_goals(ax, agent_states[a] if agent_states is not None else None, x[0, a], y[0, a], color, V)
        _finish(ax, fig, V, f"{title}   t={int(t)}", txt, scol)
        frames.append(_fig_to_rgb(fig))
        plt.close(fig)
    return np.stack(frames, axis=0)


def save_gif(frames: np.ndarray, path: str, fps: int = 10) -> str:
    """Write [T,H,W,3] uint8 frames to an animated GIF via Pillow (no moviepy dep)."""
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(fps, 1)), loop=0, optimize=True)
    return path
