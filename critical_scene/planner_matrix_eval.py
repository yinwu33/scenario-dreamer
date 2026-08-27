"""SUT x traffic-planner x scene-initialization benchmark: one cell per run.

Measures how well an ego planner (the system under test) drives to its goal when
the background traffic is driven by another planner, in scenes drawn from a
chosen initialization source. The two planner axes of the table are the
rollout's own role axes -- ``planner.sut`` is the row, ``planner.env`` the column
-- so selecting a cell is pure config composition; nothing here special-cases a
particular planner.

Deliberately does NOT go through ``ddpo.reward.RewardModel``. That class is
the consumer of the *adversarial* reward: it scores collisions as a positive
(``collision_bonus``) because DDPO is trying to manufacture critical scenes, and
it restricts collisions and TTC to the generated adversary, which log scenes do
not have. A planner benchmark needs the opposite polarity and a broader
collision notion, so it owns a ``RolloutRunner`` directly and injects its own
lean hook set:

  * ``ReachedGoalHook``      -> Succ.
  * ``EgoAnyCollisionHook``  -> Coll. (ego vs any vehicle)
  * ``EgoOffroadProxyHook``  -> Off.  (centerline-distance proxy; the maps
                                      carry no ROAD_EDGE entities)
  * ``RouteDiagnosticsHook``       -> whether the ego got a lane-following route
                                      at all, so "drove badly" can be told apart
                                      from "was never given a path".
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from omegaconf import OmegaConf

from critical_scene.metrics_common import (
    ego_goal_dist,
    mean_finite,
    num_agents_per_scene,
    rate,
    write_json,
)
from sim.scenes import GeneratedScenes
from sim.runner import RolloutRunner, SimulatorConfig
from sim.hooks import (
    MetricHook,
    EgoAnyCollisionHook,
    EgoOffroadProxyHook,
    ReachedGoalHook,
    TrajectoryHook,
    RolloutContext,
)

# Per-scene metric columns written to the CSV.
METRIC_KEYS = (
    "reached_goal",
    "ego_collision_any",
    "ego_fault_collision_any",
    "ego_collision_time",
    "ego_offroad_proxy",
    "ego_offroad_frac",
    "ego_lane_dist_max",
    "route_from_graph",
    "route_unavailable",
)


class RouteDiagnosticsHook(MetricHook):
    """Record where the ego's reference path came from.

    Only rule-based planners build routes; when the sut planner is a neural one
    the sim carries no ``_idm_route_sources`` and every scene reports NaN, which
    is the honest answer rather than a misleading zero.

    Routes always follow lane centerlines -- there is no straight-line fallback
    -- so the only failure mode is having no route at all. Those scenes measure
    the coverage of the lane-graph search, not how well the planner drives, and
    the rate is reported alongside the headline numbers so the two can be told
    apart. It is NOT excluded from the headline rates: the IDM and neural rows
    must share a denominator to stay comparable.

    Metrics (per scene, ego only):
      * ``route_from_graph``     -- 1.0 when the ego got a lane-following route
      * ``route_unavailable``    -- 1.0 when the lane graph had no path spawn->goal
      * ``route_multi_lane``     -- 1.0 when the route chained more than one lane
    """

    def after_rollout(self, ctx: RolloutContext) -> None:
        n = ctx.num_scenes
        from_graph = np.full(n, np.nan, dtype=np.float32)
        unavailable = np.full(n, np.nan, dtype=np.float32)
        multi_lane = np.full(n, np.nan, dtype=np.float32)
        for s, sim in enumerate(ctx.sims):
            source = getattr(sim, "_idm_route_sources", {}).get(0)
            if source is None:
                continue
            from_graph[s] = float(source != "none")
            unavailable[s] = float(source == "none")
            multi_lane[s] = float(source == "graph")
        ctx.metrics["route_from_graph"] = from_graph
        ctx.metrics["route_unavailable"] = unavailable
        ctx.metrics["route_multi_lane"] = multi_lane


def build_runner(cfg, *, num_workers: int = 0, batch_size: int = 0) -> RolloutRunner:
    """The benchmark rollout: per-role planners + shared dynamics, from config.

    ``num_workers`` shards each rollout across processes (``sim.parallel``); the
    result is bit-exact, so it only buys throughput.
    """
    cls = RolloutRunner
    extra = {}
    if num_workers > 0:
        from sim.parallel import ParallelRolloutRunner

        cls = ParallelRolloutRunner
        extra = {"num_workers": num_workers, "train_batch_size": batch_size}
    return cls(
        cfg.planner,
        SimulatorConfig(
            **OmegaConf.to_container(cfg.simulator, resolve=True),
            # Adversary-only knobs the planner benchmark never scores. They are
            # required fields of SimulatorConfig (strict by design) but are read
            # exclusively by the adversary hooks, which are not installed here.
            init_overlap_margin=0.0,
            goal_offlane_threshold=np.inf,
            goal_onroad_threshold=np.inf,
            approach_warmup_time=0.0,
            # No generated adversary in these scene sources, so no realized-vs-
            # requested condition to check.
            gen_invalid=None,
            ego_offroad_threshold=float(cfg.benchmark.ego_offroad_threshold),
        ),
        **extra,
    )


def make_hooks(runner: RolloutRunner, cfg, *, record_trajectories: bool = False) -> list:
    hooks: list[MetricHook] = [
        ReachedGoalHook(runner.sim_cfg.goal_radius),
        EgoAnyCollisionHook(),
        EgoOffroadProxyHook(float(cfg.benchmark.ego_offroad_threshold)),
        RouteDiagnosticsHook(),
    ]
    if record_trajectories:
        hooks.append(TrajectoryHook())
    return hooks


def evaluate_scenes(
    runner: RolloutRunner,
    cfg,
    scenes: GeneratedScenes,
    *,
    record_trajectories: bool = False,
):
    """Roll one batch out and return ``(metrics, trajectories)``."""
    result = runner.rollout(
        scenes,
        hooks=make_hooks(runner, cfg, record_trajectories=record_trajectories),
        record_trajectories=record_trajectories,
    )
    metrics = dict(result.metrics)
    metrics["ego_goal_dist"] = ego_goal_dist(_scene_view(scenes))
    metrics["num_agents"] = num_agents_per_scene(_scene_view(scenes))
    return metrics, result.trajectories


def _scene_view(scenes: GeneratedScenes) -> dict[str, Any]:
    """Adapt ``GeneratedScenes`` to the mapping ``metrics_common`` expects."""
    return {
        "agent_states": scenes.agent_states,
        "agent_scene_idx": scenes.agent_scene_idx,
        "num_scenes": scenes.num_scenes,
    }


def concat_metrics(chunks: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = [k for k, v in chunks[0].items() if isinstance(v, np.ndarray) and v.ndim == 1]
    return {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys}


def summarize(metrics: dict[str, np.ndarray], *, min_ego_drive: float) -> dict[str, float]:
    """Headline rates, plus the same rates on the driving-ego subset.

    An ego whose goal is essentially its spawn is never controlled and reaches
    its goal for free, so the unrestricted ``reached_goal_rate`` flatters any
    planner. ``*_driving`` restricts to scenes where the ego actually has
    somewhere to go and is the number to quote in the table.
    """
    driving = metrics["ego_goal_dist"] >= float(min_ego_drive)
    return {
        "num_scenes": int(metrics["reached_goal"].size),
        "num_agents_mean": float(np.mean(metrics["num_agents"])),
        "ego_goal_dist_mean": mean_finite(metrics["ego_goal_dist"]),
        "reached_goal_rate": rate(metrics["reached_goal"]),
        "ego_collision_rate": rate(metrics["ego_collision_any"]),
        "ego_offroad_rate": rate(metrics["ego_offroad_proxy"]),
        "num_driving_ego": int(driving.sum()),
        "reached_goal_rate_driving": rate(metrics["reached_goal"][driving]),
        "ego_collision_rate_driving": rate(metrics["ego_collision_any"][driving]),
        "ego_offroad_rate_driving": rate(metrics["ego_offroad_proxy"][driving]),
        "ego_fault_collision_rate_driving": rate(
            metrics["ego_fault_collision_any"][driving]
        ),
        "ego_offroad_frac_mean_driving": mean_finite(metrics["ego_offroad_frac"][driving]),
        # Diagnostics: a low success rate means something different when most
        # egos never got a lane-graph route in the first place.
        "route_from_graph_rate": mean_finite(metrics.get("route_from_graph", np.array([]))),
        "route_unavailable_rate": mean_finite(metrics.get("route_unavailable", np.array([]))),
        "route_multi_lane_rate": mean_finite(metrics.get("route_multi_lane", np.array([]))),
    }


# --------------------------------------------------------------------- gifs
def select_gif_scenes(metrics: dict[str, np.ndarray], count: int, *, min_ego_drive: float) -> list[int]:
    """Pick up to ``count`` scenes worth looking at, stratified by outcome.

    Groups are sampled ROUND-ROBIN, not filled in order: a greedy fill hands
    every slot to whichever group comes first (with ~9% collisions in 668 scenes
    there are always more crashes than slots), and a reel of nothing but crashes
    cannot answer the question you usually open the GIFs to ask -- whether the
    planner drives normally. Round-robin guarantees the successes are in there as
    the control. Within each group the longest drives come first (more to see).
    """
    driving = metrics["ego_goal_dist"] >= float(min_ego_drive)
    order = np.argsort(-np.nan_to_num(metrics["ego_goal_dist"]))

    reached = metrics["reached_goal"] > 0
    collided = metrics["ego_collision_any"] > 0
    groups = (
        driving & reached & ~collided,                  # normal driving (control)
        driving & collided,                             # crashed
        driving & ~reached & ~collided,                 # never arrived
        driving & (metrics["ego_offroad_proxy"] > 0),   # left the lane
    )
    queues = [[int(s) for s in order if g[s]] for g in groups]

    picked: list[int] = []
    seen: set[int] = set()
    while len(picked) < count and any(queues):
        progressed = False
        for queue in queues:
            while queue and queue[0] in seen:
                queue.pop(0)
            if not queue or len(picked) >= count:
                continue
            s = queue.pop(0)
            picked.append(s)
            seen.add(s)
            progressed = True
        if not progressed:
            break
    return picked


def render_cell_gifs(
    runner: RolloutRunner,
    cfg,
    scenes: GeneratedScenes,
    scene_ids: Sequence[int],
    out_dir: Path,
    *,
    fps: int = 10,
) -> list[str]:
    """Re-roll ``scenes`` with trajectory recording and write one GIF per scene.

    Rendering is a separate pass rather than part of the benchmark run because
    recording trajectories for every scene of a 1000-scene sweep costs a lot of
    memory for frames nobody looks at.
    """
    from ddpo.viz import render_rollout_frames, save_gif

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, trajectories = evaluate_scenes(runner, cfg, scenes, record_trajectories=True)
    if trajectories is None:
        return []

    states = scenes.agent_states.detach().cpu().numpy()
    agent_scene = scenes.agent_scene_idx.detach().cpu().numpy()
    lanes = scenes.lane_polylines
    lane_scene = scenes.meta["lane_scene_idx"].detach().cpu().numpy()

    paths = []
    for s in range(scenes.num_scenes):
        reached = int(metrics["reached_goal"][s])
        collided = int(metrics["ego_collision_any"][s])
        offroad = int(metrics["ego_offroad_proxy"][s])
        outcome = "collision" if collided else ("reached" if reached else "timeout")
        frames = render_rollout_frames(
            trajectories[s],
            lanes[lane_scene == s],
            agent_states=states[agent_scene == s],
            ego_collision=bool(collided),
            title=(
                f"scene={scene_ids[s]} goal_dist={metrics['ego_goal_dist'][s]:.1f}m  "
                f"reached={reached} collision={collided} offroad={offroad}"
            ),
        )
        paths.append(
            save_gif(frames, str(out_dir / f"{outcome}_scene{scene_ids[s]}.gif"), fps=fps)
        )
    return paths


# ------------------------------------------------------------------- output
SUMMARY_COLUMNS = (
    "num_scenes",
    "num_agents_mean",
    "ego_goal_dist_mean",
    "reached_goal_rate",
    "ego_collision_rate",
    "ego_offroad_rate",
    "num_driving_ego",
    "reached_goal_rate_driving",
    "ego_collision_rate_driving",
    "ego_offroad_rate_driving",
    "ego_fault_collision_rate_driving",
    "ego_offroad_frac_mean_driving",
    "route_from_graph_rate",
    "route_unavailable_rate",
    "route_multi_lane_rate",
)


def cell_label(sut: str, env: str, adv: str, source: str) -> str:
    """One table row: which planner drove the ego, the traffic, and from where.

    ``/`` rather than ``|``: the label is emitted inside a markdown table, where a
    pipe would be read as a column separator and shear the row.

    The adversary suffix is omitted when ``adv == env`` so the common case (no
    distinct adversary planner) keeps the old, shorter label.
    """
    label = f"{sut}/{env}/{source}"
    if adv != env:
        label += f"[adv={adv}]"
    return label


def write_per_scene_csv(
    path: Path, *, cell: str, metadata: dict[str, Any], metrics: dict[str, np.ndarray]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(metrics["reached_goal"].size)
    scene_idx = metadata["dataset_scene_idx"]
    rows = []
    for i in range(n):
        row: dict[str, Any] = {
            "cell": cell,
            "sut": metadata["sut"],
            "env": metadata["env"],
            "source": metadata["source"],
            "split": metadata["split"],
            "dataset_scene_idx": scene_idx[i],
            "num_agents": int(metrics["num_agents"][i]),
            "ego_goal_dist": float(metrics["ego_goal_dist"][i]),
        }
        for key in METRIC_KEYS:
            if key in metrics:
                row[key] = float(metrics[key][i])
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_table(out_dir: Path, summaries: dict[str, dict[str, Any]]) -> None:
    """Cross-cell table (CSV + markdown), one row per (sut, env, source).

    ``goal_behavior`` leads the columns because the table accumulates rows across
    invocations: two cells run under different rollout lifecycles are not
    comparable, and without the column that would sit in the file invisibly.
    """
    columns = ("goal_behavior",) + SUMMARY_COLUMNS
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = list(summaries)
    with (out_dir / "table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cell"] + list(columns))
        writer.writeheader()
        for cell in cells:
            writer.writerow({"cell": cell, **{c: summaries[cell].get(c) for c in columns}})

    def _fmt(v) -> str:
        if v is None:
            return "-"
        if isinstance(v, str):
            return v
        if isinstance(v, int):
            return str(v)
        return f"{v:.4f}" if np.isfinite(v) else "nan"

    lines = [
        "| cell | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for cell in cells:
        s = summaries[cell]
        lines.append(f"| {cell} | " + " | ".join(_fmt(s.get(c)) for c in columns) + " |")
    (out_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[table] wrote {out_dir / 'table.csv'} and {out_dir / 'table.md'}", flush=True)


__all__ = [
    "METRIC_KEYS",
    "SUMMARY_COLUMNS",
    "RouteDiagnosticsHook",
    "build_runner",
    "cell_label",
    "concat_metrics",
    "evaluate_scenes",
    "make_hooks",
    "render_cell_gifs",
    "select_gif_scenes",
    "summarize",
    "write_json",
    "write_per_scene_csv",
    "write_table",
]
