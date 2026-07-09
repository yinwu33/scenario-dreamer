"""Four-source ldm_adv scene generation + bad_driver benchmark.

Compares the frozen planner's ego metrics (reached-goal / collision / off-road
proxy) across four scene sources that share the SAME template scenes (pool
slots of an ``LDMAdvConditioningPool``, val split by default):

  * ``original``          -- the real dataset scene (GT agents + GT lanes); no
                             generated adversary, nobody gets the adversary
                             collision-factor override.
  * ``base_gen``          -- fully generated scene: stage 1 samples lane+agent
                             latents with the BASE ldm_adv checkpoint in
                             init_scene mode, stage 2 denoises one adversary
                             latent with the BASE adv branch conditioned on the
                             clean stage-1 latents (init_adv regime).
  * ``ddpo_gen``          -- the SAME stage-1 base-scene latents, adversary
                             denoised by the DDPO-fine-tuned adv branch. Paired
                             with ``base_gen`` (same template, same base scene,
                             same initial adv noise): the only difference is the
                             DDPO weights.
  * ``original_ddpo_adv`` -- the real scene plus the DDPO adversary (init_adv on
                             the real latents): exactly the DDPO training
                             distribution, upper-bound reference that separates
                             the adversary effect from the generated-scene shift.

Two-stage sampling for the generated sources (instead of running the DDPO
checkpoint in joint init_scene mode) matches the DDPO sampling regime exactly:
the fine-tuned adv branch only ever saw CLEAN (t=0) base latents during
training, and the ``la2adv`` cross-attention is one-directional (base -> adv,
base streams frozen during DDPO), so the base-scene distribution is unaffected.

Ego convention: scene agent row 0 (for generated scenes this is an arbitrary
generated agent -- the set decoder is permutation-equivariant). A generated ego
whose goal is within the static-agent threshold is never controlled and its
scene counts as trivially reached; the summary therefore also reports every
rate on the ``driving ego`` subset (ego spawn->goal distance >= min_ego_drive),
and ``ego_goal_dist`` is a per-scene CSV column.

Artifacts are plain-tensor payloads (no pickled dataclasses) with a metadata
dict recording everything needed to reproduce them (config, ckpts, seed, pool
slots and the dataset indices they resolved to, git commit).
"""

from __future__ import annotations

import copy
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cfgs.config import CONFIG_PATH
from ddpo.conditioning import LDMAdvConditioningPool
from ddpo.interfaces import GeneratedScenes
from ddpo.planners import SimulatorConfig
from ddpo.policy_ldm_adv import LDMAdvDDPOPolicy
from ddpo.reward import PufferSimulator, RewardConfig
from ddpo.train_loop import _build_gen_invalid
from utils.data_helpers import unnormalize_latents, unnormalize_scene
from utils.train_helpers import cache_latent_stats, set_latent_stats

SOURCES = ("original", "base_gen", "ddpo_gen", "original_ddpo_adv")

# Per-scene metric columns copied from PufferSimulator.evaluate into the CSV.
METRIC_KEYS = (
    "reached_goal",
    "ego_collision",
    "ego_offroad_proxy",
    "ego_offroad_frac",
    "ego_lane_dist_max",
    "ego_min_ttc",
    "ego_adv_min_dist",
    "init_invalid",
    "gen_agent_is_invalid",
    "gen_agent_is_parked",
    "reward",
    "criticality",
)


# --------------------------------------------------------------------- config
def compose_eval_cfg(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=str(Path(CONFIG_PATH).resolve()), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides or []))


def _set_dataset_name(cfg_node, dataset_name: str) -> None:
    OmegaConf.set_struct(cfg_node, False)
    cfg_node.dataset_name = dataset_name
    OmegaConf.set_struct(cfg_node, True)


def prepare_ldm_cfg(cfg_root):
    """Resolve the ldm_adv config with latent stats, mirroring the DDPO trainer."""
    _set_dataset_name(cfg_root.ldm_adv, cfg_root.dataset_name.name)
    _set_dataset_name(cfg_root.ae_goal, cfg_root.dataset_name.name)
    if not Path(cfg_root.ldm_adv.dataset.latent_stats_path).exists():
        cache_latent_stats(cfg_root.ldm_adv)
    return set_latent_stats(cfg_root.ldm_adv)


def build_pool(cfg_root, ldm_cfg, *, split: str, pool_size: int, device: str) -> LDMAdvConditioningPool:
    """The DDPO conditioning pool, configured exactly like training (driving-ego
    filter, insertion mode, per-scene deterministic adv-cond target)."""
    cfg = cfg_root.ddpo
    return LDMAdvConditioningPool(
        ldm_cfg.dataset,
        split_name=split,
        pool_size=pool_size,
        device=device,
        seed=int(cfg.seed),
        min_ego_drive=float(cfg.get("min_ego_drive", 10.0)),
        prune_base_to_ego=bool(cfg.get("prune_base_to_ego", False)),
        insert_adv_as_extra=bool(cfg.get("insert_adv_as_extra", False)),
        adv_cond_target=cfg.get("adv_cond_target", None),
    )


def build_policy(cfg_root, ldm_cfg, *, ckpt: str, device: str) -> LDMAdvDDPOPolicy:
    cfg = cfg_root.ddpo
    return LDMAdvDDPOPolicy(
        ldm_cfg,
        cfg_root.ae_goal,
        ldm_ckpt=ckpt,
        ae_ckpt=cfg.ae_ckpt,
        device=device,
        use_ema_weights=bool(cfg.get("use_ema_weights", True)),
        sampler=str(cfg.get("sampler", "ddpm")),
        ddim_steps=cfg.get("ddim_steps", None),
        ddim_eta=float(cfg.get("ddim_eta", 1.0)),
    )


def build_reward(cfg_root, ldm_cfg) -> PufferSimulator:
    """The evaluation simulator, constructed exactly like ``run_ddpo`` (same
    planner yaml incl. the adv collision-factor override, same strict configs)."""
    cfg = cfg_root.ddpo
    return PufferSimulator(
        planner_cfg=cfg.planner,
        simulator_cfg=SimulatorConfig(
            seed=int(cfg.seed),
            gen_invalid=_build_gen_invalid(cfg, ldm_cfg.dataset),
            **OmegaConf.to_container(cfg.simulator, resolve=True),
        ),
        reward_cfg=RewardConfig(**OmegaConf.to_container(cfg.reward, resolve=True)),
    )


# ------------------------------------------------------------------ tensors
def _cpu(t) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu()
    return torch.as_tensor(t)


def _type_ids(type_tensor: torch.Tensor) -> torch.Tensor:
    if type_tensor.dim() == 2:
        return type_tensor.argmax(dim=-1).long()
    return type_tensor.long()


def scenes_to_payload(scenes: GeneratedScenes) -> dict[str, Any]:
    """Plain-tensor artifact payload (robust to dataclass refactors)."""
    num_scenes = int(scenes.num_scenes)
    payload = {
        "agent_states": _cpu(scenes.agent_states),
        "agent_types": _cpu(scenes.agent_types).long(),
        "agent_scene_idx": _cpu(scenes.agent_scene_idx).long(),
        "lane_polylines": _cpu(scenes.lane_polylines),
        "lane_scene_idx": _cpu(scenes.meta["lane_scene_idx"]).long(),
        "num_scenes": num_scenes,
        "adv_local_idx": (
            _cpu(scenes.adv_local_idx).long()
            if scenes.adv_local_idx is not None
            else torch.full((num_scenes,), -1, dtype=torch.long)
        ),
    }
    if "gen_agent_mask" in scenes.meta:
        payload["gen_agent_mask"] = _cpu(scenes.meta["gen_agent_mask"]).bool()
    if "adv_cond" in scenes.meta:
        payload["adv_cond"] = _cpu(scenes.meta["adv_cond"]).long()
    return payload


def payload_to_scenes(payload: dict[str, Any]) -> GeneratedScenes:
    meta = {"lane_scene_idx": payload["lane_scene_idx"]}
    if "gen_agent_mask" in payload:
        meta["gen_agent_mask"] = payload["gen_agent_mask"]
    if "adv_cond" in payload:
        meta["adv_cond"] = payload["adv_cond"]
    return GeneratedScenes(
        agent_states=payload["agent_states"],
        agent_types=payload["agent_types"],
        agent_scene_idx=payload["agent_scene_idx"],
        lane_polylines=payload["lane_polylines"],
        num_scenes=int(payload["num_scenes"]),
        adv_local_idx=payload["adv_local_idx"],
        meta=meta,
    )


def cat_payloads(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    offset = 0
    parts: dict[str, list[torch.Tensor]] = {}
    for c in chunks:
        n = int(c["num_scenes"])
        for key in ("agent_states", "agent_types", "lane_polylines", "gen_agent_mask", "adv_cond", "adv_local_idx"):
            if key in c:
                parts.setdefault(key, []).append(c[key])
        parts.setdefault("agent_scene_idx", []).append(c["agent_scene_idx"] + offset)
        parts.setdefault("lane_scene_idx", []).append(c["lane_scene_idx"] + offset)
        offset += n
    for key, vals in parts.items():
        out[key] = torch.cat(vals, dim=0)
    out["num_scenes"] = offset
    return out


def slice_payload(payload: dict[str, Any], start: int, end: int) -> GeneratedScenes:
    a_idx = payload["agent_scene_idx"]
    l_idx = payload["lane_scene_idx"]
    a_sel = (a_idx >= start) & (a_idx < end)
    l_sel = (l_idx >= start) & (l_idx < end)
    sub = {
        "agent_states": payload["agent_states"][a_sel],
        "agent_types": payload["agent_types"][a_sel],
        "agent_scene_idx": a_idx[a_sel] - start,
        "lane_polylines": payload["lane_polylines"][l_sel],
        "lane_scene_idx": l_idx[l_sel] - start,
        "num_scenes": end - start,
        "adv_local_idx": payload["adv_local_idx"][start:end],
    }
    if "gen_agent_mask" in payload:
        sub["gen_agent_mask"] = payload["gen_agent_mask"][a_sel]
    if "adv_cond" in payload:
        sub["adv_cond"] = payload["adv_cond"][start:end]
    return payload_to_scenes(sub)


# ------------------------------------------------------------------ sources
def _gt_scenes(cond, dataset_cfg) -> GeneratedScenes:
    """Source ``original``: the real dataset scene from the conditioning batch.

    In insertion mode (``insert_adv_as_extra=true``) ``data['agent']`` already
    holds the complete real agent set (the real adversary included), so nothing
    is appended; ``adv_local_idx`` stays -1 (no generated adversary, no
    collision-factor override). Lane polylines are the pool's sorted physical
    ``road_points``.
    """
    gt_states = cond["agent"].gt_x.float().clone()
    dummy_lane = torch.zeros(
        (1, int(dataset_cfg.num_points_per_lane), 2),
        dtype=gt_states.dtype,
        device=gt_states.device,
    )
    gt_states, _ = unnormalize_scene(
        gt_states,
        dummy_lane,
        fov=dataset_cfg.fov,
        min_speed=dataset_cfg.min_speed,
        max_speed=dataset_cfg.max_speed,
        min_length=dataset_cfg.min_length,
        max_length=dataset_cfg.max_length,
        min_width=dataset_cfg.min_width,
        max_width=dataset_cfg.max_width,
        min_lane_x=dataset_cfg.min_lane_x,
        max_lane_x=dataset_cfg.max_lane_x,
        min_lane_y=dataset_cfg.min_lane_y,
        max_lane_y=dataset_cfg.max_lane_y,
    )
    num_scenes = int(cond.batch_size)
    return GeneratedScenes(
        agent_states=gt_states,
        agent_types=_type_ids(cond["agent"].gt_type),
        agent_scene_idx=cond["agent"].batch,
        lane_polylines=cond["lane"].road_points,
        num_scenes=num_scenes,
        adv_local_idx=torch.full((num_scenes,), -1, dtype=torch.long, device=gt_states.device),
        meta={"lane_scene_idx": cond["lane"].batch},
    )


@torch.no_grad()
def _decoded_lane_polylines(policy: LDMAdvDDPOPolicy, x_agent, x_lane, data) -> torch.Tensor:
    """Physical lane polylines decoded from generated (normalized) latents."""
    agent_latents, lane_latents = unnormalize_latents(
        x_agent,
        x_lane,
        policy.agent_latents_mean,
        policy.agent_latents_std,
        policy.lane_latents_mean,
        policy.lane_latents_std,
    )
    agent_states, lane_states, _, _, _ = policy.ae.forward_decoder(agent_latents, lane_latents, data)
    ds = policy.cfg_dataset
    _, lane_polys = unnormalize_scene(
        agent_states.clone(),
        lane_states.clone(),
        fov=ds.fov,
        min_speed=ds.min_speed,
        max_speed=ds.max_speed,
        min_length=ds.min_length,
        max_length=ds.max_length,
        min_width=ds.min_width,
        max_width=ds.max_width,
        min_lane_x=ds.min_lane_x,
        max_lane_x=ds.max_lane_x,
        min_lane_y=ds.min_lane_y,
        max_lane_y=ds.max_lane_y,
    )
    return lane_polys


@torch.no_grad()
def sample_base_scene_latents(policy: LDMAdvDDPOPolicy, cond) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 1: joint init_scene sampling of the base scene (lane+agent latents)
    with the base checkpoint; the jointly denoised adv latent is discarded (both
    generated sources re-sample the adversary in the init_adv regime so the ONLY
    difference between them is the adv-branch weights)."""
    data = cond.to(policy.device)
    net = policy.net
    agent_shape = (data["agent"].latents.shape[0], 1, int(net.cfg_model.agent_latent_dim))
    lane_shape = (data["lane"].latents.shape[0], 1, int(net.cfg_model.lane_latent_dim))
    adv_shape = (data["adv"].latents.shape[0], 1, int(net.cfg_model.agent_latent_dim))
    x_agent, x_lane, _ = net.p_sample_loop(
        agent_shape, lane_shape, adv_shape, data, device=policy.device, mode="init_scene"
    )
    return x_agent, x_lane


def make_generated_cond(policy: LDMAdvDDPOPolicy, cond, x_agent, x_lane):
    """Swap the template's real latents/road-points for the stage-1 generated
    ones; graph structure, node counts and all cond labels are kept, so the
    result conditions ``LDMAdvDDPOPolicy.sample`` exactly like a dataset scene."""
    gen = copy.deepcopy(cond)
    gen["agent"].latents = x_agent.detach()
    gen["lane"].latents = x_lane.detach()
    gen["lane"].road_points = _decoded_lane_polylines(policy, x_agent, x_lane, cond)
    return gen


def _seed_all(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_chunk(
    *,
    base_policy: LDMAdvDDPOPolicy,
    ddpo_policy: LDMAdvDDPOPolicy,
    pool: LDMAdvConditioningPool,
    ldm_cfg,
    slots: list[int],
    seed: int,
    chunk_id: int,
    device: str,
    sources: tuple[str, ...] = SOURCES,
) -> dict[str, dict[str, Any]]:
    """Generate one chunk of every requested source from the same template batch.

    Seeding: stage 1 uses (seed, 1000+chunk); every stage-2 adv sampling uses the
    SAME (seed, 2000+chunk) so base_gen / ddpo_gen start from identical adversary
    noise (paired comparison; the nets consume the RNG stream identically).
    """
    cond = pool.batch_from_indices(slots)
    out: dict[str, dict[str, Any]] = {}

    if "original" in sources:
        out["original"] = scenes_to_payload(_gt_scenes(cond, ldm_cfg.dataset))

    if "base_gen" in sources or "ddpo_gen" in sources:
        _seed_all(seed * 1_000_003 + 1000 + chunk_id, device)
        x_agent, x_lane = sample_base_scene_latents(base_policy, cond)
        gen_cond = make_generated_cond(base_policy, cond, x_agent, x_lane)
        if "base_gen" in sources:
            _seed_all(seed * 1_000_003 + 2000 + chunk_id, device)
            scenes, _ = base_policy.sample(gen_cond)
            out["base_gen"] = scenes_to_payload(scenes)
        if "ddpo_gen" in sources:
            _seed_all(seed * 1_000_003 + 2000 + chunk_id, device)
            scenes, _ = ddpo_policy.sample(gen_cond)
            out["ddpo_gen"] = scenes_to_payload(scenes)

    if "original_ddpo_adv" in sources:
        _seed_all(seed * 1_000_003 + 2000 + chunk_id, device)
        scenes, _ = ddpo_policy.sample(cond)
        out["original_ddpo_adv"] = scenes_to_payload(scenes)

    return out


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def build_metadata(
    *,
    source: str,
    config_name: str,
    overrides: list[str],
    split: str,
    seed: int,
    slots: list[int],
    resolved_scene_idx: list[int],
    base_ckpt: str,
    ddpo_ckpt: str,
    cfg_root,
) -> dict[str, Any]:
    cfg = cfg_root.ddpo
    return {
        "source": source,
        "config_name": config_name,
        "overrides": list(overrides),
        "split": split,
        "seed": int(seed),
        "pool_slots": [int(s) for s in slots],
        "dataset_scene_idx": [int(i) for i in resolved_scene_idx],
        "base_ckpt": str(base_ckpt),
        "ddpo_ckpt": str(ddpo_ckpt),
        "ae_ckpt": str(cfg.ae_ckpt),
        "sampler": str(cfg.get("sampler", "ddpm")),
        "use_ema_weights": bool(cfg.get("use_ema_weights", True)),
        "insert_adv_as_extra": bool(cfg.get("insert_adv_as_extra", False)),
        "prune_base_to_ego": bool(cfg.get("prune_base_to_ego", False)),
        "min_ego_drive": float(cfg.get("min_ego_drive", 10.0)),
        "adv_cond_target": OmegaConf.to_container(cfg.adv_cond_target, resolve=True)
        if cfg.get("adv_cond_target", None) is not None
        else None,
        "planner": OmegaConf.to_container(cfg.planner, resolve=True),
        "git_commit": _git_commit(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------- benchmark
def _rate(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float((arr > 0).mean()) if arr.size else float("nan")


def _mean_finite(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    return float(arr[finite].mean()) if finite.any() else float("nan")


def ego_goal_dist(payload: dict[str, Any]) -> np.ndarray:
    """Per-scene ego spawn->goal distance (agent row 0; states layout
    [x, y, speed, cos, sin, length, width, goal_x, goal_y])."""
    states = payload["agent_states"].numpy()
    scene_idx = payload["agent_scene_idx"].numpy()
    n = int(payload["num_scenes"])
    out = np.full(n, np.nan, dtype=np.float32)
    for s in range(n):
        rows = np.flatnonzero(scene_idx == s)
        if rows.size == 0:
            continue
        ego = states[rows[0]]
        out[s] = float(np.hypot(ego[7] - ego[0], ego[8] - ego[1]))
    return out


def num_agents_per_scene(payload: dict[str, Any]) -> np.ndarray:
    return np.bincount(
        payload["agent_scene_idx"].numpy(), minlength=int(payload["num_scenes"])
    ).astype(np.int64)


def benchmark_payload(
    reward: PufferSimulator, payload: dict[str, Any], *, batch_size: int, label: str = ""
) -> dict[str, np.ndarray]:
    """Roll the artifact out in batches and concatenate per-scene metrics."""
    n = int(payload["num_scenes"])
    chunks: list[dict[str, np.ndarray]] = []
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        if label:
            print(f"[benchmark] {label} scenes {start}..{end - 1} / {n}", flush=True)
        chunks.append(reward.evaluate(slice_payload(payload, start, end)))
    keys = [k for k in chunks[0] if isinstance(chunks[0][k], np.ndarray) and chunks[0][k].ndim == 1]
    metrics = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys}
    metrics["ego_goal_dist"] = ego_goal_dist(payload)
    metrics["num_agents"] = num_agents_per_scene(payload)
    return metrics


def summarize(metrics: dict[str, np.ndarray], *, min_ego_drive: float) -> dict[str, float]:
    driving = metrics["ego_goal_dist"] >= float(min_ego_drive)
    summary = {
        "num_scenes": int(metrics["reached_goal"].size),
        "reached_goal_rate": _rate(metrics["reached_goal"]),
        "ego_collision_rate": _rate(metrics["ego_collision"]),
        "ego_offroad_rate": _rate(metrics["ego_offroad_proxy"]),
        "ego_min_ttc_mean": _mean_finite(metrics["ego_min_ttc"]),
        "ego_goal_dist_mean": _mean_finite(metrics["ego_goal_dist"]),
        "num_agents_mean": float(np.mean(metrics["num_agents"])),
        # Same rates restricted to scenes whose ego actually has somewhere to
        # drive (a generated parked ego reaches its goal trivially).
        "num_driving_ego": int(driving.sum()),
        "reached_goal_rate_driving": _rate(metrics["reached_goal"][driving]),
        "ego_collision_rate_driving": _rate(metrics["ego_collision"][driving]),
        "ego_offroad_rate_driving": _rate(metrics["ego_offroad_proxy"][driving]),
        "ego_min_ttc_mean_driving": _mean_finite(metrics["ego_min_ttc"][driving]),
    }
    if "gen_agent_is_invalid" in metrics:
        summary["gen_agent_invalid_rate"] = _rate(metrics["gen_agent_is_invalid"])
    return summary


def write_per_scene_csv(
    path: Path,
    *,
    source: str,
    metadata: dict[str, Any],
    metrics: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(metrics["reached_goal"].size)
    rows = []
    for i in range(n):
        row: dict[str, Any] = {
            "source": source,
            "split": metadata["split"],
            "pool_slot": metadata["pool_slots"][i],
            "dataset_scene_idx": metadata["dataset_scene_idx"][i],
            "num_agents": int(metrics["num_agents"][i]),
            "ego_goal_dist": float(metrics["ego_goal_dist"][i]),
        }
        for key in METRIC_KEYS:
            if key in metrics:
                v = float(metrics[key][i])
                row[key] = v if np.isfinite(v) else float("nan")
        rows.append(row)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_table(out_dir: Path, summaries: dict[str, dict[str, float]]) -> None:
    """Combined cross-source table (CSV + markdown)."""
    columns = [
        "num_scenes",
        "num_agents_mean",
        "ego_goal_dist_mean",
        "reached_goal_rate",
        "ego_collision_rate",
        "ego_offroad_rate",
        "ego_min_ttc_mean",
        "num_driving_ego",
        "reached_goal_rate_driving",
        "ego_collision_rate_driving",
        "ego_offroad_rate_driving",
        "ego_min_ttc_mean_driving",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source"] + columns)
        writer.writeheader()
        for source in SOURCES:
            if source in summaries:
                writer.writerow({"source": source, **{c: summaries[source].get(c) for c in columns}})

    def _fmt(v) -> str:
        if v is None:
            return "-"
        if isinstance(v, int):
            return str(v)
        return f"{v:.4f}" if np.isfinite(v) else "nan"

    lines = [
        "| source | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for source in SOURCES:
        if source in summaries:
            s = summaries[source]
            lines.append(f"| {source} | " + " | ".join(_fmt(s.get(c)) for c in columns) + " |")
    (out_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[table] wrote {out_dir / 'table.csv'} and {out_dir / 'table.md'}", flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
