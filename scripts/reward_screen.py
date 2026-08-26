#!/usr/bin/env python
"""Reward screen: which reward variants have exploitable GRPO signal on the
FROZEN base model, BEFORE spending a training run on them.

A reward config is a pure post-rollout function of the metric table
(``ddpo.reward`` assemblers), so ONE sample+rollout sweep scores every
candidate reward. Phase 1 samples N adversaries for each of M training contexts
from the base ldm_adv checkpoint, rolls them out with the composed planner trio
and dumps the raw per-scene metrics. Phase 2 re-scores that dump under each
reward variant and reports what GRPO would actually see.

What is measured (per variant), all on the SAME rollouts:

  * support     -- does the reward ever fire? mean reward, share of samples with
    a nonzero positive term.
  * contrast    -- the GRPO precondition. Groups are the real thing: group_size
    samples from ONE context, exactly how ``sample_group_batch`` builds a batch.
    A group whose reward std is below ``algo.group_skip_std`` is degenerate --
    it has no within-group gradient of its own and is routed by
    ``algo.degenerate_group`` ("global": whitened against the whole batch, i.e.
    a uniform push-down; "skip": no gradient at all).
  * gradient targeting -- advantages come from the REAL
    ``ddpo_loss.compute_advantages`` with the run's own algo config. Reported as
    the mean advantage of collision / ego-fault / near-miss samples, and as what
    kind of sample each group crowns as its winner (argmax reward). A reward
    whose group winners are non-critical samples cannot teach criticality, no
    matter how healthy its std looks.
  * headroom    -- E[max - mean] within a group: how far the reward could rise
    if the policy learned to always produce its group's best sample.

Scope: this screens the reward at iteration 0 only, on the base distribution. It
rules OUT a reward with no contrast; it cannot rank rewards that all have
contrast, because a reward's value also lies in what it does after the policy
moves (that is what the approach-anneal and collision-bonus exist for).

Usage:
    # phase 1 + 2 (dumps to data/reward_screen/<pair>_<M>x<N>.npz, then scores)
    python scripts/reward_screen.py --sut idm --env idm --adv idm \
        --num-contexts 256 --samples-per-context 32
    # phase 2 only, on an existing dump
    python scripts/reward_screen.py --score-only --dump data/reward_screen/....npz

Read-only: no training, no wandb; results go to stdout + a JSON next to the dump.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cfgs.config import CONFIG_PATH


# Raw metric columns the reward assemblers consume, plus the diagnostics the
# screen groups by. Everything else the hooks emit is dropped from the dump.
_DUMP_KEYS = (
    "ego_min_ttc",
    "ego_adv_init_dist",
    "ego_adv_min_dist_warmup",
    "ego_collision",
    "ego_collision_time",
    "ego_fault_collision",
    "init_invalid",
    "init_overlap_frac",
    "spawn_lane_dist",
    "goal_lane_dist",
    "gen_agent_is_parked",
    "gen_agent_is_invalid",
    "gen_agent_invalid_gap",
    "reached_goal",
    "path_conflict",
    "path_following",
    "path_conflict_dist",
    "path_conflict_cos",
    "adv_spawn_offset",
    "adv_goal_offset",
    "path_conflict_pet",
    "ego_adv_spawn_dist",
)
# Stored separately: string array, not float.
_DUMP_STR_KEYS = ("gen_agent_invalid_reason",)


# ---------------------------------------------------------------- variants
def _variant_overlays(base_it: int) -> dict[str, dict]:
    """Reward variants to screen, as field overlays on the `ttc_only` yaml.

    The ladder is "min-TTC alone, then exactly one term added back", plus the
    shipped `full` at both ends of its approach anneal. Anything that looks
    worth training is then worth its own cfgs/ddpo/reward/<name>.yaml.
    """
    return {
        "ttc_only":            {},
        "ttc+approach":        {"approach_coef": 1.0, "approach_coef_final": 1.0},
        "ttc+collision":       {"collision_bonus": 0.5},
        "ttc+egofault":        {"ego_fault_bonus": 1.0},
        "ttc+coll+fault":      {"collision_bonus": 0.5, "ego_fault_bonus": 1.0},
        "ttc+lane":            {"lane_penalty": 0.25},
        "ttc+overlap":         {"init_overlap_penalty": 0.5},
        # The shipped reward, at the start of the approach anneal (w_app = 1.0)
        # and after it (w_app = 0.25) -- the two regimes a full run passes
        # through, both scored on the same base-model rollouts.
        "full@it0":            None,     # loaded from the full yaml as-is
        "full@annealed":       "anneal",  # full yaml, approach weight at its final value
        # Whole-yaml variant, scored as-is.
        "tiered":              "yaml:tiered",
        "hierarchical":        "yaml:hierarchical",
    }


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sut", default="idm", help="planner name for the ego role")
    p.add_argument("--env", default="idm", help="planner name for background traffic")
    p.add_argument("--adv", default="idm", help="planner name driving the adversary")
    p.add_argument("--config-name", default="config_ldm_adv_ddpo_idm_idm",
                   help="entrypoint config (its sampler / context_prior / conditioning are used)")
    p.add_argument("--num-contexts", type=int, default=256, help="M distinct conditioning scenes")
    p.add_argument("--samples-per-context", type=int, default=32, help="N base-model draws per context")
    p.add_argument("--group-size", type=int, default=None,
                   help="GRPO group size (default: the algo config's)")
    p.add_argument("--context-start", type=int, default=0,
                   help="first dump context to score (for held-out reward validation)")
    p.add_argument("--context-end", type=int, default=None,
                   help="exclusive last dump context to score")
    p.add_argument("--chunk-scenes", type=int, default=256, help="max scenes per sampling forward")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dump", default=None, help="npz path (default: data/reward_screen/<pair>_<M>x<N>.npz)")
    p.add_argument("--score-only", action="store_true", help="skip sampling, re-score an existing dump")
    p.add_argument("--override", action="append", default=[], help="extra hydra overrides (repeatable)")
    return p.parse_args()


def _compose_cfg(args, reward_variant: str | None = None):
    overrides = [
        f"planner@ddpo.planner.sut={args.sut}",
        f"planner@ddpo.planner.env={args.env}",
        f"planner@ddpo.planner.adv={args.adv}",
        f"experiment.planner_name={args.sut}-{args.env}",
        "experiment.output_dir=${scratch_root}/critical_scene/reward_screen",
        *( [f"ddpo/reward={reward_variant}"] if reward_variant else [] ),
        *args.override,
    ]
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        return compose(config_name=args.config_name, overrides=overrides)


# ------------------------------------------------------------------- phase 1
def run_dump(args, cfg_root, dump_path: Path) -> None:
    """Sample N adversaries per context from the base model, roll out, dump the
    per-scene metric table (no reward assembled)."""
    cfg = cfg_root.ddpo
    device = cfg.device
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    from ddpo.train_loop import _build_policy_and_pool, _build_gen_invalid
    from ddpo.reward import RewardModel, build_reward_config
    from sim.runner import SimulatorConfig

    t0 = time.time()
    _, policy, pool, eval_dataset_cfg = _build_policy_and_pool(cfg_root, cfg, device)
    sim = RewardModel(
        planner_cfg=cfg.planner,
        simulator_cfg=SimulatorConfig(
            seed=int(args.seed),
            gen_invalid=_build_gen_invalid(cfg, eval_dataset_cfg),
            **OmegaConf.to_container(cfg.simulator, resolve=True),
        ),
        # Only the metric hooks are used here; the weights are irrelevant to the
        # dump (every variant is applied offline in phase 2).
        reward_cfg=build_reward_config(cfg.reward),
    )

    M, N = int(args.num_contexts), int(args.samples_per_context)
    # Contexts are drawn the way TRAINING draws them (context_prior included),
    # so the screen sees the reward on the batch distribution DDPO will see.
    slots = pool._draw_slots(M)

    cols: dict[str, list[np.ndarray]] = {k: [] for k in _DUMP_KEYS + _DUMP_STR_KEYS}
    groups_per_chunk = max(1, int(args.chunk_scenes) // N)
    done = 0
    for lo in range(0, M, groups_per_chunk):
        chunk = slots[lo : lo + groups_per_chunk]
        cond = pool.batch_from_indices(np.repeat(chunk, N))
        scenes, _ = policy.sample(cond)
        metrics = sim.measure(scenes).metrics
        for k in _DUMP_KEYS:
            cols[k].append(np.asarray(metrics[k], dtype=np.float64))
        for k in _DUMP_STR_KEYS:
            cols[k].append(np.asarray(metrics[k], dtype=object).astype("U32"))
        done += len(chunk)
        print(f"[screen] contexts {done}/{M} ({done * N} rollouts, "
              f"{time.time() - t0:.0f}s elapsed)", flush=True)

    data = {k: np.concatenate(v).reshape(M, N) for k, v in cols.items()}
    data["pool_slot"] = np.asarray(slots)
    data["scene_idx"] = np.asarray(
        [pool.resolved_scene_idx.get(int(s), -1) for s in slots]
    )
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dump_path,
        gen_invalid_enabled=np.asarray(sim.gen_invalid_enabled),
        **data,
    )
    print(f"[screen] dumped {M}x{N} metrics -> {dump_path} ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------- phase 2
def _reward_cfgs(args) -> dict:
    """Build one reward config per screened variant (see _variant_overlays)."""
    from ddpo.reward import build_reward_config

    ttc = _compose_cfg(args, "ttc_only").ddpo.reward
    full = _compose_cfg(args, "full").ddpo.reward

    out = {}
    for name, overlay in _variant_overlays(0).items():
        if overlay is None:
            out[name] = build_reward_config(full)
        elif overlay == "anneal":
            cfg = build_reward_config(full)
            cfg.approach_coef = cfg.approach_coef_final
            out[name] = cfg
        elif isinstance(overlay, str) and overlay.startswith("yaml:"):
            variant = overlay.split(":", 1)[1]
            out[name] = build_reward_config(_compose_cfg(args, variant).ddpo.reward)
        else:
            out[name] = replace(build_reward_config(ttc), **overlay)
    return out


def _group_stats(reward: np.ndarray, adv: np.ndarray, cfg_algo, mask: dict) -> dict:
    """Per-group contrast / targeting stats. ``reward``/``adv`` are [G, S]."""
    std = reward.std(axis=1)
    degenerate = std < float(cfg_algo.group_skip_std)
    live = ~degenerate
    win = reward.argmax(axis=1)
    rows = np.arange(reward.shape[0])
    out = {
        "deg_group_frac": float(degenerate.mean()),
        "within_std_mean": float(std.mean()),
        "within_std_median": float(np.median(std)),
        "within_std_live_mean": float(std[live].mean()) if live.any() else 0.0,
        "headroom_mean": float((reward.max(axis=1) - reward.mean(axis=1)).mean()),
    }
    # Share of the total |advantage| mass produced by degenerate groups, i.e. by
    # the whole-batch fallback rather than by a real within-group comparison.
    mass = np.abs(adv).sum()
    out["deg_adv_mass_frac"] = float(np.abs(adv[degenerate]).sum() / mass) if mass > 0 else 0.0
    # What the reward crowns as a group's best sample. Only LIVE groups count:
    # in a degenerate (all-tie) group argmax returns index 0, which says nothing
    # about the reward and would otherwise dominate a sparse variant's table.
    for key, m in mask.items():
        m_g = m.reshape(reward.shape)
        out[f"winner_{key}"] = float(m_g[rows, win][live].mean()) if live.any() else float("nan")
    return out


def run_score(args, dump_path: Path) -> dict:
    from ddpo.ddpo_loss import DDPOConfig, compute_advantages
    from ddpo.reward import (
        HierarchicalRewardConfig,
        TieredRewardConfig,
        build_reward,
    )

    d = np.load(dump_path, allow_pickle=False)
    gen_invalid_enabled = bool(d["gen_invalid_enabled"])
    total_m, N = d["ego_min_ttc"].shape
    lo = max(0, int(args.context_start))
    hi = total_m if args.context_end is None else min(total_m, int(args.context_end))
    if not 0 <= lo < hi <= total_m:
        raise SystemExit(f"invalid context slice [{lo}:{hi}] for dump with {total_m} contexts")
    M = hi - lo
    cfg_root = _compose_cfg(args)
    algo = DDPOConfig(**OmegaConf.to_container(cfg_root.ddpo.algo, resolve=True))
    S = int(args.group_size or algo.group_size)
    if N % S:
        raise SystemExit(f"samples_per_context ({N}) must be divisible by group_size ({S})")
    G = M * (N // S)

    metrics = {k: d[k][lo:hi].reshape(-1) for k in _DUMP_KEYS}
    metrics["gen_agent_invalid_reason"] = d["gen_agent_invalid_reason"][lo:hi].reshape(-1)
    n = metrics["ego_min_ttc"].size

    # The raw dump may have been collected with more condition checks enabled
    # than the config being screened (the hierarchical IDM entry intentionally
    # disables goal-distance checking).  Rebuild the boolean/reason from the
    # per-field reason string so offline scoring matches the prospective run.
    gi = cfg_root.ddpo.gen_agent_invalid
    enabled_fields = {
        name
        for name in ("type", "motion", "goal_dist", "ego_dist")
        if bool(getattr(gi, f"check_{name}"))
    }
    raw_reasons = metrics["gen_agent_invalid_reason"].astype(str)
    filtered_reasons = []
    for reason in raw_reasons:
        kept = [token for token in reason.split() if token.split(":", 1)[0] in enabled_fields]
        filtered_reasons.append(" ".join(kept))
    filtered_reasons = np.asarray(filtered_reasons, dtype="U64")
    metrics["gen_agent_invalid_reason"] = filtered_reasons
    metrics["gen_agent_is_invalid"] = (filtered_reasons != "").astype(np.float32)

    # Behaviour labels, shared by every variant (they come from the rollout,
    # not from the reward): what each sample actually did.
    coll = metrics["ego_collision"] > 0
    fault = metrics["ego_fault_collision"] > 0
    ttc = metrics["ego_min_ttc"]
    near = np.isfinite(ttc) & (ttc < 1.5) & ~coll
    rejected = (metrics["gen_agent_is_invalid"] > 0) if gen_invalid_enabled else (
        metrics["gen_agent_is_parked"] > 0)
    overlapped = metrics["init_overlap_frac"] > 0
    quiet = ~(coll | near | rejected | overlapped)
    labels = {"collision": coll, "near_miss": near, "reject": rejected,
              "overlap": overlapped, "quiet": quiet}

    print(f"\n=== reward screen: sut={args.sut} env={args.env} adv={args.adv} | "
          f"{M} contexts x {N} samples, groups of {S} ({G} groups) ===")
    print(f"base rollout mix: collision {coll.mean():.1%}, ego-fault {fault.mean():.1%}, "
          f"near-miss(<1.5s) {near.mean():.1%}, rejected {rejected.mean():.1%}, "
          f"spawn-overlap {overlapped.mean():.1%}, quiet {quiet.mean():.1%}")
    finite = np.isfinite(ttc)
    for tau in (1.0, 2.0, 3.0):
        print(f"  P(min_TTC < {tau:.0f}s) = {(finite & (ttc < tau)).mean():.1%}", end="")
    print(f"   [min_TTC finite for {finite.mean():.1%} of samples]")

    # ---- path-conflict predicate: what does skipping the rollout cost? -----
    # The dump is produced WITHOUT skipping, so the events the skip would have
    # thrown away are directly countable here. Recall is the number that decides
    # whether the tiered reward is sound: a collision the predicate misses can
    # never be rewarded, so the policy would be taught that such geometries do
    # not exist.
    if "path_conflict_dist" in metrics:
        pd = metrics["path_conflict_dist"]
        print(f"\npath-conflict predicate (chord clearance), {n} samples:")
        print(f"{'thresh':>7s} {'conflict':>9s} {'coll recall':>12s} "
              f"{'TTC<3s recall':>14s} {'rollouts saved':>15s}")
        ttc_ev = np.isfinite(ttc) & (ttc < 3.0)
        for thr in (2.0, 3.0, 5.0, 8.0, 12.0, 20.0):
            sel = pd <= thr
            print(f"{thr:>7.0f} {sel.mean():>9.1%} "
                  f"{(sel & coll).sum() / max(coll.sum(), 1):>12.1%} "
                  f"{(sel & ttc_ev).sum() / max(ttc_ev.sum(), 1):>14.1%} "
                  f"{1 - sel.mean():>15.1%}")

        # Geometry classes inside the conflict set: is the chord-clearance test
        # admitting car-following (collinear, same direction, nothing happens)?
        cos_a = metrics["path_conflict_cos"]
        off = np.maximum(metrics["adv_spawn_offset"], metrics["adv_goal_offset"])
        near = pd <= 5.0
        classes = {
            "following (cos>=.8, both ends in corridor)": near & (cos_a >= 0.8) & (off <= 2.0),
            "cut-in    (cos>=.8, enters corridor)":       near & (cos_a >= 0.8) & (off > 2.0),
            "transversal (|cos|<.8)":                     near & (np.abs(cos_a) < 0.8),
            "head-on   (cos<=-.8)":                       near & (cos_a <= -0.8),
        }
        print(f"\ngeometry classes within chord clearance <= 5 m:")
        print(f"{'class':46s} {'share of all':>12s} {'collision':>10s} {'TTC<3s':>8s}")
        for name, sel in classes.items():
            k = max(sel.sum(), 1)
            print(f"{name:46s} {sel.mean():>12.1%} {coll[sel].sum() / k:>10.2%} "
                  f"{ttc_ev[sel].sum() / k:>8.1%}")
        rest = ~near
        print(f"{'(no conflict: clearance > 5 m)':46s} {rest.mean():>12.1%} "
              f"{coll[rest].mean():>10.2%} {ttc_ev[rest].mean():>8.1%}")

    # Group-level support: can a group discriminate at all, and on what.
    def _grp(m):
        return m.reshape(M, N // S, S).reshape(G, S).any(axis=1).mean()
    ttc_pos = np.isfinite(ttc) & (ttc < 3.0)
    print(f"group support ({S} samples/group): "
          f"{_grp(ttc_pos):.1%} contain a TTC-positive sample, "
          f"{_grp(coll):.1%} a collision, {_grp(rejected):.1%} a rejected one")

    def _score(name_cfgs: dict, m: dict) -> dict:
        res = {}
        for name, rcfg in name_cfgs.items():
            scorer = build_reward(rcfg, gen_invalid_enabled)
            hierarchical = isinstance(rcfg, HierarchicalRewardConfig)
            mv = m
            if isinstance(rcfg, TieredRewardConfig):
                # ``path_conflict`` was frozen at the dump's conflict_dist;
                # re-derive it at THIS variant's threshold so a screen of
                # several tier radii scores what each one would actually admit.
                # tier1_path_near is the yaml's single source of truth for it.
                mv = dict(m)
                mv["path_conflict"] = (
                    (m["path_conflict_dist"] <= rcfg.tier1_path_near)
                    & (m["path_following"] <= 0)
                ).astype(np.float32)
            elif hierarchical:
                # Recreate the prospective prefilter on this full-rollout dump.
                # The selected hierarchical config disables the following
                # exclusion (follow_cos > 1), but keep this generic for future
                # variants that turn it back on.
                hcfg = _compose_cfg(args, "hierarchical").ddpo.simulator.path_conflict
                admitted = m["path_conflict_dist"] <= float(hcfg.conflict_dist)
                if float(hcfg.follow_cos) <= 1.0:
                    admitted &= m["path_following"] <= 0
                mv = dict(m)
                mv["path_conflict"] = admitted.astype(np.float32)
                # The dump rolled EVERY scene out, but in production the
                # prefilter leaves the hooks at their defaults on a skip. A
                # variant that grades on a rollout metric (h_fallback_mode
                # "measured" ranks the fallback band by dmin) would otherwise be
                # credited offline with a measurement it never would have had.
                for key, default in (
                    ("ego_min_ttc", np.inf),
                    ("ego_adv_min_dist_warmup", np.inf),
                    ("ego_adv_init_dist", np.inf),
                    ("ego_collision_time", np.inf),
                    ("ego_collision", 0.0),
                    ("ego_fault_collision", 0.0),
                ):
                    mv[key] = np.where(admitted, m[key], default)
            reward, comp = scorer.assemble(mv)
            # Group along the SAME context, mirroring sample_group_batch.
            r_g = reward.reshape(M, N // S, S).reshape(G, S)
            group_ids = torch.arange(G).repeat_interleave(S)
            adv = compute_advantages(
                torch.as_tensor(r_g.reshape(-1), dtype=torch.float32), algo, group_ids
            ).numpy()
            stats = {
                "mean_reward": float(reward.mean()),
                "std_reward": float(reward.std()),
                "frac_reward_pos": float((reward > 0).mean()),
                "frac_criticality_pos": float((comp["criticality"] > 0).mean()),
                **_group_stats(r_g, adv.reshape(G, S), algo, labels),
            }
            if "tier" in comp:
                tier_g = comp["tier"].reshape(M, N // S, S).reshape(G, S)
                winner = r_g.argmax(axis=1)
                rows = np.arange(G)
                for level in range(int(tier_g.max()) + 1):
                    stats[f"tier{level}_frac"] = float((tier_g == level).mean())
                    stats[f"winner_tier{level}"] = float(
                        (tier_g[rows, winner] == level).mean()
                    )

                # Conditional targeting: when a desirable event is available
                # in a group, does the reward actually crown it?  This is more
                # informative than the unconditional winner mix for rare events.
                ctime = np.asarray(mv["ego_collision_time"]).reshape(G, S)
                valid_g = (~rejected & ~overlapped).reshape(G, S)
                coll_g = coll.reshape(G, S) & valid_g & (
                    ctime >= float(rcfg.collision_warmup)
                )
                ttc_g = (
                    np.isfinite(ttc).reshape(G, S)
                    & (ttc.reshape(G, S) < float(rcfg.ttc_tau))
                    & valid_g
                    & ~coll_g
                )
                close_g = (
                    np.isfinite(mv["ego_adv_min_dist_warmup"]).reshape(G, S)
                    & (np.asarray(mv["ego_adv_min_dist_warmup"]).reshape(G, S)
                       < float(rcfg.h_close_dist))
                    & valid_g
                    & ~coll_g
                    & ~ttc_g
                ) if hierarchical else np.zeros_like(valid_g)
                has_coll = coll_g.any(axis=1)
                has_ttc = ~has_coll & ttc_g.any(axis=1)
                has_close = ~has_coll & ~ttc_g.any(axis=1) & close_g.any(axis=1)
                for key, present, event in (
                    ("collision", has_coll, coll_g),
                    ("ttc", has_ttc, ttc_g),
                    ("close", has_close, close_g),
                ):
                    stats[f"group_has_{key}"] = float(present.mean())
                    stats[f"winner_{key}_given_available"] = (
                        float(event[rows, winner][present].mean())
                        if present.any() else float("nan")
                    )
            total_mass = float(np.abs(adv).sum())
            for key, lm in labels.items():
                stats[f"adv_{key}"] = float(adv[lm].mean()) if lm.any() else float("nan")
                # Share of the gradient MAGNITUDE this behaviour class carries:
                # what the update is actually about, weighted by how common the
                # class is (a huge advantage on 2% of samples moves little).
                stats[f"mass_{key}"] = (
                    float(np.abs(adv[lm]).sum() / total_mass) if total_mass > 0 else 0.0
                )
            stats["adv_abs_mean"] = float(np.abs(adv).mean())
            res[name] = stats
        return res

    results = _score(_reward_cfgs(args), metrics)
    _print_table(results)

    # What-if: the same rewards with the condition-violation gate neutralised
    # (no reject branch at all). Isolates how much of each variant's contrast is
    # criticality and how much is just "do not violate the conditioning target".
    m_nogate = dict(metrics)
    m_nogate["gen_agent_is_invalid"] = np.zeros(n)
    m_nogate["gen_agent_is_parked"] = np.zeros(n)
    all_cfgs = _reward_cfgs(args)
    nogate = _score({k: all_cfgs[k] for k in ("ttc_only", "ttc+coll+fault", "full@it0")}, m_nogate)
    print("\n--- what-if: reject gate OFF (ddpo.gen_agent_invalid.enabled=false) ---")
    _print_table(nogate)
    results.update({f"{k}|nogate": v for k, v in nogate.items()})
    out = {
        "dump": str(dump_path), "num_contexts": int(M), "samples_per_context": int(N),
        "context_start": lo, "context_end": hi,
        "group_size": S, "pair": f"sut={args.sut} env={args.env} adv={args.adv}",
        "base_mix": {k: float(v.mean()) for k, v in labels.items()},
        "variants": results,
    }
    out_path = dump_path.with_suffix(".screen.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[screen] -> {out_path}")
    return out


def _print_table(results: dict) -> None:
    def row(name, s):
        return (f"{name:16s} {s['mean_reward']:>7.3f} {s['frac_reward_pos']:>7.1%} "
                f"{s['deg_group_frac']:>8.1%} {s['within_std_live_mean']:>8.3f} "
                f"{s['headroom_mean']:>8.3f} {s['deg_adv_mass_frac']:>8.1%}")

    print(f"\n{'variant':16s} {'meanR':>7s} {'R>0':>7s} {'deg.grp':>8s} "
          f"{'std|live':>8s} {'headroom':>8s} {'degmass':>8s}")
    for name, s in results.items():
        print(row(name, s))

    print(f"\n{'mean advantage':16s} {'A|coll':>8s} {'A|near':>8s} {'A|reject':>9s} "
          f"{'A|overlap':>10s} {'A|quiet':>8s} {'|A|':>6s}")
    for name, s in results.items():
        print(f"{name:16s} {s['adv_collision']:>8.2f} {s['adv_near_miss']:>8.2f} "
              f"{s['adv_reject']:>9.2f} {s['adv_overlap']:>10.2f} {s['adv_quiet']:>8.2f} "
              f"{s['adv_abs_mean']:>6.2f}")

    print(f"\n{'share of |A| mass':16s} {'coll':>8s} {'near':>8s} {'reject':>9s} "
          f"{'overlap':>10s} {'quiet':>8s}")
    for name, s in results.items():
        print(f"{name:16s} {s['mass_collision']:>8.1%} {s['mass_near_miss']:>8.1%} "
              f"{s['mass_reject']:>9.1%} {s['mass_overlap']:>10.1%} {s['mass_quiet']:>8.1%}")

    print(f"\nlive-group winner     {'collision':>10s} {'near-miss':>10s} {'quiet':>8s} "
          f"{'reject':>8s} {'overlap':>8s}")
    for name, s in results.items():
        print(f"{name:16s}      {s['winner_collision']:>10.1%} {s['winner_near_miss']:>10.1%} "
              f"{s['winner_quiet']:>8.1%} {s['winner_reject']:>8.1%} {s['winner_overlap']:>8.1%}")

    hierarchical = {k: v for k, v in results.items() if "tier4_frac" in v}
    if hierarchical:
        print(f"\n{'hierarchical':16s} {'T0':>7s} {'T1':>7s} {'T2':>7s} "
              f"{'T3':>7s} {'T4':>7s} {'win|coll':>10s} {'win|TTC':>9s} "
              f"{'win|close':>10s}")
        for name, s in hierarchical.items():
            print(
                f"{name:16s} {s['tier0_frac']:>7.1%} {s['tier1_frac']:>7.1%} "
                f"{s['tier2_frac']:>7.1%} {s['tier3_frac']:>7.1%} "
                f"{s['tier4_frac']:>7.1%} "
                f"{s['winner_collision_given_available']:>10.1%} "
                f"{s['winner_ttc_given_available']:>9.1%} "
                f"{s['winner_close_given_available']:>10.1%}"
            )


def main():
    args = _parse_args()
    default_dump = (Path("data/reward_screen") /
                    f"{args.sut}-{args.env}-{args.adv}_{args.num_contexts}x{args.samples_per_context}.npz")
    dump_path = Path(args.dump) if args.dump else default_dump
    if not args.score_only:
        run_dump(args, _compose_cfg(args), dump_path)
    elif not dump_path.exists():
        raise SystemExit(f"no dump at {dump_path}")
    run_score(args, dump_path)


if __name__ == "__main__":
    main()
