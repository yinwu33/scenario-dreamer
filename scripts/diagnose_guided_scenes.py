"""Per-scene breakdown of a guided-vs-unguided run.

``scripts/generate_guided_scenes.py`` prints batch means, which hide the failure
modes that matter: a batch mean of ``overlap = 1.4`` can be one badly broken scene
or four mildly bad ones, and a good mean ``dmin`` can hide a conflict pinned at
t = 0 (a spawn overlap) or at the horizon (irrelevant to the ego). This prints one
row per scene so weight tuning has something to act on.

Usage::

    python scripts/diagnose_guided_scenes.py --scenes <viz_dir>_guided/guided_scenes.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfgs.config import CONFIG_PATH
from guidance.costs import (GOAL_X, GOAL_Y, footprint_radius, rollout_states,
                            soft_min, unnormalize_agent_states)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="guided_scenes.pkl from generate_guided_scenes.py")
    ap.add_argument("--config-name", default="config_dm_goal")
    args = ap.parse_args()

    with open(args.scenes, "rb") as f:
        dump = pickle.load(f)
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        cfg = compose(config_name=args.config_name).dm_goal
    gc = cfg.guidance.cost
    # The proximity objective has no rollout parameters; the informational columns
    # below still need a horizon, so fall back to the same defaults the trajectory
    # variant uses. They are a rough proxy, never the objective.
    horizon = gc.get("horizon", 5.0)
    dt = gc.get("dt", 0.5)
    tau = gc.get("soft_min_tau", 1.0)
    margin = gc.get("overlap_margin", 0.5)
    ts = torch.arange(0.0, horizon + 1e-6, dt)

    for key in sorted(dump):
        entry = dump[key]
        states = unnormalize_agent_states(torch.tensor(entry["agent_states"]), cfg.dataset)
        agent_batch = torch.tensor(entry["agent_batch"])
        adv_idx = torch.tensor(entry["adv_idx"])
        radius = footprint_radius(states)

        print(f"\n=== {key} ===")
        print(f"{'scene':>6}{'spawn_d':>9}{'goalgap':>9}{'ego_trip':>10}"
              f"{'| dmin':>9}{'t*':>7}{'speed':>8}{'ovl_max':>9}{'n_ovl':>7}")
        for i in range(len(adv_idx)):
            a = int(adv_idx[i])
            in_scene = agent_batch == i
            ego = int(in_scene.nonzero()[0])

            others = in_scene.clone()
            others[a] = False
            dist = torch.linalg.norm(states[others][:, :2] - states[a, :2], dim=-1)
            need = radius[a] + radius[others] + margin
            viol = torch.relu(need - dist)

            traj_a = rollout_states(states[a:a + 1], ts)
            traj_e = rollout_states(states[ego:ego + 1], ts)
            gap = torch.linalg.norm(traj_a - traj_e, dim=-1) - (radius[a] + radius[ego])
            weights = torch.softmax(-gap / tau, dim=-1)
            t_star = float((weights * ts).sum())
            goal_d = float(torch.linalg.norm(states[a, GOAL_X:GOAL_Y + 1] - states[a, :2]))

            # the three baseline constraints (assumption-free, these are the objective)
            spawn_d = float(torch.linalg.norm(states[a, :2] - states[ego, :2]))
            goalgap = float(torch.linalg.norm(
                states[a, GOAL_X:GOAL_Y + 1] - states[ego, GOAL_X:GOAL_Y + 1]))
            ego_trip = float(torch.linalg.norm(
                states[ego, GOAL_X:GOAL_Y + 1] - states[ego, :2]))
            print(f"{i:>6}{spawn_d:>9.2f}{goalgap:>9.2f}{ego_trip:>10.2f}"
                  f"{float(soft_min(gap, tau)):>9.2f}{t_star:>7.2f}"
                  f"{float(states[a, 2]):>8.2f}"
                  f"{(float(viol.max()) if viol.numel() else 0.0):>9.2f}{int((viol > 0).sum()):>7}")

    print("\nspawn_d / goalgap / ego_trip: the three baseline constraints (m) --")
    print("  the objective drives spawn_d and goalgap DOWN and ego_trip UP.")
    print("Right of the bar are INFORMATIONAL only and assume a constant-speed,")
    print("goal-directed rollout -- they are not optimised by the proximity objective;")
    print("a real criticality number has to come from a PufferDrive rollout.")


if __name__ == "__main__":
    main()
