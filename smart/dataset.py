"""Training samples: short logged CHAINS -> observations, actions, and a reference.

Each item is ONE scene. For a few randomly chosen start steps it takes a chain of
``chain_steps`` consecutive timesteps and, for every agent labelled across the
whole chain, builds that agent's observation at each step through the SAME
``smart.observation.build`` the planner calls at rollout time.

A chain rather than an independent timestep because the loss integrates the
model's actions forward and scores the resulting trajectory
(``smart.trajectory``). Per-step cross-entropy cannot see accumulated drift; a
chain can. The item therefore also carries the logged state at the chain start
and the logged poses along it, which are what the integrated trajectory is
scored against.

**History masking.** The visible history length is drawn uniformly from
``[0, HISTORY_STEPS]`` per sampled timestep, which is the whole reason this
model can drive a generated scene. A generated initial scene has a complete
state but no past, so ``first_valid_delta = HISTORY_STEPS`` must be an ordinary
input rather than a cold start. One length per timestep, not per agent, because
the planner keeps a single ``filled`` counter per scene: every agent in a
rollout has seen exactly as many steps as the rollout is old.

**Perturbation.** With probability ``perturb_prob`` the chain is SYNTHESISED off
the logged one: the agent is displaced sideways and rotated at the chain start,
and the offset decays linearly back to zero by the chain end. Observations,
action labels and the trajectory reference are all rebuilt from that perturbed
chain, so the sample reads "you are half a metre off the lane, here is how to get
back". Without it every observation the model ever sees comes from a pose the
logged driver actually occupied, and at rollout time -- where nothing restores
the truth -- it has never been taught to recover from its own drift. Measured
motivation: three architectures with perception radii of 10 m, 26 m and 41 m all
land at 66-69% off-road against a log rate of 11%, so the failure is not missing
information.

Masking uniformly over-represents short histories relative to a rollout, where
90% of the 91 steps have a full window. That is deliberate -- the short ones are
the hard case and the reason the model exists in this form. Within a chain the
visible history GROWS by one per step, exactly as it does during a rollout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sim.world import _compute_grid

from .actions import label_transitions
from .net import HISTORY_STEPS, OBS_DIM
from .observation import POSE_SLOTS, Frame, build
from .preprocess import NO_LABEL
from .records import load_scene, scene_paths

DT = 0.1  # cfgs/rollout/base.yaml


def _window_poses(state: np.ndarray, valid: np.ndarray, t: int) -> np.ndarray:
    """[A, POSE_SLOTS, 3] of (x, y, heading) ending at step ``t``.

    Steps before the record starts are clamped to the first one, and a step where
    an agent is not valid holds its last valid pose -- which is what a removed
    agent's entry looks like in the planner, since a removed agent stops moving
    and its pose is still written every step.
    """
    idx = np.clip(np.arange(t - POSE_SLOTS + 1, t + 1), 0, state.shape[1] - 1)
    poses = state[:, idx, :3].copy()
    ok = valid[:, idx]
    for slot in range(1, POSE_SLOTS):
        stale = ~ok[:, slot]
        poses[stale, slot] = poses[stale, slot - 1]
    return poses


class SMARTScenes(Dataset):
    """Waymo scenes as labelled action chains, on the simulator's agent set."""

    def __init__(self, split: str, chain_steps: int = 5, starts_per_scene: int = 2,
                 seed: int = 0, perturb_prob: float = 0.0, perturb_lat: float = 0.5,
                 perturb_yaw_deg: float = 3.0,
                 records: str = "data/advscene_preprocess_waymo",
                 labels: str = "data/smart_action_labels"):
        self.paths = scene_paths(split, records)
        self.label_dir = Path(labels, split)
        if not self.label_dir.is_dir():
            raise FileNotFoundError(
                f"no action labels at {self.label_dir}; run "
                f"`python smart/preprocess.py --split {split}` first"
            )
        # Only scenes whose labels are cached, so a partially preprocessed split
        # trains on what exists instead of failing halfway through an epoch.
        cached = {p.stem for p in self.label_dir.glob("*.npy")}
        self.paths = [p for p in self.paths if p.stem in cached]
        if not self.paths:
            raise FileNotFoundError(f"{self.label_dir} holds no labels for {split}")
        self.chain_steps = chain_steps
        self.starts_per_scene = starts_per_scene
        self.seed = seed
        self.perturb_prob = perturb_prob
        self.perturb_lat = perturb_lat
        self.perturb_yaw = np.radians(perturb_yaw_deg)

    def __len__(self) -> int:
        return len(self.paths)

    def _chain_starts(self, labels: np.ndarray) -> np.ndarray:
        """[A, T] mask: agent a has a label at every step of the chain from t."""
        k_steps, t_total = self.chain_steps, labels.shape[1]
        ok = labels != NO_LABEL
        chain = ok.copy()
        for k in range(1, k_steps):
            chain[:, : t_total - k] &= ok[:, k:]
        # tail entries were only partially ANDed; a chain must fit entirely
        chain[:, t_total - k_steps + 1 :] = False
        return chain

    def __getitem__(self, i: int):
        path = self.paths[i]
        scene = load_scene(path)
        labels = np.load(self.label_dir / f"{path.stem}.npy")
        state, valid = scene["state"], scene["valid"]
        k_steps = self.chain_steps

        chain = self._chain_starts(labels)
        starts = np.flatnonzero(chain.any(axis=0))
        if not len(starts):
            return EMPTY_ITEM

        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % (1 << 31)))
        starts = rng.choice(starts, size=min(self.starts_per_scene, len(starts)),
                            replace=False)
        grid = _compute_grid(scene["lanes"])
        grid["lanes"] = scene["lanes"]
        ptype = np.argmax(scene["types"], axis=1) + 1

        obs_out, act_out, s0_out, ref_out, len_out, wid_out = [], [], [], [], [], []
        for t in starts:
            t = int(t)
            ids = np.flatnonzero(chain[:, t])
            visible0 = int(rng.integers(0, HISTORY_STEPS + 1))
            # Window of true states for this chain, [A, k_steps+1, 4].
            window = state[:, t : t + k_steps + 1].copy()
            perturbed = rng.random() < self.perturb_prob
            if perturbed:
                # Sideways displacement + yaw error at the start, decaying to zero
                # by the end: a trajectory that begins off the lane and rejoins it.
                lat = rng.normal(0.0, self.perturb_lat, size=len(ids))
                yaw = rng.normal(0.0, self.perturb_yaw, size=len(ids))
                decay = 1.0 - np.arange(k_steps + 1) / k_steps
                head = window[ids][:, :, 2]
                window[ids, :, 0] += (lat[:, None] * decay[None, :]) * -np.sin(head)
                window[ids, :, 1] += (lat[:, None] * decay[None, :]) * np.cos(head)
                window[ids, :, 2] += yaw[:, None] * decay[None, :]
            chain_obs = np.empty((len(ids), k_steps, OBS_DIM), dtype=np.float32)
            for k in range(k_steps):
                step = t + k
                poses = _window_poses(state, valid, step)
                if perturbed:
                    poses[ids, -1, 0] = window[ids, k, 0]
                    poses[ids, -1, 1] = window[ids, k, 1]
                    poses[ids, -1, 2] = window[ids, k, 2]
                # history grows along the chain, and is bounded by the record start
                visible = min(visible0 + k, HISTORY_STEPS, step)
                cur = window[:, k]
                frame = Frame(
                    x=cur[:, 0], y=cur[:, 1],
                    heading_x=np.cos(cur[:, 2]), heading_y=np.sin(cur[:, 2]),
                    vx=cur[:, 3] * np.cos(cur[:, 2]), vy=cur[:, 3] * np.sin(cur[:, 2]),
                    length=scene["length"], width=scene["width"], ptype=ptype,
                    active=valid[:, step],
                    collision=np.zeros(len(state), dtype=bool),
                    removed=~valid[:, step],
                )
                chain_obs[:, k] = build(frame, grid, ids, poses, HISTORY_STEPS - visible)
            obs_out.append(chain_obs)
            if perturbed:
                # The cached labels describe the LOGGED chain; this one is
                # different, so relabel it -- one vectorised search over the 91
                # actions per transition, and the result is a recovery action.
                flat_from = window[ids, :-1].reshape(-1, 4)
                flat_to = window[ids, 1:].reshape(-1, 4)
                rep = np.repeat(np.arange(len(ids)), k_steps)
                acts, _ = label_transitions(flat_from, flat_to,
                                            scene["length"][ids][rep],
                                            scene["width"][ids][rep], DT)
                act_out.append(acts.reshape(len(ids), k_steps))
            else:
                act_out.append(labels[ids, t : t + k_steps].astype(np.int64))
            s0_out.append(window[ids, 0])
            ref_out.append(window[ids, 1:, :3])
            len_out.append(scene["length"][ids])
            wid_out.append(scene["width"][ids])

        f32 = lambda a: torch.from_numpy(np.concatenate(a).astype(np.float32))
        return (
            f32(obs_out),
            torch.from_numpy(np.concatenate(act_out)),
            f32(s0_out), f32(ref_out), f32(len_out), f32(wid_out),
        )


EMPTY_ITEM = (
    torch.zeros((0, 0, 0)), torch.zeros((0, 0), dtype=torch.long),
    torch.zeros((0, 4)), torch.zeros((0, 0, 3)), torch.zeros(0), torch.zeros(0),
)


def collate(batch):
    """Concatenate the per-scene chain blocks; scenes contribute different counts."""
    kept = [b for b in batch if b[0].numel()]
    if not kept:
        return EMPTY_ITEM
    return tuple(torch.cat([b[j] for b in kept]) for j in range(len(kept[0])))
