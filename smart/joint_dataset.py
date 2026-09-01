"""Training samples for the joint net: one scene at one timestep.

One item is a whole SCENE, not an agent, because the model attends across agents
and cannot be fed them one at a time. A forward therefore yields one target per
agent present, which is the same target count per unit of work as the per-agent
design -- SMART's extra efficiency comes from running the TIME axis inside the
model too, which this first version does not.

Deliberately cross-entropy only, no trajectory term. The joint model would need
one full-scene forward per chain step, so keeping the chain would multiply the
cost by K and, worse, would change two things at once against the per-agent
baseline. The architecture is the variable under test here.

History masking carries over unchanged: the visible length is drawn uniformly
from [0, HISTORY_STEPS] so an empty past is an ordinary input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sim.world import _compute_grid

from .joint_net import HISTORY_STEPS, KEYS, MAX_AGENTS
from .joint_observation import POSE_SLOTS, build_scene, self_features
from .preprocess import NO_LABEL
from .records import load_scene, scene_paths

def _window_poses(state, valid, t):
    """[A, POSE_SLOTS, 3] ending at ``t``; an invalid step holds the last valid
    pose, which is what a retired agent looks like to the rollout planner."""
    idx = np.clip(np.arange(t - POSE_SLOTS + 1, t + 1), 0, state.shape[1] - 1)
    poses = state[:, idx, :3].copy()
    ok = valid[:, idx]
    for slot in range(1, POSE_SLOTS):
        stale = ~ok[:, slot]
        poses[stale, slot] = poses[stale, slot - 1]
    return poses


class JointScenes(Dataset):
    def __init__(self, split: str, steps_per_scene: int = 4, seed: int = 0,
                 records: str = "data/advscene_preprocess_waymo",
                 labels: str = "data/smart_action_labels"):
        self.paths = scene_paths(split, records)
        self.label_dir = Path(labels, split)
        cached = {p.stem for p in self.label_dir.glob("*.npy")}
        self.paths = [p for p in self.paths if p.stem in cached]
        if not self.paths:
            raise FileNotFoundError(f"{self.label_dir} holds no labels for {split}")
        self.steps_per_scene = steps_per_scene
        self.seed = seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        scene = load_scene(path)
        labels = np.load(self.label_dir / f"{path.stem}.npy")
        state, valid = scene["state"], scene["valid"]
        usable = np.flatnonzero((labels != NO_LABEL).any(axis=0))
        if not len(usable):
            return None

        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % (1 << 31)))
        steps = rng.choice(usable, size=min(self.steps_per_scene, len(usable)), replace=False)
        grid = _compute_grid(scene["lanes"])
        ptype = np.argmax(scene["types"], axis=1) + 1

        items = []
        for t in steps:
            t = int(t)
            visible = min(int(rng.integers(0, HISTORY_STEPS + 1)), t)
            pose = state[:, t, :3]
            sf = self_features(
                np.abs(state[:, t, 3]), scene["width"], scene["length"], ptype,
                np.zeros(len(state)), (~valid[:, t]).astype(np.float32),
            )
            obs = build_scene(pose, sf, _window_poses(state, valid, t),
                              HISTORY_STEPS - visible,
                              grid["seg_mid"], grid["seg_dir"], grid["seg_half_len"])
            act = np.full(MAX_AGENTS, NO_LABEL, dtype=np.int64)
            n = min(len(state), MAX_AGENTS)
            act[:n] = labels[:n, t]
            obs["action"] = act
            # supervise only agents that HAVE a label and are present this step
            obs["target_valid"] = (act != NO_LABEL) & obs["agent_valid"]
            items.append(obs)
        return items


def collate(batch):
    flat = [it for item in batch if item for it in item]
    if not flat:
        return None
    out = {k: torch.from_numpy(np.stack([it[k] for it in flat])) for k in KEYS}
    out["action"] = torch.from_numpy(np.stack([it["action"] for it in flat])).clamp(min=0)
    out["target_valid"] = torch.from_numpy(np.stack([it["target_valid"] for it in flat]))
    return out
