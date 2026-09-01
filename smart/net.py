"""SMART-style traffic network: next action from a masked-length history.

This is a NETWORK, not a planner: it maps a flat agent-centric observation to a
distribution over the SHARED 7x13 accel/steer table, and knows nothing about
roles, scenes or rollouts. ``smart.planner`` is the planner that wraps it.

Two design points, both deliberate and both different from ``ctrl_sim``:

  * **The observation is one flat row per agent.** ``sim.parallel`` shards a
    rollout by shuttling a ``[rows, obs_dim]`` matrix through shared memory
    (``sim/parallel.py``), so a planner whose gather is that shape can be
    sharded exactly like the PPO one -- workers run the CPU halves, the parent
    runs ONE batched forward. ``ctrl_sim`` cannot be sharded only because its
    input is a set of agent-centric buffers plus lanes; that is an artifact of
    its shape, not of agent-centric models. Keeping the row flat is what makes
    this planner usable inside DDPO at all.
  * **It emits an index into the shared action table**, not a motion token, so
    ``SimScene.step_dynamics`` integrates it like every other planner and the
    paper keeps a single integrator exception (``ctrl_sim``) rather than two.

The history block carries a per-step validity flag, so a masked-length history
-- including the empty one a freshly generated scene starts from -- is an
ordinary input rather than a cold start.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

# ---- observation layout ----------------------------------------------------
# [ self | history | neighbors | road ], all in the agent's current frame.
SELF_FEATURES = 8            # signed speed, width, length, type one-hot(3), collision, removed
HISTORY_STEPS = 10           # 1.0 s at 10 Hz
HISTORY_FEATURES = 5         # dx, dy, cos dyaw, sin dyaw, valid
MAX_NEIGHBORS = 16
NEIGHBOR_FEATURES = 8        # dx, dy, cos dyaw, sin dyaw, signed speed, width, length, valid
MAX_ROAD_SEGMENTS = 64
ROAD_FEATURES = 6            # rel mid x/y, half length, dir x/y, valid
# The road budget is split. The segments stored in a record are ~1.2 m long
# (20 points per lane over the FOV), so taking the 64 NEAREST of them buys 1.2 m
# resolution over a perception radius of only ~10 m -- four times finer than
# SMART's 5 m tokens while covering a twentieth of its 50 m radius. Half the
# budget therefore keeps that local resolution, and half is spent walking the
# nearest lanes end to end, which is where range comes from. Both selections are
# purely geometric, so the row stays SE(2) invariant; an earlier attempt that
# strided the distance-RANKED candidate pool did not, because that pool is built
# in world coordinates.
ROAD_NEAR_SEGMENTS = 32
ROAD_FAR_LANES = 8
ROAD_FAR_POINTS = 4          # ROAD_FAR_LANES * ROAD_FAR_POINTS = the other half

SELF_OFF = 0
HISTORY_OFF = SELF_OFF + SELF_FEATURES
NEIGHBOR_OFF = HISTORY_OFF + HISTORY_STEPS * HISTORY_FEATURES
ROAD_OFF = NEIGHBOR_OFF + MAX_NEIGHBORS * NEIGHBOR_FEATURES
OBS_DIM = ROAD_OFF + MAX_ROAD_SEGMENTS * ROAD_FEATURES

# The shared accel/steer table (sim/world.py: 7 accelerations x 13 steering angles).
NUM_ACTIONS = 7 * 13

# Token groups, in the order they are concatenated into the sequence.
NUM_TOKENS = 1 + HISTORY_STEPS + MAX_NEIGHBORS + MAX_ROAD_SEGMENTS


@dataclass(frozen=True)
class NetConfig:
    """Fully resolved architecture + weights of one traffic net.

    Frozen and hashable so ``load_net`` can key its cache on it: two roles that
    compose the same planner yaml share one loaded network.
    """

    weights: str          # "random" (cost probe only) or an absolute checkpoint path
    device: str
    hidden_size: int
    num_layers: int
    num_heads: int
    seed: int


def _require(cfg, key: str):
    node = cfg
    for part in key.split("."):
        if part not in node:
            raise KeyError(f"smart planner yaml is missing required key {key!r}")
        node = node[part]
    if node is None:
        raise ValueError(f"smart planner yaml: {key!r} must have a value, got null")
    return node


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_net_config(cfg: Any) -> NetConfig:
    """Resolve one planner yaml node into a ``NetConfig``. Strict: every key required."""
    cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    )
    weights = str(_require(cfg, "weights"))
    if weights != "random":
        path = Path(weights)
        if not path.is_absolute():
            raise ValueError(
                f"smart planner weights must be 'random' or an absolute path, got {weights}; "
                "use ${project_root}/checkpoints/... in the planner yaml"
            )
        if not path.exists():
            raise FileNotFoundError(f"smart planner weights do not exist: {path}")
    return NetConfig(
        weights=weights,
        device=_resolve_device(str(_require(cfg, "device"))),
        hidden_size=int(_require(cfg, "policy.hidden_size")),
        num_layers=int(_require(cfg, "policy.num_layers")),
        num_heads=int(_require(cfg, "policy.num_heads")),
        seed=int(_require(cfg, "seed")),
    )


class SMARTTrafficNet(nn.Module):
    """Per-group token projections -> transformer encoder -> action logits.

    The self token is the read-out position. Padding is expressed through each
    group's trailing ``valid`` feature AND a key-padding mask, so an agent with
    no history, no neighbours or no nearby road is a normal input.
    """

    def __init__(self, cfg: NetConfig):
        super().__init__()
        d = cfg.hidden_size
        self.hidden_size = d
        self.proj_self = nn.Linear(SELF_FEATURES, d)
        self.proj_history = nn.Linear(HISTORY_FEATURES, d)
        self.proj_neighbor = nn.Linear(NEIGHBOR_FEATURES, d)
        self.proj_road = nn.Linear(ROAD_FEATURES, d)
        # One learned embedding per token group, plus a step embedding that
        # orders the history (neighbours and road segments are sets).
        self.group_embed = nn.Embedding(4, d)
        self.step_embed = nn.Embedding(HISTORY_STEPS, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.num_heads, dim_feedforward=4 * d,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.head = nn.Linear(d, NUM_ACTIONS)

        group = torch.zeros(NUM_TOKENS, dtype=torch.long)
        group[1 : 1 + HISTORY_STEPS] = 1
        group[1 + HISTORY_STEPS : 1 + HISTORY_STEPS + MAX_NEIGHBORS] = 2
        group[1 + HISTORY_STEPS + MAX_NEIGHBORS :] = 3
        self.register_buffer("group_ids", group, persistent=False)
        self.register_buffer("step_ids", torch.arange(HISTORY_STEPS), persistent=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """``obs`` [B, OBS_DIM] -> action logits [B, NUM_ACTIONS]."""
        b = obs.shape[0]
        s = obs[:, SELF_OFF:HISTORY_OFF]
        h = obs[:, HISTORY_OFF:NEIGHBOR_OFF].view(b, HISTORY_STEPS, HISTORY_FEATURES)
        n = obs[:, NEIGHBOR_OFF:ROAD_OFF].view(b, MAX_NEIGHBORS, NEIGHBOR_FEATURES)
        r = obs[:, ROAD_OFF:].view(b, MAX_ROAD_SEGMENTS, ROAD_FEATURES)

        tokens = torch.cat(
            [
                self.proj_self(s).unsqueeze(1),
                self.proj_history(h) + self.step_embed(self.step_ids).unsqueeze(0),
                self.proj_neighbor(n),
                self.proj_road(r),
            ],
            dim=1,
        )
        tokens = tokens + self.group_embed(self.group_ids).unsqueeze(0)

        # Trailing feature of every padded group is its validity flag; the self
        # token is always present.
        valid = torch.cat(
            [
                torch.ones(b, 1, dtype=torch.bool, device=obs.device),
                h[:, :, -1] > 0.5,
                n[:, :, -1] > 0.5,
                r[:, :, -1] > 0.5,
            ],
            dim=1,
        )
        out = self.encoder(tokens, src_key_padding_mask=~valid)
        return self.head(out[:, 0])


_NET_CACHE: dict[NetConfig, SMARTTrafficNet] = {}


def load_net(cfg: Any) -> SMARTTrafficNet:
    """Build (or reuse) the traffic net described by a planner yaml node.

    ``weights: random`` builds the target architecture with untrained weights.
    That mode exists for ONE purpose -- measuring what this planner costs in a
    sharded rollout before any training is spent (see the Phase 0 probe) -- and
    is seeded so the measurement is reproducible.
    """
    resolved = load_net_config(cfg)
    cached = _NET_CACHE.get(resolved)
    if cached is not None:
        return cached
    torch.manual_seed(resolved.seed)
    net = SMARTTrafficNet(resolved).to(resolved.device)
    if resolved.weights != "random":
        sd = torch.load(resolved.weights, map_location=resolved.device, weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    _NET_CACHE[resolved] = net
    return net
