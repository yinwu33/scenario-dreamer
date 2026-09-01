"""Joint traffic net: SMART's factorised attention, adapted to this repo's data.

The per-agent net in ``smart.net`` bakes neighbours into each agent's own flat
row as seven handcrafted numbers, so two agents are coupled only through
features computed before the model runs. This one follows SMART: a scene is
processed ONCE, and an agent attends to the ENCODED representation of the other
agents and of the map.

Three attention stages, in SMART's order:

  1. **temporal** -- each agent over its own recent motion, in its own frame.
     Produces one embedding per agent, and it is rotation and translation
     invariant because the history is stored as frame-local deltas.
  2. **agent-agent** -- each agent over every other agent's stage-1 embedding,
     with the other agent's pose expressed IN THE QUERY'S FRAME and added to the
     key. This is the part the flat-row design cannot express: agent j is
     represented by what it has been doing, not by a fixed feature vector.
  3. **agent-map** -- each agent over the map, encoded once per scene rather
     than replicated into every agent's row. That is what makes full map
     coverage affordable: the per-agent design spent its whole budget on the 64
     nearest segments and reached 10% of the map.

Everything a query sees is expressed in the query's own frame, so the network is
equivariant by construction rather than by convention.

Cost, stated plainly: the input is per-SCENE, so this planner cannot use the
flat ``[rows, obs_dim]`` shared-memory path in ``sim/parallel.py``. It is
unshardable for the same reason ``ctrl_sim`` is, and it gives up the +21%
sharded-rollout result the per-agent design was built around.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

SELF_FEATURES = 8         # signed speed, width, length, type one-hot(3), collision, removed
HISTORY_STEPS = 10
HISTORY_FEATURES = 5      # dx, dy, cos dyaw, sin dyaw, valid   (own frame)
REL_AGENT_FEATURES = 6    # rel x, rel y, cos dyaw, sin dyaw, distance, valid
REL_MAP_FEATURES = 6      # rel x, rel y, dir x, dir y, half length, valid
MAP_FEATURES = 2          # half length, valid -- intrinsic to the segment

MAX_AGENTS = 32
MAX_MAP_TOKENS = 256      # encoded once per scene, so this can cover the map

NUM_ACTIONS = 7 * 13      # the shared accel/steer table (sim/world.py)

# The tensors a scene is made of. Defined HERE, in the one module that imports
# nothing from ``sim``: the planner needs them, and the planner is imported from
# inside ``sim.planners``, so sourcing them from the dataset would close an
# import cycle through ``sim.world``.
KEYS = ("agent_self", "agent_hist", "agent_valid", "map_feat",
        "rel_agent", "rel_map", "mask_agent", "mask_map")


@dataclass(frozen=True)
class JointNetConfig:
    weights: str
    device: str
    hidden_size: int
    num_layers: int
    num_heads: int
    seed: int


def _require(cfg, key: str):
    node = cfg
    for part in key.split("."):
        if part not in node:
            raise KeyError(f"joint smart planner yaml is missing required key {key!r}")
        node = node[part]
    if node is None:
        raise ValueError(f"joint smart planner yaml: {key!r} must have a value, got null")
    return node


def load_net_config(cfg: Any) -> JointNetConfig:
    cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    )
    weights = str(_require(cfg, "weights"))
    if weights != "random" and not Path(weights).exists():
        raise FileNotFoundError(f"joint smart weights do not exist: {weights}")
    device = str(_require(cfg, "device"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return JointNetConfig(
        weights=weights, device=device,
        hidden_size=int(_require(cfg, "policy.hidden_size")),
        num_layers=int(_require(cfg, "policy.num_layers")),
        num_heads=int(_require(cfg, "policy.num_heads")),
        seed=int(_require(cfg, "seed")),
    )


class RelativeAttention(nn.Module):
    """Query agents attend to a context whose geometry is given per (query, key).

    The relative embedding is added to BOTH key and value, which is how the
    query gets to see where the other thing is without either side being
    expressed in a shared global frame.
    """

    def __init__(self, d: int, heads: int, rel_features: int):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.rel = nn.Sequential(nn.Linear(rel_features, d), nn.GELU(), nn.Linear(d, d))
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, query, context, rel, mask):
        """query [B, A, d]; context [B, N, d]; rel [B, A, N, R]; mask [B, A, N] bool."""
        b, a, _ = query.shape
        n = context.shape[1]
        r = self.rel(rel)                                     # [B, A, N, d]
        q = self.q(self.norm(query)).view(b, a, self.h, self.dk)
        k = (self.k(context)[:, None] + r).view(b, a, n, self.h, self.dk)
        v = (self.v(context)[:, None] + r).view(b, a, n, self.h, self.dk)
        att = torch.einsum("bahd,banhd->bahn", q, k) / self.dk ** 0.5
        att = att.masked_fill(~mask[:, :, None, :], float("-inf"))
        # An agent with no valid context at all would softmax over all -inf.
        empty = ~mask.any(-1)
        att = torch.where(empty[:, :, None, None], torch.zeros_like(att), att)
        w = att.softmax(-1)
        o = torch.einsum("bahn,banhd->bahd", w, v).reshape(b, a, -1)
        return query + self.out(o) * (~empty)[:, :, None].to(query.dtype)


class JointTrafficNet(nn.Module):
    def __init__(self, cfg: JointNetConfig):
        super().__init__()
        d = cfg.hidden_size
        self.hidden_size = d
        self.self_proj = nn.Linear(SELF_FEATURES, d)
        self.hist_proj = nn.Linear(HISTORY_FEATURES, d)
        self.step_embed = nn.Embedding(HISTORY_STEPS, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=cfg.num_heads,
                                           dim_feedforward=4 * d, batch_first=True,
                                           norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        # Map tokens are encoded from the segment's OWN properties, so the scene's
        # map is embedded once; where each segment sits relative to a given agent
        # enters through the attention's relative term, not through this.
        self.map_proj = nn.Sequential(nn.Linear(MAP_FEATURES, d), nn.GELU(),
                                      nn.Linear(d, d))
        self.a2a = nn.ModuleList([RelativeAttention(d, cfg.num_heads, REL_AGENT_FEATURES)
                                  for _ in range(cfg.num_layers)])
        self.a2m = nn.ModuleList([RelativeAttention(d, cfg.num_heads, REL_MAP_FEATURES)
                                  for _ in range(cfg.num_layers)])
        self.ff = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            for _ in range(cfg.num_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, NUM_ACTIONS))
        self.register_buffer("steps", torch.arange(HISTORY_STEPS), persistent=False)

    def forward(self, batch: dict) -> torch.Tensor:
        """``batch`` holds per-scene tensors; returns action logits [B, A, 91]."""
        self_f, hist = batch["agent_self"], batch["agent_hist"]
        b, a = self_f.shape[:2]

        # ---- 1. temporal: each agent over its own motion, in its own frame ----
        tok = torch.cat([
            self.self_proj(self_f).reshape(b * a, 1, -1),
            (self.hist_proj(hist) + self.step_embed(self.steps)).reshape(b * a, HISTORY_STEPS, -1),
        ], dim=1)
        valid = torch.cat([
            torch.ones(b * a, 1, dtype=torch.bool, device=tok.device),
            (hist[..., -1] > 0.5).reshape(b * a, HISTORY_STEPS),
        ], dim=1)
        h = self.temporal(tok, src_key_padding_mask=~valid)[:, 0].view(b, a, -1)

        # ---- 2/3. interaction: other agents, then the map -------------------
        m = self.map_proj(batch["map_feat"])                   # [B, M, d], once per scene
        for a2a, a2m, ff in zip(self.a2a, self.a2m, self.ff):
            h = a2a(h, h, batch["rel_agent"], batch["mask_agent"])
            h = a2m(h, m, batch["rel_map"], batch["mask_map"])
            h = h + ff(h)
        return self.head(h)


_CACHE: dict[JointNetConfig, JointTrafficNet] = {}


def load_net(cfg: Any) -> JointTrafficNet:
    resolved = load_net_config(cfg)
    if resolved in _CACHE:
        return _CACHE[resolved]
    torch.manual_seed(resolved.seed)
    net = JointTrafficNet(resolved).to(resolved.device)
    if resolved.weights != "random":
        sd = torch.load(resolved.weights, map_location=resolved.device, weights_only=False)
        net.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    _CACHE[resolved] = net
    return net
