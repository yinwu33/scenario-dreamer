"""Frozen PufferDrive planner policy, re-implemented in plain torch.

Exact architectural port of ``pufferlib.pacific.torch.Drive`` (PufferDrive repo)
for the configuration used by the DDPO reward rollout:

  * dynamics_model = "classic"  -> ego features = 11
  * action_type    = "discrete" -> single MultiDiscrete head of 7*13 = 91 actions
  * obs layout = [ego(11) | partners(63*7) | road(512*7)]

The state_dict produced by PufferDrive training (``selfplay_drive_*.pt``) loads
directly: layer names/shapes match the original module. Planner-specific settings
are read from ``planner/selfplay_drive/config.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf

# Constants mirrored from PufferDrive pacific/drive.h (classic dynamics, discrete actions).
MAX_AGENTS = 64
EGO_FEATURES = 11                  # EGO_FEATURES_CLASSIC
PARTNER_FEATURES = 7
MAX_PARTNER_OBJECTS = MAX_AGENTS - 1
ROAD_FEATURES = 7
MAX_ROAD_OBJECTS = 512             # MAX_ROAD_SEGMENT_OBSERVATIONS
OBS_DIM = EGO_FEATURES + MAX_PARTNER_OBJECTS * PARTNER_FEATURES + MAX_ROAD_OBJECTS * ROAD_FEATURES
NUM_ACTIONS = 7 * 13               # accel_idx * 13 + steer_idx
CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class PlannerConfig:
    checkpoint: Path
    device: str
    input_size: int
    hidden_size: int
    deterministic: bool


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_planner_config() -> PlannerConfig:
    raw = OmegaConf.load(CONFIG_PATH)
    checkpoint = Path(str(raw.get("checkpoint", "selfplay_drive_178121292262.pt")))
    if not checkpoint.is_absolute():
        checkpoint = CONFIG_PATH.parent / checkpoint
    return PlannerConfig(
        checkpoint=checkpoint,
        device=_resolve_device(str(raw.get("device", "auto"))),
        input_size=int(raw.get("input_size", 64)),
        hidden_size=int(raw.get("hidden_size", 256)),
        deterministic=bool(raw.get("deterministic", True)),
    )


class DrivePlanner(nn.Module):
    """Port of pufferlib.pacific.torch.Drive (discrete / classic only)."""

    def __init__(self):
        super().__init__()
        cfg = load_planner_config()
        input_size = cfg.input_size
        hidden_size = cfg.hidden_size
        self.deterministic = cfg.deterministic
        self.ego_dim = EGO_FEATURES
        self.partner_features = PARTNER_FEATURES
        self.max_partner_objects = MAX_PARTNER_OBJECTS
        self.road_features = ROAD_FEATURES
        self.max_road_objects = MAX_ROAD_OBJECTS
        road_features_after_onehot = ROAD_FEATURES + 6  # categorical -> 7-way one-hot

        self.ego_encoder = nn.Sequential(
            nn.Linear(self.ego_dim, input_size),
            nn.LayerNorm(input_size),
            nn.Linear(input_size, input_size),
        )
        self.road_encoder = nn.Sequential(
            nn.Linear(road_features_after_onehot, input_size),
            nn.LayerNorm(input_size),
            nn.Linear(input_size, input_size),
        )
        self.partner_encoder = nn.Sequential(
            nn.Linear(self.partner_features, input_size),
            nn.LayerNorm(input_size),
            nn.Linear(input_size, input_size),
        )
        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            nn.Linear(3 * input_size, hidden_size),
        )
        self.actor = nn.Linear(hidden_size, NUM_ACTIONS)
        self.value_fn = nn.Linear(hidden_size, 1)

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)

        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)
        return F.relu(self.shared_embedding(concat_features))

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_observations(observations)
        return self.actor(hidden), self.value_fn(hidden)

    @torch.no_grad()
    def act(self, observations: torch.Tensor, deterministic: bool | None = None) -> torch.Tensor:
        if deterministic is None:
            deterministic = self.deterministic
        logits, _ = self.forward(observations)
        if deterministic:
            return logits.argmax(dim=-1)
        return torch.distributions.Categorical(logits=logits).sample()


def load_planner() -> DrivePlanner:
    cfg = load_planner_config()
    planner = DrivePlanner().to(cfg.device)
    sd = torch.load(cfg.checkpoint, map_location=cfg.device, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    planner.load_state_dict(sd)
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)
    return planner
