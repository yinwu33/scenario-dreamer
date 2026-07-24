"""Frozen PufferDrive planner policy, re-implemented in plain torch.

Exact architectural port of ``pufferlib.pacific.torch.Drive`` (PufferDrive repo)
for the configuration used by the DDPO reward rollout:

  * dynamics_model = "classic"  -> ego features = 11
  * action_type    = "discrete" -> single MultiDiscrete head of 7*13 = 91 actions
  * obs layout = [ego(11) | partners(63*7) | road(512*7)]

The state_dict produced by PufferDrive training (``selfplay_drive_*.pt`` or a
recurrent bad_driver checkpoint) loads directly: layer names/shapes match the
original module. Planner-specific settings are read from
``planner/selfplay_drive/config.yaml`` and can be overridden by Hydra planner
configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    rnn_name: str | None = None
    rnn_input_size: int | None = None
    rnn_hidden_size: int | None = None

    @property
    def recurrent(self) -> bool:
        return self.rnn_name is not None


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _to_config(overrides: Any | None):
    if overrides is None:
        return OmegaConf.create({})
    if OmegaConf.is_config(overrides):
        return OmegaConf.create(OmegaConf.to_container(overrides, resolve=True))
    return OmegaConf.create(overrides)


def _none_string(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() == "none":
        return None
    return text


def load_planner_config(overrides: Any | None = None) -> PlannerConfig:
    user = _to_config(overrides)
    raw = OmegaConf.load(CONFIG_PATH)
    base_dir = CONFIG_PATH.parent

    config_path = user.get("config_path", None)
    if config_path is not None:
        config_path = Path(str(config_path))
        raw = OmegaConf.merge(raw, OmegaConf.load(config_path))
        base_dir = config_path.parent

    raw = OmegaConf.merge(raw, user)
    policy_cfg = raw.get("policy", {})
    rnn_cfg = raw.get("rnn", {})

    checkpoint = Path(str(raw.get("checkpoint", "selfplay_drive_178121292262.pt")))
    if not checkpoint.is_absolute():
        checkpoint = base_dir / checkpoint
    rnn_name = _none_string(raw.get("rnn_name", None))
    return PlannerConfig(
        checkpoint=checkpoint,
        device=_resolve_device(str(raw.get("device", "auto"))),
        input_size=int(raw.get("input_size", policy_cfg.get("input_size", 64))),
        hidden_size=int(raw.get("hidden_size", policy_cfg.get("hidden_size", 256))),
        deterministic=bool(raw.get("deterministic", True)),
        rnn_name=rnn_name,
        rnn_input_size=int(raw.get("rnn_input_size", rnn_cfg.get("input_size", 256))),
        rnn_hidden_size=int(raw.get("rnn_hidden_size", rnn_cfg.get("hidden_size", 256))),
    )


class DrivePlanner(nn.Module):
    """Port of pufferlib.pacific.torch.Drive (discrete / classic only)."""

    def __init__(self, cfg: PlannerConfig | None = None):
        super().__init__()
        cfg = cfg or load_planner_config()
        input_size = cfg.input_size
        hidden_size = cfg.hidden_size
        self.deterministic = cfg.deterministic
        self.recurrent = False
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


class RecurrentDrivePlanner(nn.Module):
    """Inference-only port of pufferlib.models.LSTMWrapper around Drive."""

    def __init__(self, cfg: PlannerConfig):
        super().__init__()
        if cfg.rnn_name != "Recurrent":
            raise ValueError(f"unsupported rnn_name {cfg.rnn_name!r}; expected 'Recurrent'")
        self.policy = DrivePlanner(cfg)
        self.input_size = int(cfg.rnn_input_size or cfg.hidden_size)
        self.hidden_size = int(cfg.rnn_hidden_size or cfg.hidden_size)
        self.deterministic = cfg.deterministic
        self.recurrent = True
        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.cell = nn.LSTMCell(self.input_size, self.hidden_size)
        # Match pufferlib.models.LSTMWrapper state_dict names while sharing params.
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0

    def initial_state(self, rows: int, *, device: str | torch.device | None = None) -> dict[str, torch.Tensor]:
        device = device or next(self.parameters()).device
        return {
            "lstm_h": torch.zeros(rows, self.hidden_size, device=device),
            "lstm_c": torch.zeros(rows, self.hidden_size, device=device),
        }

    def forward_eval(
        self, observations: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.policy.encode_observations(observations)
        h = state.get("lstm_h")
        c = state.get("lstm_c")
        lstm_state = None if h is None else (h, c)
        hidden, c = self.cell(hidden, lstm_state)
        state["hidden"] = hidden
        state["lstm_h"] = hidden
        state["lstm_c"] = c
        return self.policy.actor(hidden), self.policy.value_fn(hidden)

    @torch.no_grad()
    def act(
        self,
        observations: torch.Tensor,
        *,
        state: dict[str, torch.Tensor],
        deterministic: bool | None = None,
    ) -> torch.Tensor:
        if deterministic is None:
            deterministic = self.deterministic
        logits, _ = self.forward_eval(observations, state)
        if deterministic:
            return logits.argmax(dim=-1)
        return torch.distributions.Categorical(logits=logits).sample()


def _load_state_dict(planner: nn.Module, state_dict: dict[str, torch.Tensor], *, recurrent: bool) -> None:
    try:
        planner.load_state_dict(state_dict)
        return
    except RuntimeError:
        if not recurrent:
            raise
    missing, unexpected = planner.load_state_dict(state_dict, strict=False)
    allowed_missing = [key for key in missing if key.startswith("cell.")]
    if len(allowed_missing) != len(missing) or unexpected:
        raise RuntimeError(
            "checkpoint did not match recurrent Drive planner; "
            f"missing={missing}, unexpected={unexpected}"
        )


# Loaded nets keyed by their resolved (frozen, hashable) config: the per-role
# rollout planners typically share one checkpoint, and the net is frozen/eval
# with all recurrent state held by the caller, so sharing one instance is safe.
_PLANNER_CACHE: dict[PlannerConfig, "DrivePlanner | RecurrentDrivePlanner"] = {}


def load_planner(overrides: Any | None = None) -> DrivePlanner | RecurrentDrivePlanner:
    cfg = load_planner_config(overrides)
    cached = _PLANNER_CACHE.get(cfg)
    if cached is not None:
        return cached
    planner: DrivePlanner | RecurrentDrivePlanner
    planner = RecurrentDrivePlanner(cfg) if cfg.recurrent else DrivePlanner(cfg)
    planner = planner.to(cfg.device)
    sd = torch.load(cfg.checkpoint, map_location=cfg.device, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in sd.items()
    }
    _load_state_dict(planner, sd, recurrent=cfg.recurrent)
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)
    _PLANNER_CACHE[cfg] = planner
    return planner
