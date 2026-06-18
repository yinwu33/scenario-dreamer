"""Declarative model registry shared by train.py and eval.py.

Replaces the duplicated ``model_name`` if/elif chains that (a) selected which
root-config child node to collapse to (``cfg.ae`` / ``cfg.ldm`` / ``cfg.dm_goal``
/ ...) and injected ``dataset_name``, and (b) dispatched to a trainer/eval
function with the right ``model_cls``.

``ddpo`` is intentionally NOT in this registry: it has its own RL loop and needs
the full root cfg, so train.py dispatches it before collapsing (see train.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import OmegaConf

from models.scenario_dreamer_autoencoder import ScenarioDreamerAutoEncoder
from models.scenario_dreamer_autoencoder_bezier import ScenarioDreamerAutoEncoderBezier
from models.scenario_dreamer_ldm import ScenarioDreamerLDM
from models.scenario_dreamer_dm import ScenarioDreamerDM
from models.scenario_dreamer_dm_goal import ScenarioDreamerDMGoal
from models.scenario_dreamer_dm_fixed_map_agent_goal import ScenarioDreamerDMFixedMapAgentGoal
from models.scenario_dreamer_cldm import ScenarioDreamerCLDM
from models.ctrl_sim import CtRLSim


@dataclass(frozen=True)
class ModelSpec:
    """How to collapse the root cfg and which trainer/model to use for a model_name.

    ``cfg_attr``  : root-config child node carrying this model's train/eval/model/...
    ``kind``      : trainer/eval family ('autoencoder' | 'ldm' | 'dm' | 'ctrl_sim').
    ``model_cls`` : Lightning module class to instantiate.
    ``ae_attr``   : root-config child node with the (frozen) autoencoder cfg, for the
                    latent-diffusion family ('ldm'); ``None`` otherwise.
    """

    cfg_attr: str
    kind: str
    model_cls: type
    ae_attr: str | None = None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "autoencoder": ModelSpec("ae", "autoencoder", ScenarioDreamerAutoEncoder),
    "autoencoder_goal": ModelSpec("ae_goal", "autoencoder", ScenarioDreamerAutoEncoder),
    "autoencoder_bezier": ModelSpec("ae", "autoencoder", ScenarioDreamerAutoEncoderBezier),
    "ldm": ModelSpec("ldm", "ldm", ScenarioDreamerLDM, ae_attr="ae"),
    "ldm_goal": ModelSpec("ldm_goal", "ldm", ScenarioDreamerLDM, ae_attr="ae_goal"),
    "cldm": ModelSpec("ldm", "ldm", ScenarioDreamerCLDM, ae_attr="ae"),
    "dm": ModelSpec("dm", "dm", ScenarioDreamerDM),
    "dm_goal": ModelSpec("dm_goal", "dm", ScenarioDreamerDMGoal),
    "dm_fixed_map_agent_goal": ModelSpec(
        "dm_fixed_map_agent_goal", "dm", ScenarioDreamerDMFixedMapAgentGoal
    ),
    "ctrl_sim": ModelSpec("ctrl_sim", "ctrl_sim", CtRLSim),
}


def _inject_dataset_name(cfg_node, dataset_name: str) -> None:
    """Set ``cfg_node.dataset_name`` through the struct lock (nuplan vs waymo)."""
    OmegaConf.set_struct(cfg_node, False)
    cfg_node.dataset_name = dataset_name
    OmegaConf.set_struct(cfg_node, True)


def collapse_cfg(cfg, model_name: str):
    """Collapse the root cfg to one model's node, injecting ``dataset_name``.

    Returns ``(spec, cfg_node, cfg_ae)`` where ``cfg_ae`` is ``None`` unless the
    model is in the latent-diffusion family. Mirrors the old per-branch logic in
    train.py / eval.py but driven by ``MODEL_REGISTRY``.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name: {model_name!r}")
    spec = MODEL_REGISTRY[model_name]
    dataset_name = cfg.dataset_name.name

    cfg_node = getattr(cfg, spec.cfg_attr)
    _inject_dataset_name(cfg_node, dataset_name)

    cfg_ae = None
    if spec.ae_attr is not None:
        cfg_ae = getattr(cfg, spec.ae_attr)
        _inject_dataset_name(cfg_ae, dataset_name)

    return spec, cfg_node, cfg_ae
