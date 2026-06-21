import torch
from torch_ema import ExponentialMovingAverage

from models.scenario_dreamer_dm import ScenarioDreamerDM
from nn_modules.dm_adv import DMAdv
from utils.data_helpers import unnormalize_scene
from utils.viz import visualize_batch


def _unnormalize_agent_like(agent_states, lane_states, cfg_dataset):
    """Unnormalize agent-layout states (``[x, y, speed, cosθ, sinθ, length, width]``).

    ``unnormalize_scene`` transforms the agent columns independently of the lane
    tensor, so we can reuse it to unnormalize either the normal agents or the
    single adversarial agent (same layout). The lane tensor is unnormalized in
    place too; callers that only want the agent result pass a throwaway clone.
    """
    return unnormalize_scene(
        agent_states,
        lane_states,
        fov=cfg_dataset.fov,
        min_speed=cfg_dataset.min_speed,
        max_speed=cfg_dataset.max_speed,
        min_length=cfg_dataset.min_length,
        max_length=cfg_dataset.max_length,
        min_width=cfg_dataset.min_width,
        max_width=cfg_dataset.max_width,
        min_lane_x=cfg_dataset.min_lane_x,
        min_lane_y=cfg_dataset.min_lane_y,
        max_lane_x=cfg_dataset.max_lane_x,
        max_lane_y=cfg_dataset.max_lane_y,
    )


class ScenarioDreamerDMAdv(ScenarioDreamerDM):
    """Lightning wrapper for the direct diffusion model with one adversarial
    agent per scene. Reuses the entire ``ScenarioDreamerDM`` training /
    optimization / checkpointing machinery and only swaps in the ``DMAdv``
    diffusion module and an ``adv``-aware ``forward``."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.diff_model = DMAdv(self.cfg)
        self.ema = ExponentialMovingAverage(self.diff_model.parameters(), decay=self.cfg.train.ema_decay)

    def forward(
        self,
        data,
        mode,
        batch_idx,
        viz_dir=None,
        visualize=False,
        save_wandb=False,
        num_samples_to_visualize=None,
    ):
        data = data.to(self.device)
        (
            agent_samples,
            lane_samples,
            agent_types,
            lane_types,
            lane_conn_samples,
            adv_samples,
            adv_types,
        ) = self.diff_model.forward(data, mode=mode)

        # Unnormalize the adversary first (using a throwaway copy of the still
        # normalized lane), then the normal agents and lanes in place.
        adv_samples, _ = _unnormalize_agent_like(adv_samples, lane_samples.clone(), self.cfg_dataset)
        agent_samples, lane_samples = _unnormalize_agent_like(agent_samples, lane_samples, self.cfg_dataset)

        if visualize:
            if num_samples_to_visualize is None:
                num_samples_to_visualize = data.batch_size
            images_to_log_batch = visualize_batch(
                num_samples_to_visualize,
                agent_samples,
                lane_samples,
                agent_types,
                lane_types,
                lane_conn_samples,
                data,
                viz_dir,
                epoch=self.current_epoch,
                batch_idx=batch_idx,
                save_wandb=save_wandb,
            )
        else:
            images_to_log_batch = None

        data["agent"].x = agent_samples
        data["lane"].x = lane_samples
        data["agent"].type = torch.nn.functional.one_hot(agent_types, num_classes=self.cfg_dataset.num_agent_types)
        data["lane", "to", "lane"].type = lane_conn_samples
        data["adv"].x = adv_samples
        data["adv"].type = torch.nn.functional.one_hot(adv_types, num_classes=self.cfg_dataset.num_agent_types)
        return data, images_to_log_batch
