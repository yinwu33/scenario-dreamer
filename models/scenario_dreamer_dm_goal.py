import torch
from torch_ema import ExponentialMovingAverage

from models.scenario_dreamer_dm import ScenarioDreamerDM
from nn_modules.dm_goal import DMGoal
from utils.data_helpers import unnormalize_scene
from utils.viz import visualize_batch


def unnormalize_scene_with_goal(agent_states, lane_states, cfg_dataset):
    agent_states, lane_states = unnormalize_scene(
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
    if agent_states.shape[-1] >= 9:
        agent_states[:, 7] = ((torch.clip(agent_states[:, 7], -1, 1) + 1) / 2) * cfg_dataset.fov - cfg_dataset.fov / 2
        agent_states[:, 8] = ((torch.clip(agent_states[:, 8], -1, 1) + 1) / 2) * cfg_dataset.fov - cfg_dataset.fov / 2
    return agent_states, lane_states


class ScenarioDreamerDMGoal(ScenarioDreamerDM):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.diff_model = DMGoal(self.cfg)
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
        agent_samples, lane_samples, agent_types, lane_types, lane_conn_samples = self.diff_model.forward(data, mode=mode)
        agent_samples, lane_samples = unnormalize_scene_with_goal(agent_samples, lane_samples, self.cfg_dataset)

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
        return data, images_to_log_batch
