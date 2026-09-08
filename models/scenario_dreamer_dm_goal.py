import torch
from torch_ema import ExponentialMovingAverage

from cfgs.config import NON_PARTITIONED
from models.scenario_dreamer_dm import ScenarioDreamerDM
from nn_modules.dm_goal import DMGoal
from utils.data_helpers import unnormalize_scene
from utils.viz import visualize_batch


def unnormalize_scene_with_goal(agent_states, lane_states, cfg_dataset):
    # NOTE: goal columns [7, 8] are unnormalized by ``unnormalize_scene`` itself
    # (FOV-frame, same as init position). Do NOT re-apply the goal transform here:
    # the second clip(-1, 1) would treat the already-metre-scale goal as a
    # normalized value and saturate it to the four FOV corners (±fov/2, ±fov/2).
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


class ScenarioDreamerDMGoal(ScenarioDreamerDM):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.diff_model = DMGoal(self.cfg)
        self.ema = ExponentialMovingAverage(self.diff_model.parameters(), decay=self.cfg.train.ema_decay)

    def _initialize_pyg_dset(self, mode, num_samples, conditioning_path=None, nocturne_compatible_only=False):
        """Build the generation dataset by sampling ``(num_lanes, num_agents)``
        layouts from ``eval.init_prob_matrix_path``.

        Overrides the baseline DM path, which indexes a **3D**
        ``(num_map_ids, num_lanes, num_agents)`` prior by ``map_id``. The v2 goal
        records carry no nocturne metadata, so their prior
        (``initial_prob_matrix_goal_waymo.pt``, built by
        ``scripts/create_goal_init_prob_matrix.py``) is a **2D** joint
        distribution with no map_id dimension -- the same shape ldm_adv uses, so
        the two baselines sample layouts from an identical prior.
        """
        if mode != "initial_scene":
            return super()._initialize_pyg_dset(mode, num_samples, conditioning_path, nocturne_compatible_only)
        if self.cfg.dataset_name != "waymo":
            raise ValueError("dm_goal generation currently supports Waymo only.")

        prior = self.init_prob_matrix
        expected = (self.cfg_dataset.max_num_lanes + 1, self.cfg_dataset.max_num_agents + 1)
        assert tuple(prior.shape) == expected, (
            f"Expected a 2D init_prob_matrix with shape {expected}, got "
            f"{tuple(prior.shape)} from {self.cfg.eval.init_prob_matrix_path}"
        )

        flatten_probs = prior.reshape(-1)
        if not torch.isfinite(flatten_probs).all():
            raise ValueError("init_prob_matrix contains NaN or Inf")
        if (flatten_probs < 0).any():
            raise ValueError("init_prob_matrix contains negative probabilities")
        if flatten_probs.sum() <= 0:
            raise ValueError(f"Empty init_prob_matrix at {self.cfg.eval.init_prob_matrix_path}")

        data_list = []
        for _ in range(num_samples):
            prob_idx = int(torch.multinomial(flatten_probs, 1).item())
            num_lanes = prob_idx // (self.cfg_dataset.max_num_agents + 1)
            num_agents = prob_idx % (self.cfg_dataset.max_num_agents + 1)
            assert num_lanes > 0, "Generating scene with no lanes"
            # The guidance baseline needs the ego plus at least one other agent to
            # push; the prior is built with the same >= 2 constraint.
            assert num_agents >= 2, (
                "Guided adversarial generation needs the ego plus at least one "
                "non-ego agent; rebuild the prior with --min-num-agents 2."
            )
            data_list.append(self._empty_scene(num_lanes, num_agents, 0, NON_PARTITIONED))

        return data_list, None

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
        # Snapshot the (normalized) ground-truth scene before diffusion overwrites it,
        # so we can render it alongside the generated samples for comparison.
        gt_agent = data["agent"].x.clone().float()
        gt_lane = data["lane"].x.clone().float()
        gt_agent_types = torch.argmax(data["agent"].type, dim=-1)

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
                tag="scene_plot",
            )

            gt_agent, gt_lane = unnormalize_scene_with_goal(gt_agent, gt_lane, self.cfg_dataset)
            gt_images = visualize_batch(
                num_samples_to_visualize,
                gt_agent,
                gt_lane,
                gt_agent_types,
                lane_types,
                lane_conn_samples,
                data,
                viz_dir,
                epoch=self.current_epoch,
                batch_idx=batch_idx,
                save_wandb=save_wandb,
                tag="scene_plot_gt",
            )
            if save_wandb and images_to_log_batch is not None and gt_images is not None:
                images_to_log_batch.update(gt_images)
        else:
            images_to_log_batch = None

        data["agent"].x = agent_samples
        data["lane"].x = lane_samples
        data["agent"].type = torch.nn.functional.one_hot(agent_types, num_classes=self.cfg_dataset.num_agent_types)
        data["lane", "to", "lane"].type = lane_conn_samples
        return data, images_to_log_batch
