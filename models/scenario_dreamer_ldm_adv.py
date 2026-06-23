import torch
from torch_ema import ExponentialMovingAverage

from models.scenario_dreamer_ldm import ScenarioDreamerLDM
from nn_modules.ldm_adv import LDMAdv
from utils.data_container import ScenarioDreamerData
from utils.data_helpers import unnormalize_latents, unnormalize_scene
from utils.viz import visualize_batch


class ScenarioDreamerLDMAdv(ScenarioDreamerLDM):
    """Lightning wrapper for the latent diffusion model with one adversarial
    agent per scene.

    Reuses the entire :class:`ScenarioDreamerLDM` training / optimization /
    checkpointing machinery (including the frozen goal autoencoder loaded in the
    base ``__init__``) and only swaps in the :class:`LDMAdv` diffusion module and
    an ``adv``-aware ``forward`` that decodes the adversary alongside the scene.
    """

    def __init__(self, cfg, cfg_ae):
        super().__init__(cfg, cfg_ae)
        self.diff_model = LDMAdv(self.cfg)
        self.ema = ExponentialMovingAverage(self.diff_model.parameters(), decay=self.cfg.train.ema_decay)

    def _unnormalize_agent_like(self, agent_states, lane_states):
        """Unnormalize agent-layout decoded states (``[x, y, speed, cosθ, sinθ,
        length, width, goal_x, goal_y]``). ``unnormalize_scene`` transforms the
        agent columns independently of the lane tensor, so we can reuse it for
        the single adversarial agent (same layout) by passing a throwaway lane."""
        return unnormalize_scene(
            agent_states,
            lane_states,
            fov=self.cfg_dataset.fov,
            min_speed=self.cfg_dataset.min_speed,
            max_speed=self.cfg_dataset.max_speed,
            min_length=self.cfg_dataset.min_length,
            max_length=self.cfg_dataset.max_length,
            min_width=self.cfg_dataset.min_width,
            max_width=self.cfg_dataset.max_width,
            min_lane_x=self.cfg_dataset.min_lane_x,
            min_lane_y=self.cfg_dataset.min_lane_y,
            max_lane_x=self.cfg_dataset.max_lane_x,
            max_lane_y=self.cfg_dataset.max_lane_y,
        )

    def _decode_scene_and_adv(self, agent_latents, lane_latents, adv_latents, data):
        """Unnormalize the (normalized) latents and decode the scene with the
        adversary re-inserted into the full agent set.

        The adversary was split out of the agent set at load time, but the goal
        autoencoder's agent decoder is a permutation-equivariant set model
        (complete agent<->agent attention + lane->agent bipartite, no per-agent
        positional embedding and no special ego token). So we append each scene's
        adversary back onto that scene's agents, decode the full set in one pass
        -- exactly the distribution the autoencoder was trained on -- and split
        the adversary rows back out. This decodes the adversary *in context*
        rather than in isolation, which fixes the case where a context-less
        adversary collapsed onto the ego at the origin."""
        agent_latents, lane_latents = unnormalize_latents(
            agent_latents,
            lane_latents,
            self.cfg_dataset.agent_latents_mean,
            self.cfg_dataset.agent_latents_std,
            self.cfg_dataset.lane_latents_mean,
            self.cfg_dataset.lane_latents_std,
        )
        # The adversary shares the agent latent statistics (it is an agent latent).
        adv_latents = adv_latents * self.cfg_dataset.agent_latents_std + self.cfg_dataset.agent_latents_mean

        # Scene-grouping batch vectors (fall back to a single unbatched scene).
        agent_batch = data["agent"].batch if "batch" in data["agent"] else torch.zeros(
            agent_latents.shape[0], device=agent_latents.device, dtype=torch.long)
        lane_batch = data["lane"].batch if "batch" in data["lane"] else torch.zeros(
            lane_latents.shape[0], device=lane_latents.device, dtype=torch.long)
        adv_batch = data["adv"].batch if "batch" in data["adv"] else torch.arange(
            adv_latents.shape[0], device=adv_latents.device, dtype=torch.long)

        num_agents = agent_latents.shape[0]

        # Append the per-scene advs as a contiguous tail block. Order within a
        # scene is irrelevant to the set decoder, so appending avoids any
        # mid-array reindexing while keeping the ego at each scene's first row.
        combined_latents = torch.cat([agent_latents, adv_latents], dim=0)
        combined_batch = torch.cat([agent_batch, adv_batch], dim=0)

        # Rebuild the decode graph over the combined agent set from the batch
        # vectors: a per-scene complete agent graph (with self-loops, matching
        # get_edge_index_complete_graph) and an all-lanes->all-agents bipartite
        # graph. Agent counts per scene are small, so the dense masks are cheap.
        same_scene = combined_batch.unsqueeze(0) == combined_batch.unsqueeze(1)
        a2a_edge_index = same_scene.nonzero(as_tuple=False).t().contiguous()
        l2a_src, l2a_dst = (lane_batch.unsqueeze(1) == combined_batch.unsqueeze(0)).nonzero(as_tuple=True)
        l2a_edge_index = torch.stack([l2a_src, l2a_dst], dim=0)

        dec_data = ScenarioDreamerData()
        dec_data["lane"].x = lane_latents
        dec_data["lane", "to", "lane"].edge_index = data["lane", "to", "lane"].edge_index
        dec_data["agent", "to", "agent"].edge_index = a2a_edge_index
        dec_data["lane", "to", "agent"].edge_index = l2a_edge_index

        agent_samples, lane_samples, agent_types, lane_types, lane_conn_samples = self.autoencoder.model.forward_decoder(
            combined_latents, lane_latents, dec_data
        )
        agent_samples, lane_samples = self._unnormalize_agent_like(agent_samples, lane_samples)

        # Split the appended adversary rows back out; the scene-agent rows keep
        # their original order (ego stays at each scene's first row).
        adv_samples, adv_types = agent_samples[num_agents:], agent_types[num_agents:]
        agent_samples, agent_types = agent_samples[:num_agents], agent_types[:num_agents]
        return agent_samples, lane_samples, agent_types, lane_types, lane_conn_samples, adv_samples, adv_types

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

        if "batch" in data["adv"]:
            adv_batch = data["adv"].batch
        else:
            adv_batch = torch.arange(data.batch_size, device=self.device, dtype=torch.long)

        # Snapshot the ground-truth (normalized) latents before diffusion runs so
        # we can decode and render them next to the generated samples.
        if visualize:
            gt_agent_latents = data["agent"].latents.clone().float()
            gt_lane_latents = data["lane"].latents.clone().float()
            gt_adv_latents = data["adv"].latents.clone().float()

        agent_latents, lane_latents, adv_latents = self.diff_model.forward(data, mode=mode)
        (
            agent_samples,
            lane_samples,
            agent_types,
            lane_types,
            lane_conn_samples,
            adv_samples,
            adv_types,
        ) = self._decode_scene_and_adv(agent_latents, lane_latents, adv_latents, data)

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
                adv_samples=adv_samples,
                adv_batch=adv_batch,
                adv_types=adv_types,
            )

            # Decode + render the ground-truth latents under a separate tag.
            (
                gt_agent,
                gt_lane,
                gt_agent_types,
                gt_lane_types,
                gt_lane_conn,
                gt_adv,
                gt_adv_types,
            ) = self._decode_scene_and_adv(gt_agent_latents, gt_lane_latents, gt_adv_latents, data)
            gt_images = visualize_batch(
                num_samples_to_visualize,
                gt_agent,
                gt_lane,
                gt_agent_types,
                gt_lane_types,
                gt_lane_conn,
                data,
                viz_dir,
                epoch=self.current_epoch,
                batch_idx=batch_idx,
                save_wandb=save_wandb,
                tag="scene_plot_gt",
                adv_samples=gt_adv,
                adv_batch=adv_batch,
                adv_types=gt_adv_types,
            )
            if save_wandb and images_to_log_batch is not None and gt_images is not None:
                images_to_log_batch.update(gt_images)
        else:
            images_to_log_batch = None

        data["agent"].x = agent_samples
        data["lane"].x = lane_samples
        data["agent"].type = torch.nn.functional.one_hot(agent_types, num_classes=self.cfg_dataset.num_agent_types)
        if self.cfg.dataset_name == "nuplan":
            data["lane"].type = torch.nn.functional.one_hot(lane_types, num_classes=self.cfg_dataset.num_lane_types)
        data["lane", "to", "lane"].type = lane_conn_samples
        data["adv"].x = adv_samples
        data["adv"].type = torch.nn.functional.one_hot(adv_types, num_classes=self.cfg_dataset.num_agent_types)
        return data, images_to_log_batch
