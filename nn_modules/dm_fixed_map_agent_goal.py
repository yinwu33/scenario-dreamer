import torch
import torch.nn.functional as F

from cfgs.config import BEFORE_PARTITION
from nn_modules.dm import DM


class DMFixedMapAgentGoal(DM):
    """Direct diffusion with a fixed map and generated agent init/goal state.

    Agent latent layout:
      [agent_state(9), parking_logits(2), agent_type_logits(num_agent_types)].
    The lane graph is always supplied by the dataset and is not diffused.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.parking_dim = int(self.cfg_model.get("parking_dim", 2))
        self.parking_class_id = int(self.cfg_model.get("parking_class_id", 1))
        self.goal_loss_for_parking = bool(self.cfg_model.get("goal_loss_for_parking", False))

    def _agent_target(self, data):
        return torch.cat(
            [
                data["agent"].x.float(),
                data["agent"].parking.float(),
                data["agent"].type.float(),
            ],
            dim=-1,
        ).unsqueeze(1)

    def _split_agent(self, x_agent):
        state_dim = self.cfg_model.state_dim
        parking_start = state_dim
        parking_end = parking_start + self.parking_dim
        return (
            x_agent[:, :state_dim],
            x_agent[:, parking_start:parking_end],
            x_agent[:, parking_end:],
        )

    def _mean_per_scene(self, values, batch, batch_size=None):
        if batch_size is None:
            batch_size = int(batch.max().detach()) + 1
        out = torch.zeros(batch_size, device=values.device, dtype=values.dtype)
        count = torch.zeros(batch_size, device=values.device, dtype=values.dtype)
        out.scatter_add_(0, batch.long(), values)
        count.scatter_add_(0, batch.long(), torch.ones_like(values))
        return out / count.clamp_min(1.0)

    def _masked_agent_noise_loss(self, pred, target, data):
        loss = 0.5 * F.mse_loss(pred, target, reduction="none")
        dim_mask = torch.ones_like(loss)

        if not self.goal_loss_for_parking:
            parking_label = data["agent"].parking_label.long()
            parking_agent = parking_label == self.parking_class_id
            dim_mask[parking_agent, :, 7:9] = 0.0

        if "partition_mask" in data["agent"]:
            agent_mask = data["agent"].partition_mask == BEFORE_PARTITION
            dim_mask[agent_mask] = 0.0

        per_agent = (loss * dim_mask).sum(dim=(1, 2)) / dim_mask.sum(dim=(1, 2)).clamp_min(1.0)
        return self._mean_per_scene(per_agent, data["agent"].batch, data.batch_size)

    def _masked_parking_loss(self, parking_logits, data):
        parking_label = data["agent"].parking_label.long()
        loss = F.cross_entropy(parking_logits, parking_label, reduction="none")
        if "partition_mask" in data["agent"]:
            keep = data["agent"].partition_mask != BEFORE_PARTITION
            loss = loss * keep.float()
            denom = self._mean_per_scene(keep.float(), data["agent"].batch, data.batch_size).clamp_min(1e-6)
            return self._mean_per_scene(loss, data["agent"].batch, data.batch_size) / denom
        return self._mean_per_scene(loss, data["agent"].batch, data.batch_size)

    def _masked_goal_loss(self, agent_states, data):
        parking_label = data["agent"].parking_label.long()
        keep = parking_label != self.parking_class_id
        if "partition_mask" in data["agent"]:
            keep = keep & (data["agent"].partition_mask != BEFORE_PARTITION)
        per_agent = 0.5 * F.mse_loss(agent_states[:, 7:9], data["agent"].x.float()[:, 7:9], reduction="none").mean(dim=-1)
        per_agent = per_agent * keep.float()
        denom = self._mean_per_scene(keep.float(), data["agent"].batch, data.batch_size).clamp_min(1e-6)
        return self._mean_per_scene(per_agent, data["agent"].batch, data.batch_size) / denom

    def p_mean_variance(self, x_agent, x_lane, data, t_agent, t_lane):
        t_lane = torch.zeros_like(t_lane)
        return super().p_mean_variance(x_agent, x_lane, data, t_agent, t_lane)

    @torch.no_grad()
    def p_sample_loop(self, agent_shape, lane_shape, data, device="cuda", mode="initial_scene"):
        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        batch_size = data.batch_size

        x_agent = torch.randn(agent_shape, device=device)
        x_lane = self._lane_target(data).to(device)

        if mode in ("train", "inpainting"):
            target_agent = self._agent_target(data).to(device)
            if "partition_mask" in data["agent"]:
                agent_mask = data["agent"].partition_mask == BEFORE_PARTITION
                x_agent[agent_mask] = target_agent[agent_mask]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_agent = timesteps[agent_batch]
            t_lane = torch.zeros_like(timesteps[lane_batch])
            x_agent, _ = self.p_sample(x_agent, x_lane, data, t_agent, t_lane)
            x_agent = torch.clip(x_agent, -self.cfg_model.diffusion_clip, self.cfg_model.diffusion_clip)

            if mode in ("train", "inpainting") and "partition_mask" in data["agent"]:
                agent_mask = data["agent"].partition_mask == BEFORE_PARTITION
                x_agent[agent_mask] = target_agent[agent_mask]

        return x_agent[:, 0], x_lane[:, 0]

    def p_losses(self, x_agent, x_lane, data, t_agent, t_lane):
        agent_noise = torch.randn_like(x_agent)
        x_agent_noisy = self.q_sample(x_start=x_agent, t=t_agent, noise=agent_noise)

        if "partition_mask" in data["agent"]:
            agent_mask = data["agent"].partition_mask == BEFORE_PARTITION
            x_agent_noisy[agent_mask] = x_agent[agent_mask]

        t_lane = torch.zeros_like(t_lane)
        agent_noise_pred, _ = self.model(
            x_agent_noisy,
            x_lane,
            data,
            t_agent,
            t_lane,
        )

        if "partition_mask" in data["agent"]:
            agent_noise = agent_noise.clone()
            agent_noise[agent_mask] = 0.0

        agent_loss = self._masked_agent_noise_loss(agent_noise_pred, agent_noise, data)

        x_agent_recon = self.predict_start_from_noise(x_agent_noisy, t_agent.to(torch.int64), agent_noise_pred)[:, 0]
        agent_states, parking_logits, agent_type_logits = self._split_agent(x_agent_recon)
        parking_loss = self._masked_parking_loss(parking_logits, data)
        goal_loss = self._masked_goal_loss(agent_states, data)
        agent_type_loss = self.agent_type_loss_fn(agent_type_logits, data["agent"].type.float(), data["agent"].batch)
        lane_loss = torch.zeros_like(agent_loss)

        loss = (
            agent_loss
            + self.cfg.train.parking_cls_weight * parking_loss
            + self.cfg.train.goal_weight * goal_loss
            + self.cfg.train.agent_type_weight * agent_type_loss
        )
        return loss, agent_loss, lane_loss, agent_type_loss, parking_loss, goal_loss

    def loss(self, data):
        x_agent = self._agent_target(data)
        x_lane = self._lane_target(data)

        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        batch_size = data.batch_size
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_agent.device).long()
        t_agent = t[agent_batch]
        t_lane = torch.zeros_like(t[lane_batch])

        loss, agent_loss, lane_loss, agent_type_loss, parking_loss, goal_loss = self.p_losses(
            x_agent, x_lane, data, t_agent, t_lane
        )
        return {
            "loss": loss.mean(),
            "agent_loss": agent_loss.mean().detach(),
            "lane_loss": lane_loss.mean().detach(),
            "agent_type_loss": agent_type_loss.mean().detach(),
            "parking_loss": parking_loss.mean().detach(),
            "goal_loss": goal_loss.mean().detach(),
        }

    @torch.no_grad()
    def decode_outputs(self, x_agent, x_lane, data):
        agent_states, parking_logits, agent_type_logits = self._split_agent(x_agent)
        parking = torch.argmax(parking_logits, dim=1) == self.parking_class_id
        agent_states = agent_states.clone()
        agent_states[parking, 7:9] = agent_states[parking, 0:2]

        agent_types = torch.argmax(agent_type_logits, dim=1)
        lane_states = self._reshape_lane(x_lane)

        lane_conn_pred = torch.zeros(
            data["lane", "to", "lane"].edge_index.shape[1],
            self.cfg_dataset.num_lane_connection_types,
            device=x_agent.device,
            dtype=torch.float32,
        )
        lane_conn_pred[:, 0] = 1.0
        return agent_states, lane_states, agent_types, None, lane_conn_pred
