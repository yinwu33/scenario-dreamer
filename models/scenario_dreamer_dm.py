import glob
import os
import pickle

import pytorch_lightning as pl
import torch
from torch import nn
from torch_ema import ExponentialMovingAverage
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from pytorch_lightning.utilities import grad_norm

from cfgs.config import NON_PARTITIONED, NOCTURNE_COMPATIBLE, PROPORTION_NOCTURNE_COMPATIBLE
from datasets.waymo.dataset_dm_waymo import WaymoDatasetDM
from nn_modules.dm import DM
from utils.data_container import ScenarioDreamerData
from utils.data_helpers import convert_batch_to_scenarios, unnormalize_scene
from utils.inpainting_helpers import normalize_and_crop_scene
from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph
from utils.sim_env_helpers import get_default_route_center_yaw, sample_route
from utils.lane_graph_helpers import estimate_heading
from utils.train_helpers import create_lambda_lr_constant, create_lambda_lr_cosine, create_lambda_lr_linear
from utils.viz import visualize_batch


class ScenarioDreamerDM(pl.LightningModule):
    def __init__(self, cfg):
        super(ScenarioDreamerDM, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.cfg_model = cfg.model
        self.cfg_dataset = cfg.dataset
        self.diff_model = DM(self.cfg)
        self.init_prob_matrix = torch.load(self.cfg.eval.init_prob_matrix_path)
        self.ema = ExponentialMovingAverage(self.diff_model.parameters(), decay=self.cfg.train.ema_decay)

    def on_train_start(self):
        self.ema.to(self.device)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()

    def _log_losses(self, loss_dict, split="train", batch_size=None):
        if split == "train":
            on_step = True
            on_epoch = False
            key_lambda = lambda s: s
        elif split == "val":
            on_step = False
            on_epoch = True
            key_lambda = lambda s: f"val_{s}"
        else:
            on_step = False
            on_epoch = True
            key_lambda = lambda s: f"test_{s}"

        for k, v in loss_dict.items():
            if k == "loss":
                v = v.item()
            self.log(key_lambda(k), v, prog_bar=True, on_step=on_step, on_epoch=on_epoch, sync_dist=True, batch_size=batch_size)

        if split == "train":
            cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", cur_lr, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

    def training_step(self, data, batch_idx):
        loss_dict = self.diff_model.loss(data)
        self._log_losses(loss_dict, split="train")
        return loss_dict["loss"]

    def validation_step(self, data, batch_idx):
        with self.ema.average_parameters():
            loss_dict = self.diff_model.loss(data)
            self._log_losses(loss_dict, split="val", batch_size=data.batch_size)

            visualize = self.cfg.train.num_samples_to_visualize > 0 and self.trainer.is_global_zero and batch_idx == 0
            if not visualize:
                return

            num_samples = self.cfg.train.num_samples_to_visualize
            assert num_samples <= data.batch_size
            subset_data = Batch.from_data_list(data.index_select(torch.arange(num_samples)))
            _, images_to_log = self.forward(
                subset_data,
                "train",
                batch_idx,
                viz_dir=self.cfg.train.viz_dir,
                visualize=True,
                save_wandb=self.cfg.train.track,
                num_samples_to_visualize=num_samples,
            )
            if self.cfg.train.track:
                self.logger.experiment.log(images_to_log)

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
        agent_samples, lane_samples = unnormalize_scene(
            agent_samples,
            lane_samples,
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

    def _empty_scene(self, num_lanes, num_agents, map_id, lg_type=NON_PARTITIONED):
        d = ScenarioDreamerData()
        d["map_id"] = int(map_id)
        d["lg_type"] = int(lg_type)
        d["num_lanes"] = int(num_lanes)
        d["num_agents"] = int(num_agents)
        d["agent"].x = torch.empty((num_agents, self.cfg_model.state_dim))
        d["agent"].type = torch.zeros((num_agents, self.cfg_dataset.num_agent_types))
        d["lane"].x = torch.empty((num_lanes, self.cfg_model.num_points_per_lane, self.cfg_model.lane_attr))
        d["agent"].partition_mask = torch.zeros(num_agents).bool()
        d["lane"].partition_mask = torch.zeros(num_lanes).bool()
        d["lane", "to", "lane"].edge_index = get_edge_index_complete_graph(num_lanes)
        d["agent", "to", "agent"].edge_index = get_edge_index_complete_graph(num_agents)
        d["lane", "to", "agent"].edge_index = get_edge_index_bipartite(num_lanes, num_agents)
        d["lane", "to", "lane"].type = torch.zeros(
            (num_lanes * num_lanes, self.cfg_dataset.num_lane_connection_types)
        )
        return d

    def _initialize_pyg_dset(self, mode, num_samples, conditioning_path=None, nocturne_compatible_only=False):
        data_list = []
        conditioning_files = None

        if mode == "initial_scene":
            for _ in range(num_samples):
                if nocturne_compatible_only:
                    map_id = torch.tensor(NOCTURNE_COMPATIBLE)
                else:
                    map_id = torch.multinomial(
                        torch.tensor([1 - PROPORTION_NOCTURNE_COMPATIBLE, PROPORTION_NOCTURNE_COMPATIBLE]), 1
                    )
                lane_agent_probs = self.init_prob_matrix[map_id].reshape(1, -1)
                folded_num_lanes_agents = torch.multinomial(lane_agent_probs, 1).squeeze(-1)
                num_lanes = (folded_num_lanes_agents // (self.cfg_dataset.max_num_agents + 1)).item()
                num_agents = (folded_num_lanes_agents % (self.cfg_dataset.max_num_agents + 1)).item()
                assert num_lanes > 0 and num_agents > 0
                data_list.append(self._empty_scene(num_lanes, num_agents, map_id, NON_PARTITIONED))

        elif mode == "lane_conditioned":
            assert conditioning_path is not None
            conditioning_files = sorted(glob.glob(os.path.join(conditioning_path, "*.pkl")))[:num_samples]
            dset = WaymoDatasetDM(self.cfg_dataset, split_name="val", mode="eval")
            for i, conditioning_file in enumerate(conditioning_files):
                with open(conditioning_file, "rb") as f:
                    cond_d = pickle.load(f)
                d = dset.get_data(cond_d, i, conditioning_file)
                data_list.append(d)

        elif mode == "inpainting":
            assert conditioning_path is not None
            conditioning_files = sorted(glob.glob(os.path.join(conditioning_path, "*.pkl")))[:num_samples]
            for conditioning_file in conditioning_files:
                with open(conditioning_file, "rb") as f:
                    cond_d = pickle.load(f)
                if "route" in cond_d:
                    route = cond_d["route"]
                    center = route[-1]
                    _, yaw = estimate_heading(route)
                else:
                    route, found_route = sample_route(cond_d, dataset=self.cfg.dataset_name)
                    if found_route:
                        center = route[-1]
                        _, yaw = estimate_heading(route)
                    else:
                        center, yaw = get_default_route_center_yaw(dataset=self.cfg.dataset_name)
                d = ScenarioDreamerData()
                d = normalize_and_crop_scene(cond_d, d, {"center": center, "yaw": yaw}, self.cfg_dataset, self.cfg.dataset_name)
                d["agent"].partition_mask = torch.ones(d["num_agents"]).bool()
                data_list.append(d)
        else:
            raise ValueError(f"Unsupported DM generation mode: {mode}")

        conditioning_filenames = (
            [os.path.splitext(os.path.basename(f))[0] for f in conditioning_files] if conditioning_files is not None else None
        )
        return data_list, conditioning_filenames

    def generate(
        self,
        mode,
        num_samples,
        batch_size,
        cache_samples=False,
        visualize=False,
        conditioning_path=None,
        cache_dir=None,
        viz_dir=None,
        save_wandb=False,
        return_samples=False,
        nocturne_compatible_only=False,
    ):
        self.eval()
        with torch.no_grad():
            with self.ema.average_parameters():
                images_to_log = {}
                dset, conditioning_filenames = self._initialize_pyg_dset(
                    mode, num_samples, conditioning_path, nocturne_compatible_only
                )
                dataloader = DataLoader(dset, batch_size=batch_size, shuffle=False, drop_last=False)
                scenarios = {}
                for batch_idx, data in enumerate(dataloader):
                    data, images_to_log_batch = self.forward(
                        data,
                        mode,
                        batch_idx,
                        viz_dir=viz_dir,
                        visualize=visualize,
                        save_wandb=save_wandb,
                    )
                    if visualize and save_wandb:
                        images_to_log.update(images_to_log_batch)
                    batch_of_scenarios = convert_batch_to_scenarios(
                        data,
                        batch_size=batch_size,
                        batch_idx=batch_idx,
                        cache_dir=cache_dir,
                        conditioning_filenames=conditioning_filenames,
                        cache_samples=cache_samples,
                        cache_lane_types=False,
                        mode=mode,
                    )
                    scenarios.update(batch_of_scenarios)
        return scenarios if return_samples else None

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grad_norm(self.diff_model.model, norm_type=2))

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self.ema.load_state_dict(checkpoint["ema_state_dict"])

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.diff_model.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = "%s.%s" % (module_name, param_name) if module_name else param_name
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)

        param_dict = {param_name: param for param_name, param in self.diff_model.named_parameters()}
        assert len(decay & no_decay) == 0
        assert len(param_dict.keys() - (decay | no_decay)) == 0
        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))], "weight_decay": self.cfg.train.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            betas=(self.cfg.train.beta_1, self.cfg.train.beta_2),
            eps=self.cfg.train.epsilon,
        )
        if self.cfg.train.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lambda_lr_cosine(self.cfg))
        elif self.cfg.train.lr_schedule == "linear":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lambda_lr_linear(self.cfg))
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lambda_lr_constant(self.cfg))
        return [optimizer], {"scheduler": scheduler, "interval": "step", "frequency": 1}
