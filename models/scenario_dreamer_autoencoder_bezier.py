import os
import pickle
from utils.train_helpers import create_lambda_lr_cosine, create_lambda_lr_linear
from datasets.waymo.dataset_autoencoder_waymo import WaymoDatasetAutoEncoder
from datasets.nuplan.dataset_autoencoder_nuplan import NuplanDatasetAutoEncoder
from nn_modules.autoencoder_bezier import AutoEncoderBezier
from torch_geometric.loader import DataLoader

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from torch_geometric.data import Batch
torch.set_printoptions(sci_mode=False)

from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph
from utils.data_helpers import unnormalize_scene
from utils.viz import visualize_batch


# this ensures CPUs are not suboptimally utilized
def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))


class ScenarioDreamerAutoEncoderBezier(pl.LightningModule):
    """PyTorch Lightning module for the Bezier lane-graph AutoEncoder.

    Mirrors ``ScenarioDreamerAutoEncoder`` but wraps :class:`AutoEncoderBezier`.
    The encoder (and therefore the latent space / latent cache) is unchanged, so
    latent caching for LDM training remains compatible.
    """

    def __init__(self, cfg):
        super(ScenarioDreamerAutoEncoderBezier, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.cfg_dataset = self.cfg.dataset
        self.model = AutoEncoderBezier(cfg.model)

        # nocturne-compatible metadata (stored in latent cache)
        if self.cfg.eval.cache_latents.enable_caching and self.cfg.dataset_name == 'waymo':
            with open(self.cfg.eval.cache_latents.nocturne_train_filenames_path, 'rb') as f:
                nocturne_train_filenames = pickle.load(f)
            with open(self.cfg.eval.cache_latents.nocturne_val_filenames_path, 'rb') as f:
                nocturne_val_filenames = pickle.load(f)
            self.nocturne_compatible_filenames = nocturne_train_filenames + nocturne_val_filenames

    def forward(self, data):
        """Reconstruct a batch and unnormalize for visualization.

        Mirrors ``ScenarioDreamerAutoEncoder.forward``: returns per-lane
        reconstructed polylines (each GT lane shown as its Hungarian-matched
        predicted bezier edge), so the existing ``visualize_batch`` can be
        reused. ``lane_conn_samples`` carries the GT connectivity (this model
        does not predict lane connectivity — it is structural via shared nodes).
        """
        agent_states_pred, lane_samples, agent_types_pred, lane_cond_dis = self.model.reconstruct(data)
        agent_samples, lane_samples = unnormalize_scene(
            agent_states_pred,
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
            max_lane_y=self.cfg_dataset.max_lane_y)

        lane_conn_samples = data['lane', 'to', 'lane'].type.float()
        lane_types = None  # waymo (num_lane_types == 0); this model has no lane-type head
        return agent_samples, lane_samples, agent_types_pred, lane_types, lane_conn_samples, lane_cond_dis

    def test_dataloader(self):
        """If caching enabled, returns Dataloader for dataset we wish to cache latents, otherwise returns a DataLoader for the test dataset."""
        if self.cfg.eval.cache_latents.enable_caching:
            if self.cfg.dataset_name == 'waymo':
                test_dataset = WaymoDatasetAutoEncoder(self.cfg_dataset, split_name=self.cfg.eval.cache_latents.split_name, mode='eval')
            else:
                test_dataset = NuplanDatasetAutoEncoder(self.cfg_dataset, split_name=self.cfg.eval.cache_latents.split_name, mode='eval')
            latent_dir = os.path.join(self.cfg.eval.cache_latents.latent_dir, self.cfg.eval.cache_latents.split_name)
            if not os.path.exists(latent_dir):
                os.makedirs(latent_dir, exist_ok=True)
            self.files = test_dataset.files.copy()
        else:
            if self.cfg.dataset_name == 'waymo':
                test_dataset = WaymoDatasetAutoEncoder(self.cfg_dataset, split_name='test', mode='eval')
            else:
                test_dataset = NuplanDatasetAutoEncoder(self.cfg_dataset, split_name='test', mode='eval')
        test_dataloader = DataLoader(test_dataset,
                            batch_size=self.cfg.datamodule.val_batch_size,
                            shuffle=False,
                            num_workers=self.cfg.datamodule.num_workers,
                            pin_memory=self.cfg.datamodule.pin_memory,
                            drop_last=False,
                            worker_init_fn=worker_init_fn)
        return test_dataloader

    def _log_losses(self, loss_dict, split='train', batch_size=None):
        """Log the losses to WandB."""
        if split == 'train':
            on_step = True
            on_epoch = False
            key_lambda = lambda s: s
        elif split == 'val':
            on_step = False
            on_epoch = True
            key_lambda = lambda s: f'val_{s}'
        elif split == 'test':
            on_step = False
            on_epoch = True
            key_lambda = lambda s: f'test_{s}'

        for k, v in loss_dict.items():
            if k == 'loss':
                v = v.item()
            self.log(key_lambda(k), v, prog_bar=True, on_step=on_step, on_epoch=on_epoch, sync_dist=True, batch_size=batch_size)

        if split == 'train':
            cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('lr', cur_lr, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

    def _cache_latents(self, data):
        """Cache the autoencoder latents to disk for ldm training (encoder is unchanged)."""
        agent_mu, lane_mu, agent_log_var, lane_log_var = self.model.forward(data, return_latents=True)

        agent_mu = agent_mu.detach().cpu().numpy()
        lane_mu = lane_mu.detach().cpu().numpy()
        agent_log_var = agent_log_var.detach().cpu().numpy()
        lane_log_var = lane_log_var.detach().cpu().numpy()
        scene_type = data['lg_type'].cpu().int()
        road_points = data['lane'].x.cpu().numpy()
        agent_states = data['agent'].x.cpu().numpy()
        agent_batch = data['agent'].batch.cpu().numpy()
        lane_batch = data['lane'].batch.cpu().numpy()
        if self.cfg.dataset_name == 'nuplan':
            map_id = data['map_id'].cpu().numpy()

        for i in range(data.batch_size):
            idx = data.idx[i].item()
            file_path = self.files[idx]

            agent_mu_i = agent_mu[agent_batch == i]
            lane_mu_i = lane_mu[lane_batch == i]
            agent_log_var_i = agent_log_var[agent_batch == i]
            lane_log_var_i = lane_log_var[lane_batch == i]
            scene_type_i = scene_type[i].item()
            road_points_i = road_points[lane_batch == i]
            agent_states_i = agent_states[agent_batch == i]
            if self.cfg.dataset_name == 'nuplan':
                map_id_i = int(map_id[i].item())

            n_lanes = lane_mu_i.shape[0]
            n_agents = agent_mu_i.shape[0]
            edge_index_i_l2l = get_edge_index_complete_graph(n_lanes).numpy()
            edge_index_i_l2a = get_edge_index_bipartite(n_lanes, n_agents).numpy()
            edge_index_i_a2a = get_edge_index_complete_graph(n_agents).numpy()

            prefix = f"{self.cfg_dataset.preprocess_dir}/{self.cfg.eval.cache_latents.split_name}/"
            new_prefix = f"{self.cfg.eval.cache_latents.latent_dir}/{self.cfg.eval.cache_latents.split_name}/"
            file_path = file_path.replace(prefix, new_prefix)

            d = {
                'idx': idx,
                'agent_mu': agent_mu_i,
                'lane_mu': lane_mu_i,
                'agent_log_var': agent_log_var_i,
                'lane_log_var': lane_log_var_i,
                'edge_index_lane_to_lane': edge_index_i_l2l,
                'edge_index_agent_to_agent': edge_index_i_a2a,
                'edge_index_lane_to_agent': edge_index_i_l2a,
                'scene_type': scene_type_i,
                'road_points': road_points_i,
                'agent_states': agent_states_i,
            }

            if self.cfg.dataset_name == 'waymo':
                filename = file_path.split('/')[-1]
                train_filename_parts = filename.split('.')[1].split('_')[:2]
                train_filename = f'{train_filename_parts[0]}_{train_filename_parts[1]}'
                noct_compatible = 1 if train_filename in self.nocturne_compatible_filenames else 0
                d['nocturne_compatible'] = noct_compatible
            else:
                d['map_id'] = map_id_i

            with open(file_path, 'wb') as f:
                pickle.dump(d, f)

    def training_step(self, data, batch_idx):
        loss_dict = self.model.loss(data)
        self._log_losses(loss_dict, split='train')
        return loss_dict['loss']

    def validation_step(self, data, batch_idx):
        loss_dict = self.model.loss(data)
        self._log_losses(loss_dict, split='val', batch_size=data.batch_size)

        if self.cfg.train.num_samples_to_visualize > 0 and batch_idx == 0 and self.trainer.is_global_zero:
            num_samples = self.cfg.train.num_samples_to_visualize
            assert num_samples <= data.batch_size, f"num_samples ({num_samples}) must be <= batch size ({data.batch_size})"

            indices = torch.arange(num_samples)
            subset_data_list = data.index_select(indices)
            subset_data = Batch.from_data_list(subset_data_list)

            # each GT lane is visualized as its matched predicted bezier edge
            agent_samples, lane_samples, agent_types, lane_types, lane_conn_samples, _ = self.forward(subset_data)
            save_dir = self.cfg.train.viz_dir

            print(f"Visualizing batch {batch_idx}...")
            images_to_log = visualize_batch(num_samples,
                            agent_samples,
                            lane_samples,
                            agent_types,
                            lane_types,
                            lane_conn_samples,
                            subset_data,
                            save_dir,
                            self.current_epoch,
                            batch_idx,
                            self.cfg.train.track)
            if self.cfg.train.track:
                self.logger.experiment.log(images_to_log)

    def test_step(self, data, batch_idx):
        if self.cfg.eval.cache_latents.enable_caching:
            self._cache_latents(data)
        else:
            loss_dict = self.model.loss(data)
            self._log_losses(loss_dict, split='test', batch_size=data.batch_size)

    def on_before_optimizer_step(self, optimizer):
        norms_encoder = grad_norm(self.model, norm_type=2)
        self.log_dict(norms_encoder)

    ### Taken largely from QCNet repository: https://github.com/ZikangZhou/QCNet
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.cfg.train.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay, betas=(self.cfg.train.beta_1, self.cfg.train.beta_2), eps=self.cfg.train.epsilon)

        if self.cfg.train.lr_schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lambda_lr_cosine(self.cfg))
        elif self.cfg.train.lr_schedule == 'linear':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lambda_lr_linear(self.cfg))

        return [optimizer], {"scheduler": scheduler,
                             "interval": "step",
                             "frequency": 1}
