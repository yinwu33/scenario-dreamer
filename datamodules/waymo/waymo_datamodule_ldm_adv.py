import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater

from datasets.waymo.dataset_ldm_adv_waymo import WaymoDatasetLDMAdv


def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))


class _FilterNoneCollater(Collater):
    """PyG collater that drops samples the dataset returned as ``None``.

    ``WaymoDatasetLDMAdv`` returns ``None`` for scenes with fewer than two agents
    (no non-ego agent to act as the adversary). Those entries must be removed
    before collating -- the default PyG ``Collater`` crashes on ``None``
    (``'NoneType' object has no attribute 'stores'``). We use a plain ``torch``
    ``DataLoader`` with this collater because PyG's own ``DataLoader`` discards
    any ``collate_fn`` it is given.
    """

    def __call__(self, batch):
        batch = [data for data in batch if data is not None]
        return super().__call__(batch)


class WaymoDataModuleLDMAdv(pl.LightningDataModule):
    def __init__(
        self,
        train_batch_size,
        val_batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
        dataset_cfg,
    ):
        super(WaymoDataModuleLDMAdv, self).__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.cfg_dataset = dataset_cfg

    def setup(self, stage):
        self.train_dataset = WaymoDatasetLDMAdv(self.cfg_dataset, split_name="train", mode="train")
        self.val_dataset = WaymoDatasetLDMAdv(self.cfg_dataset, split_name="val", mode="eval")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            collate_fn=_FilterNoneCollater(self.train_dataset),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            collate_fn=_FilterNoneCollater(self.val_dataset),
        )
