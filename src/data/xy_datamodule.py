from typing import Any, Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler

from src.tasks.base import OfflineBBOTask


class XYDataModule(LightningDataModule):

    def __init__(
        self,
        *,
        task: OfflineBBOTask,
        val_ratio: float = 0.2,
        batch_size: int = 128,
        num_workers: int = 0,
        persistent_workers: bool = True,
        pin_memory: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        assert 0 < val_ratio < 1

        self.task = task
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.save_hyperparameters(logger=False)

        # TODO: More flexible setting of transforms
        self.x_transforms = []
        self.y_transforms = []

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )
            # if self.trainer.accelerator:
            #     self.device = self.trainer.accelerator.device

        if not self.data_train or not self.data_val:
            self.x_values = self.task.x_np
            self.y_values = self.task.y_np
            assert len(self.x_values) == len(self.y_values)

            # self.y_values = self.task.task.normalize_y(self.y_values)
            self.y_values = (self.y_values - self.y_values.mean(axis=0)) / (
                self.y_values.std(axis=0) + 1e-10
            )
            if self.task.task_type == "Categorical":
                self.x_values = self.task.task.to_logits(self.x_values)
                self.x_values = self.x_values.reshape(self.x_values.shape[0], -1)
            elif self.task.task_type in ["Interger", "Permutation"]:
                self.x_values = self.x_values.astype(np.float32)

            self.x_values = torch.from_numpy(self.x_values).to(dtype=torch.float32)
            self.y_values = torch.from_numpy(self.y_values).to(dtype=torch.float32)

            dataset = TensorDataset(self.x_values, self.y_values)
            lengths = [
                len(self.x_values) - int(len(self.x_values) * self.hparams.val_ratio),
                int(len(self.x_values) * self.hparams.val_ratio),
            ]
            self.data_train, self.data_val = random_split(
                dataset=dataset, lengths=lengths
            )

    def train_dataloader(self) -> DataLoader[Any]:
        sampler = None
        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(
                self.data_train,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True,
            )

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=(sampler is None),
            persistent_workers=self.hparams.persistent_workers,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        sampler = None
        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(
                self.data_val,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=False,
            )

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.persistent_workers,
            sampler=sampler,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
