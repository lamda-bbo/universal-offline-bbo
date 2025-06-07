from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, SpearmanCorrCoef

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class RegressorModule(LightningModule):

    def __init__(
        self,
        input_dim: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = model(input_dim=input_dim)
        self.batch_norm = nn.BatchNorm1d(input_dim)

        self.criterion = nn.MSELoss()

        self.train_rank_corr = SpearmanCorrCoef()
        self.val_rank_corr = SpearmanCorrCoef()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_rank_corr_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(x)
        return self.model(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_rank_corr.reset()
        self.val_rank_corr_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds.squeeze(), y.squeeze())
        return loss, preds, y

    def training_step(
        self, batch: Tuple[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_rank_corr(preds.squeeze(), targets.squeeze())
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/rank_corr",
            self.train_rank_corr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_rank_corr(preds.squeeze(), targets.squeeze())
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/rank_corr",
            self.val_rank_corr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        rank_corr = self.val_rank_corr.compute()
        self.val_rank_corr_best(rank_corr)

        self.log(
            "val/rank_corr_best",
            self.val_rank_corr_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self) -> None:
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer
