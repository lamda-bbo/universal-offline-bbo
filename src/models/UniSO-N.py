from contextlib import nullcontext
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, SpearmanCorrCoef
from transformers.tokenization_utils_base import BatchEncoding

from src.utils import RankedLogger
from src.utils.io_utils import load_task_names

log = RankedLogger(__name__, rank_zero_only=True)


class EmbedRegressorModule(LightningModule):
    def __init__(
        self,
        embedder: nn.Module,
        embedder_output_dim: int,
        regressor: nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        data_dir: Path = None,
        task_names: List[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.embedder = embedder
        self.regressor = regressor
        self.batch_norm = nn.BatchNorm1d(embedder_output_dim)

        for param in self.embedder.parameters():
            param.requires_grad = False

        self.criterion = nn.MSELoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.task_names = load_task_names(task_names, data_dir)
        self.train_rank_corr = {}
        self.val_rank_corr = {}
        
    def forward(self, x: BatchEncoding) -> torch.Tensor:
        with torch.no_grad():
            encoded_input = x.to(self.device)
            x_emb = self.embedder(**encoded_input)
            x_emb = self._mean_pooling(x_emb, encoded_input["attention_mask"])

        x_emb = self.batch_norm(x_emb)
        return self.regressor(x_emb)
    
    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:

        x, y = batch["text"], batch["value"]

        preds = self.forward(x)
        loss = self.criterion(preds.squeeze(), y.squeeze())
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        x, y = batch["text"], batch["value"]
        preds = self.forward(x)
        loss = self.criterion(preds.squeeze(), y.squeeze())
        
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.regressor.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return {"optimizer": optimizer}

    def setup(self, stage: str) -> None:
        if stage == "fit":
            for task_name in self.task_names:
                self.train_rank_corr[task_name] = SpearmanCorrCoef()
                self.val_rank_corr[task_name] = SpearmanCorrCoef()