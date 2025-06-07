from contextlib import nullcontext
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningDataModule, LightningModule
from torchmetrics import MaxMetric, MeanMetric, SpearmanCorrCoef
from transformers.tokenization_utils_base import BatchEncoding
from transformers import T5EncoderModel

from src.utils import RankedLogger
from src.utils.io_utils import load_task_names

log = RankedLogger(__name__, rank_zero_only=True)


class T5FineTuner(LightningModule):
    def __init__(
        self,
        model_name: str,
        non_shuffled_datamodule: LightningDataModule,
        embedder_output_dim: int,
        compile: bool = True,
        metadata_embedder: Optional[nn.Module] = None,
        metadata_embedder_output_dim: Optional[int] = None,
        learning_rate: float = 2e-5,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.embedder = T5EncoderModel.from_pretrained(model_name)
        self.metadata_embedder = metadata_embedder
        self.temperature = temperature
        self.non_shuffled_datamodule = non_shuffled_datamodule
        self.automatic_optimization = False

        self.projection_head = nn.Sequential(
            nn.Linear(embedder_output_dim, embedder_output_dim),
            nn.ReLU(),
            nn.Linear(embedder_output_dim, 128),
        )

        self.metadata_projection_head = nn.Sequential(
            nn.Linear(metadata_embedder_output_dim, metadata_embedder_output_dim),
            nn.ReLU(),
            nn.Linear(metadata_embedder_output_dim, 128),
        )

        self.train_loss = MeanMetric()
        self.train_lip_loss = MeanMetric()
        self.train_con_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def contrastive_loss(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> torch.Tensor:
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)

        metadata_sim = torch.matmul(embeddings2, embeddings2.T)
        metadata_sim_max, _ = metadata_sim.max(dim=1, keepdim=True)
        metadata_sim_min, _ = metadata_sim.min(dim=1, keepdim=True)
        metadata_sim = (metadata_sim - metadata_sim_min) / (
            metadata_sim_max - metadata_sim_min + 1e-10
        )
        
        threshold = 0.2  
        mask_diag = torch.eye(metadata_sim.shape[0], device=metadata_sim.device)
        metadata_sim = torch.where(
            mask_diag == 1,
            metadata_sim,
            threshold * torch.tanh(metadata_sim / threshold)
        )

        similarity_matrix = torch.matmul(embeddings1, embeddings1.T)
        similarity_matrix = similarity_matrix / self.temperature

        log_prob = F.log_softmax(similarity_matrix, dim=1)

        mask = ~torch.eye(
            embeddings1.shape[0], dtype=torch.bool, device=embeddings1.device
        )
        loss = -(metadata_sim[mask] * log_prob[mask]).mean()

        return loss

    def lipschitz_loss(self, z, y, recon_weight=None):
        z = z.to(self.device)
        y = y.to(self.device)
        
        # if all ys are the same, return the mean of all zs
        if torch.all(y == y[0]):
            dif_z = torch.sqrt(
                torch.sum((z.unsqueeze(1) - z.unsqueeze(0)) ** 2, dim=2) + 1e-10
            )
            return dif_z.mean()
        
        dif_y = (y.unsqueeze(1) - y.unsqueeze(0)).squeeze(-1)
        dif_z = torch.sqrt(
            torch.sum((z.unsqueeze(1) - z.unsqueeze(0)) ** 2, dim=2) + 1e-10
        )
        
        lips = abs(dif_y / (dif_z + 1e-10))
        ratio = lips - torch.median(lips)
        ratio = ratio[ratio > 0]
        
        if len(ratio) == 0:
            return torch.tensor(0.0, device=self.device)
            
        loss = ratio.mean()
        return loss
    
    def _mean_pooling(
        self, model_output: Tuple[torch.Tensor], attention_mask: torch.Tensor, require_grad: bool
    ) -> torch.Tensor:
        context = torch.no_grad() if not require_grad else nullcontext()
        with context:
            # First element of model_output contains all token embeddings
            token_embeddings: torch.Tensor = model_output[
                0
            ]  # Shape: [batch_size, sequence_length, hidden_size]
            input_mask_expanded: torch.Tensor = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )  # Shape: [batch_size, sequence_length, hidden_size]

            sum_embeddings: torch.Tensor = torch.sum(
                token_embeddings * input_mask_expanded, 1
            )  # Shape: [batch_size, hidden_size]
            sum_mask: torch.Tensor = torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )  # Shape: [batch_size, hidden_size]

            return sum_embeddings / sum_mask  # Shape: [batch_size, hidden_size]

    def _emb_metadata(self, m: Tuple[BatchEncoding]) -> torch.Tensor:
        context = torch.no_grad()
        with context:
            encoded_input = m.to(self.device)
            emb_m = self.metadata_embedder(**encoded_input)
            emb_m = self._mean_pooling(emb_m, encoded_input["attention_mask"], require_grad=False)
        return emb_m

    def forward(self, input_ids, attention_mask):
        pass 
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        
        try:
            non_shuffled_batch = next(self.non_shuffled_train_iter)
        except:
            self.non_shuffled_train_iter = iter(
                self.non_shuffled_datamodule.train_dataloader()
            )
            non_shuffled_batch = next(self.non_shuffled_train_iter)
        
        x = batch["text"]
        m = batch["metadata"]

        encoded_input = x.to(self.device)
        x_embeddings = self.embedder(**encoded_input)
        x_embeddings = self._mean_pooling(x_embeddings, encoded_input["attention_mask"], require_grad=True)
        x_projected = self.projection_head(x_embeddings)

        with torch.no_grad():
            m_embeddings = self._emb_metadata(m)
        m_projected = self.metadata_projection_head(m_embeddings)

        loss = self.contrastive_loss(x_projected, m_projected)

        x_n = non_shuffled_batch["text"]
        y_n = non_shuffled_batch["value"]
        task_name_n = non_shuffled_batch["task_names"]

        encoded_input_n = x_n.to(self.device)
        x_embeddings_n = self.embedder(**encoded_input_n)
        x_embeddings_n = self._mean_pooling(
            x_embeddings_n, encoded_input_n["attention_mask"], require_grad=True
        )

        # Combine
        task_data = {}
        for i in range(len(task_name_n)):
            task = task_name_n[i]
            if task not in task_data:
                task_data[task] = {"text": [], "value": []}

            task_data[task]["text"].append(x_embeddings_n[i])
            if y_n.dim() > 1:
                task_data[task]["value"].append(y_n[i : i + 1])
            else:
                task_data[task]["value"].append(y_n[i])

        loss_lip = 0
        for task in task_data:
            if len(task_data[task]["text"]) > 0:
                task_data[task]["text"] = torch.stack(task_data[task]["text"], dim=0)

            if len(task_data[task]["value"]) > 0:
                try:
                    task_data[task]["value"] = torch.cat(
                        task_data[task]["value"], dim=0
                    )
                except:
                    task_data[task]["value"] = torch.tensor(task_data[task]["value"])

            loss_lip += self.lipschitz_loss(
                z=task_data[task]["text"], y=task_data[task]["value"]
            ) * (len(task_data[task]["value"]) / len(y_n))

        self.train_lip_loss(loss_lip)
        self.log(
            "train/lip_loss",
            self.train_lip_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="train_lip_loss",
        )

        self.train_con_loss(loss)
        self.log(
            "train/con_loss",
            self.train_con_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="train_con_loss",
        )

        total_loss = loss + loss_lip / ((loss_lip / loss).detach() + 1e-10)
        
        self.train_loss(total_loss)
        self.log('train/loss', self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="train_loss",)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), max_norm=0.5)
        
        opt.step()
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["text"]
        m = batch["metadata"]

        encoded_input = x.to(self.device)
        x_embeddings = self.embedder(**encoded_input)
        x_embeddings = self._mean_pooling(x_embeddings, encoded_input["attention_mask"], require_grad=False)
        x_projected = self.projection_head(x_embeddings)

        with torch.no_grad():
            m_embeddings = self._emb_metadata(m)
        m_projected = self.metadata_projection_head(m_embeddings)

        loss = self.contrastive_loss(x_projected, m_projected)
        self.val_loss(loss)
        self.log(
            "val/con_loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="val_con_loss",
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.non_shuffled_datamodule.setup("fit")
            loader = self.non_shuffled_datamodule.train_dataloader()
            self.non_shuffled_train_iter = iter(loader)
            log.info("Finish iterate non_shuffled dataloader")
        
        if self.hparams.compile and stage == "fit":
            self.embedder = torch.compile(self.embedder)
            self.projection_head = torch.compile(self.projection_head)
            self.metadata_projection_head = torch.compile(self.metadata_projection_head)

            self.metadata_embedder = torch.compile(self.metadata_embedder)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(chain(
            self.embedder.parameters(),
            self.projection_head.parameters(),
            self.metadata_projection_head.parameters()
        ), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        
        return optimizer