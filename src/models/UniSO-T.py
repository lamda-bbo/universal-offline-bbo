import copy
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.nn import CrossEntropyLoss
from torchmetrics import MaxMetric, MeanMetric, SpearmanCorrCoef
from transformers import T5EncoderModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack


class OmniPredModule(LightningModule):
    def __init__(
        self,
        encoder_model: T5EncoderModel,
        decoder_model: Callable[..., T5Stack],
        input_tokenizer: Any,
        output_tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        compile: bool,
        scheduler=None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.encoder_hidden_size = encoder_model.config.hidden_size
        self.decoder_hidden_size = self.encoder_hidden_size

        self.encoder = encoder_model.encoder
        self.shared = encoder_model.shared

        decoder_config = decoder_model.keywords["config"]
        decoder_config.vocab_size = output_tokenizer.vocab_size
        decoder_config.d_model = encoder_model.config.d_model
        decoder_config.is_decoder = True
        decoder_config.use_cache = True
        decoder_config.add_cross_attention = True

        self.decoder = decoder_model()

        self.decoder_input_proj = nn.Linear(
            self.encoder_hidden_size, self.decoder_hidden_size
        )

        self.lm_head = nn.Linear(
            self.decoder_hidden_size, decoder_config.vocab_size, bias=False
        )

        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Encoder
        input_embeds = self.shared(input_ids)
        encoder_outputs = self.encoder(
            inputs_embeds=input_embeds, attention_mask=attention_mask
        ).last_hidden_state

        # Decoder
        if decoder_input_ids is not None:
            decoder_inputs = self.shared(decoder_input_ids)
            decoder_inputs = self.decoder_input_proj(decoder_inputs)

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        lm_logits = self.lm_head(decoder_outputs)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            encoder_last_hidden_state=encoder_outputs,
        )

    def on_train_start(self) -> None:
        self.val_loss.reset()

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        return loss, outputs, batch["labels"]

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        # Update metrics
        self.train_loss(loss)

        # Log metrics
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="train/loss",
        )

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # Update metrics
        self.val_loss(loss)

        # Log metrics
        self.log(
            "val/loss",
            self.val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="val/loss",
        )

    def on_validation_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)
            self.shared = torch.compile(self.shared)
            self.decoder_input_proj = torch.compile(self.decoder_input_proj)
            self.lm_head = torch.compile(self.lm_head)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            # Calculate total steps
            total_steps = self.trainer.estimated_stepping_batches

            scheduler = self.hparams.scheduler(
                optimizer=optimizer, num_training_steps=total_steps
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # Update lr every step instead of epoch
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 32,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.95,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]

        # Encoder
        input_embeds = self.shared(input_ids)
        encoder_outputs = self.encoder(
            inputs_embeds=input_embeds, attention_mask=attention_mask
        ).last_hidden_state

        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.output_tokenizer.bos_token_id,
            dtype=torch.long,
            device=self.device,
        )

        past_key_values = None

        for i in range(max_length - 1):
            if i == 0:
                decoder_inputs = self.shared(decoder_input_ids)
                decoder_inputs = self.decoder_input_proj(decoder_inputs)
            else:
                last_token = decoder_input_ids[:, -1:]
                decoder_inputs = self.shared(last_token)
                decoder_inputs = self.decoder_input_proj(decoder_inputs)

            # cache for acceleration
            decoder_outputs = self.decoder(
                inputs_embeds=decoder_inputs,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = decoder_outputs.past_key_values
            hidden_states = decoder_outputs.last_hidden_state

            next_token_logits = self.lm_head(hidden_states[:, -1:]) / temperature
            next_token_logits = next_token_logits.squeeze(1)

            if i == 0:
                scores = torch.empty_like(next_token_logits)

            # Top-K + Top-P
            if top_k > 0 or top_p < 1.0:
                scores.copy_(next_token_logits)

                if top_k > 0:
                    indices_to_remove = (
                        scores < torch.topk(scores, top_k)[0][..., -1, None]
                    )
                    scores[indices_to_remove] = float("-inf")

                if top_p < 1.0:
                    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_scores, dim=-1), dim=-1
                    )

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = torch.zeros_like(
                        scores, dtype=torch.bool
                    ).scatter_(1, sorted_indices, sorted_indices_to_remove)
                    scores[indices_to_remove] = float("-inf")

            probs = F.softmax(scores, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)

            if (next_token == self.output_tokenizer.eos_token_id).all():
                break

        return decoder_input_ids

    @torch.inference_mode()
    def generate_numbers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 32,
        decoding_strategy: str = "top_k",  # Add decoding strategy parameter
        **kwargs,
    ):
        if decoding_strategy != "top_k":
            raise NotImplementedError

        predictions = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            decoding_strategy=decoding_strategy,
            **kwargs,
        )

        pred_numbers = []
        for pred in predictions:
            # print("Token IDs:", pred[pred != -100].tolist())

            # tokens = [
            #     self.output_tokenizer._convert_id_to_token(idx)
            #     for idx in pred[pred != -100]
            # ]
            # print("Tokens:", tokens)

            text = self.output_tokenizer.decode(
                pred[pred != -100], skip_special_tokens=True
            )
            # print("Decoded text:", text)

            try:
                num = float(text)
                pred_numbers.append(num)
            except ValueError:
                if len(pred_numbers) > 0:
                    mean_value = sum(pred_numbers) / len(pred_numbers)
                    pred_numbers.append(mean_value)
                else:
                    pred_numbers.append(0.0)
            # print("---")

        # Return a two-dimensional vector
        return torch.tensor(pred_numbers, device=self.device).reshape(-1, 1)