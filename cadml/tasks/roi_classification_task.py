from typing import Any

import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig

import torch
from torch import nn
from torch.nn import functional as F
from torchtyping import TensorType

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from ..utils import calculate_metrics


class ROIClassificationTask(pl.LightningModule):
    def __init__(
            self,
            cfg: DictConfig,
            model: nn.Module,
    ):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.model = model

        self.loss_weights = torch.tensor([cfg.task.positive_class_weight])
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.loss_weights)

    def common_step(
            self,
            batch: tuple[TensorType["batch_size", "channels", "height", "width"], TensorType["batch_size"]]
    ) -> tuple[TensorType["batch_size"], TensorType["batch_size"], torch.float]:  # FIXME how do we annotate type for loss?
        """
        :param batch: tuple containing images tensor and a tensor of classes
        :return: tuple containing predictions, gold labels and aggregated loss
        """

        X, gold = batch
        pred_logits = self.model(X)
        loss = self.loss(pred_logits, gold.unsqueeze(1).float())
        return torch.sigmoid(pred_logits), gold, loss

    def training_step(self, batch, batch_idx):
        pred, gold, loss = self.common_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(pred))
        return {
            'loss': loss,
            'pred': pred,
            'gold': gold
        }

    def validation_step(self, batch, batch_idx):
        pred, gold, loss = self.common_step(batch)
        self.log('val_loss', loss, batch_size=len(pred))
        return {
            'pred': pred,
            'gold': gold
        }

    def test_step(self, batch, batch_idx):
        pred, gold, loss = self.common_step(batch)
        self.log('test_loss', loss)
        return {
            'pred': pred,
            'gold': gold
        }

    def _eval_epoch_end(self, outputs: list[dict[str, Any]]) -> None:
        # flattening outputs from dataloaders if there are multiple
        if outputs and isinstance(outputs[0], list):
            outputs = [
                step_output
                for dataloader_output in outputs
                for step_output in dataloader_output
            ]

        preds_prob: torch.Tensor = torch.tensor([pred for step_output in outputs for pred in step_output['pred']])
        preds: torch.Tensor = torch.where(preds_prob >= self.hparams.task.decision_threshold, 1, 0)
        golds: torch.Tensor = torch.tensor([gold for step_output in outputs for gold in step_output['gold']])

        preds_numpy = preds.cpu().numpy()
        golds_numpy = golds.cpu().numpy()

        metrics = calculate_metrics(preds_numpy, golds_numpy)
        self.log_dict(metrics, on_epoch=True)

    def validation_epoch_end(self, validation_epoch_outputs: list[dict[str, Any]]) -> None:
        return self._eval_epoch_end(validation_epoch_outputs)

    def test_epoch_end(self, test_epoch_outputs: list[dict[str, Any]]) -> None:
        return self._eval_epoch_end(test_epoch_outputs)

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(
        #     self.parameters(),
        #     lr=self.hparams.task.learning_rate,
        #     weight_decay=self.hparams.task.weight_decay
        # )
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.task.learning_rate,
            weight_decay=self.hparams.task.weight_decay,
            momentum=self.hparams.task.momentum,
            nesterov=self.hparams.task.nesterov
        )
        return optimizer
