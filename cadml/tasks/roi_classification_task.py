from typing import Any

import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig

import torch
from torch import nn
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

        self.loss_weights = torch.tensor([cfg.task.zero_class_weight, 1.0])
        self.loss = nn.CrossEntropyLoss(self.loss_weights)

    def common_step(
            self,
            batch: tuple[TensorType["batch_size", "channels", "height", "width"], TensorType["batch_size"]]
    ) -> tuple[TensorType["batch_size"], TensorType["batch_size"], torch.float]:  # FIXME how do we annotate type for loss?
        """
        :param batch: tuple containing images tensor and a tensor of classes
        :return: tuple containing predictions, gold labels and aggregated loss
        """

        X, gold = batch
        pred = self.model(X)
        loss = self.loss(pred, gold)
        return pred, gold, loss

    def training_step(self, batch, batch_idx):
        pred, gold, loss = self.common_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(pred))
        return {
            'loss': loss,
            'pred': pred,
            'gold': gold
        }

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            pred, gold, loss = self.common_step(batch)
        self.log('val_loss', loss, batch_size=len(pred))
        return {
            'pred': pred,
            'gold': gold
        }

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            pred, gold, loss = self.common_step(batch)
        self.log('test_loss', loss)

    # FIXME shouldn't it be validation_step_end instead? https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#validating-with-dataparallel
    def validation_epoch_end(self, validation_step_outputs: list[dict[str, Any]]) -> None:
        # flattening outputs from dataloaders if there are multiple
        if validation_step_outputs and isinstance(validation_step_outputs[0], list):
            outputs = [
                step_output
                for dataloader_output in validation_step_outputs
                for step_output in dataloader_output
            ]
        else:
            outputs = validation_step_outputs

        preds_logits: torch.Tensor = torch.tensor([pred for step_output in outputs for pred in step_output['pred']])
        preds: torch.Tensor = torch.where(preds_logits >= 0.5, 1, 0)  # TODO move threshold to the hparams
        golds: torch.Tensor = torch.tensor([gold for step_output in outputs for gold in step_output['gold']])

        preds_numpy = preds.cpu().numpy()
        golds_numpy = golds.cpu().numpy()

        # FIXME remove this debug after works alright
        rank_zero_info(f'tp: {np.sum(np.where(golds_numpy != 0, preds_numpy == golds_numpy, False))}')
        rank_zero_info(f'nonzero count:, {np.count_nonzero(preds_numpy)}')
        rank_zero_info(f'preds_logits: {preds_logits[0]}')

        metrics = calculate_metrics(preds_numpy, golds_numpy)
        self.log_dict(metrics, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.task.learning_rate,
            weight_decay=self.hparams.task.weight_decay
        )
        return optimizer
