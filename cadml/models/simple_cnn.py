import torch
from torch import nn
from torchtyping import TensorType

from omegaconf import DictConfig

from ..models.cnn_blocks import SimpleCNNBlock


class SimpleCNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.in_channels = cfg.model.in_channels

        self.encoder = nn.Sequential(
            SimpleCNNBlock(self.in_channels, 20, 0.1),
            SimpleCNNBlock(20, 40, 0.2),
            SimpleCNNBlock(40, 80, 0.3),
            SimpleCNNBlock(80, 160, 0.4),
            nn.Flatten()
        )

        self.head = nn.Sequential(
            nn.Linear(640, 1),
            nn.Sigmoid()
        )

    def forward(self, X: TensorType["batch_size", "channels", "height", "width"]) -> TensorType["batch_size"]:
        embeddings = self.encoder(X)
        logits = self.head(embeddings)
        return logits