import torch
from torch import nn
from torchtyping import TensorType

from torchvision import models

from omegaconf import DictConfig


class ResNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.in_channels = cfg.model.in_channels
        self.use_pretrained = cfg.model.use_pretrained

        self.model = models.resnet18(pretrained=self.use_pretrained)
        embedding_size = self.model.fc.in_features
        self.model.fc = nn.Linear(embedding_size, 1)

    def forward(self, X: TensorType["batch_size", "channels", "height", "width"]) -> TensorType["batch_size"]:
        # TODO try out: make 1st channel as the roi a little to the left, and 3rd channel as the roi a little to the right
        X = X.repeat(1, 3, 1, 1)  # increases the number of in_channels to 3
        # TODO resize? normalize?
        logits = self.model(X)
        return logits