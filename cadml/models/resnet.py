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

        if cfg.model.version == '18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model_class = models.resnet18
        elif cfg.model.version == '50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model_class = models.resnet50
        else:
            raise ValueError('unknown model version, can be either `18` or `50`')

        self.preprocess = weights.transforms()
        if self.use_pretrained:
            self.model = model_class(weights=weights)
        else:
            self.model = model_class(weights=None)
        embedding_size = self.model.fc.in_features
        self.model.fc = nn.Linear(embedding_size, 1)

    def forward(self, X: TensorType["batch_size", "channels", "height", "width"]) -> TensorType["batch_size"]:
        X = X.repeat(1, 3, 1, 1)  # increases the number of in_channels to 3
        X = self.preprocess(X)
        # TODO try out: make 1st channel as the roi a little to the left, and 3rd channel as the roi a little to the right
        # TODO resize? normalize?
        logits = self.model(X)
        return logits