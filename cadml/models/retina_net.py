import torch
from torch import nn
from torchtyping import TensorType

from torchvision import models

from omegaconf import DictConfig


class RetinaNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.in_channels = cfg.model.in_channels
        self.use_pretrained = cfg.model.use_pretrained

        weights = models.detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        self.preprocess = weights.transforms()
        if self.use_pretrained:
            self.model = models.detection.retinanet_resnet50_fpn_v2(weights=weights, num_classes=2)
        else:
            self.model = models.detection.retinanet_resnet50_fpn_v2(weights=None, num_classes=2)

        # TODO

    def forward(self, X: TensorType["batch_size", "channels", "height", "width"]) -> TensorType["batch_size"]:
        X = X.repeat(1, 3, 1, 1)  # increases the number of in_channels to 3
        X = self.preprocess(X)
        # TODO
