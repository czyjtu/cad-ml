import torch
from torch import nn
from torchtyping import TensorType

from torchvision import models

from omegaconf import DictConfig


class FasterRCNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.in_channels = cfg.model.in_channels
        self.num_classes = 2  # background and stenosis
        self.use_pretrained = cfg.model.use_pretrained

        if str(cfg.model.version) == '1':
            weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            model_class = models.detection.fasterrcnn_resnet50_fpn
        elif str(cfg.model.version) == '2':
            weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            model_class = models.detection.fasterrcnn_resnet50_fpn_v2
        else:
            raise ValueError(f'unknown model version - {cfg.model.version}, can be either `1` or `2`')

        self.preprocess = weights.transforms()
        if self.use_pretrained:
            self.model = model_class(weights=weights, num_classes=2)
        else:
            self.model = model_class(weights=None, num_classes=2)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = \
            models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)

    def forward(self, X: TensorType["batch_size", "channels", "height", "width"]) -> TensorType["batch_size"]:
        X = X.repeat(1, 3, 1, 1)  # increases the number of in_channels to 3
        X = self.preprocess(X)

        return self.model(X)
