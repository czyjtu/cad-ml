import torch
from torch import nn
from torchtyping import TensorType


class SimpleCNNBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout_rate: float
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, X: TensorType["batch_size", "channels", "height", "width"]) -> \
            TensorType["batch_size", "channels", "height", "width"]:

        return self.block(X)
