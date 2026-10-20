"""Baseline convolutional network for plant disease classification."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """Two convolutions followed by spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )


class BaselineCNN(nn.Module):
    """Compact CNN baseline with input-size-independent global pooling."""

    def __init__(self, num_classes: int, dropout: float = 0.30) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        self.num_classes = num_classes
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one logit per disease class."""
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("inputs must have shape (batch, 3, height, width)")
        features = self.features(inputs)
        return self.classifier(self.pool(features))


def build_baseline_cnn(num_classes: int = 64, dropout: float = 0.30) -> BaselineCNN:
    """Construct the Phase 5 baseline classifier."""
    return BaselineCNN(num_classes=num_classes, dropout=dropout)
