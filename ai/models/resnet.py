"""ResNet50 transfer-learning model."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50(
    num_classes: int = 64,
    dropout: float = 0.30,
    pretrained: bool = True,
) -> nn.Module:
    """Create ResNet50 with a regularized Plant Guard classification head."""
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in the range [0, 1)")

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
