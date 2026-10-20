"""Vision Transformer B/16 transfer-learning model."""

from __future__ import annotations

from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


def build_vit_b_16(
    num_classes: int = 64,
    dropout: float = 0.10,
    pretrained: bool = True,
) -> nn.Module:
    """Create ViT-B/16 with a Plant Guard classification head."""
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in the range [0, 1)")

    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
