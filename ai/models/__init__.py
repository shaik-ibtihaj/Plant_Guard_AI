"""Model registry for all Phase 5 classification architectures."""

from __future__ import annotations

from typing import Callable, Dict

from torch import nn

from ai.models.cnn import BaselineCNN, build_baseline_cnn
from ai.models.efficientnet import build_efficientnet_b0
from ai.models.resnet import build_resnet50
from ai.models.vit import build_vit_b_16

SUPPORTED_MODELS = (
    "baseline_cnn",
    "efficientnet_b0",
    "resnet50",
    "vit_b_16",
)

MODEL_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    "baseline_cnn": build_baseline_cnn,
    "efficientnet_b0": build_efficientnet_b0,
    "resnet50": build_resnet50,
    "vit_b_16": build_vit_b_16,
}


def build_model(
    name: str,
    num_classes: int,
    dropout: float = 0.30,
    pretrained: bool = True,
) -> nn.Module:
    """Build a registered model by its canonical name."""
    try:
        builder = MODEL_BUILDERS[name]
    except KeyError as exc:
        choices = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unknown model '{name}'. Choose one of: {choices}") from exc

    if name == "baseline_cnn":
        return builder(num_classes=num_classes, dropout=dropout)
    return builder(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )


def freeze_feature_extractor(model: nn.Module, name: str) -> None:
    """Freeze a transfer model backbone while leaving its head trainable."""
    if name == "baseline_cnn":
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
    if name == "efficientnet_b0":
        head = model.classifier
    elif name == "resnet50":
        head = model.fc
    elif name == "vit_b_16":
        head = model.heads
    else:
        raise ValueError(f"Unknown model '{name}'")
    for parameter in head.parameters():
        parameter.requires_grad = True


__all__ = [
    "BaselineCNN",
    "MODEL_BUILDERS",
    "SUPPORTED_MODELS",
    "build_baseline_cnn",
    "build_efficientnet_b0",
    "build_model",
    "build_resnet50",
    "build_vit_b_16",
    "freeze_feature_extractor",
]
