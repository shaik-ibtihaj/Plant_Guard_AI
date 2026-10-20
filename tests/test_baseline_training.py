"""Focused tests for the baseline model and training engine."""

from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset

    from ai.models import SUPPORTED_MODELS, build_model, freeze_feature_extractor
    from ai.models.cnn import BaselineCNN
    from ai.training.train import run_epoch
    from ai.utils.helpers import load_checkpoint, save_checkpoint


requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")


@requires_torch
def test_baseline_cnn_output_shape() -> None:
    model = BaselineCNN(num_classes=7, dropout=0.1)
    inputs = torch.randn(2, 3, 64, 64)

    outputs = model(inputs)

    assert outputs.shape == (2, 7)


@requires_torch
@pytest.mark.parametrize("model_name", SUPPORTED_MODELS if torch is not None else ())
def test_registered_models_have_correct_output_head(model_name: str) -> None:
    model = build_model(model_name, num_classes=7, pretrained=False)
    output_layers = [
        module for module in model.modules() if isinstance(module, nn.Linear)
    ]

    assert output_layers[-1].out_features == 7


@requires_torch
def test_freeze_feature_extractor_keeps_transfer_head_trainable() -> None:
    model = build_model("efficientnet_b0", num_classes=7, pretrained=False)

    freeze_feature_extractor(model, "efficientnet_b0")

    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


@requires_torch
def test_run_epoch_returns_finite_metrics() -> None:
    model = BaselineCNN(num_classes=3, dropout=0.0)
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 1])),
        batch_size=2,
    )
    optimizer = AdamW(model.parameters(), lr=1e-3)

    loss, accuracy = run_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        optimizer=optimizer,
        description="test",
    )

    assert loss >= 0
    assert 0 <= accuracy <= 1


@requires_torch
def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = BaselineCNN(num_classes=3)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    checkpoint = tmp_path / "model.pth"
    original = model.classifier[-1].weight.detach().clone()
    save_checkpoint(model, optimizer, 4, checkpoint, metrics={"val_loss": 0.5})
    with torch.no_grad():
        model.classifier[-1].weight.zero_()

    epoch = load_checkpoint(checkpoint, model, optimizer)

    assert epoch == 4
    assert torch.equal(model.classifier[-1].weight, original)
