"""Tests for training and data configuration validation."""

import pytest

from ai.configs.config import DataConfig, TrainingConfig


def test_default_configs_are_valid() -> None:
    DataConfig().validate()
    TrainingConfig().validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epochs", 0),
        ("learning_rate", 0),
        ("weight_decay", -0.1),
        ("dropout", 1.0),
        ("early_stopping_patience", 0),
    ],
)
def test_training_config_rejects_invalid_values(field: str, value: float) -> None:
    config = TrainingConfig(**{field: value})

    with pytest.raises(ValueError):
        config.validate()
