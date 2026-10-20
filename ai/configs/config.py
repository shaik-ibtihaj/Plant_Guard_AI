"""Configuration defaults for the Plant Guard AI training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class DataConfig:
    """Dataset and DataLoader settings shared by training scripts."""

    data_dir: Path = Path("datasets/processed")
    statistics_path: Path = Path("datasets/metadata/rgb_stats.json")
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    use_weighted_sampler: bool = True
    seed: int = 42
    max_train_samples_per_class: Optional[int] = None
    max_eval_samples_per_class: Optional[int] = None

    def validate(self) -> None:
        """Reject invalid values early, before workers are started."""
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        for name, value in (
            ("max_train_samples_per_class", self.max_train_samples_per_class),
            ("max_eval_samples_per_class", self.max_eval_samples_per_class),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be greater than zero when provided")


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for baseline model training."""

    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.30
    early_stopping_patience: int = 5
    checkpoint_path: Path = Path("models/checkpoints/baseline_cnn_best.pth")
    history_path: Path = Path("models/runs/baseline_cnn_history.json")
    use_amp: bool = True

    def validate(self) -> None:
        """Validate training hyperparameters before allocating the model."""
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be greater than zero")


DEFAULT_MEAN: Tuple[float, float, float] = (0.229681, 0.263303, 0.234913)
DEFAULT_STD: Tuple[float, float, float] = (0.137491, 0.160573, 0.124940)
