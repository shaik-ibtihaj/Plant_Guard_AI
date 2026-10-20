"""Reproducibility, device selection, and checkpoint helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators."""
    if seed < 0:
        raise ValueError("seed cannot be negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA when available, otherwise return CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    filepath: str | Path,
    *,
    scheduler: Optional[Any] = None,
    metrics: Optional[Mapping[str, float]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically save enough state to resume training and serve the model."""
    if epoch < 0:
        raise ValueError("epoch cannot be negative")

    destination = Path(filepath)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "metrics": dict(metrics or {}),
        "metadata": dict(metadata or {}),
    }
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_checkpoint(
    filepath: str | Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    *,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """Restore a checkpoint and return its completed epoch number."""
    source = Path(filepath)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")

    checkpoint = torch.load(source, map_location=map_location, weights_only=True)
    if "model_state_dict" not in checkpoint or "epoch" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {source}")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"])
