"""Deterministic preprocessing used for validation, testing, and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Tuple

from torchvision import transforms

from ai.configs.config import DEFAULT_MEAN, DEFAULT_STD


def load_normalization_stats(
    path: str | Path,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Load three-channel mean and standard deviation values from JSON."""
    stats_path = Path(path)
    if not stats_path.is_file():
        raise FileNotFoundError(f"Normalization statistics not found: {stats_path}")

    with stats_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    try:
        mean = tuple(float(value) for value in payload["mean"])
        std = tuple(float(value) for value in payload["std"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid normalization statistics in {stats_path}") from exc

    if len(mean) != 3 or len(std) != 3:
        raise ValueError("Normalization mean and std must each contain three values")
    if any(value <= 0 for value in std):
        raise ValueError("Normalization standard deviations must be positive")
    return mean, std


def build_eval_transform(
    image_size: int = 224,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> transforms.Compose:
    """Build deterministic RGB preprocessing for evaluation and inference."""
    if image_size <= 0:
        raise ValueError("image_size must be greater than zero")

    resize_size = round(image_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
