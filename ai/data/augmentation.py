"""Training-time image augmentation for plant disease classification."""

from __future__ import annotations

from typing import Sequence

from torchvision import transforms

from ai.configs.config import DEFAULT_MEAN, DEFAULT_STD


def build_train_transform(
    image_size: int = 224,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> transforms.Compose:
    """Build augmentation suited to both leaf scans and field photographs."""
    if image_size <= 0:
        raise ValueError("image_size must be greater than zero")

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(
                degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
