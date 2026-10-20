"""Dataset, class-balancing, and DataLoader construction for Phase 5."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ai.configs.config import DataConfig
from ai.data.augmentation import build_train_transform
from ai.data.preprocessing import build_eval_transform, load_normalization_stats

VALID_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


class PlantDiseaseDataset(Dataset):
    """Read an image-folder split while keeping a stable class mapping."""

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Mapping[str, int]] = None,
        max_samples_per_class: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        if max_samples_per_class is not None and max_samples_per_class <= 0:
            raise ValueError("max_samples_per_class must be greater than zero")

        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {self.root}")

        discovered_classes = sorted(
            path.name for path in self.root.iterdir() if path.is_dir()
        )
        if not discovered_classes:
            raise ValueError(f"No class directories found in: {self.root}")

        if class_to_idx is None:
            self.class_to_idx = {
                class_name: index for index, class_name in enumerate(discovered_classes)
            }
        else:
            self.class_to_idx = dict(class_to_idx)
            expected = set(self.class_to_idx)
            discovered = set(discovered_classes)
            if discovered != expected:
                missing = sorted(expected - discovered)
                unexpected = sorted(discovered - expected)
                raise ValueError(
                    f"Class mismatch in {self.root}; "
                    f"missing={missing}, unexpected={unexpected}"
                )

        indices = sorted(self.class_to_idx.values())
        if indices != list(range(len(indices))):
            raise ValueError("class_to_idx values must be contiguous and start at zero")

        ordered_classes = sorted(
            self.class_to_idx.items(), key=lambda item: item[1]
        )
        self.classes = [name for name, _ in ordered_classes]
        self.samples: list[Tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            label = self.class_to_idx[class_name]
            paths = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
            )
            if max_samples_per_class is not None and len(paths) > max_samples_per_class:
                class_random = random.Random(seed + label)
                paths = sorted(class_random.sample(paths, max_samples_per_class))
            self.samples.extend((path, label) for path in paths)

        if not self.samples:
            raise ValueError(f"No supported images found in: {self.root}")
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Unable to read image: {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)
        return image, label


@dataclass(frozen=True)
class DataPipeline:
    """DataLoaders and metadata needed by training and evaluation."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: Sequence[str]
    class_to_idx: Mapping[str, int]
    class_weights: torch.Tensor


def compute_class_weights(targets: Sequence[int], num_classes: int) -> torch.Tensor:
    """Return inverse-frequency weights for a class-weighted loss function."""
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero")
    if not targets:
        raise ValueError("targets cannot be empty")

    target_tensor = torch.as_tensor(targets, dtype=torch.long)
    if target_tensor.min().item() < 0 or target_tensor.max().item() >= num_classes:
        raise ValueError("targets contain an out-of-range class index")

    counts = torch.bincount(target_tensor, minlength=num_classes).to(torch.float64)
    if torch.any(counts == 0):
        empty_classes = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Cannot weight empty classes: {empty_classes}")

    weights = target_tensor.numel() / (num_classes * counts)
    return weights.to(torch.float32)


def create_weighted_sampler(
    targets: Sequence[int], class_weights: torch.Tensor, seed: int = 42
) -> WeightedRandomSampler:
    """Create a reproducible sampler assigning each sample its class weight."""
    if not targets:
        raise ValueError("targets cannot be empty")
    target_tensor = torch.as_tensor(targets, dtype=torch.long)
    sample_weights = class_weights.to(torch.float64)[target_tensor]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(targets),
        replacement=True,
        generator=generator,
    )


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_pipeline(config: DataConfig = DataConfig()) -> DataPipeline:
    """Construct train/validation/test datasets and their DataLoaders."""
    config.validate()
    mean, std = load_normalization_stats(config.statistics_path)
    train_transform = build_train_transform(config.image_size, mean, std)
    eval_transform = build_eval_transform(config.image_size, mean, std)

    train_dataset = PlantDiseaseDataset(
        config.data_dir / "train",
        train_transform,
        max_samples_per_class=config.max_train_samples_per_class,
        seed=config.seed,
    )
    val_dataset = PlantDiseaseDataset(
        config.data_dir / "val",
        eval_transform,
        train_dataset.class_to_idx,
        max_samples_per_class=config.max_eval_samples_per_class,
        seed=config.seed,
    )
    test_dataset = PlantDiseaseDataset(
        config.data_dir / "test",
        eval_transform,
        train_dataset.class_to_idx,
        max_samples_per_class=config.max_eval_samples_per_class,
        seed=config.seed,
    )

    class_weights = compute_class_weights(
        train_dataset.targets, len(train_dataset.classes)
    )
    sampler = (
        create_weighted_sampler(train_dataset.targets, class_weights, config.seed)
        if config.use_weighted_sampler
        else None
    )
    generator = torch.Generator().manual_seed(config.seed)
    common: Dict[str, object] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }

    train_loader = DataLoader(
        train_dataset, sampler=sampler, shuffle=sampler is None, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)

    return DataPipeline(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_names=tuple(train_dataset.classes),
        class_to_idx=dict(train_dataset.class_to_idx),
        class_weights=class_weights,
    )
