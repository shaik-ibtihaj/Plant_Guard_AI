"""Tests for the Phase 5 data pipeline."""

from pathlib import Path

import pytest

from PIL import Image

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    pytest.importorskip("torchvision")
    from ai.configs.config import DataConfig
    from ai.data.dataset import (
        PlantDiseaseDataset,
        compute_class_weights,
        create_data_pipeline,
    )


requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def _build_tiny_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        _write_image(root / split / "Apple___healthy" / "one.jpg", (20, 180, 20))
        _write_image(root / split / "Apple___scab" / "one.png", (100, 60, 20))
    _write_image(root / "train" / "Apple___healthy" / "two.jpeg", (30, 190, 30))


@requires_torch
def test_dataset_discovers_images_and_converts_rgb(tmp_path: Path) -> None:
    _build_tiny_dataset(tmp_path)
    dataset = PlantDiseaseDataset(tmp_path / "train")

    assert dataset.classes == ["Apple___healthy", "Apple___scab"]
    assert len(dataset) == 3
    image, label = dataset[0]
    assert image.mode == "RGB"
    assert label in (0, 1)


@requires_torch
def test_class_weights_are_inverse_frequency() -> None:
    weights = compute_class_weights([0, 0, 0, 1], num_classes=2)

    assert weights.tolist() == pytest.approx([2 / 3, 2.0])


@requires_torch
def test_pipeline_builds_consistent_loaders(tmp_path: Path) -> None:
    data_dir = tmp_path / "processed"
    _build_tiny_dataset(data_dir)
    stats_path = tmp_path / "rgb_stats.json"
    stats_path.write_text(
        '{"mean": [0.2, 0.3, 0.2], "std": [0.1, 0.2, 0.1]}',
        encoding="utf-8",
    )
    config = DataConfig(
        data_dir=data_dir,
        statistics_path=stats_path,
        image_size=24,
        batch_size=2,
        use_weighted_sampler=True,
    )

    pipeline = create_data_pipeline(config)
    images, labels = next(iter(pipeline.train))

    assert images.shape == (2, 3, 24, 24)
    assert labels.shape == (2,)
    assert pipeline.class_names == ("Apple___healthy", "Apple___scab")
    assert pipeline.class_weights.tolist() == pytest.approx([0.75, 1.5])


@requires_torch
def test_dataset_rejects_split_class_mismatch(tmp_path: Path) -> None:
    _write_image(tmp_path / "Apple___healthy" / "one.jpg", (20, 180, 20))

    with pytest.raises(ValueError, match="Class mismatch"):
        PlantDiseaseDataset(
            tmp_path,
            class_to_idx={"Apple___healthy": 0, "Apple___scab": 1},
        )


@requires_torch
def test_dataset_limits_each_class_reproducibly(tmp_path: Path) -> None:
    for index in range(4):
        _write_image(
            tmp_path / "Apple___healthy" / f"{index}.jpg",
            (index, 180, 20),
        )
    first = PlantDiseaseDataset(tmp_path, max_samples_per_class=2, seed=7)
    second = PlantDiseaseDataset(tmp_path, max_samples_per_class=2, seed=7)

    assert len(first) == 2
    assert first.samples == second.samples
