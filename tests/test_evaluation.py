"""Tests for Phase 6 evaluation metrics and artifacts."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ai.training.evaluate import evaluate_classifier, write_evaluation_artifacts


def test_evaluation_writes_complete_artifact_set(tmp_path: Path) -> None:
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)

    results = evaluate_classifier(
        nn.Identity(),
        loader,
        torch.device("cpu"),
        ["class_a", "class_b", "class_c"],
    )
    results.update(
        {
            "model": "test_model",
            "checkpoint": "test.pth",
            "split": "test",
        }
    )
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (16, 16), color="green").save(image_path)
    for prediction in results["predictions"]:
        prediction["image_path"] = str(image_path)
    write_evaluation_artifacts(results, tmp_path)

    assert results["accuracy"] == 0.75
    assert results["top5_accuracy"] == 1.0
    assert len(results["per_class"]) == 3
    assert (tmp_path / "per_class_metrics.csv").is_file()
    assert (tmp_path / "confusion_matrix.png").is_file()
    assert (tmp_path / "confusion_matrix_normalized.png").is_file()
    assert (tmp_path / "predictions.csv").is_file()
    assert (tmp_path / "error_analysis.json").is_file()
    assert (tmp_path / "reliability_diagram.png").is_file()
    assert (tmp_path / "misclassification_gallery.png").is_file()
    assert (tmp_path / "evaluation_report.md").is_file()
