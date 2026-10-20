"""Tests for model ranking and production selection."""

import json
from pathlib import Path

from ai.training.compare import compare_and_select, rank_model_results


def _result(model: str, checkpoint: Path, macro_f1: float) -> dict:
    return {
        "model": model,
        "checkpoint": str(checkpoint),
        "parameter_count": 100,
        "validation_metrics": {
            "accuracy": macro_f1 + 0.01,
            "precision_macro": macro_f1,
            "recall_macro": macro_f1,
            "f1_macro": macro_f1,
            "f1_weighted": macro_f1 + 0.005,
            "loss": 1 - macro_f1,
        },
    }


def test_rank_model_results_prefers_macro_f1(tmp_path: Path) -> None:
    first = _result("resnet50", tmp_path / "resnet.pth", 0.80)
    second = _result("efficientnet_b0", tmp_path / "efficientnet.pth", 0.85)

    ranked = rank_model_results([first, second])

    assert ranked[0]["model"] == "efficientnet_b0"


def test_compare_and_select_writes_artifacts(tmp_path: Path) -> None:
    metric_paths = []
    for model, score in (("baseline_cnn", 0.70), ("resnet50", 0.82)):
        checkpoint = tmp_path / f"{model}.pth"
        checkpoint.write_bytes(model.encode("utf-8"))
        metrics = tmp_path / f"{model}_metrics.json"
        metrics.write_text(
            json.dumps(_result(model, checkpoint, score)), encoding="utf-8"
        )
        metric_paths.append(metrics)

    output_json = tmp_path / "comparison.json"
    output_markdown = tmp_path / "comparison.md"
    manifest = tmp_path / "production.json"
    production = tmp_path / "best.pth"
    result = compare_and_select(
        metric_paths,
        output_json,
        output_markdown,
        manifest,
        production,
    )

    assert result["selected_model"] == "resnet50"
    assert output_json.is_file()
    assert "resnet50" in output_markdown.read_text(encoding="utf-8")
    assert json.loads(manifest.read_text(encoding="utf-8"))["model"] == "resnet50"
    assert production.read_bytes() == b"resnet50"
