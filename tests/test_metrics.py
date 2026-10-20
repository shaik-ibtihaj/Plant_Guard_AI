"""Tests for standardized classification metrics."""

import pytest

from ai.utils.metrics import (
    compute_accuracy,
    compute_calibration_metrics,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_f1_score,
    compute_per_class_metrics,
)


def test_classification_metrics_return_expected_values() -> None:
    truth = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]

    metrics = compute_classification_metrics(truth, predictions)

    assert compute_accuracy(truth, predictions) == pytest.approx(0.75)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert compute_f1_score(truth, predictions, "macro") == pytest.approx(
        metrics["f1_macro"]
    )
    assert compute_confusion_matrix(truth, predictions).tolist() == [[1, 1], [0, 2]]


def test_metrics_reject_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_accuracy([0, 1], [0])


def test_per_class_metrics_include_missing_predictions() -> None:
    metrics = compute_per_class_metrics(
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        ["healthy", "diseased"],
    )

    assert metrics[0]["recall"] == pytest.approx(1.0)
    assert metrics[1]["precision"] == pytest.approx(0.0)
    assert metrics[1]["support"] == 2


def test_calibration_metrics_compute_expected_error() -> None:
    calibration = compute_calibration_metrics(
        [0.9, 0.8, 0.2, 0.1],
        [True, False, True, False],
        num_bins=2,
    )

    assert calibration["ece"] == pytest.approx(0.35)
    assert sum(item["count"] for item in calibration["bins"]) == 4
