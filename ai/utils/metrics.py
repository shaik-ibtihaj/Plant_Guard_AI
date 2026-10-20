"""Validated classification metrics shared by evaluation and comparison."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def _validate_labels(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if truth.ndim != 1 or predictions.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if truth.size == 0:
        raise ValueError("y_true and y_pred cannot be empty")
    if truth.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have the same length")
    return truth, predictions


def compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute classification accuracy."""
    truth, predictions = _validate_labels(y_true, y_pred)
    return float(accuracy_score(truth, predictions))


def compute_f1_score(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: str = "weighted",
) -> float:
    """Compute an F1 score with zero-division safety."""
    truth, predictions = _validate_labels(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(
        truth,
        predictions,
        average=average,
        zero_division=0,
    )
    return float(f1)


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Return a consistently ordered confusion matrix."""
    truth, predictions = _validate_labels(y_true, y_pred)
    labels = range(len(class_names)) if class_names is not None else None
    return confusion_matrix(truth, predictions, labels=labels)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, float]:
    """Compute accuracy plus macro and weighted precision/recall/F1."""
    truth, predictions = _validate_labels(y_true, y_pred)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(truth, predictions))
    }
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth,
            predictions,
            average=average,
            zero_division=0,
        )
        metrics[f"precision_{average}"] = float(precision)
        metrics[f"recall_{average}"] = float(recall)
        metrics[f"f1_{average}"] = float(f1)
    return metrics


def compute_per_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> list[Dict[str, float | int | str]]:
    """Return precision, recall, F1, and support for every known class."""
    truth, predictions = _validate_labels(y_true, y_pred)
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return [
        {
            "class_index": index,
            "class_name": class_name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    ]


def compute_calibration_metrics(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    num_bins: int = 10,
) -> Dict[str, object]:
    """Compute expected calibration error and reliability-bin statistics."""
    confidence_array = np.asarray(confidences, dtype=np.float64)
    correct_array = np.asarray(correctness, dtype=np.float64)
    if confidence_array.ndim != 1 or correct_array.ndim != 1:
        raise ValueError("confidences and correctness must be one-dimensional")
    if confidence_array.size == 0 or confidence_array.shape != correct_array.shape:
        raise ValueError("confidences and correctness must be non-empty and aligned")
    if num_bins <= 0:
        raise ValueError("num_bins must be greater than zero")
    if np.any((confidence_array < 0) | (confidence_array > 1)):
        raise ValueError("confidence values must be in the range [0, 1]")

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.minimum(
        np.digitize(confidence_array, edges[1:-1], right=True), num_bins - 1
    )
    bins = []
    expected_calibration_error = 0.0
    for index in range(num_bins):
        mask = bin_indices == index
        count = int(mask.sum())
        mean_confidence = float(confidence_array[mask].mean()) if count else 0.0
        accuracy = float(correct_array[mask].mean()) if count else 0.0
        expected_calibration_error += (
            count / confidence_array.size
        ) * abs(accuracy - mean_confidence)
        bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {"ece": float(expected_calibration_error), "bins": bins}
