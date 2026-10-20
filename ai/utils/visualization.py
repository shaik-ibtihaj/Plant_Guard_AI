"""Visualization utilities for training and model evaluation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence

_matplotlib_config = Path(tempfile.gettempdir()) / "plant_guard_matplotlib"
_matplotlib_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_config))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_accs: Sequence[float],
    val_accs: Sequence[float],
    save_path: Optional[str | Path] = None,
):
    """Plot loss and accuracy histories."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label="train")
    axes[0].plot(val_losses, label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(train_accs, label="train")
    axes[1].plot(val_accs, label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    return figure


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[str | Path] = None,
    *,
    normalize: bool = False,
):
    """Plot a raw or row-normalized confusion matrix for many classes."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (len(class_names), len(class_names)):
        raise ValueError("matrix shape must match the number of class names")
    if normalize:
        row_totals = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values,
            row_totals,
            out=np.zeros_like(values),
            where=row_totals != 0,
        )

    size = max(12, min(28, len(class_names) * 0.35))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(values, interpolation="nearest", cmap="YlGn")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title="Normalized Confusion Matrix" if normalize else "Confusion Matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=6)
    axis.tick_params(axis="y", labelsize=6)
    figure.tight_layout()
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    return figure


def plot_reliability_diagram(
    calibration_bins: Sequence[dict], save_path: Optional[str | Path] = None
):
    """Plot observed accuracy against confidence for populated bins."""
    populated = [item for item in calibration_bins if item["count"] > 0]
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    if populated:
        axis.plot(
            [item["mean_confidence"] for item in populated],
            [item["accuracy"] for item in populated],
            marker="o",
            linewidth=2,
            label="model",
        )
    axis.set(
        title="Reliability Diagram",
        xlabel="Mean confidence",
        ylabel="Observed accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    return figure


def plot_misclassification_gallery(
    errors: Sequence[dict],
    save_path: Optional[str | Path] = None,
    max_images: int = 16,
):
    """Plot the highest-confidence readable misclassifications."""
    readable = [item for item in errors if item.get("image_path")][:max_images]
    if not readable:
        raise ValueError("No misclassification images were provided")
    columns = 4
    rows = int(np.ceil(len(readable) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, rows * 4))
    axes_array = np.asarray(axes).reshape(-1)
    for axis, item in zip(axes_array, readable):
        with Image.open(item["image_path"]) as source:
            axis.imshow(source.convert("RGB"))
        true_name = str(item["true_class"])[:30]
        predicted_name = str(item["predicted_class"])[:30]
        axis.set_title(
            f"True: {true_name}\nPred: {predicted_name}\n"
            f"Confidence: {item['confidence']:.1%}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes_array[len(readable) :]:
        axis.axis("off")
    figure.suptitle("Highest-Confidence Misclassifications", fontsize=14)
    figure.tight_layout()
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    return figure
