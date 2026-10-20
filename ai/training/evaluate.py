"""Evaluate a trained Plant Guard classifier with standardized metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import torch
from torch import nn
from tqdm import tqdm

from ai.configs.config import DataConfig
from ai.data import create_data_pipeline
from ai.models import SUPPORTED_MODELS, build_model
from ai.utils.helpers import get_device, load_checkpoint
from ai.utils.metrics import (
    compute_classification_metrics,
    compute_calibration_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
)
from ai.utils.visualization import (
    plot_confusion_matrix,
    plot_misclassification_gallery,
    plot_reliability_diagram,
)


def evaluate_classifier(
    model: nn.Module,
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    class_names: Sequence[str],
) -> Dict[str, object]:
    """Evaluate a classifier and return JSON-serializable metrics."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    total_samples = 0
    top5_correct = 0
    confidence_total = 0.0
    confidences: list[float] = []
    correctness: list[bool] = []
    prediction_records: list[Dict[str, object]] = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(batches, desc="evaluate", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            y_true.extend(labels.cpu().tolist())
            probabilities = logits.softmax(dim=1)
            confidence, predictions = probabilities.max(dim=1)
            top_count = min(5, logits.shape[1])
            top_predictions = logits.topk(top_count, dim=1).indices
            top5_correct += (
                top_predictions.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            )
            confidence_total += confidence.sum().item()
            label_values = labels.cpu().tolist()
            prediction_values = predictions.cpu().tolist()
            confidence_values = confidence.cpu().tolist()
            top_values = top_predictions.cpu().tolist()
            y_pred.extend(prediction_values)
            for true_label, predicted_label, score, top_indices in zip(
                label_values,
                prediction_values,
                confidence_values,
                top_values,
            ):
                is_correct = true_label == predicted_label
                confidences.append(float(score))
                correctness.append(is_correct)
                prediction_records.append(
                    {
                        "sample_index": len(prediction_records),
                        "true_index": true_label,
                        "true_class": class_names[true_label],
                        "predicted_index": predicted_label,
                        "predicted_class": class_names[predicted_label],
                        "confidence": float(score),
                        "correct": is_correct,
                        "top5_indices": top_indices,
                        "top5_classes": [class_names[index] for index in top_indices],
                    }
                )

    if total_samples == 0:
        raise ValueError("DataLoader produced no samples")
    metrics: Dict[str, object] = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / total_samples
    metrics["samples"] = total_samples
    metrics["top5_accuracy"] = top5_correct / total_samples
    metrics["mean_confidence"] = confidence_total / total_samples
    metrics["confusion_matrix"] = compute_confusion_matrix(
        y_true, y_pred, class_names
    ).tolist()
    metrics["class_names"] = list(class_names)
    metrics["per_class"] = compute_per_class_metrics(y_true, y_pred, class_names)
    metrics["calibration"] = compute_calibration_metrics(confidences, correctness)
    metrics["predictions"] = prediction_records
    return metrics


def write_evaluation_artifacts(results: Dict[str, object], output_dir: Path) -> None:
    """Write metrics, diagnostic plots, predictions, and a Markdown summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    per_class = results["per_class"]
    csv_path = output_dir / "per_class_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "class_index",
                "class_name",
                "precision",
                "recall",
                "f1",
                "support",
            ),
        )
        writer.writeheader()
        writer.writerows(per_class)

    predictions = results["predictions"]
    prediction_fields = (
        "sample_index",
        "image_path",
        "true_index",
        "true_class",
        "predicted_index",
        "predicted_class",
        "confidence",
        "correct",
        "top5_classes",
    )
    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for item in predictions:
            row = {field: item.get(field, "") for field in prediction_fields}
            row["top5_classes"] = " | ".join(item.get("top5_classes", []))
            writer.writerow(row)

    matrix = results["confusion_matrix"]
    class_names = results["class_names"]
    raw_figure = plot_confusion_matrix(
        matrix, class_names, output_dir / "confusion_matrix.png"
    )
    normalized_figure = plot_confusion_matrix(
        matrix,
        class_names,
        output_dir / "confusion_matrix_normalized.png",
        normalize=True,
    )
    import matplotlib.pyplot as plt

    plt.close(raw_figure)
    plt.close(normalized_figure)

    calibration = results["calibration"]
    reliability_figure = plot_reliability_diagram(
        calibration["bins"], output_dir / "reliability_diagram.png"
    )
    plt.close(reliability_figure)

    errors = sorted(
        (item for item in predictions if not item["correct"]),
        key=lambda item: item["confidence"],
        reverse=True,
    )
    confusion_counts = Counter(
        (item["true_class"], item["predicted_class"]) for item in errors
    )
    dominant_confusions = [
        {"true_class": truth, "predicted_class": predicted, "count": count}
        for (truth, predicted), count in confusion_counts.most_common(20)
    ]
    error_analysis = {
        "total_errors": len(errors),
        "error_rate": len(errors) / results["samples"],
        "expected_calibration_error": calibration["ece"],
        "dominant_confusions": dominant_confusions,
        "highest_confidence_errors": errors[:25],
    }
    with (output_dir / "error_analysis.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(error_analysis, handle, indent=2)

    readable_errors = [item for item in errors if item.get("image_path")]
    if readable_errors:
        gallery_figure = plot_misclassification_gallery(
            readable_errors,
            output_dir / "misclassification_gallery.png",
        )
        plt.close(gallery_figure)

    weakest = sorted(per_class, key=lambda item: (item["recall"], item["f1"]))[:10]
    lines = [
        "# Phase 6 Model Evaluation",
        "",
        f"- Model: `{results['model']}`",
        f"- Checkpoint: `{results['checkpoint']}`",
        f"- Split: `{results['split']}`",
        f"- Samples: {results['samples']}",
        f"- Loss: {results['loss']:.4f}",
        f"- Top-1 accuracy: {results['accuracy']:.2%}",
        f"- Top-5 accuracy: {results['top5_accuracy']:.2%}",
        f"- Macro F1: {results['f1_macro']:.2%}",
        f"- Weighted F1: {results['f1_weighted']:.2%}",
        f"- Mean confidence: {results['mean_confidence']:.2%}",
        f"- Expected calibration error: {calibration['ece']:.2%}",
        f"- Errors: {len(errors)} ({len(errors) / results['samples']:.2%})",
        "",
        "## Ten Lowest-Recall Classes",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in weakest:
        lines.append(
            f"| {row['class_name']} | {row['precision']:.2%} | "
            f"{row['recall']:.2%} | {row['f1']:.2%} | {row['support']} |"
        )
    lines.extend(
        [
            "",
            "## Most Frequent Confusions",
            "",
            "| True class | Predicted class | Count |",
            "|---|---|---:|",
        ]
    )
    for row in dominant_confusions[:10]:
        lines.append(
            f"| {row['true_class']} | {row['predicted_class']} | "
            f"{row['count']} |"
        )
    lines.extend(
        [
            "",
            "> Results must be interpreted using the training and evaluation scope ",
            "> documented in the project README.",
            "",
        ]
    )
    (output_dir / "evaluation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def evaluate_checkpoint(
    model_name: str,
    checkpoint_path: Path,
    output_path: Path,
    data_config: DataConfig,
    split: str = "val",
) -> Dict[str, object]:
    """Load and evaluate a model checkpoint on validation or test data."""
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    pipeline = create_data_pipeline(data_config)
    device = get_device()
    model = build_model(
        model_name,
        num_classes=len(pipeline.class_names),
        pretrained=False,
    ).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    loader = pipeline.val if split == "val" else pipeline.test
    results = evaluate_classifier(model, loader, device, pipeline.class_names)
    samples = getattr(loader.dataset, "samples", ())
    for index, prediction in enumerate(results["predictions"]):
        if index < len(samples):
            prediction["image_path"] = str(samples[index][0])
    results.update(
        {
            "model": model_name,
            "split": split,
            "checkpoint": str(checkpoint_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--eval-samples-per-class", type=int)
    parser.add_argument("--artifacts-dir", type=Path)
    args = parser.parse_args()
    config = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_weighted_sampler=False,
        max_eval_samples_per_class=args.eval_samples_per_class,
    )
    results = evaluate_checkpoint(
        args.model,
        args.checkpoint,
        args.output,
        config,
        args.split,
    )
    if args.artifacts_dir is not None:
        write_evaluation_artifacts(results, args.artifacts_dir)


if __name__ == "__main__":
    main()
