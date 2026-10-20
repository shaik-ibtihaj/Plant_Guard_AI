"""Compare trained models and select a production checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

REQUIRED_METRICS = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "loss",
)


def _compact_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    """Exclude class-level arrays from comparison and manifest artifacts."""
    names = (*REQUIRED_METRICS, "precision_weighted", "recall_weighted", "samples")
    return {name: metrics[name] for name in names if name in metrics}


def load_model_result(path: Path) -> Dict[str, object]:
    """Load and validate one training metrics document."""
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not result.get("model") or not result.get("checkpoint"):
        raise ValueError(f"Missing model or checkpoint in {path}")
    metrics = result.get("validation_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing validation_metrics in {path}")
    missing = [name for name in REQUIRED_METRICS if name not in metrics]
    if missing:
        raise ValueError(f"Missing metrics in {path}: {missing}")
    return result


def rank_model_results(results: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Rank by macro-F1, then weighted-F1, accuracy, and lower loss."""
    ranked = list(results)
    if not ranked:
        raise ValueError("No model results were provided")

    def ranking_key(result: Dict[str, object]) -> tuple[float, float, float, float]:
        metrics = result["validation_metrics"]
        return (
            float(metrics["f1_macro"]),
            float(metrics["f1_weighted"]),
            float(metrics["accuracy"]),
            -float(metrics["loss"]),
        )

    return sorted(ranked, key=ranking_key, reverse=True)


def _write_markdown(ranked: List[Dict[str, object]], destination: Path) -> None:
    lines = [
        "# Phase 5 Model Comparison",
        "",
        "Models are ranked using validation macro-F1, with weighted-F1, accuracy,",
        "and validation loss as successive tie-breakers.",
        "",
        (
            "| Rank | Model | Accuracy | Macro Precision | Macro Recall | "
            "Macro F1 | Weighted F1 | Loss |"
        ),
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, result in enumerate(ranked, start=1):
        metrics = result["validation_metrics"]
        lines.append(
            f"| {rank} | {result['model']} | {metrics['accuracy']:.4f} | "
            f"{metrics['precision_macro']:.4f} | {metrics['recall_macro']:.4f} | "
            f"{metrics['f1_macro']:.4f} | {metrics['f1_weighted']:.4f} | "
            f"{metrics['loss']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Selected production model: **{ranked[0]['model']}**",
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def compare_and_select(
    metric_paths: Iterable[Path],
    output_json: Path,
    output_markdown: Path,
    manifest_path: Path,
    production_checkpoint: Path | None = None,
) -> Dict[str, object]:
    """Rank models, write reports, and optionally copy the winning checkpoint."""
    ranked = rank_model_results(load_model_result(path) for path in metric_paths)
    winner = ranked[0]
    source_checkpoint = Path(str(winner["checkpoint"]))
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Winning checkpoint not found: {source_checkpoint}")

    selected_checkpoint = source_checkpoint
    if production_checkpoint is not None:
        production_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint, production_checkpoint)
        selected_checkpoint = production_checkpoint

    comparison = {
        "selection_metric": "validation.f1_macro",
        "selected_model": winner["model"],
        "selected_checkpoint": str(selected_checkpoint),
        "rankings": [
            {
                "rank": rank,
                "model": result["model"],
                "checkpoint": result["checkpoint"],
                "parameter_count": result.get("parameter_count"),
                "validation_metrics": _compact_metrics(result["validation_metrics"]),
            }
            for rank, result in enumerate(ranked, start=1)
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    _write_markdown(ranked, output_markdown)

    manifest = {
        "model": winner["model"],
        "checkpoint": str(selected_checkpoint),
        "selected_by": "validation.f1_macro",
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "validation_metrics": _compact_metrics(winner["validation_metrics"]),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        nargs="+",
        type=Path,
        help="Per-model *_metrics.json files produced by training",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/model_comparison.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/model_comparison.md"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("models/production_model.json"),
    )
    parser.add_argument(
        "--production-checkpoint",
        type=Path,
        default=Path("models/best_model.pth"),
    )
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()
    compare_and_select(
        args.metrics,
        args.output_json,
        args.output_markdown,
        args.manifest,
        None if args.no_copy else args.production_checkpoint,
    )


if __name__ == "__main__":
    main()
