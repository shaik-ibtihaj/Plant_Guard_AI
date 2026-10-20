"""Run the complete Phase 5 train, compare, and select workflow."""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import torch

from ai.configs.config import DataConfig, TrainingConfig
from ai.data import create_data_pipeline
from ai.models import SUPPORTED_MODELS
from ai.training.compare import compare_and_select
from ai.training.train import fit_model
from ai.utils.helpers import get_device, set_seed

LEARNING_RATES = {
    "baseline_cnn": 1e-3,
    "efficientnet_b0": 3e-4,
    "resnet50": 3e-4,
    "vit_b_16": 1e-4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--freeze-backbones", action="store_true")
    parser.add_argument("--train-samples-per-class", type=int)
    parser.add_argument("--eval-samples-per-class", type=int)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    data_config = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_weighted_sampler=True,
        max_train_samples_per_class=args.train_samples_per_class,
        max_eval_samples_per_class=args.eval_samples_per_class,
    )
    set_seed(data_config.seed)
    device = get_device()
    pipeline = create_data_pipeline(data_config)
    metric_paths: list[Path] = []

    for model_name in args.models:
        logging.info("Starting Phase 5 model: %s", model_name)
        set_seed(data_config.seed)
        sampler_generator = getattr(pipeline.train.sampler, "generator", None)
        if sampler_generator is not None:
            sampler_generator.manual_seed(data_config.seed)
        if pipeline.train.generator is not None:
            pipeline.train.generator.manual_seed(data_config.seed)
        config = TrainingConfig(
            epochs=args.epochs,
            learning_rate=LEARNING_RATES[model_name],
            dropout=0.10 if model_name == "vit_b_16" else 0.30,
            early_stopping_patience=args.patience,
            checkpoint_path=Path(f"models/checkpoints/{model_name}_best.pth"),
            history_path=Path(f"models/runs/{model_name}_history.json"),
            use_amp=not args.no_amp,
        )
        fit_model(
            pipeline,
            config,
            model_name=model_name,
            pretrained=not args.no_pretrained,
            freeze_backbone=args.freeze_backbones,
            device=device,
        )
        metric_paths.append(Path(f"models/runs/{model_name}_metrics.json"))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison = compare_and_select(
        metric_paths,
        Path("reports/model_comparison.json"),
        Path("reports/model_comparison.md"),
        Path("models/production_model.json"),
        Path("models/best_model.pth"),
    )
    logging.info("Selected production model: %s", comparison["selected_model"])


if __name__ == "__main__":
    main()
