"""Train any Phase 5 model on the prepared Plant Guard dataset."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from ai.configs.config import DataConfig, TrainingConfig
from ai.data import DataPipeline, create_data_pipeline
from ai.models import SUPPORTED_MODELS, build_model, freeze_feature_extractor
from ai.training.evaluate import evaluate_classifier
from ai.utils.helpers import (
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)

logger = logging.getLogger(__name__)


def run_epoch(
    model: nn.Module,
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    description: str = "epoch",
) -> Tuple[float, float]:
    """Run one training or evaluation epoch and return loss and accuracy."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_amp = scaler is not None and scaler.is_enabled()

    progress = tqdm(batches, desc=description, leave=False)
    context = torch.enable_grad if is_training else torch.no_grad
    with context():
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.detach().item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")

    if total_samples == 0:
        raise ValueError("DataLoader produced no samples")
    return total_loss / total_samples, total_correct / total_samples


def fit_model(
    pipeline: DataPipeline,
    training_config: TrainingConfig,
    *,
    model_name: str = "baseline_cnn",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: torch.device | None = None,
) -> Dict[str, list[float]]:
    """Train a registered model with checkpointing and early stopping."""
    training_config.validate()
    device = device or get_device()
    model = build_model(
        model_name,
        num_classes=len(pipeline.class_names),
        dropout=training_config.dropout,
        pretrained=pretrained,
    ).to(device)
    if freeze_backbone:
        freeze_feature_extractor(model, model_name)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # Weighted sampling already balances training batches. Do not also weight
    # the loss, which would over-correct minority classes.
    uses_weighted_sampler = isinstance(pipeline.train.sampler, WeightedRandomSampler)
    loss_weights = None if uses_weighted_sampler else pipeline.class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    amp_enabled = training_config.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    history: Dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, training_config.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            pipeline.train,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            description=f"train {epoch}/{training_config.epochs}",
        )
        val_loss, val_accuracy = run_epoch(
            model,
            pipeline.val,
            criterion,
            device,
            description=f"val {epoch}/{training_config.epochs}",
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        logger.info(
            "Epoch %d/%d - train loss %.4f acc %.4f - val loss %.4f acc %.4f",
            epoch,
            training_config.epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                optimizer,
                epoch,
                training_config.checkpoint_path,
                scheduler=scheduler,
                metrics={"val_loss": val_loss, "val_accuracy": val_accuracy},
                metadata={
                    "architecture": model_name,
                    "num_classes": len(pipeline.class_names),
                    "class_to_idx": dict(pipeline.class_to_idx),
                    "image_size": 224,
                    "pretrained": pretrained,
                    "backbone_frozen": freeze_backbone,
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.early_stopping_patience:
                logger.info("Early stopping after epoch %d", epoch)
                break

    training_config.history_path.parent.mkdir(parents=True, exist_ok=True)
    with training_config.history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    load_checkpoint(training_config.checkpoint_path, model, map_location=device)
    validation_metrics = evaluate_classifier(
        model, pipeline.val, device, pipeline.class_names
    )
    summary = {
        "model": model_name,
        "checkpoint": str(training_config.checkpoint_path),
        "history": str(training_config.history_path),
        "epochs_completed": len(history["train_loss"]),
        "training_samples": len(pipeline.train.dataset),
        "validation_samples": len(pipeline.val.dataset),
        "pretrained": pretrained,
        "backbone_frozen": freeze_backbone,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "validation_metrics": validation_metrics,
    }
    metrics_path = training_config.history_path.with_name(f"{model_name}_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return history


def fit_baseline(
    pipeline: DataPipeline,
    training_config: TrainingConfig,
    *,
    device: torch.device | None = None,
) -> Dict[str, list[float]]:
    """Backward-compatible wrapper for baseline CNN training."""
    return fit_model(
        pipeline,
        training_config,
        model_name="baseline_cnn",
        pretrained=False,
        freeze_backbone=False,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides for model training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="baseline_cnn")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--train-samples-per-class", type=int)
    parser.add_argument("--eval-samples-per-class", type=int)
    return parser.parse_args()


def main() -> None:
    """Build the full pipeline and train the requested classifier."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    data_config = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_weighted_sampler=not args.no_weighted_sampler,
        max_train_samples_per_class=args.train_samples_per_class,
        max_eval_samples_per_class=args.eval_samples_per_class,
    )
    default_learning_rates = {
        "baseline_cnn": 1e-3,
        "efficientnet_b0": 3e-4,
        "resnet50": 3e-4,
        "vit_b_16": 1e-4,
    }
    learning_rate = args.learning_rate or default_learning_rates[args.model]
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=learning_rate,
        early_stopping_patience=args.patience,
        dropout=0.10 if args.model == "vit_b_16" else 0.30,
        checkpoint_path=Path(f"models/checkpoints/{args.model}_best.pth"),
        history_path=Path(f"models/runs/{args.model}_history.json"),
        use_amp=not args.no_amp,
    )
    set_seed(data_config.seed)
    device = get_device()
    logger.info(
        "Training on %s with configuration: %s", device, asdict(training_config)
    )
    pipeline = create_data_pipeline(data_config)
    fit_model(
        pipeline,
        training_config,
        model_name=args.model,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        device=device,
    )


if __name__ == "__main__":
    main()
