#!/usr/bin/env python3
"""
03_dataset_insights.py

PlantGuard AI — Phase 4 EDA: Actionable Dataset Insights

Reads four pre-existing metadata files, computes a comprehensive set of
dataset statistics, and derives actionable recommendations for:

  • Training pipeline configuration
  • Data augmentation strategy
  • Class weight computation
  • Model input size selection
  • Model architecture selection

All findings are saved to:

    datasets/metadata/eda_insights.json

Usage:
    python scripts/eda/03_dataset_insights.py

Inputs:
    datasets/metadata/class_distribution.csv
    datasets/metadata/class_imbalance_report.json
    datasets/metadata/image_resolution_stats.json
    datasets/metadata/dataset_summary.csv

Output:
    datasets/metadata/eda_insights.json

Author: PlantGuard AI ML Engineering Team
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
METADATA_DIR: Path = PROJECT_ROOT / "datasets" / "metadata"

CLASS_DISTRIBUTION_CSV: Path  = METADATA_DIR / "class_distribution.csv"
IMBALANCE_REPORT_JSON: Path   = METADATA_DIR / "class_imbalance_report.json"
RESOLUTION_STATS_JSON: Path   = METADATA_DIR / "image_resolution_stats.json"
DATASET_SUMMARY_CSV: Path     = METADATA_DIR / "dataset_summary.csv"

OUTPUT_JSON: Path = METADATA_DIR / "eda_insights.json"

# Keyword for healthy/diseased classification (case-insensitive).
HEALTHY_KEYWORD: str = "healthy"

# Imbalance thresholds (mirrors detect_class_imbalance.py).
MINORITY_THRESHOLD: int = 500
MAJORITY_THRESHOLD: int = 5_000

# Severity bands for the imbalance ratio (largest / smallest class).
_SEVERITY_CRITICAL:  float = 100.0   # ratio ≥ 100  → Critical
_SEVERITY_HIGH:      float = 20.0    # ratio ≥ 20   → High
_SEVERITY_MODERATE:  float = 5.0     # ratio ≥ 5    → Moderate
# ratio < 5 → Low

# Standard input sizes for common architectures.
_INPUT_SIZES: Tuple[int, ...] = (224, 256, 384, 512)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def _configure_logger() -> logging.Logger:
    """Configure and return the module logger with file + console handlers.

    Returns:
        The configured ``plantguard.dataset_insights`` logger.
    """
    log = logging.getLogger("plantguard.dataset_insights")
    log.setLevel(logging.DEBUG)
    log.propagate = False

    if log.handlers:
        return log

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file: Path = PROJECT_ROOT / "scripts" / "eda" / "03_dataset_insights.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(ch)
    return log


logger: logging.Logger = _configure_logger()


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _require_file(path: Path) -> None:
    """Raise FileNotFoundError with a clear message if ``path`` is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required metadata file not found: {path}\n"
            "Run the appropriate Phase 3 analysis scripts first."
        )


def load_class_distribution(csv_path: Path) -> pd.DataFrame:
    """Load and validate the class-distribution CSV.

    Args:
        csv_path: Path to ``class_distribution.csv``.

    Returns:
        Validated ``DataFrame`` with columns ``['class', 'count']``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: On schema or data errors.
    """
    _require_file(csv_path)
    logger.info("Loading class distribution: %s", csv_path)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Cannot parse '{csv_path}': {exc}") from exc

    for col in ("class", "count"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in '{csv_path}'.")

    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["class", "count"])
    df = df[df["class"].str.strip() != ""]
    df["count"] = df["count"].astype(int)

    if df.empty:
        raise ValueError(f"No valid rows in '{csv_path}'.")

    logger.info("Loaded %d class entries.", len(df))
    return df.reset_index(drop=True)


def load_json(path: Path, label: str) -> Dict[str, Any]:
    """Load a JSON metadata file.

    Args:
        path:  Absolute path to the JSON file.
        label: Human-readable name used in error messages.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON.
    """
    _require_file(path)
    logger.info("Loading %s: %s", label, path)
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{path}': {exc}") from exc


def load_dataset_summary(csv_path: Path) -> pd.DataFrame:
    """Load and validate the dataset-summary CSV.

    Args:
        csv_path: Path to ``dataset_summary.csv``.

    Returns:
        Validated ``DataFrame`` with columns ``['dataset', 'class', 'images']``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: On schema or data errors.
    """
    _require_file(csv_path)
    logger.info("Loading dataset summary: %s", csv_path)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Cannot parse '{csv_path}': {exc}") from exc

    for col in ("dataset", "images"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in '{csv_path}'.")

    df["images"] = pd.to_numeric(df["images"], errors="coerce")
    df = df.dropna(subset=["images"])
    df["images"] = df["images"].astype(int)

    if df.empty:
        raise ValueError(f"No valid rows in '{csv_path}'.")

    logger.info("Loaded %d source dataset entries.", len(df))
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Core statistics
# --------------------------------------------------------------------------- #


def compute_class_stats(
    df: pd.DataFrame,
    imbalance: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive class-level statistics from distribution data.

    Args:
        df:         Class-distribution DataFrame.
        imbalance:  Pre-parsed imbalance report dict.

    Returns:
        Dictionary of class statistics.
    """
    mask_healthy = df["class"].str.contains(HEALTHY_KEYWORD, case=False, na=False)
    healthy_df   = df[mask_healthy]
    diseased_df  = df[~mask_healthy]

    total_classes    = int(len(df))
    total_images     = int(df["count"].sum())
    healthy_classes  = int(len(healthy_df))
    diseased_classes = int(len(diseased_df))
    healthy_images   = int(healthy_df["count"].sum())
    diseased_images  = int(diseased_df["count"].sum())

    healthy_class_pct  = round(healthy_classes  / total_classes  * 100, 2) if total_classes  else 0.0
    diseased_class_pct = round(diseased_classes / total_classes  * 100, 2) if total_classes  else 0.0
    healthy_image_pct  = round(healthy_images   / total_images   * 100, 2) if total_images   else 0.0
    diseased_image_pct = round(diseased_images  / total_images   * 100, 2) if total_images   else 0.0

    imbalance_ratio  = float(imbalance.get("imbalance_ratio", 0))
    num_minority     = int(imbalance.get("num_minority_classes", 0))
    num_majority     = int(imbalance.get("num_majority_classes", 0))
    mean_class_size  = float(imbalance.get("mean_class_size", 0))
    median_class_size = float(imbalance.get("median_class_size", 0))
    std_class_size   = float(imbalance.get("std_class_size", 0))

    largest  = imbalance.get("largest_class",  {})
    smallest = imbalance.get("smallest_class", {})

    # Coefficient of Variation — useful for expressing spread relative to mean.
    cv = round(std_class_size / mean_class_size * 100, 2) if mean_class_size else 0.0

    # Severity classification.
    if imbalance_ratio >= _SEVERITY_CRITICAL:
        severity = "Critical"
        severity_note = (
            f"Imbalance ratio {imbalance_ratio:.1f}x is critical. "
            "Weighted loss, oversampling, and aggressive augmentation are required."
        )
    elif imbalance_ratio >= _SEVERITY_HIGH:
        severity = "High"
        severity_note = (
            f"Imbalance ratio {imbalance_ratio:.1f}x is high. "
            "Weighted cross-entropy or focal loss is strongly recommended."
        )
    elif imbalance_ratio >= _SEVERITY_MODERATE:
        severity = "Moderate"
        severity_note = (
            f"Imbalance ratio {imbalance_ratio:.1f}x is moderate. "
            "Mild class weighting or oversampling is advised."
        )
    else:
        severity = "Low"
        severity_note = (
            f"Imbalance ratio {imbalance_ratio:.1f}x is low. "
            "Standard training may proceed without special balancing."
        )

    return {
        "total_classes": total_classes,
        "total_images_train_split": total_images,
        "healthy_classes": healthy_classes,
        "diseased_classes": diseased_classes,
        "healthy_class_percentage": healthy_class_pct,
        "diseased_class_percentage": diseased_class_pct,
        "healthy_image_count": healthy_images,
        "diseased_image_count": diseased_images,
        "healthy_image_percentage": healthy_image_pct,
        "diseased_image_percentage": diseased_image_pct,
        "largest_class": largest,
        "smallest_class": smallest,
        "mean_class_size": round(mean_class_size, 2),
        "median_class_size": round(median_class_size, 2),
        "std_class_size": round(std_class_size, 2),
        "coefficient_of_variation_pct": cv,
        "imbalance_ratio": imbalance_ratio,
        "imbalance_severity": severity,
        "imbalance_severity_note": severity_note,
        "minority_classes_count": num_minority,
        "majority_classes_count": num_majority,
        "minority_threshold_used": int(imbalance.get("minority_threshold", MINORITY_THRESHOLD)),
        "majority_threshold_used": int(imbalance.get("majority_threshold", MAJORITY_THRESHOLD)),
    }


def compute_resolution_stats(res: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and annotate image-dimension statistics.

    Args:
        res: Parsed ``image_resolution_stats.json`` dict.

    Returns:
        Dictionary of resolution statistics with annotations.
    """
    total   = int(res.get("total_images", 0))
    corrupt = int(res.get("corrupted_images", 0))

    width  = res.get("width",        {})
    height = res.get("height",       {})
    aspect = res.get("aspect_ratio", {})
    resol  = res.get("resolution",   {})

    median_w = float(width.get("median", 0))
    median_h = float(height.get("median", 0))
    mean_ar  = float(aspect.get("mean", 1.0))

    # Describe the dominant image size.
    if median_w == median_h:
        shape_note = (
            f"Dataset is predominantly square ({int(median_w)}×{int(median_h)} px). "
            "Standard square-crop resizing is appropriate."
        )
    else:
        shape_note = (
            f"Dominant image shape is {int(median_w)}×{int(median_h)} px "
            f"(aspect ratio ≈ {mean_ar:.2f}). "
            "Padding or aspect-preserving resize may be beneficial."
        )

    return {
        "total_images_full_dataset": total,
        "corrupted_images": corrupt,
        "data_quality_note": (
            "Dataset is clean — 0 corrupted images detected."
            if corrupt == 0
            else f"{corrupt} corrupted images detected and excluded."
        ),
        "width":        {k: round(v, 2) for k, v in width.items()},
        "height":       {k: round(v, 2) for k, v in height.items()},
        "aspect_ratio": {k: round(v, 4) for k, v in aspect.items()},
        "resolution":   {k: round(v, 2) for k, v in resol.items()},
        "dominant_shape_note": shape_note,
        "median_width_px":  int(median_w),
        "median_height_px": int(median_h),
    }


def compute_source_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Summarise per-source dataset contributions.

    Args:
        df: Parsed ``dataset_summary.csv`` DataFrame.

    Returns:
        Dictionary of source statistics.
    """
    total_source_images = int(df["images"].sum())
    sources: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        count = int(row["images"])
        pct   = round(count / total_source_images * 100, 2) if total_source_images else 0.0
        sources.append({
            "dataset": str(row["dataset"]),
            "image_count": count,
            "percentage": pct,
        })

    # Sort by image count descending.
    sources.sort(key=lambda x: x["image_count"], reverse=True)

    dominant = sources[0]["dataset"] if sources else "unknown"

    return {
        "num_source_datasets": len(sources),
        "total_images_across_sources": total_source_images,
        "dominant_source": dominant,
        "sources": sources,
        "multi_source_note": (
            f"Dataset aggregated from {len(sources)} sources. "
            f"'{dominant}' contributes the most images. "
            "Domain shift across sources may affect model generalisation."
        ),
    }


# --------------------------------------------------------------------------- #
# Recommendation engines
# --------------------------------------------------------------------------- #


def build_training_recommendations(
    class_stats: Dict[str, Any],
    res_stats: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Generate actionable training-pipeline recommendations.

    Each recommendation has a ``priority`` (Critical / High / Medium / Low),
    a ``category``, and a ``recommendation`` string.

    Args:
        class_stats: Output of :func:`compute_class_stats`.
        res_stats:   Output of :func:`compute_resolution_stats`.

    Returns:
        Sorted list of recommendation dicts (Critical first).
    """
    recs: List[Dict[str, str]] = []

    severity = class_stats["imbalance_severity"]
    ratio    = class_stats["imbalance_ratio"]
    n_min    = class_stats["minority_classes_count"]

    # --- Loss function ------------------------------------------------------ #
    if severity in ("Critical", "High"):
        recs.append({
            "priority": "Critical",
            "category": "Loss Function",
            "recommendation": (
                f"Use Focal Loss (gamma=2, alpha tuned per class) instead of "
                f"standard cross-entropy. Imbalance ratio is {ratio:.1f}x — "
                "standard CE will over-fit to majority classes."
            ),
        })
    elif severity == "Moderate":
        recs.append({
            "priority": "High",
            "category": "Loss Function",
            "recommendation": (
                "Use weighted cross-entropy with inverse-frequency class weights. "
                f"Imbalance ratio is {ratio:.1f}x."
            ),
        })
    else:
        recs.append({
            "priority": "Low",
            "category": "Loss Function",
            "recommendation": "Standard cross-entropy is appropriate for this dataset.",
        })

    # --- Sampling strategy -------------------------------------------------- #
    if n_min > 0:
        recs.append({
            "priority": "High",
            "category": "Sampling",
            "recommendation": (
                f"{n_min} minority classes (≤{class_stats['minority_threshold_used']} images) "
                "detected. Use WeightedRandomSampler in PyTorch DataLoader to oversample "
                "minority classes during training."
            ),
        })

    # --- Epoch count -------------------------------------------------------- #
    total = class_stats["total_images_train_split"]
    if total > 100_000:
        recs.append({
            "priority": "Medium",
            "category": "Training Duration",
            "recommendation": (
                f"Dataset has {total:,} training images. Start with 30–50 epochs "
                "and use early stopping (patience=5) to avoid over-training."
            ),
        })
    else:
        recs.append({
            "priority": "Medium",
            "category": "Training Duration",
            "recommendation": (
                f"Dataset has {total:,} training images. "
                "50–100 epochs with early stopping (patience=10) is recommended."
            ),
        })

    # --- Learning rate scheduler -------------------------------------------- #
    recs.append({
        "priority": "Medium",
        "category": "Learning Rate",
        "recommendation": (
            "Use CosineAnnealingLR or OneCycleLR scheduler. "
            "Initial LR: 1e-3 (scratch) or 1e-4 (fine-tuning pretrained). "
            "Warm-up for 3–5 epochs."
        ),
    })

    # --- Batch size --------------------------------------------------------- #
    recs.append({
        "priority": "Medium",
        "category": "Batch Size",
        "recommendation": (
            "Batch size 32 is recommended for 224×224 inputs on a single GPU. "
            "Increase to 64 if GPU VRAM ≥ 16 GB. "
            "Use gradient accumulation if VRAM is limited."
        ),
    })

    # --- Mixed precision ---------------------------------------------------- #
    recs.append({
        "priority": "Low",
        "category": "Compute Efficiency",
        "recommendation": (
            "Enable AMP (torch.cuda.amp.autocast) for ~2× training speed-up "
            "on NVIDIA GPUs with Tensor Cores (Volta+)."
        ),
    })

    # --- Validation frequency ----------------------------------------------- #
    recs.append({
        "priority": "Low",
        "category": "Validation",
        "recommendation": (
            "Validate every epoch. Track macro-averaged F1 and per-class recall "
            "alongside accuracy — accuracy is misleading under class imbalance."
        ),
    })

    # Sort: Critical → High → Medium → Low
    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    recs.sort(key=lambda r: order.get(r["priority"], 99))
    return recs


def build_augmentation_recommendations(
    class_stats: Dict[str, Any],
    res_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate augmentation strategy recommendations.

    Args:
        class_stats: Output of :func:`compute_class_stats`.
        res_stats:   Output of :func:`compute_resolution_stats`.

    Returns:
        Dictionary with train/val transform specs and minority-class targets.
    """
    smallest_count = class_stats["smallest_class"].get("count", 0)
    mean_size      = class_stats["mean_class_size"]
    severity       = class_stats["imbalance_severity"]
    n_minority     = class_stats["minority_classes_count"]

    # Target image count for minority-class augmentation.
    aug_target = int(min(mean_size, 2_000))

    # Compute multiplier for the most extreme minority class.
    max_multiplier = (
        round(aug_target / smallest_count, 1) if smallest_count > 0 else 0
    )

    train_transforms = [
        "RandomHorizontalFlip(p=0.5)",
        "RandomVerticalFlip(p=0.5)",
        "RandomRotation(degrees=30)",
        "ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)",
        "RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))",
        "RandomResizedCrop(size=224, scale=(0.8, 1.0))",
        "Normalize(mean=[0.2297, 0.2633, 0.2349], std=[0.1375, 0.1606, 0.1249])",
    ]

    val_transforms = [
        "Resize(256)",
        "CenterCrop(224)",
        "Normalize(mean=[0.2297, 0.2633, 0.2349], std=[0.1375, 0.1606, 0.1249])",
    ]

    minority_strategy: List[str] = []
    if severity in ("Critical", "High"):
        minority_strategy = [
            f"Augment all {n_minority} minority classes (≤500 images) to ~{aug_target} images.",
            f"Maximum augmentation multiplier needed: ×{max_multiplier} "
            f"(for '{class_stats['smallest_class'].get('name', 'unknown')}' with "
            f"{smallest_count} images).",
            "Apply ElasticTransform and GridDistortion for additional visual diversity.",
            "Consider MixUp or CutMix at the batch level (alpha=0.4) for regularisation.",
        ]
    elif severity == "Moderate":
        minority_strategy = [
            f"Augment minority classes to ~{aug_target} images.",
            "Standard geometric + colour augmentation is sufficient.",
        ]
    else:
        minority_strategy = [
            "Standard augmentation is sufficient. No minority-targeted augmentation needed."
        ]

    return {
        "augmentation_target_per_minority_class": aug_target,
        "max_augmentation_multiplier": max_multiplier,
        "train_transforms": train_transforms,
        "val_transforms": val_transforms,
        "test_transforms": val_transforms,
        "minority_class_strategy": minority_strategy,
        "normalization_values": {
            "mean": [0.2297, 0.2633, 0.2349],
            "std":  [0.1375, 0.1606, 0.1249],
            "source": "datasets/metadata/rgb_stats.json (computed from train split)",
        },
        "advanced_techniques": [
            "RandAugment(n=2, m=9) — automatic policy search, reduces manual tuning.",
            "TrivialAugmentWide — state-of-the-art single-policy augmentation.",
            "MixUp (alpha=0.4) — improves calibration and generalisation.",
        ],
    }


def build_class_weight_recommendations(
    class_stats: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Generate class weight computation recommendations.

    Provides the formula, strategy, and example weights for the five most
    extreme classes.

    Args:
        class_stats: Output of :func:`compute_class_stats`.
        df:          Class-distribution DataFrame.

    Returns:
        Dictionary with formula, strategy, and example weights.
    """
    total   = int(df["count"].sum())
    n_cls   = int(len(df))
    severity = class_stats["imbalance_severity"]

    # Compute inverse-frequency weights (sklearn balanced mode).
    df = df.copy()
    df["weight"] = total / (n_cls * df["count"].astype(float))
    df["weight"] = df["weight"].round(4)

    # Top-5 highest weight (most under-represented).
    top_minority = (
        df.sort_values("weight", ascending=False)
        .head(5)[["class", "count", "weight"]]
        .rename(columns={"class": "class_name", "count": "image_count"})
        .to_dict(orient="records")
    )

    # Top-5 lowest weight (most over-represented).
    top_majority = (
        df.sort_values("weight", ascending=True)
        .head(5)[["class", "count", "weight"]]
        .rename(columns={"class": "class_name", "count": "image_count"})
        .to_dict(orient="records")
    )

    # Choose loss strategy based on severity.
    if severity in ("Critical", "High"):
        loss_strategy = "Focal Loss (gamma=2) with per-class alpha weights derived from inverse frequency."
    elif severity == "Moderate":
        loss_strategy = "Weighted CrossEntropyLoss with inverse-frequency class weights."
    else:
        loss_strategy = "Standard CrossEntropyLoss. Class weights optional."

    return {
        "formula": "weight_c = N_total / (N_classes × N_c)   [sklearn 'balanced' mode]",
        "formula_variables": {
            "N_total":   "Total training images",
            "N_classes": "Total number of classes",
            "N_c":       "Images in class c",
        },
        "implementation": (
            "sklearn.utils.class_weight.compute_class_weight("
            "class_weight='balanced', classes=np.arange(N_classes), y=labels)"
        ),
        "pytorch_usage": (
            "pos_weights = torch.tensor(weights).to(device); "
            "criterion = nn.CrossEntropyLoss(weight=pos_weights)"
        ),
        "recommended_loss_strategy": loss_strategy,
        "top_5_highest_weight_minority_classes": top_minority,
        "top_5_lowest_weight_majority_classes":  top_majority,
        "expected_weight_range": {
            "min_weight": float(df["weight"].min()),
            "max_weight": float(df["weight"].max()),
            "ratio":      round(float(df["weight"].max() / df["weight"].min()), 2),
        },
        "note": (
            "Persist final weights to datasets/metadata/class_weights.json "
            "after running scripts/analysis/compute_class_weights.py."
        ),
    }


def build_input_size_recommendations(
    res_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Recommend model input sizes based on image dimension statistics.

    Args:
        res_stats: Output of :func:`compute_resolution_stats`.

    Returns:
        Dictionary with recommended input sizes and rationale.
    """
    median_w = res_stats["median_width_px"]
    median_h = res_stats["median_height_px"]
    mean_ar  = res_stats["aspect_ratio"]["mean"]
    min_w    = int(res_stats["width"]["min"])

    # Standard size that fits within the native resolution with no upscaling.
    # Native median is 256, so 224 is the natural choice (no upscaling needed).
    primary_size   = 224
    secondary_size = 256
    high_res_size  = 384

    resize_note = (
        f"Native median resolution is {median_w}×{median_h} px. "
        f"Resizing to {primary_size}×{primary_size} requires a slight downscale "
        f"(≈{round((median_w - primary_size) / median_w * 100, 1)}% reduction). "
        "No quality-degrading upscaling is required."
    )

    return {
        "native_median_resolution": f"{median_w}×{median_h}",
        "native_min_resolution":    f"{min_w}×{min_w}",
        "mean_aspect_ratio":        round(mean_ar, 4),
        "primary_recommended_size": primary_size,
        "secondary_size":           secondary_size,
        "high_resolution_size":     high_res_size,
        "resize_note":              resize_note,
        "resize_pipeline": [
            f"Train:  RandomResizedCrop({primary_size}) — preserves scale variation.",
            f"Val/Test: Resize({secondary_size}) → CenterCrop({primary_size}).",
        ],
        "rationale": {
            f"{primary_size}×{primary_size}": (
                "Matches EfficientNet-B0, ResNet50, ViT-B/16 expected input. "
                "Optimal for batch size 32 on 8 GB VRAM."
            ),
            f"{secondary_size}×{secondary_size}": (
                "Minimal downscale — preserves more detail. "
                "~25% more VRAM required vs 224."
            ),
            f"{high_res_size}×{high_res_size}": (
                "EfficientNet-B4+ territory. "
                "~3× VRAM vs 224; use only with ≥16 GB VRAM."
            ),
        },
    }


def build_model_recommendations(
    class_stats: Dict[str, Any],
    res_stats: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Recommend model architectures ordered by suitability.

    Args:
        class_stats: Output of :func:`compute_class_stats`.
        res_stats:   Output of :func:`compute_resolution_stats`.

    Returns:
        Ordered list of model recommendation dicts.
    """
    total_images = class_stats["total_images_train_split"]
    n_classes    = class_stats["total_classes"]
    severity     = class_stats["imbalance_severity"]

    models = [
        {
            "rank": 1,
            "architecture": "EfficientNet-B0",
            "type": "Transfer Learning (ImageNet)",
            "input_size": "224×224",
            "parameters_M": 5.3,
            "recommended": True,
            "reason": (
                f"Best accuracy/efficiency trade-off for {n_classes}-class classification. "
                f"Pre-trained on ImageNet, fine-tune all layers for {total_images:,} images. "
                "Compound scaling generalises well to leaf disease patterns."
            ),
            "training_config": {
                "pretrained": True,
                "fine_tune_all_layers": True,
                "freeze_backbone_first_n_epochs": 3,
                "learning_rate": "1e-4 (backbone) / 1e-3 (head)",
                "dropout": 0.3,
            },
        },
        {
            "rank": 2,
            "architecture": "ResNet-50",
            "type": "Transfer Learning (ImageNet)",
            "input_size": "224×224",
            "parameters_M": 25.6,
            "recommended": True,
            "reason": (
                "Strong benchmark architecture. Residual connections mitigate vanishing "
                "gradients on deep features. Widely cited in plant disease literature."
            ),
            "training_config": {
                "pretrained": True,
                "fine_tune_all_layers": True,
                "freeze_backbone_first_n_epochs": 5,
                "learning_rate": "1e-4",
                "dropout": 0.4,
            },
        },
        {
            "rank": 3,
            "architecture": "ViT-B/16",
            "type": "Vision Transformer (ImageNet-21k)",
            "input_size": "224×224",
            "parameters_M": 86.6,
            "recommended": True,
            "reason": (
                f"Transformer attention captures global texture patterns critical for "
                f"fine-grained disease discrimination. Best suited after baseline CNN models "
                f"are established. Requires more VRAM."
            ),
            "training_config": {
                "pretrained": True,
                "fine_tune_all_layers": True,
                "freeze_backbone_first_n_epochs": 5,
                "learning_rate": "1e-4",
                "dropout": 0.1,
                "label_smoothing": 0.1,
            },
        },
        {
            "rank": 4,
            "architecture": "Custom CNN (Baseline)",
            "type": "Trained from scratch",
            "input_size": "224×224",
            "parameters_M": "~2–5",
            "recommended": False,
            "reason": (
                f"Useful as a performance baseline. "
                f"With {total_images:,} training images, transfer learning will "
                "significantly outperform a scratch-trained CNN. "
                "Implement first to establish a lower-bound benchmark."
            ),
            "training_config": {
                "pretrained": False,
                "learning_rate": "1e-3",
                "dropout": 0.5,
                "weight_decay": "1e-4",
            },
        },
    ]

    # Add imbalance-specific note to all models if severity is critical/high.
    if severity in ("Critical", "High"):
        note = (
            f"All models must use Focal Loss or weighted CE due to {severity.lower()} "
            f"class imbalance (ratio {class_stats['imbalance_ratio']:.1f}x)."
        )
        for m in models:
            m["imbalance_note"] = note

    return models


# --------------------------------------------------------------------------- #
# JSON assembly
# --------------------------------------------------------------------------- #


def assemble_insights(
    class_stats:     Dict[str, Any],
    res_stats:       Dict[str, Any],
    source_stats:    Dict[str, Any],
    training_recs:   List[Dict[str, str]],
    aug_recs:        Dict[str, Any],
    weight_recs:     Dict[str, Any],
    input_recs:      Dict[str, Any],
    model_recs:      List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble all computed sections into the final insights dictionary.

    Args:
        class_stats:   Class-level statistics.
        res_stats:     Resolution statistics.
        source_stats:  Source dataset statistics.
        training_recs: Training recommendations.
        aug_recs:      Augmentation recommendations.
        weight_recs:   Class weight recommendations.
        input_recs:    Input size recommendations.
        model_recs:    Model recommendations.

    Returns:
        Complete ``eda_insights`` dictionary.
    """
    import datetime
    return {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "schema_version": "1.0",
        "project": "PlantGuard AI",
        "phase": "Phase 4 — Training Pipeline Preparation",
        "dataset_statistics": {
            "class_statistics":       class_stats,
            "resolution_statistics":  res_stats,
            "source_statistics":      source_stats,
        },
        "training_recommendations":    training_recs,
        "augmentation_recommendations": aug_recs,
        "class_weight_recommendations": weight_recs,
        "input_size_recommendations":   input_recs,
        "model_recommendations":        model_recs,
    }


# --------------------------------------------------------------------------- #
# JSON persistence
# --------------------------------------------------------------------------- #


def save_json(payload: Dict[str, Any], output_path: Path) -> None:
    """Persist the insights dictionary to a JSON file.

    Args:
        payload:     Dictionary to serialise.
        output_path: Destination file path.

    Raises:
        OSError: If the file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        logger.info("Insights saved to: %s", output_path)
    except OSError as exc:
        logger.error("Failed to write JSON to '%s': %s", output_path, exc)
        raise


# --------------------------------------------------------------------------- #
# Console summary
# --------------------------------------------------------------------------- #


def print_summary(insights: Dict[str, Any]) -> None:
    """Print a human-readable console summary of the generated insights.

    Args:
        insights: Assembled insights dictionary.
    """
    cs  = insights["dataset_statistics"]["class_statistics"]
    rs  = insights["dataset_statistics"]["resolution_statistics"]
    ss  = insights["dataset_statistics"]["source_statistics"]
    recs = insights["training_recommendations"]

    sep = "=" * 60
    print()
    print(sep)
    print("  PLANTGUARD AI — EDA INSIGHTS SUMMARY")
    print(sep)
    print(f"  Total classes           : {cs['total_classes']}")
    print(f"  Total images (train)    : {cs['total_images_train_split']:,}")
    print(f"  Total images (full)     : {rs['total_images_full_dataset']:,}")
    print(f"  Source datasets         : {ss['num_source_datasets']}")
    print()
    print(f"  Healthy classes         : {cs['healthy_classes']}  ({cs['healthy_class_percentage']:.1f}%)")
    print(f"  Diseased classes        : {cs['diseased_classes']} ({cs['diseased_class_percentage']:.1f}%)")
    print(f"  Healthy images          : {cs['healthy_image_count']:,} ({cs['healthy_image_percentage']:.1f}%)")
    print(f"  Diseased images         : {cs['diseased_image_count']:,} ({cs['diseased_image_percentage']:.1f}%)")
    print()
    print(f"  Largest class           : {cs['largest_class'].get('name')}  ({cs['largest_class'].get('count'):,})")
    print(f"  Smallest class          : {cs['smallest_class'].get('name')} ({cs['smallest_class'].get('count'):,})")
    print(f"  Mean class size         : {cs['mean_class_size']:,.2f}")
    print(f"  Imbalance ratio         : {cs['imbalance_ratio']:.2f}x")
    print(f"  Imbalance severity      : {cs['imbalance_severity']}")
    print(f"  Minority classes        : {cs['minority_classes_count']}")
    print(f"  Majority classes        : {cs['majority_classes_count']}")
    print()
    print(f"  Median image size       : {rs['median_width_px']}×{rs['median_height_px']} px")
    print(f"  Corrupted images        : {rs['corrupted_images']}")
    print()
    print("  Training Recommendations:")
    for r in recs:
        print(f"    [{r['priority']:8s}] {r['category']}: {r['recommendation'][:72]}…")
    print()
    print("  Output:")
    print(f"  ✓ {OUTPUT_JSON.relative_to(PROJECT_ROOT)}")
    print(sep)
    print()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Run the full dataset insights pipeline."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    logger.info("Starting dataset insights generation.")

    try:
        # --- Load inputs ---------------------------------------------------- #
        df_dist    = load_class_distribution(CLASS_DISTRIBUTION_CSV)
        imbalance  = load_json(IMBALANCE_REPORT_JSON,  "imbalance report")
        resolution = load_json(RESOLUTION_STATS_JSON,  "resolution stats")
        df_summary = load_dataset_summary(DATASET_SUMMARY_CSV)

        # --- Compute statistics --------------------------------------------- #
        class_stats  = compute_class_stats(df_dist, imbalance)
        res_stats    = compute_resolution_stats(resolution)
        source_stats = compute_source_stats(df_summary)

        # --- Build recommendations ------------------------------------------ #
        training_recs = build_training_recommendations(class_stats, res_stats)
        aug_recs      = build_augmentation_recommendations(class_stats, res_stats)
        weight_recs   = build_class_weight_recommendations(class_stats, df_dist)
        input_recs    = build_input_size_recommendations(res_stats)
        model_recs    = build_model_recommendations(class_stats, res_stats)

        # --- Assemble and save ---------------------------------------------- #
        insights = assemble_insights(
            class_stats, res_stats, source_stats,
            training_recs, aug_recs, weight_recs, input_recs, model_recs,
        )

        save_json(insights, OUTPUT_JSON)
        print_summary(insights)
        logger.info("Dataset insights complete.")

    except FileNotFoundError as exc:
        logger.error("Input file error: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Data validation error: %s", exc)
        sys.exit(1)
    except OSError as exc:
        logger.error("I/O error: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 — top-level safety net
        logger.exception("Unexpected failure: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

