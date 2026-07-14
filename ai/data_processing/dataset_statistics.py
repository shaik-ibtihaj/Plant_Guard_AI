"""
========================================
PLANT GUARD AI - DATASET STATISTICS
========================================

Phase 3.1: Dataset Statistics
Phase 3.2: Class Distribution Analysis

This module computes:
    1. Channel-wise RGB mean and standard deviation statistics across
       the entire training dataset (used for normalization).
    2. Per-class image counts across the training dataset (used for
       understanding class balance / imbalance).

Usage:
    python -m ai.data_processing.dataset_statistics
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dataset_statistics")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
VALID_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
TRAIN_DATA_DIR: Path = Path("datasets/processed/train")
OUTPUT_METADATA_DIR: Path = Path("datasets/metadata")
OUTPUT_FILE_PATH: Path = OUTPUT_METADATA_DIR / "rgb_stats.json"
CLASS_DISTRIBUTION_FILE_PATH: Path = OUTPUT_METADATA_DIR / "class_distribution.csv"
PROGRESS_LOG_INTERVAL: int = 1000


def get_image_paths(root_dir: Path) -> List[Path]:
    """
    Recursively scan a directory for valid image files.

    Args:
        root_dir: Root directory to recursively search for images.

    Returns:
        A list of Path objects pointing to valid image files
        (.jpg, .jpeg, .png), sorted for deterministic ordering.

    Raises:
        FileNotFoundError: If the root directory does not exist.
        NotADirectoryError: If the root path is not a directory.
    """
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, got a file: {root_dir}")

    logger.info("Scanning for images in: %s", root_dir.resolve())

    image_paths: List[Path] = [
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    image_paths.sort()

    logger.info("Found %d image(s) with extensions %s", len(image_paths), VALID_EXTENSIONS)
    return image_paths


def calculate_rgb_statistics(image_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute channel-wise RGB mean and standard deviation across a list of images.

    Each image is opened, converted to RGB, and normalized to the [0, 1]
    pixel range. Running per-channel sums and sums of squares are
    accumulated in a single pass so the entire dataset does not need to
    be held in memory at once. Corrupted or unreadable images are
    skipped with a logged warning.

    Args:
        image_paths: List of image file paths to process.

    Returns:
        A tuple of (mean, std), each a numpy array of shape (3,)
        representing the R, G, B channel statistics.

    Raises:
        ValueError: If no valid images could be processed.
    """
    if not image_paths:
        raise ValueError("No image paths provided for statistics calculation.")

    channel_sum: np.ndarray = np.zeros(3, dtype=np.float64)
    channel_sum_sq: np.ndarray = np.zeros(3, dtype=np.float64)
    total_pixel_count: int = 0

    processed_count: int = 0
    skipped_count: int = 0
    total_images: int = len(image_paths)

    logger.info("Starting RGB statistics calculation over %d image(s)...", total_images)

    for index, image_path in enumerate(image_paths, start=1):
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                pixel_array = np.asarray(img, dtype=np.float64) / 255.0

            if pixel_array.ndim != 3 or pixel_array.shape[2] != 3:
                logger.warning("Skipping image with unexpected shape: %s", image_path)
                skipped_count += 1
                continue

            channel_sum += pixel_array.sum(axis=(0, 1))
            channel_sum_sq += np.square(pixel_array).sum(axis=(0, 1))
            total_pixel_count += pixel_array.shape[0] * pixel_array.shape[1]
            processed_count += 1

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            logger.warning("Skipping corrupted or unreadable image: %s (%s)", image_path, exc)
            skipped_count += 1
            continue

        if index % PROGRESS_LOG_INTERVAL == 0 or index == total_images:
            logger.info("Progress: %d/%d images processed", index, total_images)

    if processed_count == 0 or total_pixel_count == 0:
        raise ValueError("No valid images were processed; cannot compute statistics.")

    mean: np.ndarray = channel_sum / total_pixel_count
    variance: np.ndarray = (channel_sum_sq / total_pixel_count) - np.square(mean)
    variance = np.clip(variance, a_min=0.0, a_max=None)
    std: np.ndarray = np.sqrt(variance)

    logger.info(
        "Finished processing. Successful: %d, Skipped: %d, Total pixels: %d",
        processed_count,
        skipped_count,
        total_pixel_count,
    )
    logger.info("Computed mean: %s", mean.tolist())
    logger.info("Computed std: %s", std.tolist())

    return mean, std


def save_rgb_statistics(mean: np.ndarray, std: np.ndarray, output_path: Path) -> None:
    """
    Save computed RGB mean and standard deviation statistics to a JSON file.

    Args:
        mean: Numpy array of shape (3,) with per-channel mean values.
        std: Numpy array of shape (3,) with per-channel std values.
        output_path: Destination file path for the JSON output.

    Raises:
        OSError: If the output file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "mean": [round(float(value), 6) for value in mean],
        "std": [round(float(value), 6) for value in std],
    }

    try:
        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(stats, json_file, indent=4)
    except OSError as exc:
        logger.error("Failed to write statistics to %s: %s", output_path, exc)
        raise

    logger.info("Saved RGB statistics to: %s", output_path.resolve())


def compute_class_distribution(root_dir: Path) -> Tuple[Dict[str, int], int, int]:
    """
    Compute per-class image counts from the training dataset directory.

    Each immediate subdirectory of `root_dir` is treated as a distinct
    class. Image files (.jpg, .jpeg, .png) are counted recursively within
    each class subdirectory. Results are saved to a CSV file sorted by
    image count in descending order.

    Args:
        root_dir: Root training dataset directory containing one
            subdirectory per class.

    Returns:
        A tuple of (class_counts, total_classes, total_images), where:
            class_counts: Mapping of class name to image count, sorted
                by count in descending order.
            total_classes: Total number of class subdirectories found.
            total_images: Total number of images across all classes.

    Raises:
        FileNotFoundError: If the root directory does not exist.
        NotADirectoryError: If the root path is not a directory.
        ValueError: If no class subdirectories are found or processed.
    """
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, got a file: {root_dir}")

    logger.info("Scanning for class subdirectories in: %s", root_dir.resolve())

    class_dirs = sorted([path for path in root_dir.iterdir() if path.is_dir()])

    if not class_dirs:
        raise ValueError(f"No class subdirectories found in: {root_dir}")

    class_counts: Dict[str, int] = {}

    for class_dir in class_dirs:
        try:
            image_count = sum(
                1
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
            )
        except OSError as exc:
            logger.warning("Skipping unreadable class directory: %s (%s)", class_dir, exc)
            continue

        class_counts[class_dir.name] = image_count
        logger.info("Class '%s': %d image(s)", class_dir.name, image_count)

    if not class_counts:
        raise ValueError("No valid class directories could be processed.")

    total_classes: int = len(class_counts)
    total_images: int = sum(class_counts.values())

    sorted_class_counts: Dict[str, int] = dict(
        sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
    )

    try:
        distribution_df = pd.DataFrame(
            list(sorted_class_counts.items()), columns=["class", "count"]
        )
        CLASS_DISTRIBUTION_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        distribution_df.to_csv(CLASS_DISTRIBUTION_FILE_PATH, index=False)
    except OSError as exc:
        logger.error(
            "Failed to write class distribution to %s: %s",
            CLASS_DISTRIBUTION_FILE_PATH,
            exc,
        )
        raise

    logger.info("Saved class distribution to: %s", CLASS_DISTRIBUTION_FILE_PATH.resolve())

    return sorted_class_counts, total_classes, total_images


def main() -> None:
    """
    Entry point for computing and saving dataset RGB and class distribution
    statistics.

    Orchestrates the full pipeline:
        1. Discover image paths in the training dataset directory.
        2. Compute channel-wise mean and standard deviation.
        3. Persist the RGB statistics to a JSON metadata file.
        4. Compute per-class image distribution.
        5. Persist the class distribution to a CSV metadata file.
        6. Print a summary of the class distribution.
    """
    logger.info("=" * 40)
    logger.info("PLANT GUARD AI - DATASET STATISTICS")
    logger.info("=" * 40)

    try:
        image_paths = get_image_paths(TRAIN_DATA_DIR)
        mean, std = calculate_rgb_statistics(image_paths)
        save_rgb_statistics(mean, std, OUTPUT_FILE_PATH)

        class_counts, total_classes, total_images = compute_class_distribution(TRAIN_DATA_DIR)
        average_images_per_class: float = (
            total_images / total_classes if total_classes > 0 else 0.0
        )

        print("=" * 40)
        print("CLASS DISTRIBUTION SUMMARY")
        print("=" * 40)
        print(f"Total Classes: {total_classes}")
        print(f"Total Images: {total_images}")
        print(f"Average Images/Class: {average_images_per_class:.2f}")
        print("Saved:")
        print(f"{CLASS_DISTRIBUTION_FILE_PATH}")

    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        logger.error("Dataset statistics calculation failed: %s", exc)
        raise
    except Exception as exc:
        logger.exception("Unexpected error during dataset statistics calculation: %s", exc)
        raise

    logger.info("Dataset statistics calculation completed successfully.")


if __name__ == "__main__":
    main()