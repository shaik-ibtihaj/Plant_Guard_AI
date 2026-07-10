#!/usr/bin/env python3
"""
split_dataset.py

Create a stratified train/validation/test split for the PlantGuard AI
image dataset.

For every class folder found under the intermediate dataset directory,
images are shuffled with a fixed random seed and split independently
into train (70%), validation (15%), and test (15%) subsets. Files are
copied (never moved) into a mirrored class-folder structure under the
processed dataset directory. A per-class CSV report and a plain-text
summary are generated under the metadata directory.

Project: PlantGuard AI
Author: Senior Machine Learning Engineer
Python: 3.11
"""

from __future__ import annotations

import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

INPUT_DIR: Path = Path("datasets/intermediate")
OUTPUT_DIR: Path = Path("datasets/processed")
METADATA_DIR: Path = Path("datasets/metadata")

REPORT_CSV_PATH: Path = METADATA_DIR / "split_report.csv"
SUMMARY_TXT_PATH: Path = METADATA_DIR / "split_summary.txt"

SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)

TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

RANDOM_SEED: int = 42

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("split_dataset")


@dataclass(frozen=True)
class ClassSplit:
    """Container holding the file lists for a single class split.

    Attributes:
        class_name: Name of the class (source folder name).
        train_files: List of file paths assigned to the train split.
        val_files: List of file paths assigned to the validation split.
        test_files: List of file paths assigned to the test split.
    """

    class_name: str
    train_files: list[Path]
    val_files: list[Path]
    test_files: list[Path]


def get_class_folders(input_dir: Path) -> list[Path]:
    """Retrieve all class folders from the input dataset directory.

    Args:
        input_dir: Path to the intermediate dataset directory containing
            one subfolder per class.

    Returns:
        A sorted list of paths to class folders.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        NotADirectoryError: If the input path is not a directory.
        ValueError: If no class folders are found in the input directory.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    class_folders = sorted(
        [entry for entry in input_dir.iterdir() if entry.is_dir()]
    )

    if not class_folders:
        raise ValueError(f"No class folders found in: {input_dir}")

    logger.info("Discovered %d class folder(s) in %s", len(class_folders), input_dir)
    return class_folders


def split_class(
    class_folder: Path,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> ClassSplit:
    """Shuffle and split a single class folder's images into train/val/test.

    Args:
        class_folder: Path to the class folder containing images.
        train_ratio: Fraction of images assigned to the train split.
        val_ratio: Fraction of images assigned to the validation split.
        test_ratio: Fraction of images assigned to the test split.
        seed: Random seed used for reproducible shuffling.

    Returns:
        A ClassSplit instance containing the file lists for each split.

    Raises:
        ValueError: If the ratios do not sum to 1.0 (within tolerance).
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio, val_ratio, and test_ratio must sum to 1.0 "
            f"(got {train_ratio + val_ratio + test_ratio})"
        )

    images = sorted(
        entry
        for entry in class_folder.iterdir()
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        logger.warning("No supported images found in class folder: %s", class_folder)

    random.seed(seed)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    return ClassSplit(
        class_name=class_folder.name,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
    )


def copy_images(class_split: ClassSplit, output_dir: Path) -> None:
    """Copy images for a class split into the processed dataset directory.

    Original files are never moved or deleted; all files are copied using
    `shutil.copy2` to preserve metadata.

    Args:
        class_split: The ClassSplit instance holding file lists per split.
        output_dir: Root path of the processed dataset directory.

    Raises:
        OSError: If a file cannot be copied due to a filesystem error.
    """
    split_to_files: dict[str, list[Path]] = {
        "train": class_split.train_files,
        "val": class_split.val_files,
        "test": class_split.test_files,
    }

    for split_name, files in split_to_files.items():
        destination_dir = output_dir / split_name / class_split.class_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        for source_path in tqdm(
            files,
            desc=f"{class_split.class_name[:30]:<30} -> {split_name}",
            unit="img",
            leave=False,
        ):
            destination_path = destination_dir / source_path.name
            try:
                shutil.copy2(source_path, destination_path)
            except OSError as exc:
                logger.error(
                    "Failed to copy %s -> %s: %s",
                    source_path,
                    destination_path,
                    exc,
                )
                raise


def generate_report(
    class_splits: list[ClassSplit],
    report_csv_path: Path,
    summary_txt_path: Path,
) -> None:
    """Generate the CSV report and text summary for the dataset split.

    Args:
        class_splits: List of ClassSplit instances, one per class.
        report_csv_path: Destination path for the per-class CSV report.
        summary_txt_path: Destination path for the overall text summary.

    Raises:
        OSError: If the report or summary files cannot be written.
    """
    report_rows = []
    for class_split in class_splits:
        train_count = len(class_split.train_files)
        val_count = len(class_split.val_files)
        test_count = len(class_split.test_files)
        report_rows.append(
            {
                "class_name": class_split.class_name,
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count,
                "total_count": train_count + val_count + test_count,
            }
        )

    report_df = pd.DataFrame(
        report_rows,
        columns=[
            "class_name",
            "train_count",
            "val_count",
            "test_count",
            "total_count",
        ],
    )

    report_csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report_df.to_csv(report_csv_path, index=False)
    except OSError as exc:
        logger.error("Failed to write CSV report to %s: %s", report_csv_path, exc)
        raise

    total_classes = len(report_df)
    total_train = int(report_df["train_count"].sum())
    total_val = int(report_df["val_count"].sum())
    total_test = int(report_df["test_count"].sum())
    total_images = int(report_df["total_count"].sum())

    summary_lines = [
        "PlantGuard AI - Dataset Split Summary",
        "=" * 40,
        f"Total Classes:      {total_classes}",
        f"Train Images:       {total_train}",
        f"Validation Images:  {total_val}",
        f"Test Images:        {total_test}",
        f"Total Images:       {total_images}",
    ]

    summary_txt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary_txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to write summary to %s: %s", summary_txt_path, exc)
        raise

    logger.info("Split report written to: %s", report_csv_path)
    logger.info("Split summary written to: %s", summary_txt_path)


def main() -> None:
    """Run the full stratified dataset split pipeline."""
    logger.info("Starting PlantGuard AI dataset split")
    logger.info("Input directory:  %s", INPUT_DIR)
    logger.info("Output directory: %s", OUTPUT_DIR)

    try:
        class_folders = get_class_folders(INPUT_DIR)

        for split_name in SPLIT_NAMES:
            (OUTPUT_DIR / split_name).mkdir(parents=True, exist_ok=True)

        class_splits: list[ClassSplit] = []

        for class_folder in tqdm(class_folders, desc="Processing classes", unit="class"):
            class_split = split_class(
                class_folder,
                train_ratio=TRAIN_RATIO,
                val_ratio=VAL_RATIO,
                test_ratio=TEST_RATIO,
                seed=RANDOM_SEED,
            )
            copy_images(class_split, OUTPUT_DIR)
            class_splits.append(class_split)

            logger.info(
                "Class '%s': train=%d, val=%d, test=%d",
                class_split.class_name,
                len(class_split.train_files),
                len(class_split.val_files),
                len(class_split.test_files),
            )

        generate_report(class_splits, REPORT_CSV_PATH, SUMMARY_TXT_PATH)

        logger.info("Dataset split completed successfully")

    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        logger.error("Dataset split failed: %s", exc)
        raise
    except OSError as exc:
        logger.error("Filesystem error during dataset split: %s", exc)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error during dataset split: %s", exc)
        raise


if __name__ == "__main__":
    main()