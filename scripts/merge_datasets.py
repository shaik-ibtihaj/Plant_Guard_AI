"""
PlantGuard AI - Dataset Merge Utility
======================================

Merges heterogeneous raw plant-disease image datasets located under
``datasets/raw/`` into a single, unified dataset under
``datasets/intermediate/`` using a standardized class-name mapping
defined in ``datasets/metadata/class_mapping.json``.

Typical usage
-------------
    python scripts/merge_datasets.py

Author: Senior ML Engineering Team
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA_DIR: Path = PROJECT_ROOT / "datasets" / "raw"
INTERMEDIATE_DATA_DIR: Path = PROJECT_ROOT / "datasets" / "intermediate"
CLASS_MAPPING_PATH: Path = PROJECT_ROOT / "datasets" / "metadata" / "class_mapping.json"
MERGE_REPORT_PATH: Path = PROJECT_ROOT / "datasets" / "metadata" / "merge_report.csv"

SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)

# Folder names that must always be ignored, regardless of the class mapping.
IGNORED_FOLDER_NAMES: Tuple[str, ...] = (
    "train",
    "test",
    "valid",
    "validation",
    "images to predict",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("merge_datasets")


# --------------------------------------------------------------------------- #
# Core functions
# --------------------------------------------------------------------------- #

def load_mapping(mapping_path: Path = CLASS_MAPPING_PATH) -> Dict[str, str]:
    """
    Load the standardized class-name mapping from a JSON file.

    Parameters
    ----------
    mapping_path : Path
        Path to the ``class_mapping.json`` file. The JSON file is expected
        to be a flat dictionary mapping ``{original_class_name: standardized_class_name}``.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping original (raw) class folder names to their
        standardized class names.

    Raises
    ------
    FileNotFoundError
        If the mapping file does not exist.
    ValueError
        If the mapping file is not valid JSON or does not contain a
        flat string-to-string dictionary.
    """
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Class mapping file not found at: {mapping_path}"
        )

    try:
        with mapping_path.open("r", encoding="utf-8") as fh:
            mapping = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse JSON in class mapping file: {mapping_path}"
        ) from exc

    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in mapping.items()
    ):
        raise ValueError(
            "class_mapping.json must be a flat dictionary of "
            "{original_class_name: standardized_class_name} string pairs."
        )

    logger.info("Loaded %d class mappings from %s", len(mapping), mapping_path)
    return mapping


def _folder_contains_images(folder: Path) -> bool:
    """
    Check whether a folder directly contains at least one supported image file.

    Parameters
    ----------
    folder : Path
        Directory to inspect (non-recursive check of direct children).

    Returns
    -------
    bool
        True if the folder directly contains at least one image file with a
        supported extension, False otherwise.
    """
    try:
        for entry in folder.iterdir():
            if entry.is_file() and entry.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                return True
    except PermissionError:
        logger.warning("Permission denied while scanning: %s", folder)
    return False


def find_class_folders(raw_dir: Path = RAW_DATA_DIR) -> List[Path]:
    """
    Recursively scan the raw dataset directory and identify class folders
    that directly contain image files.

    A "class folder" is any directory that directly holds one or more
    supported image files (images nested inside further subdirectories do
    not count towards the parent).

    Parameters
    ----------
    raw_dir : Path
        Root directory containing the raw, unmerged datasets.

    Returns
    -------
    List[Path]
        List of directories that directly contain supported image files.

    Raises
    ------
    FileNotFoundError
        If the raw data directory does not exist.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found at: {raw_dir}")

    class_folders: List[Path] = []
    for candidate in raw_dir.rglob("*"):
        if candidate.is_dir() and _folder_contains_images(candidate):
            class_folders.append(candidate)

    logger.info("Discovered %d candidate class folder(s) under %s", len(class_folders), raw_dir)
    return class_folders


def safe_copy(src_file: Path, dest_dir: Path) -> Path:
    """
    Copy a single image file into a destination directory, preventing
    filename collisions by appending an incrementing numeric suffix.

    For example, if ``image.jpg`` already exists in ``dest_dir``, the new
    file will be saved as ``image_1.jpg``; if that also exists, ``image_2.jpg``,
    and so on.

    Parameters
    ----------
    src_file : Path
        Path to the source image file to copy.
    dest_dir : Path
        Destination directory in which to place the copied file.

    Returns
    -------
    Path
        The final path of the copied file.

    Raises
    ------
    OSError
        If the copy operation fails for I/O related reasons.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    destination = dest_dir / src_file.name
    stem = src_file.stem
    suffix = src_file.suffix
    counter = 1

    while destination.exists():
        destination = dest_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    try:
        shutil.copy2(src_file, destination)
    except OSError as exc:
        raise OSError(f"Failed to copy '{src_file}' to '{destination}': {exc}") from exc

    return destination


def merge_dataset(
    raw_dir: Path,
    intermediate_dir: Path,
    mapping: Dict[str, str],
) -> List[Dict[str, object]]:
    """
    Merge all valid class folders from the raw dataset directory into the
    unified intermediate dataset directory, using standardized class names.

    Parameters
    ----------
    raw_dir : Path
        Root directory containing the raw, unmerged datasets.
    intermediate_dir : Path
        Root directory in which the merged, standardized dataset will be
        written.
    mapping : Dict[str, str]
        Mapping of original class folder names to standardized class names.

    Returns
    -------
    List[Dict[str, object]]
        A list of per-folder merge records, each containing:
        ``original_class``, ``merged_class``, and ``image_count``.
    """
    class_folders = find_class_folders(raw_dir)
    records: List[Dict[str, object]] = []

    for folder in tqdm(class_folders, desc="Merging class folders", unit="folder"):
        folder_name = folder.name

        if folder_name.lower() in IGNORED_FOLDER_NAMES:
            logger.info("Ignoring reserved folder: %s", folder)
            continue

        if folder_name not in mapping:
            logger.info(
                "Skipping folder not present in class_mapping.json: %s", folder_name
            )
            continue

        standardized_class = mapping[folder_name]
        dest_dir = intermediate_dir / standardized_class

        image_files = [
            f
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

        copied_count = 0
        for image_file in tqdm(
            image_files,
            desc=f"  {folder_name} -> {standardized_class}",
            unit="img",
            leave=False,
        ):
            try:
                safe_copy(image_file, dest_dir)
                copied_count += 1
            except OSError as exc:
                logger.error("Error copying file '%s': %s", image_file, exc)

        records.append(
            {
                "original_class": folder_name,
                "merged_class": standardized_class,
                "image_count": copied_count,
            }
        )

    return records


def generate_report(
    records: List[Dict[str, object]],
    report_path: Path = MERGE_REPORT_PATH,
) -> pd.DataFrame:
    """
    Generate and persist a CSV report summarizing the dataset merge process.

    Parameters
    ----------
    records : List[Dict[str, object]]
        Per-folder merge records produced by ``merge_dataset``.
    report_path : Path
        Destination path for the generated CSV report.

    Returns
    -------
    pd.DataFrame
        DataFrame representation of the generated report.

    Raises
    ------
    OSError
        If the report cannot be written to disk.
    """
    columns = ["original_class", "merged_class", "image_count"]
    report_df = pd.DataFrame(records, columns=columns)

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write merge report to '{report_path}': {exc}") from exc

    logger.info("Merge report saved to: %s", report_path)
    return report_df


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    """
    Execute the full PlantGuard AI dataset merge pipeline:

    1. Load the standardized class mapping.
    2. Discover and merge all valid raw class folders into the
       intermediate dataset directory.
    3. Generate a CSV merge report.
    4. Print a human-readable summary.
    """
    try:
        mapping = load_mapping(CLASS_MAPPING_PATH)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load class mapping: %s", exc)
        sys.exit(1)

    try:
        records = merge_dataset(RAW_DATA_DIR, INTERMEDIATE_DATA_DIR, mapping)
    except FileNotFoundError as exc:
        logger.error("Failed to merge dataset: %s", exc)
        sys.exit(1)

    try:
        report_df = generate_report(records, MERGE_REPORT_PATH)
    except OSError as exc:
        logger.error("Failed to generate merge report: %s", exc)
        sys.exit(1)

    classes_merged = len(report_df)
    images_copied = int(report_df["image_count"].sum()) if not report_df.empty else 0
    destination_classes = report_df["merged_class"].nunique() if not report_df.empty else 0

    print("=" * 50)
    print("PlantGuard AI Dataset Merge Complete")
    print("=" * 50)
    print(f"Classes Merged      : {classes_merged}")
    print(f"Images Copied       : {images_copied}")
    print(f"Destination Classes : {destination_classes}")
    print("Report Saved:")
    print(f"{MERGE_REPORT_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()