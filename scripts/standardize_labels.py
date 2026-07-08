#!/usr/bin/env python3
"""
standardize_labels.py

PlantGuard AI - Dataset Label Standardization Utility

This script scans a raw dataset directory tree, discovers every class
folder that directly contains image files, and derives a unified
"Plant___Disease" naming convention for each discovered class name.

It handles three distinct source naming styles:
    1. PlantLeaves-style names with a trailing scan code, e.g.
       "Mango healthy (P0a)" or "Pongamia Pinnata diseased (P7b)".
    2. Already-standardized names containing a triple-underscore
       separator, e.g. "Potato___healthy" or
       "Cherry_(including_sour)___healthy".
    3. Legacy PlantVillage-style names with single/double underscores,
       e.g. "Tomato_Early_blight".

It does NOT rename any folders on disk. Instead, it produces a JSON
mapping file (old_name -> new_name) at:

    datasets/metadata/class_mapping.json

That mapping can later be consumed by a separate renaming script once
it has been reviewed.

Usage:
    python scripts/standardize_labels.py

Author: Senior ML Engineering Team
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Set

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DATASET_ROOT = Path("datasets/raw")
METADATA_DIR = Path("datasets/metadata")
MAPPING_FILE = METADATA_DIR / "class_mapping.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def find_class_folders(dataset_root: Path) -> List[Path]:
    """Recursively locate every folder that directly contains images.

    A "class folder" is defined as any directory that contains at least
    one file with a recognized image extension directly inside it
    (not in a subdirectory). Dataset-structure folders (train/test/valid
    splits, or prediction staging folders) are never treated as class
    folders themselves, even if they happen to contain loose images —
    but their subdirectories are still scanned normally.

    Args:
        dataset_root: Root directory under which to search recursively.

    Returns:
        A sorted list of unique Path objects, each representing a
        folder that directly contains one or more image files.

    Raises:
        FileNotFoundError: If dataset_root does not exist.
        NotADirectoryError: If dataset_root is not a directory.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    # Structural / split folders that must never be treated as classes.
    ignored_folder_names = {"train", "test", "valid", "images to predict"}

    class_folders: Set[Path] = set()

    for path in dataset_root.rglob("*"):
        try:
            if not path.is_dir():
                continue

            if path.name.strip().lower() in ignored_folder_names:
                # Skip treating this folder as a class, but its children
                # will still be visited by rglob().
                continue

            has_image = any(
                child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
                for child in path.iterdir()
            )
            if has_image:
                class_folders.add(path)
        except PermissionError as exc:
            logger.warning("Permission denied while scanning %s: %s", path, exc)
            continue
        except OSError as exc:
            logger.warning("OS error while scanning %s: %s", path, exc)
            continue

    return sorted(class_folders)


def standardize_name(raw_name: str) -> str:
    """Convert a raw class folder name into the Plant___Disease convention.

    Handles three distinct naming styles found across datasets:

        1. PlantLeaves-style names with a trailing code, e.g.
           "Mango healthy (P0a)" or "Pongamia Pinnata diseased (P7b)".
           Multi-word plant names are preserved by joining words with a
           single underscore, the trailing "(P<digits><a|b>)" code is
           discarded, and the result becomes "Plant_Name___status".

        2. Names that already contain a triple-underscore separator,
           e.g. "Potato___healthy" or "Cherry_(including_sour)___healthy".
           These are left structurally untouched (only trimmed) so that
           legitimate single underscores inside the plant or disease
           segment (including underscores inside parentheses) are never
           re-split.

        3. Legacy PlantVillage-style names with single/double
           underscores and no existing triple-underscore separator,
           e.g. "Tomato_Early_blight" -> "Tomato___Early_blight".

    Args:
        raw_name: The original folder name.

    Returns:
        A standardized class name in the form "Plant___Disease".

    Raises:
        ValueError: If raw_name is empty or contains no usable
            alphanumeric content after cleaning.
    """
    if not raw_name or not raw_name.strip():
        raise ValueError("raw_name must be a non-empty, non-whitespace string")

    # Collapse any internal whitespace runs to a single space first.
    cleaned = re.sub(r"\s+", " ", raw_name.strip())

    # --- 1. PlantLeaves-style: "<Plant Name> healthy/diseased (P0a)" ---
    plantleaves_pattern = re.compile(
        r"^(?P<plant>.+?)\s+(?P<status>healthy|diseased)\s*\(P\d+[ab]\)\s*$",
        re.IGNORECASE,
    )
    match = plantleaves_pattern.match(cleaned)
    if match:
        plant_part = match.group("plant").strip()
        status_part = match.group("status").strip().lower()
        plant_tokens = [tok for tok in re.split(r"\s+", plant_part) if tok]
        if not plant_tokens:
            raise ValueError(f"No usable plant name found in raw_name: {raw_name!r}")
        plant_normalized = "_".join(plant_tokens)
        return f"{plant_normalized}___{status_part}"

    # --- 2. Already-standardized names containing "___" -----------------
    if "___" in cleaned:
        plant_segment, _, disease_segment = cleaned.partition("___")
        plant_segment = plant_segment.strip(" _")
        disease_segment = disease_segment.strip(" _")
        if not plant_segment or not disease_segment:
            raise ValueError(f"Malformed standardized name: {raw_name!r}")
        return f"{plant_segment}___{disease_segment}"

    # --- 3. Legacy PlantVillage single/double underscore style -----------
    cleaned = cleaned.replace(" ", "_")
    tokens = [token for token in re.split(r"_+", cleaned) if token]

    if not tokens:
        raise ValueError(f"No usable tokens found in raw_name: {raw_name!r}")

    if len(tokens) == 1:
        # Only a plant name with no disease label present.
        return tokens[0]

    # First token is treated as the plant name; the remainder joined
    # by single underscores forms the disease/condition name.
    plant = tokens[0]
    disease = "_".join(tokens[1:])

    standardized = f"{plant}___{disease}"

    # Final safety pass: collapse any accidental duplicate underscores
    # beyond the canonical triple separator, and strip stray edges.
    standardized = re.sub(r"_{4,}", "___", standardized)
    standardized = standardized.strip("_")

    return standardized


def save_mapping(mapping: Dict[str, str], output_path: Path) -> None:
    """Persist the old-to-new class name mapping as JSON.

    Args:
        mapping: Dictionary of {old_name: new_name}.
        output_path: Destination JSON file path. Parent directories
            are created if they do not already exist.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(mapping, handle, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info("Saved class mapping to %s", output_path)
    except OSError as exc:
        logger.error("Failed to write mapping file %s: %s", output_path, exc)
        raise


def main() -> None:
    """Entry point: discover classes, standardize names, and save mapping."""
    try:
        class_folders = find_class_folders(DATASET_ROOT)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error("Dataset scan failed: %s", exc)
        return

    if not class_folders:
        logger.warning("No class folders with images were found under %s", DATASET_ROOT)
        return

    mapping: "OrderedDict[str, str]" = OrderedDict()
    discovered_names: List[str] = []

    for folder in class_folders:
        original_name = folder.name
        discovered_names.append(original_name)
        try:
            new_name = standardize_name(original_name)
        except ValueError as exc:
            logger.warning("Skipping folder %s: %s", folder, exc)
            continue
        mapping[original_name] = new_name

    try:
        save_mapping(mapping, MAPPING_FILE)
    except OSError:
        logger.error("Could not save class mapping. Aborting.")
        return

    unique_standardized = set(mapping.values())

    print("=" * 50)
    print("PlantGuard AI - Label Standardization Summary")
    print("=" * 50)
    print(f"Total discovered classes   : {len(discovered_names)}")
    print(f"Unique standardized classes: {len(unique_standardized)}")
    print(f"Number of mappings         : {len(mapping)}")
    print("=" * 50)


if __name__ == "__main__":
    main()