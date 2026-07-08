"""
scripts/clean_dataset.py

Project: PlantGuard AI

Recursively scans datasets/raw/ for image files, validates each image using
Pillow, and quarantines corrupted or invalid files. Generates a CSV report
and a plain-text summary describing the results of the cleaning process.

Author: Senior Machine Learning Engineer
Python: 3.11+
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RAW_DATA_DIR = Path("datasets/raw")
METADATA_DIR = Path("datasets/metadata")
QUARANTINE_DIR = Path("datasets/quarantine")

REPORT_PATH = METADATA_DIR / "cleaning_report.csv"
SUMMARY_PATH = METADATA_DIR / "cleaning_summary.txt"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Directory setup
# --------------------------------------------------------------------------- #

def create_directories() -> None:
    """
    Ensure that required output directories exist.

    Creates the metadata and quarantine directories if they are missing.
    Does nothing if the directories already exist.
    """
    try:
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(
            f"Failed to create required directories: {error}"
        ) from error


# --------------------------------------------------------------------------- #
# Image validation
# --------------------------------------------------------------------------- #

def validate_image(file_path: Path) -> Tuple[str, str]:
    """
    Validate a single image file.

    The validation pipeline checks:
        1. File extension is supported.
        2. File size is greater than zero bytes.
        3. File can be opened by Pillow.
        4. File passes Pillow's internal verify() integrity check.

    Args:
        file_path: Path to the image file to validate.

    Returns:
        A tuple of (status, reason) where status is one of
        "valid", "invalid", or "corrupted", and reason is a short
        human-readable explanation.
    """
    # 1. Extension check
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return "invalid", f"Unsupported file extension: {file_path.suffix}"

    # 2. File size check
    try:
        file_size = file_path.stat().st_size
    except OSError as error:
        return "invalid", f"Unable to read file size: {error}"

    if file_size <= 0:
        return "corrupted", "Zero-byte file"

    # 3 & 4. Open and verify with Pillow
    try:
        with Image.open(file_path) as img:
            img.verify()
    except UnidentifiedImageError:
        return "corrupted", "Unidentified image format (unreadable)"
    except (OSError, ValueError, SyntaxError) as error:
        return "corrupted", f"Corrupted or unreadable image: {error}"
    except Exception as error:  # noqa: BLE001 - final safety net
        return "corrupted", f"Unexpected error during verification: {error}"

    return "valid", "Image passed all validation checks"


# --------------------------------------------------------------------------- #
# Quarantine handling
# --------------------------------------------------------------------------- #

def move_to_quarantine(file_path: Path) -> Path:
    """
    Move a corrupted or invalid file into the quarantine directory.

    The original filename is preserved. If a file with the same name
    already exists in the quarantine directory, a numeric suffix is
    appended to avoid overwriting.

    Args:
        file_path: Path to the file to quarantine.

    Returns:
        The destination path where the file was moved.

    Raises:
        RuntimeError: If the file could not be moved.
    """
    destination = QUARANTINE_DIR / file_path.name

    if destination.exists():
        counter = 1
        stem = file_path.stem
        suffix = file_path.suffix
        while destination.exists():
            destination = QUARANTINE_DIR / f"{stem}_{counter}{suffix}"
            counter += 1

    try:
        shutil.move(str(file_path), str(destination))
    except (OSError, shutil.Error) as error:
        raise RuntimeError(
            f"Failed to move '{file_path}' to quarantine: {error}"
        ) from error

    return destination


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def generate_report(records: List[dict]) -> pd.DataFrame:
    """
    Generate the CSV report and plain-text summary from validation records.

    Args:
        records: A list of dictionaries, each containing
            'file_path', 'status', and 'reason' keys.

    Returns:
        The pandas DataFrame constructed from the records, useful for
        further inspection or testing.

    Raises:
        RuntimeError: If the report or summary files could not be written.
    """
    df = pd.DataFrame(records, columns=["file_path", "status", "reason"])

    try:
        df.to_csv(REPORT_PATH, index=False, encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Failed to write CSV report: {error}") from error

    total_files = len(df)
    valid_count = int((df["status"] == "valid").sum())
    corrupted_count = int((df["status"] == "corrupted").sum())
    invalid_count = int((df["status"] == "invalid").sum())

    summary_lines = [
        "PlantGuard AI - Dataset Cleaning Summary",
        "=" * 45,
        f"Total files scanned   : {total_files}",
        f"Valid images          : {valid_count}",
        f"Corrupted images      : {corrupted_count}",
        f"Invalid files         : {invalid_count}",
        "=" * 45,
        f"CSV report            : {REPORT_PATH}",
        f"Quarantine directory  : {QUARANTINE_DIR}",
    ]
    summary_text = "\n".join(summary_lines) + "\n"

    try:
        SUMMARY_PATH.write_text(summary_text, encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"Failed to write summary report: {error}"
        ) from error

    return df


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    """
    Execute the full dataset cleaning pipeline.

    Steps:
        1. Ensure required directories exist.
        2. Recursively collect all files under datasets/raw/.
        3. Validate each file and quarantine corrupted/invalid ones.
        4. Generate CSV report and text summary.
        5. Print a professional final summary to the console.
    """
    create_directories()

    if not RAW_DATA_DIR.exists():
        print(f"Error: raw data directory '{RAW_DATA_DIR}' does not exist.")
        return

    all_files = [
        path for path in RAW_DATA_DIR.rglob("*") if path.is_file()
    ]

    if not all_files:
        print(f"No files found under '{RAW_DATA_DIR}'. Nothing to process.")
        return

    records: List[dict] = []

    for file_path in tqdm(
        all_files, desc="Validating images", unit="file", colour="green"
    ):
        try:
            status, reason = validate_image(file_path)
        except Exception as error:  # noqa: BLE001 - final safety net
            status, reason = "invalid", f"Unhandled validation error: {error}"

        if status in ("corrupted", "invalid"):
            try:
                quarantined_path = move_to_quarantine(file_path)
                records.append(
                    {
                        "file_path": str(quarantined_path),
                        "status": status,
                        "reason": reason,
                    }
                )
            except RuntimeError as error:
                records.append(
                    {
                        "file_path": str(file_path),
                        "status": "invalid",
                        "reason": f"Quarantine failed: {error}",
                    }
                )
        else:
            records.append(
                {
                    "file_path": str(file_path),
                    "status": status,
                    "reason": reason,
                }
            )

    df = generate_report(records)

    total_files = len(df)
    valid_count = int((df["status"] == "valid").sum())
    corrupted_count = int((df["status"] == "corrupted").sum())
    invalid_count = int((df["status"] == "invalid").sum())

    print("\n" + "=" * 45)
    print("PlantGuard AI - Dataset Cleaning Complete")
    print("=" * 45)
    print(f"Total files scanned   : {total_files}")
    print(f"Valid images          : {valid_count}")
    print(f"Corrupted images      : {corrupted_count}")
    print(f"Invalid files         : {invalid_count}")
    print(f"CSV report saved to   : {REPORT_PATH}")
    print(f"Summary saved to      : {SUMMARY_PATH}")
    print(f"Quarantine directory  : {QUARANTINE_DIR}")
    print("=" * 45)


if __name__ == "__main__":
    main()