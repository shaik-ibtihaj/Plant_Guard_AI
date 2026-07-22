"""
detect_class_imbalance.py

PlantGuard AI - Phase 3.3: Class Imbalance Detection

This script analyzes the class distribution of the PlantGuard AI dataset,
computes descriptive statistics, identifies minority and majority classes,
generates a JSON imbalance report, produces a histogram visualization, and
prints a professional console summary.

Usage:
    python scripts/analysis/detect_class_imbalance.py

Author: PlantGuard AI ML Engineering Team
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CLASS_DISTRIBUTION_CSV: Path = PROJECT_ROOT/"datasets" / "metadata" / "class_distribution.csv"
MINORITY_CLASSES_CSV: Path = PROJECT_ROOT /"datasets" / "metadata" / "minority_classes.csv"
MAJORITY_CLASSES_CSV: Path = PROJECT_ROOT / "datasets" / "metadata" / "majority_classes.csv"
IMBALANCE_REPORT_JSON: Path = PROJECT_ROOT / "datasets" / "metadata" / "class_imbalance_report.json"
HISTOGRAM_PNG: Path = PROJECT_ROOT / "datasets" / "reports" / "class_imbalance_histogram.png"

MINORITY_THRESHOLD: int = 500
MAJORITY_THRESHOLD: int = 5000

REQUIRED_COLUMNS = ("class", "count")

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("plantguard.detect_class_imbalance")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate the class distribution CSV file.

    Args:
        csv_path: Path to the class_distribution.csv file.

    Returns:
        A validated pandas DataFrame with columns ['class', 'count'].

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing or counts are non-numeric.
    """
    logger.info("Loading class distribution data from: %s", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Class distribution file not found at: {csv_path}"
        )

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse CSV file '{csv_path}': {exc}") from exc

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required column(s) in '{csv_path}': {missing_columns}. "
            f"Expected columns: {list(REQUIRED_COLUMNS)}"
        )

    if not pd.api.types.is_numeric_dtype(df["count"]):
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
        if df["count"].isna().any():
            raise ValueError(
                "Column 'count' contains non-numeric or missing values. "
                "Please ensure all counts are valid integers."
            )

    df["count"] = df["count"].astype(int)

    if df.empty:
        raise ValueError(f"Class distribution file '{csv_path}' contains no data.")

    logger.info("Successfully loaded %d class entries.", len(df))
    return df


# --------------------------------------------------------------------------- #
# Statistics computation
# --------------------------------------------------------------------------- #

def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute descriptive statistics for the class distribution.

    Args:
        df: DataFrame containing 'class' and 'count' columns.

    Returns:
        A dictionary containing dataset-level statistics.
    """
    logger.info("Computing dataset statistics.")

    total_classes = int(df.shape[0])
    total_images = int(df["count"].sum())

    largest_row = df.loc[df["count"].idxmax()]
    smallest_row = df.loc[df["count"].idxmin()]

    largest_class_name = str(largest_row["class"])
    largest_class_count = int(largest_row["count"])
    smallest_class_name = str(smallest_row["class"])
    smallest_class_count = int(smallest_row["count"])

    mean_class_size = round(float(df["count"].mean()), 2)
    median_class_size = round(float(df["count"].median()), 2)
    std_class_size = round(float(df["count"].std()), 2)

    if smallest_class_count == 0:
        raise ValueError(
            f"Smallest class '{smallest_class_name}' has a count of 0. "
            "Cannot compute imbalance ratio (division by zero)."
        )

    imbalance_ratio = round(largest_class_count / smallest_class_count, 2)

    stats: Dict[str, Any] = {
        "total_classes": total_classes,
        "total_images": total_images,
        "largest_class_name": largest_class_name,
        "largest_class_count": largest_class_count,
        "smallest_class_name": smallest_class_name,
        "smallest_class_count": smallest_class_count,
        "mean_class_size": mean_class_size,
        "median_class_size": median_class_size,
        "std_class_size": std_class_size,
        "imbalance_ratio": imbalance_ratio,
    }

    logger.info("Statistics computed successfully.")
    return stats


# --------------------------------------------------------------------------- #
# Minority / Majority class detection
# --------------------------------------------------------------------------- #

def save_minority_classes(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Identify and save minority classes (count < MINORITY_THRESHOLD).

    Args:
        df: DataFrame containing 'class' and 'count' columns.
        output_path: Destination path for the minority classes CSV.

    Returns:
        The DataFrame of minority classes, sorted ascending by count.
    """
    logger.info("Detecting minority classes (threshold < %d).", MINORITY_THRESHOLD)

    minority_df = (
        df[df["count"] < MINORITY_THRESHOLD]
        .sort_values(by="count", ascending=True)
        .loc[:, ["class", "count"]]
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    minority_df.to_csv(output_path, index=False)

    logger.info(
        "Saved %d minority class(es) to: %s", len(minority_df), output_path
    )
    return minority_df


def save_majority_classes(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Identify and save majority classes (count > MAJORITY_THRESHOLD).

    Args:
        df: DataFrame containing 'class' and 'count' columns.
        output_path: Destination path for the majority classes CSV.

    Returns:
        The DataFrame of majority classes, sorted descending by count.
    """
    logger.info("Detecting majority classes (threshold > %d).", MAJORITY_THRESHOLD)

    majority_df = (
        df[df["count"] > MAJORITY_THRESHOLD]
        .sort_values(by="count", ascending=False)
        .loc[:, ["class", "count"]]
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    majority_df.to_csv(output_path, index=False)

    logger.info(
        "Saved %d majority class(es) to: %s", len(majority_df), output_path
    )
    return majority_df


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #

def save_report(
    stats: Dict[str, Any],
    num_minority_classes: int,
    num_majority_classes: int,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Build and save the full class imbalance JSON report.

    Args:
        stats: Dictionary of computed dataset statistics.
        num_minority_classes: Number of detected minority classes.
        num_majority_classes: Number of detected majority classes.
        output_path: Destination path for the JSON report.

    Returns:
        The report dictionary that was written to disk.
    """
    logger.info("Building class imbalance report.")

    report: Dict[str, Any] = {
        "total_classes": stats["total_classes"],
        "total_images": stats["total_images"],
        "largest_class": {
            "name": stats["largest_class_name"],
            "count": stats["largest_class_count"],
        },
        "smallest_class": {
            "name": stats["smallest_class_name"],
            "count": stats["smallest_class_count"],
        },
        "mean_class_size": stats["mean_class_size"],
        "median_class_size": stats["median_class_size"],
        "std_class_size": stats["std_class_size"],
        "imbalance_ratio": stats["imbalance_ratio"],
        "minority_threshold": MINORITY_THRESHOLD,
        "majority_threshold": MAJORITY_THRESHOLD,
        "num_minority_classes": num_minority_classes,
        "num_majority_classes": num_majority_classes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
    except OSError as exc:
        raise OSError(f"Failed to write report to '{output_path}': {exc}") from exc

    logger.info("Report saved to: %s", output_path)
    return report


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #

def plot_histogram(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate and save a histogram of class image counts.

    Args:
        df: DataFrame containing 'class' and 'count' columns.
        output_path: Destination path for the histogram PNG.

    Raises:
        OSError: If the figure cannot be saved to disk.
    """
    logger.info("Generating class count distribution histogram.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["count"], bins=30, color="#4C9A2A", edgecolor="black", alpha=0.8)
    ax.set_title("PlantGuard AI - Class Count Distribution")
    ax.set_xlabel("Number of Images")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()

    try:
        fig.savefig(output_path, dpi=300)
    except OSError as exc:
        raise OSError(f"Failed to save histogram to '{output_path}': {exc}") from exc
    finally:
        plt.close(fig)

    logger.info("Histogram saved to: %s", output_path)


# --------------------------------------------------------------------------- #
# Console summary
# --------------------------------------------------------------------------- #

def print_summary(
    stats: Dict[str, Any],
    num_minority_classes: int,
    num_majority_classes: int,
) -> None:
    """
    Print a professional console summary of the class imbalance analysis.

    Args:
        stats: Dictionary of computed dataset statistics.
        num_minority_classes: Number of detected minority classes.
        num_majority_classes: Number of detected majority classes.
    """
    separator = "=" * 50

    print(f"\n{separator}")
    print("PLANTGUARD AI - CLASS IMBALANCE REPORT")
    print(f"{separator}\n")

    print(f"Total Classes: {stats['total_classes']}")
    print(f"Total Images: {stats['total_images']}\n")

    print("Largest Class:")
    print(f"  {stats['largest_class_name']}")
    print(f"  {stats['largest_class_count']} images\n")

    print("Smallest Class:")
    print(f"  {stats['smallest_class_name']}")
    print(f"  {stats['smallest_class_count']} images\n")

    print(f"Imbalance Ratio: {stats['imbalance_ratio']}\n")

    print(f"Minority Classes: {num_minority_classes}")
    print(f"Majority Classes: {num_majority_classes}\n")

    print("Outputs Generated:")
    print("✓ metadata/class_imbalance_report.json")
    print("✓ metadata/minority_classes.csv")
    print("✓ metadata/majority_classes.csv")
    print("✓ reports/class_imbalance_histogram.png\n")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    """
    Execute the full class imbalance detection pipeline:
    load data, compute statistics, detect minority/majority classes,
    generate the JSON report, plot the histogram, and print a summary.
    """
    try:
        df = load_data(CLASS_DISTRIBUTION_CSV)
        stats = compute_statistics(df)

        minority_df = save_minority_classes(df, MINORITY_CLASSES_CSV)
        majority_df = save_majority_classes(df, MAJORITY_CLASSES_CSV)

        save_report(
            stats=stats,
            num_minority_classes=len(minority_df),
            num_majority_classes=len(majority_df),
            output_path=IMBALANCE_REPORT_JSON,
        )

        plot_histogram(df, HISTOGRAM_PNG)

        print_summary(
            stats=stats,
            num_minority_classes=len(minority_df),
            num_majority_classes=len(majority_df),
        )

    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Class imbalance detection failed: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during class imbalance detection: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()