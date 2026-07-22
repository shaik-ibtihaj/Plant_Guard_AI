#!/usr/bin/env python3
"""
analyze_image_resolution.py

PlantGuard AI — Phase 3.4: Image Resolution Statistics

Recursively scans the processed dataset, extracts image dimension
statistics (width, height, aspect ratio, resolution), persists a
metadata JSON summary, and generates distribution visualizations.

Usage:
    python scripts/analysis/analyze_image_resolution.py

Author: PlantGuard AI Engineering
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "datasets" / "processed"
METADATA_DIR: Path = PROJECT_ROOT / "datasets" / "metadata"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

METADATA_PATH: Path = METADATA_DIR / "image_resolution_stats.json"
WIDTH_PLOT_PATH: Path = REPORTS_DIR / "width_distribution.png"
HEIGHT_PLOT_PATH: Path = REPORTS_DIR / "height_distribution.png"
SCATTER_PLOT_PATH: Path = REPORTS_DIR / "resolution_scatter.png"

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")

LOG_PATH: Path = PROJECT_ROOT / "scripts" / "analysis" / "analyze_image_resolution.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _configure_logger() -> logging.Logger:
    """Configure and return the module logger with file + console handlers.

    Attaches handlers directly to the named logger (rather than relying on
    ``logging.basicConfig``, which only configures the root logger) so the
    handler list is populated regardless of what other libraries have
    already touched the root logger.

    Returns:
        The configured ``analyze_image_resolution`` logger.
    """
    log = logging.getLogger("analyze_image_resolution")
    log.setLevel(logging.INFO)
    log.propagate = False

    if log.handlers:
        return log  # Avoid duplicate handlers on repeated imports/runs.

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")

    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    # Keep console output clean; detailed per-file warnings go to the log file only.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(fmt)

    log.addHandler(file_handler)
    log.addHandler(console_handler)
    return log


logger = _configure_logger()


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class ScanResult:
    """Container for raw per-image measurements collected during scanning."""

    widths: List[int] = field(default_factory=list)
    heights: List[int] = field(default_factory=list)
    aspect_ratios: List[float] = field(default_factory=list)
    resolutions: List[int] = field(default_factory=list)
    corrupted_count: int = 0
    total_images: int = 0


# --------------------------------------------------------------------------- #
# Core functions
# --------------------------------------------------------------------------- #

def scan_images(data_dir: Path) -> ScanResult:
    """Recursively scan ``data_dir`` for supported images and extract dimensions.

    Args:
        data_dir: Root directory containing class-organized image folders.

    Returns:
        A ``ScanResult`` populated with per-image measurements.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    logger.info("Scanning dataset directory: %s", data_dir)

    image_paths: List[Path] = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    logger.info("Discovered %d candidate image files.", len(image_paths))

    result = ScanResult()

    for image_path in tqdm(image_paths, desc="Analyzing images", unit="img"):
        try:
            with Image.open(image_path) as img:
                img.verify()
            # Re-open after verify(), since verify() invalidates the file handle.
            with Image.open(image_path) as img:
                width, height = img.size

            if width <= 0 or height <= 0:
                raise ValueError(f"Non-positive dimensions: {width}x{height}")

            result.widths.append(width)
            result.heights.append(height)
            result.aspect_ratios.append(width / height)
            result.resolutions.append(width * height)
            result.total_images += 1

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            result.corrupted_count += 1
            logger.warning("Skipping corrupted/unreadable image: %s (%s)", image_path, exc)
            continue

    logger.info(
        "Scan complete. Valid images: %d | Corrupted/skipped: %d",
        result.total_images,
        result.corrupted_count,
    )
    return result


def _describe(values: List[float]) -> Dict[str, float]:
    """Compute rounded descriptive statistics for a list of numeric values.

    Args:
        values: Numeric samples to summarize.

    Returns:
        Dictionary with min, max, mean, median, and std, each rounded to 2
        decimal places. Returns zeros if ``values`` is empty.
    """
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}

    return {
        "min": round(float(min(values)), 2),
        "max": round(float(max(values)), 2),
        "mean": round(float(statistics.mean(values)), 2),
        "median": round(float(statistics.median(values)), 2),
        "std": round(float(statistics.stdev(values)) if len(values) > 1 else 0.0, 2),
    }


def compute_statistics(scan_result: ScanResult) -> Dict[str, Dict[str, float]]:
    """Compute descriptive statistics for width, height, aspect ratio, resolution.

    Args:
        scan_result: Populated ``ScanResult`` from ``scan_images``.

    Returns:
        Dictionary keyed by metric name, each mapping to its statistics dict.
    """
    logger.info("Computing descriptive statistics.")
    return {
        "width": _describe(scan_result.widths),
        "height": _describe(scan_result.heights),
        "aspect_ratio": _describe(scan_result.aspect_ratios),
        "resolution": _describe(scan_result.resolutions),
    }


def save_metadata(
    scan_result: ScanResult,
    stats: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Persist the analysis summary to a JSON metadata file.

    Args:
        scan_result: Populated ``ScanResult`` from ``scan_images``.
        stats: Statistics dictionary from ``compute_statistics``.
        output_path: Destination path for the JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "total_images": scan_result.total_images,
        "width": stats["width"],
        "height": stats["height"],
        "aspect_ratio": stats["aspect_ratio"],
        "resolution": stats["resolution"],
        "corrupted_images": scan_result.corrupted_count,
    }

    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        logger.info("Metadata saved to: %s", output_path)
    except OSError as exc:
        logger.error("Failed to save metadata to %s: %s", output_path, exc)
        raise


def plot_width_distribution(widths: List[int], output_path: Path) -> None:
    """Generate and save a histogram of image widths.

    Args:
        widths: List of image widths in pixels.
        output_path: Destination path for the PNG file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(widths, bins=50, color="#4C72B0", edgecolor="black")
    ax.set_title("Width Distribution")
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Width distribution plot saved to: %s", output_path)


def plot_height_distribution(heights: List[int], output_path: Path) -> None:
    """Generate and save a histogram of image heights.

    Args:
        heights: List of image heights in pixels.
        output_path: Destination path for the PNG file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(heights, bins=50, color="#55A868", edgecolor="black")
    ax.set_title("Height Distribution")
    ax.set_xlabel("Height (pixels)")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Height distribution plot saved to: %s", output_path)


def plot_resolution_scatter(
    widths: List[int], heights: List[int], output_path: Path
) -> None:
    """Generate and save a scatter plot of width vs. height.

    Args:
        widths: List of image widths in pixels.
        heights: List of image heights in pixels.
        output_path: Destination path for the PNG file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(widths, heights, alpha=0.3, s=5, color="#C44E52")
    ax.set_title("Resolution Scatter (Width vs. Height)")
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Resolution scatter plot saved to: %s", output_path)


def print_summary(scan_result: ScanResult, stats: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted console summary of the analysis.

    Args:
        scan_result: Populated ``ScanResult`` from ``scan_images``.
        stats: Statistics dictionary from ``compute_statistics``.
    """
    width = stats["width"]
    height = stats["height"]
    aspect = stats["aspect_ratio"]

    print("=" * 50)
    print("PLANTGUARD AI - IMAGE RESOLUTION ANALYSIS")
    print("=" * 50)
    print()
    print(f"Total Images: {scan_result.total_images}")
    print(f"Corrupted Images: {scan_result.corrupted_count}")
    print()
    print("Width:")
    print(f"  Min:    {width['min']}")
    print(f"  Max:    {width['max']}")
    print(f"  Mean:   {width['mean']}")
    print(f"  Median: {width['median']}")
    print()
    print("Height:")
    print(f"  Min:    {height['min']}")
    print(f"  Max:    {height['max']}")
    print(f"  Mean:   {height['mean']}")
    print(f"  Median: {height['median']}")
    print()
    print("Aspect Ratio:")
    print(f"  Mean:   {aspect['mean']}")
    print(f"  Median: {aspect['median']}")
    print()
    print("Outputs:")
    print(f"✓ {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"✓ {WIDTH_PLOT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"✓ {HEIGHT_PLOT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"✓ {SCATTER_PLOT_PATH.relative_to(PROJECT_ROOT)}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    """Run the full image resolution analysis pipeline."""
    try:
        scan_result = scan_images(DATA_DIR)

        if scan_result.total_images == 0:
            logger.error("No valid images found in %s. Aborting.", DATA_DIR)
            sys.exit(1)

        stats = compute_statistics(scan_result)
        save_metadata(scan_result, stats, METADATA_PATH)

        plot_width_distribution(scan_result.widths, WIDTH_PLOT_PATH)
        plot_height_distribution(scan_result.heights, HEIGHT_PLOT_PATH)
        plot_resolution_scatter(scan_result.widths, scan_result.heights, SCATTER_PLOT_PATH)

        print_summary(scan_result, stats)

    except FileNotFoundError as exc:
        logger.error("Dataset error: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        logger.exception("Unexpected failure during analysis: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()