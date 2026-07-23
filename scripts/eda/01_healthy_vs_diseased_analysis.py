#!/usr/bin/env python3
"""
01_healthy_vs_diseased_analysis.py

PlantGuard AI — Phase 4 EDA: Healthy vs. Diseased Class Analysis

Reads the merged class-distribution metadata and splits the 64 classes into
two groups based on the presence of the keyword "healthy" in the class name.
Computes per-group image counts and percentages, writes a JSON summary, and
produces a dual-panel visualisation (pie chart + horizontal bar chart).

Usage:
    python scripts/eda/01_healthy_vs_diseased_analysis.py

Outputs:
    datasets/metadata/disease_health_summary.json
    reports/disease_vs_healthy.png

Author: PlantGuard AI ML Engineering Team
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CLASS_DISTRIBUTION_CSV: Path = (
    PROJECT_ROOT / "datasets" / "metadata" / "class_distribution.csv"
)
OUTPUT_JSON: Path = (
    PROJECT_ROOT / "datasets" / "metadata" / "disease_health_summary.json"
)
OUTPUT_PNG: Path = PROJECT_ROOT / "reports" / "disease_vs_healthy.png"

# The keyword used to identify healthy classes (case-insensitive substring match).
HEALTHY_KEYWORD: str = "healthy"

REQUIRED_COLUMNS: Tuple[str, ...] = ("class", "count")

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def _configure_logger() -> logging.Logger:
    """Configure and return the module logger with file + console handlers.

    A dedicated file handler captures DEBUG-level detail; the console handler
    is limited to INFO so normal runs remain readable.

    Returns:
        The configured ``plantguard.healthy_vs_diseased`` logger.
    """
    log = logging.getLogger("plantguard.healthy_vs_diseased")
    log.setLevel(logging.DEBUG)
    log.propagate = False

    if log.handlers:
        return log  # Prevent duplicate handlers on repeated imports.

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file: Path = (
        PROJECT_ROOT / "scripts" / "eda" / "01_healthy_vs_diseased.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    log.addHandler(file_handler)
    log.addHandler(console_handler)
    return log


logger: logging.Logger = _configure_logger()


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_class_distribution(csv_path: Path) -> pd.DataFrame:
    """Load and validate the class-distribution CSV.

    Args:
        csv_path: Absolute path to ``class_distribution.csv``.

    Returns:
        Validated ``DataFrame`` with columns ``['class', 'count']``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If required columns are absent or counts are non-numeric.
    """
    logger.info("Loading class distribution from: %s", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Class distribution file not found: {csv_path}\n"
            "Run scripts/analysis/detect_class_imbalance.py first."
        )

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Failed to parse CSV at '{csv_path}': {exc}"
        ) from exc

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) {missing} in '{csv_path}'. "
            f"Expected: {list(REQUIRED_COLUMNS)}"
        )

    # Coerce count column to int; raise on any unparseable values.
    if not pd.api.types.is_numeric_dtype(df["count"]):
        df["count"] = pd.to_numeric(df["count"], errors="coerce")

    if df["count"].isna().any():
        raise ValueError(
            "Column 'count' contains non-numeric or missing values. "
            "Ensure all entries are valid integers."
        )

    df["count"] = df["count"].astype(int)
    df = df.dropna(subset=["class"])
    df = df[df["class"].str.strip() != ""]

    if df.empty:
        raise ValueError(f"No valid rows found in '{csv_path}'.")

    logger.info("Loaded %d class entries successfully.", len(df))
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

def classify_classes(
    df: pd.DataFrame,
    keyword: str = HEALTHY_KEYWORD,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into healthy and diseased sub-frames.

    Detection is case-insensitive substring matching on the ``class`` column.
    Any class name containing ``keyword`` is considered healthy; all others
    are considered diseased.

    Args:
        df:      Full class-distribution DataFrame with columns
                 ``['class', 'count']``.
        keyword: Substring that marks a healthy class. Default: ``"healthy"``.

    Returns:
        A tuple ``(healthy_df, diseased_df)`` of filtered DataFrames.
    """
    mask_healthy: pd.Series = df["class"].str.contains(
        keyword, case=False, na=False
    )
    healthy_df: pd.DataFrame = df[mask_healthy].copy()
    diseased_df: pd.DataFrame = df[~mask_healthy].copy()

    logger.info(
        "Classification complete — healthy classes: %d | diseased classes: %d",
        len(healthy_df),
        len(diseased_df),
    )
    logger.debug("Healthy classes:\n%s", healthy_df["class"].tolist())
    logger.debug("Diseased classes:\n%s", diseased_df["class"].tolist())

    return healthy_df, diseased_df


# --------------------------------------------------------------------------- #
# Statistics computation
# --------------------------------------------------------------------------- #

def compute_summary(
    df: pd.DataFrame,
    healthy_df: pd.DataFrame,
    diseased_df: pd.DataFrame,
) -> Dict:
    """Compute healthy vs. diseased summary statistics.

    Args:
        df:          Full class-distribution DataFrame.
        healthy_df:  Subset of healthy classes.
        diseased_df: Subset of diseased classes.

    Returns:
        Dictionary containing counts, image totals, and percentage breakdowns.
    """
    total_classes: int = len(df)
    total_images: int = int(df["count"].sum())

    healthy_class_count: int = len(healthy_df)
    diseased_class_count: int = len(diseased_df)

    healthy_image_count: int = int(healthy_df["count"].sum())
    diseased_image_count: int = int(diseased_df["count"].sum())

    healthy_class_pct: float = round(
        healthy_class_count / total_classes * 100, 2
    )
    diseased_class_pct: float = round(
        diseased_class_count / total_classes * 100, 2
    )

    healthy_image_pct: float = (
        round(healthy_image_count / total_images * 100, 2)
        if total_images else 0.0
    )
    diseased_image_pct: float = (
        round(diseased_image_count / total_images * 100, 2)
        if total_images else 0.0
    )

    summary: Dict = {
        "total_classes": total_classes,
        "total_images": total_images,
        "healthy": {
            "class_count": healthy_class_count,
            "image_count": healthy_image_count,
            "class_percentage": healthy_class_pct,
            "image_percentage": healthy_image_pct,
            "classes": (
                healthy_df
                .sort_values("count", ascending=False)
                [["class", "count"]]
                .to_dict(orient="records")
            ),
        },
        "diseased": {
            "class_count": diseased_class_count,
            "image_count": diseased_image_count,
            "class_percentage": diseased_class_pct,
            "image_percentage": diseased_image_pct,
            "classes": (
                diseased_df
                .sort_values("count", ascending=False)
                [["class", "count"]]
                .to_dict(orient="records")
            ),
        },
        "detection_keyword": HEALTHY_KEYWORD,
    }

    logger.info(
        "Summary — Total images: %d | Healthy: %d images (%.1f%%) "
        "across %d classes | Diseased: %d images (%.1f%%) across %d classes",
        total_images,
        healthy_image_count,
        healthy_image_pct,
        healthy_class_count,
        diseased_image_count,
        diseased_image_pct,
        diseased_class_count,
    )

    return summary


# --------------------------------------------------------------------------- #
# JSON persistence
# --------------------------------------------------------------------------- #

def save_json(summary: Dict, output_path: Path) -> None:
    """Persist the summary dictionary to a JSON file.

    Args:
        summary:     Dictionary returned by :func:`compute_summary`.
        output_path: Destination file path.

    Raises:
        OSError: If the file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        logger.info("JSON summary saved to: %s", output_path)
    except OSError as exc:
        logger.error("Failed to write JSON to '%s': %s", output_path, exc)
        raise


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #

_COLOR_DISEASED: str = "#C0392B"
_COLOR_HEALTHY: str  = "#27AE60"
_COLOR_GRID: str     = "#E8E8E8"


def _build_bar_data(
    healthy_df: pd.DataFrame,
    diseased_df: pd.DataFrame,
) -> Tuple[List[str], List[int], List[str]]:
    """Prepare sorted bar-chart data combining both groups.

    Diseased classes are placed first (sorted ascending by count), then
    healthy classes follow, giving a clear visual separation between the
    two groups with the largest bars at the top.

    Args:
        healthy_df:  Healthy class sub-frame.
        diseased_df: Diseased class sub-frame.

    Returns:
        Tuple of ``(labels, counts, colors)`` aligned lists ready for
        ``ax.barh``.
    """
    diseased_sorted = diseased_df.sort_values("count", ascending=True)
    healthy_sorted  = healthy_df.sort_values("count", ascending=True)

    combined = pd.concat([diseased_sorted, healthy_sorted], ignore_index=True)
    mask = combined["class"].str.contains(HEALTHY_KEYWORD, case=False, na=False)

    labels: List[str] = combined["class"].tolist()
    counts: List[int] = combined["count"].tolist()
    colors: List[str] = [
        _COLOR_HEALTHY if m else _COLOR_DISEASED for m in mask
    ]
    return labels, counts, colors


def generate_plot(
    summary: Dict,
    healthy_df: pd.DataFrame,
    diseased_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate and save the dual-panel healthy vs. diseased visualisation.

    Panel 1 (left):  Pie chart showing image-count proportions.
    Panel 2 (right): Horizontal bar chart of per-class image counts,
                     colour-coded by group (red = diseased, green = healthy).

    Args:
        summary:     Pre-computed summary dict from :func:`compute_summary`.
        healthy_df:  Healthy class sub-frame.
        diseased_df: Diseased class sub-frame.
        output_path: Destination PNG path.

    Raises:
        OSError: If the figure cannot be saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    healthy_images:  int = summary["healthy"]["image_count"]
    diseased_images: int = summary["diseased"]["image_count"]
    total_images:    int = summary["total_images"]

    labels, counts, colors = _build_bar_data(healthy_df, diseased_df)
    n_classes: int = len(labels)

    # Give each class ~0.30 inches of vertical space; minimum 10 inches.
    bar_height: float = max(10.0, n_classes * 0.30)
    fig, axes = plt.subplots(
        1, 2,
        figsize=(18, bar_height),
        gridspec_kw={"width_ratios": [1, 2.5]},
    )
    fig.patch.set_facecolor("#FAFAFA")

    # ------------------------------------------------------------------ #
    # Panel 1 — Pie chart
    # ------------------------------------------------------------------ #
    ax_pie = axes[0]
    ax_pie.set_facecolor("#FAFAFA")

    pie_values = [diseased_images, healthy_images]
    pie_labels = [
        f"Diseased\n{diseased_images:,}\n"
        f"({summary['diseased']['image_percentage']:.1f}%)",
        f"Healthy\n{healthy_images:,}\n"
        f"({summary['healthy']['image_percentage']:.1f}%)",
    ]
    pie_colors = [_COLOR_DISEASED, _COLOR_HEALTHY]

    ax_pie.pie(
        pie_values,
        labels=pie_labels,
        colors=pie_colors,
        explode=(0.04, 0.04),
        startangle=90,
        textprops={"fontsize": 10, "color": "#2C2C2C"},
        wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
    )
    ax_pie.set_title(
        f"Image Distribution\nTotal: {total_images:,}",
        fontsize=13,
        fontweight="bold",
        color="#1A1A1A",
        pad=14,
    )

    # ------------------------------------------------------------------ #
    # Panel 2 — Horizontal bar chart
    # ------------------------------------------------------------------ #
    ax_bar = axes[1]
    ax_bar.set_facecolor("#FAFAFA")

    y_pos = range(n_classes)
    bars  = ax_bar.barh(y_pos, counts, color=colors, height=0.72, edgecolor="white")

    # Inline count labels to the right of each bar.
    for bar, count in zip(bars, counts):
        ax_bar.text(
            bar.get_width() + total_images * 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center",
            ha="left",
            fontsize=7,
            color="#2C2C2C",
        )

    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(labels, fontsize=7.5)
    ax_bar.set_xlabel("Number of Images", fontsize=11, color="#2C2C2C")
    ax_bar.set_title(
        f"Per-Class Image Counts  —  "
        f"Healthy: {summary['healthy']['class_count']} classes  |  "
        f"Diseased: {summary['diseased']['class_count']} classes",
        fontsize=13,
        fontweight="bold",
        color="#1A1A1A",
        pad=14,
    )
    ax_bar.set_xlim(0, max(counts) * 1.18)
    ax_bar.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax_bar.grid(axis="x", color=_COLOR_GRID, linestyle="--", linewidth=0.7)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=_COLOR_DISEASED, label="Diseased"),
        mpatches.Patch(color=_COLOR_HEALTHY,  label="Healthy"),
    ]
    ax_bar.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=9,
        framealpha=0.85,
        edgecolor="#CCCCCC",
    )

    # ------------------------------------------------------------------ #
    # Figure title and layout
    # ------------------------------------------------------------------ #
    fig.suptitle(
        "PlantGuard AI — Healthy vs. Diseased Class Analysis",
        fontsize=15,
        fontweight="bold",
        color="#1A1A1A",
        y=1.005,
    )
    fig.tight_layout()

    try:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Visualisation saved to: %s", output_path)
    except OSError as exc:
        logger.error("Failed to save plot to '%s': %s", output_path, exc)
        raise
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Console summary
# --------------------------------------------------------------------------- #

def print_summary(summary: Dict) -> None:
    """Print a formatted console summary of the analysis results.

    Args:
        summary: Dictionary returned by :func:`compute_summary`.
    """
    sep = "=" * 56
    print()
    print(sep)
    print("  PLANTGUARD AI — HEALTHY vs. DISEASED ANALYSIS")
    print(sep)
    print(f"  Total classes  : {summary['total_classes']}")
    print(f"  Total images   : {summary['total_images']:,}")
    print(sep)
    print(
        f"  {'Group':<12} {'Classes':>8} {'%':>7}   "
        f"{'Images':>10} {'%':>7}"
    )
    print(f"  {'-'*52}")
    for group in ("healthy", "diseased"):
        g = summary[group]
        print(
            f"  {group.capitalize():<12} "
            f"{g['class_count']:>8} "
            f"{g['class_percentage']:>6.1f}%   "
            f"{g['image_count']:>10,} "
            f"{g['image_percentage']:>6.1f}%"
        )
    print(sep)
    print(f"  Detection keyword : \"{summary['detection_keyword']}\"")
    print(sep)
    print()
    print("  Outputs:")
    print(f"  ✓ {OUTPUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"  ✓ {OUTPUT_PNG.relative_to(PROJECT_ROOT)}")
    print(sep)
    print()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    """Run the full healthy vs. diseased EDA pipeline."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    logger.info("Starting healthy vs. diseased analysis.")

    try:
        df = load_class_distribution(CLASS_DISTRIBUTION_CSV)
        healthy_df, diseased_df = classify_classes(df)
        summary = compute_summary(df, healthy_df, diseased_df)
        save_json(summary, OUTPUT_JSON)
        generate_plot(summary, healthy_df, diseased_df, OUTPUT_PNG)
        print_summary(summary)
        logger.info("Analysis complete.")

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

