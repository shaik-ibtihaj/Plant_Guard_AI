#!/usr/bin/env python3
"""
02_sample_image_grid.py

PlantGuard AI — Phase 4 EDA: Sample Image Grid Overview

Randomly samples one image per class (preferring class diversity) from the
intermediate dataset directory, then assembles them into a 5×5 grid figure.
Each cell displays the image with its class name as a title. When more than
25 classes exist, 25 classes are sampled without replacement; when fewer exist,
the grid is filled with additional random samples from remaining classes.

Usage:
    python scripts/eda/02_sample_image_grid.py

Outputs:
    reports/sample_grid_overview.png

Author: PlantGuard AI ML Engineering Team
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Primary source: scan the intermediate merged dataset (all 64 classes).
# Falls back to datasets/processed/train if intermediate is empty.
DATASET_DIRS: Tuple[Path, ...] = (
    PROJECT_ROOT / "datasets" / "intermediate",
    PROJECT_ROOT / "datasets" / "processed" / "train",
)

OUTPUT_PNG: Path = PROJECT_ROOT / "reports" / "sample_grid_overview.png"

# Grid dimensions
GRID_ROWS: int = 5
GRID_COLS: int = 5
GRID_TOTAL: int = GRID_ROWS * GRID_COLS  # 25 cells

# Thumbnail size used for display inside each cell (pixels, square).
THUMBNAIL_SIZE: int = 256

# Image file extensions to accept (case-insensitive).
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")

# Deterministic seed for reproducible sampling (set to None for true random).
RANDOM_SEED: Optional[int] = 42

# Healthy keyword — used for colour-coding cell borders.
HEALTHY_KEYWORD: str = "healthy"

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def _configure_logger() -> logging.Logger:
    """Configure and return the module logger with file + console handlers.

    Returns:
        The configured ``plantguard.sample_image_grid`` logger.
    """
    log = logging.getLogger("plantguard.sample_image_grid")
    log.setLevel(logging.DEBUG)
    log.propagate = False

    if log.handlers:
        return log

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file: Path = PROJECT_ROOT / "scripts" / "eda" / "02_sample_image_grid.log"
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
# Data types
# --------------------------------------------------------------------------- #


class SampledImage(NamedTuple):
    """A single sampled image with its resolved class label."""

    path: Path
    class_name: str  # Folder name — already the canonical label.
    is_healthy: bool


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #


def _find_dataset_root() -> Path:
    """Return the first non-empty dataset directory from ``DATASET_DIRS``.

    Iterates the configured candidate roots and returns the first one that
    contains at least one subdirectory with at least one image file.

    Returns:
        An existing dataset root ``Path``.

    Raises:
        FileNotFoundError: If no valid dataset root is found.
    """
    for candidate in DATASET_DIRS:
        if not candidate.exists():
            logger.debug("Candidate root not found, skipping: %s", candidate)
            continue

        class_dirs = [d for d in candidate.iterdir() if d.is_dir()]
        if class_dirs:
            logger.info("Using dataset root: %s (%d class dirs)", candidate, len(class_dirs))
            return candidate

        logger.debug("Candidate root exists but has no class dirs: %s", candidate)

    searched = ", ".join(str(d) for d in DATASET_DIRS)
    raise FileNotFoundError(
        f"No valid dataset root found. Searched:\n  {searched}\n"
        "Ensure the dataset has been merged and placed in one of these locations."
    )


def discover_classes(dataset_root: Path) -> List[Path]:
    """Return a sorted list of class subdirectory paths.

    Only directories that contain at least one supported image file are
    included. This excludes empty placeholders (e.g., ``.gitkeep`` only dirs).

    Args:
        dataset_root: Root directory whose immediate children are class folders.

    Returns:
        Sorted list of non-empty class directory ``Path`` objects.
    """
    logger.info("Discovering class directories under: %s", dataset_root)
    class_dirs: List[Path] = []

    for d in sorted(dataset_root.iterdir()):
        if not d.is_dir():
            continue
        has_image = any(
            f.suffix.lower() in SUPPORTED_EXTENSIONS
            for f in d.iterdir()
            if f.is_file()
        )
        if has_image:
            class_dirs.append(d)
        else:
            logger.debug("Skipping empty class dir: %s", d.name)

    logger.info("Found %d non-empty class directories.", len(class_dirs))
    if not class_dirs:
        raise ValueError(
            f"No class directories with images found under '{dataset_root}'."
        )
    return class_dirs


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def _pick_image_from_class(class_dir: Path, rng: random.Random) -> Optional[Path]:
    """Pick one random supported image from a class directory.

    Args:
        class_dir: Path to a single class subdirectory.
        rng:       Seeded ``random.Random`` instance.

    Returns:
        A ``Path`` to the chosen image, or ``None`` if none exist.
    """
    images: List[Path] = [
        f
        for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not images:
        return None
    return rng.choice(images)


def sample_images(
    class_dirs: List[Path],
    n: int = GRID_TOTAL,
    seed: Optional[int] = RANDOM_SEED,
) -> List[SampledImage]:
    """Sample ``n`` images, preferring one image per unique class.

    Strategy:
    1. Shuffle the class list (reproducibly if ``seed`` is set).
    2. Pick one image per class until ``n`` images are collected or all
       classes are exhausted.
    3. If more images are needed, cycle through classes again picking
       additional images until ``n`` is reached.

    Args:
        class_dirs: List of class directory paths.
        n:          Total number of images to sample (default: 25).
        seed:       RNG seed for reproducibility. ``None`` = non-deterministic.

    Returns:
        List of up to ``n`` :class:`SampledImage` named tuples.

    Raises:
        ValueError: If no images can be sampled from any class.
    """
    rng = random.Random(seed)

    shuffled = class_dirs[:]
    rng.shuffle(shuffled)

    # --- Round 1: one image per class (class diversity pass) ----------------
    sampled: List[SampledImage] = []
    exhausted: List[Path] = []  # classes that returned an image

    for class_dir in shuffled:
        if len(sampled) >= n:
            break
        img_path = _pick_image_from_class(class_dir, rng)
        if img_path is None:
            logger.warning("No images found in class dir: %s", class_dir.name)
            continue
        is_healthy = HEALTHY_KEYWORD.lower() in class_dir.name.lower()
        sampled.append(SampledImage(img_path, class_dir.name, is_healthy))
        exhausted.append(class_dir)
        logger.debug("Sampled [%s] → %s", class_dir.name, img_path.name)

    # --- Round 2: fill remaining slots with extra images from any class ------
    extra_pool = exhausted[:]
    rng.shuffle(extra_pool)
    pool_idx = 0

    while len(sampled) < n and extra_pool:
        class_dir = extra_pool[pool_idx % len(extra_pool)]
        pool_idx += 1
        img_path = _pick_image_from_class(class_dir, rng)
        if img_path is None:
            continue
        is_healthy = HEALTHY_KEYWORD.lower() in class_dir.name.lower()
        sampled.append(SampledImage(img_path, class_dir.name, is_healthy))
        logger.debug(
            "Extra fill [%s] → %s", class_dir.name, img_path.name
        )
        if pool_idx > len(extra_pool) * 10:
            # Safety valve to prevent infinite loop if all classes are tiny.
            logger.warning("Extra-fill loop safety limit reached.")
            break

    if not sampled:
        raise ValueError("No images could be sampled from the dataset.")

    logger.info(
        "Sampled %d images from %d unique classes.",
        len(sampled),
        len({s.class_name for s in sampled}),
    )
    return sampled[:n]


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #


def load_thumbnail(
    path: Path,
    size: int = THUMBNAIL_SIZE,
) -> Optional[Image.Image]:
    """Load an image from disk, convert to RGB, and resize to a square thumbnail.

    Args:
        path: Absolute path to the image file.
        size: Target side length for the square thumbnail (pixels).

    Returns:
        An RGB ``PIL.Image`` resized to ``(size, size)``, or ``None`` if the
        image cannot be opened or decoded.
    """
    try:
        with Image.open(path) as img:
            img.verify()          # Raises on corruption.
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize((size, size), Image.LANCZOS)
            return img
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        logger.warning("Cannot load image '%s': %s", path, exc)
        return None


# --------------------------------------------------------------------------- #
# Label formatting
# --------------------------------------------------------------------------- #

_MAX_LABEL_LEN: int = 28  # Maximum characters before wrapping.


def _format_class_label(class_name: str) -> str:
    """Convert a raw folder name to a human-readable, wrapped cell label.

    Replaces underscores and triple-underscore separators with spaces, then
    splits into at most two lines if the name is long.

    Args:
        class_name: Raw class folder name, e.g.
                    ``"Tomato___Early_blight"`` or
                    ``"Corn_(maize)___Common_rust"``.

    Returns:
        A short, readable label (one or two lines).
    """
    # Split on the canonical separator "___" to get plant + condition.
    if "___" in class_name:
        parts = class_name.split("___", 1)
        plant = parts[0].replace("_", " ").strip("()")
        condition = parts[1].replace("_", " ")
        label = f"{plant}\n{condition}"
    else:
        label = class_name.replace("_", " ")

    # Truncate individual lines that are still too long.
    lines = label.split("\n")
    trimmed = []
    for line in lines:
        if len(line) > _MAX_LABEL_LEN:
            line = line[: _MAX_LABEL_LEN - 1] + "…"
        trimmed.append(line)

    return "\n".join(trimmed)


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #

# Border colours: green = healthy, red = diseased.
_BORDER_HEALTHY: str   = "#27AE60"
_BORDER_DISEASED: str  = "#C0392B"
_BG_COLOR: str         = "#1A1A2E"   # Deep dark background
_TITLE_COLOR: str      = "#E8E8E8"
_LABEL_HEALTHY: str    = "#A8E6B8"   # Muted green for healthy labels
_LABEL_DISEASED: str   = "#F1A1A1"   # Muted red for diseased labels
_LABEL_FALLBACK: str   = "#CCCCCC"   # Fallback if no thumb


def _make_placeholder(size: int = THUMBNAIL_SIZE) -> Image.Image:
    """Return a dark grey placeholder image for cells where loading fails.

    Args:
        size: Side length in pixels.

    Returns:
        A solid-colour RGB PIL Image.
    """
    placeholder = Image.new("RGB", (size, size), color=(50, 50, 60))
    return placeholder


def generate_grid(
    samples: List[SampledImage],
    output_path: Path,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    thumb_size: int = THUMBNAIL_SIZE,
) -> None:
    """Build and save the sample image grid figure.

    Each cell shows a thumbnail image with the class label below it.  Cell
    borders are colour-coded: green for healthy classes, red for diseased.

    Args:
        samples:     List of :class:`SampledImage` entries (length ≤ rows×cols).
        output_path: Destination PNG path.
        rows:        Number of grid rows.
        cols:        Number of grid columns.
        thumb_size:  Pixel size of each square thumbnail.

    Raises:
        OSError: If the figure cannot be written to disk.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cell_size_in: float = 2.4   # Inches per cell (width and height).
    fig_w: float = cols * cell_size_in + 0.4
    fig_h: float = rows * cell_size_in + 1.0  # Extra for suptitle + legend

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.08, "hspace": 0.55},
    )
    fig.patch.set_facecolor(_BG_COLOR)

    # Flatten to a 1-D list for easy indexing.
    axes_flat = axes.flatten()
    n_cells = rows * cols

    logger.info("Rendering %d-cell grid (%d×%d).", n_cells, rows, cols)

    for idx in range(n_cells):
        ax = axes_flat[idx]
        ax.set_facecolor(_BG_COLOR)

        if idx < len(samples):
            sample = samples[idx]
            thumb = load_thumbnail(sample.path, size=thumb_size)

            if thumb is None:
                thumb = _make_placeholder(thumb_size)
                label_color = _LABEL_FALLBACK
                border_color = "#555555"
            else:
                border_color = (
                    _BORDER_HEALTHY if sample.is_healthy else _BORDER_DISEASED
                )
                label_color = (
                    _LABEL_HEALTHY if sample.is_healthy else _LABEL_DISEASED
                )

            ax.imshow(thumb, aspect="equal")
            label_text = _format_class_label(sample.class_name)

            # Class name below the image.
            ax.set_title(
                label_text,
                fontsize=5.5,
                color=label_color,
                pad=3,
                linespacing=1.3,
                fontweight="medium",
            )

            # Coloured border.
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(1.8)

        else:
            # Empty cell — render as a dark blank.
            ax.imshow(
                Image.new("RGB", (thumb_size, thumb_size), color=(25, 25, 35)),
                aspect="equal",
            )
            for spine in ax.spines.values():
                spine.set_edgecolor("#333344")
                spine.set_linewidth(0.8)

        ax.set_xticks([])
        ax.set_yticks([])

    # ----------------------------------------------------------------------- #
    # Figure-level decorations
    # ----------------------------------------------------------------------- #

    n_unique = len({s.class_name for s in samples})
    fig.suptitle(
        f"PlantGuard AI — Dataset Sample Overview\n"
        f"{len(samples)} images · {n_unique} unique classes",
        fontsize=12,
        fontweight="bold",
        color=_TITLE_COLOR,
        y=0.995,
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=_BORDER_HEALTHY,  label="Healthy class"),
        mpatches.Patch(color=_BORDER_DISEASED, label="Diseased class"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        fontsize=8,
        framealpha=0.25,
        edgecolor="#555555",
        facecolor="#2C2C3E",
        labelcolor=_TITLE_COLOR,
        bbox_to_anchor=(0.5, 0.002),
    )

    try:
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=_BG_COLOR,
        )
        logger.info("Grid saved to: %s", output_path)
    except OSError as exc:
        logger.error("Failed to save grid to '%s': %s", output_path, exc)
        raise
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Console summary
# --------------------------------------------------------------------------- #


def print_summary(samples: List[SampledImage], output_path: Path) -> None:
    """Print a formatted console summary.

    Args:
        samples:     List of sampled images.
        output_path: Path to the saved PNG.
    """
    unique_classes = sorted({s.class_name for s in samples})
    healthy_count  = sum(1 for s in samples if s.is_healthy)
    diseased_count = len(samples) - healthy_count

    sep = "=" * 54
    print()
    print(sep)
    print("  PLANTGUARD AI — SAMPLE IMAGE GRID")
    print(sep)
    print(f"  Grid layout     : {GRID_ROWS} × {GRID_COLS} ({GRID_TOTAL} cells)")
    print(f"  Images sampled  : {len(samples)}")
    print(f"  Unique classes  : {len(unique_classes)}")
    print(f"  Healthy cells   : {healthy_count}")
    print(f"  Diseased cells  : {diseased_count}")
    print(sep)
    print("  Classes shown:")
    for cls in unique_classes:
        tag = "✓ healthy " if HEALTHY_KEYWORD.lower() in cls.lower() else "✗ diseased"
        print(f"    [{tag}] {cls}")
    print(sep)
    print()
    print("  Output:")
    print(f"  ✓ {output_path.relative_to(PROJECT_ROOT)}")
    print(sep)
    print()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Run the full sample image grid EDA pipeline."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    logger.info("Starting sample image grid generation.")

    try:
        dataset_root = _find_dataset_root()
        class_dirs   = discover_classes(dataset_root)
        samples      = sample_images(class_dirs, n=GRID_TOTAL, seed=RANDOM_SEED)
        generate_grid(samples, OUTPUT_PNG)
        print_summary(samples, OUTPUT_PNG)
        logger.info("Done.")

    except FileNotFoundError as exc:
        logger.error("Dataset not found: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Data error: %s", exc)
        sys.exit(1)
    except OSError as exc:
        logger.error("I/O error: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 — top-level safety net
        logger.exception("Unexpected failure: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

