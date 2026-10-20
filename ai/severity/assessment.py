"""Image and mask based disease-severity assessment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ai.severity.severity import SeverityAssessor


def _read_binary_mask(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Mask not found: {path}")
    with Image.open(path) as source:
        mask_image = source.convert("L")
    if mask_image.size != expected_size:
        raise ValueError(
            f"Mask {path} has size {mask_image.size}; expected {expected_size}"
        )
    return np.asarray(mask_image, dtype=np.uint8) > 0


def measure_affected_area(
    leaf_mask: np.ndarray,
    lesion_mask: np.ndarray,
) -> dict:
    """Measure lesion pixels inside a leaf ROI from aligned binary masks."""
    leaf_values = np.asarray(leaf_mask)
    lesion_values = np.asarray(lesion_mask)
    if leaf_values.ndim != 2 or lesion_values.ndim != 2:
        raise ValueError("leaf_mask and lesion_mask must be two-dimensional")
    if leaf_values.shape != lesion_values.shape:
        raise ValueError("leaf_mask and lesion_mask must have the same shape")
    if np.issubdtype(leaf_values.dtype, np.number) and not np.isfinite(
        leaf_values
    ).all():
        raise ValueError("leaf_mask values must be finite")
    if np.issubdtype(lesion_values.dtype, np.number) and not np.isfinite(
        lesion_values
    ).all():
        raise ValueError("lesion_mask values must be finite")
    leaf = leaf_values.astype(bool)
    lesion = lesion_values.astype(bool)
    leaf_pixels = int(leaf.sum())
    if leaf_pixels == 0:
        raise ValueError("leaf_mask contains no foreground pixels")
    affected = leaf & lesion
    affected_pixels = int(affected.sum())
    return {
        "leaf_pixels": leaf_pixels,
        "affected_pixels": affected_pixels,
        "lesion_pixels_outside_leaf": int((lesion & ~leaf).sum()),
        "affected_area_ratio": affected_pixels / leaf_pixels,
        "affected_mask": affected,
    }


def assess_image(
    image_path: Path,
    leaf_mask_path: Path,
    lesion_mask_path: Path,
    prediction_confidence: float,
    output_path: Path,
    assessor: Optional[SeverityAssessor] = None,
    overlay_alpha: float = 0.55,
) -> dict:
    """Assess an image with supplied masks and write overlay plus JSON report.

    Mask files are grayscale images where every non-black pixel is foreground.
    They must match the source image dimensions.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not 0 <= overlay_alpha <= 1:
        raise ValueError("overlay_alpha must be in the range [0, 1]")
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    height, width = image.shape[:2]
    leaf_mask = _read_binary_mask(leaf_mask_path, (width, height))
    lesion_mask = _read_binary_mask(lesion_mask_path, (width, height))
    measurement = measure_affected_area(leaf_mask, lesion_mask)
    assessor = assessor or SeverityAssessor()
    details = assessor.assess_with_details(
        prediction_confidence,
        measurement["affected_area_ratio"],
    )

    # Highlight only lesion pixels that overlap the leaf ROI.
    overlay = image.astype(np.float32)
    affected_mask = measurement.pop("affected_mask")
    red = np.array([255.0, 25.0, 25.0], dtype=np.float32)
    overlay[affected_mask] = (
        (1 - overlay_alpha) * overlay[affected_mask] + overlay_alpha * red
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)

    report = {
        "image": str(image_path),
        "leaf_mask": str(leaf_mask_path),
        "lesion_mask": str(lesion_mask_path),
        "overlay": str(output_path),
        **measurement,
        **details,
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--leaf-mask", type=Path, required=True)
    parser.add_argument("--lesion-mask", type=Path, required=True)
    parser.add_argument("--prediction-confidence", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    args = parser.parse_args()
    report = assess_image(
        args.image,
        args.leaf_mask,
        args.lesion_mask,
        args.prediction_confidence,
        args.output,
        overlay_alpha=args.overlay_alpha,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
