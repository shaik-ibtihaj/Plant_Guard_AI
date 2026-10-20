"""Rule-based severity estimates from measured affected leaf area."""

from __future__ import annotations

import math
from typing import Optional, Sequence


class SeverityAssessor:
    """Map affected-area measurements to transparent severity categories.

    The thresholds are initial project heuristics and require agronomic
    validation. Classifier confidence is validated and reported, but does not
    determine biological severity.
    """

    SEVERITY_LEVELS = ("Healthy", "Mild", "Moderate", "Severe", "Critical")
    UNKNOWN = "Unknown"
    RECOMMENDATIONS = {
        "Healthy": "No disease treatment indicated; continue routine monitoring.",
        "Mild": "Isolate affected leaves if practical and monitor for progression.",
        "Moderate": (
            "Remove heavily affected tissue and consult local plant-care guidance."
        ),
        "Severe": (
            "Isolate the plant and seek advice from an extension service or agronomist."
        ),
        "Critical": "Quarantine the plant promptly and seek expert assessment.",
        "Unknown": "Measure affected leaf area before making a severity decision.",
    }

    def __init__(self, area_thresholds: Optional[Sequence[float]] = None) -> None:
        raw_thresholds = tuple(
            (0.05, 0.20, 0.40)
            if area_thresholds is None
            else area_thresholds
        )
        if len(raw_thresholds) != 3:
            raise ValueError("area_thresholds must contain exactly three boundaries")
        try:
            thresholds = tuple(float(value) for value in raw_thresholds)
        except (TypeError, ValueError) as exc:
            raise ValueError("area thresholds must be numeric") from exc
        if any(not math.isfinite(value) or not 0 < value < 1 for value in thresholds):
            raise ValueError("area thresholds must be finite values between 0 and 1")
        if any(left >= right for left, right in zip(thresholds, thresholds[1:])):
            raise ValueError("area thresholds must be strictly increasing")
        self.area_thresholds = thresholds

    @staticmethod
    def _validate_unit_interval(value: float, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number between 0 and 1") from exc
        if not math.isfinite(result) or not 0 <= result <= 1:
            raise ValueError(f"{name} must be a finite number between 0 and 1")
        return result

    def assess(
        self,
        prediction_confidence: float,
        affected_area_ratio: Optional[float] = None,
    ) -> str:
        """Return severity from affected-area ratio, or Unknown if not measured."""
        self._validate_unit_interval(prediction_confidence, "prediction_confidence")
        if affected_area_ratio is None:
            return self.UNKNOWN
        area = self._validate_unit_interval(affected_area_ratio, "affected_area_ratio")
        if area == 0:
            return "Healthy"
        for threshold, level in zip(self.area_thresholds, self.SEVERITY_LEVELS[1:-1]):
            if area <= threshold:
                return level
        return "Critical"

    def assess_with_details(
        self,
        prediction_confidence: float,
        affected_area_ratio: Optional[float] = None,
    ) -> dict:
        """Return the severity, inputs, recommendation, and method limitation."""
        confidence = self._validate_unit_interval(
            prediction_confidence, "prediction_confidence"
        )
        area = (
            None
            if affected_area_ratio is None
            else self._validate_unit_interval(
                affected_area_ratio, "affected_area_ratio"
            )
        )
        level = self.assess(confidence, area)
        return {
            "severity": level,
            "prediction_confidence": confidence,
            "affected_area_ratio": area,
            "recommendation": self.get_recommendation(level),
            "method": "affected_area_thresholds",
            "validated_for_field_use": False,
        }

    def get_recommendation(self, severity_level: str) -> str:
        """Return a conservative next-step suggestion for a severity category."""
        try:
            return self.RECOMMENDATIONS[severity_level]
        except KeyError as exc:
            valid = ", ".join((*self.SEVERITY_LEVELS, self.UNKNOWN))
            raise ValueError(
                f"Unknown severity level '{severity_level}'. Choose: {valid}"
            ) from exc
