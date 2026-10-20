"""Environment-backed API settings."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _path_setting(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


def _model_name() -> str:
    configured = os.getenv("PLANT_GUARD_MODEL")
    if configured:
        return configured
    manifest_path = _path_setting(
        "PLANT_GUARD_MODEL_MANIFEST", "models/production_model.json"
    )
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            return str(json.load(handle)["model"])
    except (OSError, KeyError, ValueError, TypeError):
        return "efficientnet_b0"


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, with paths resolved from the repository root."""

    project_root: Path = PROJECT_ROOT
    app_name: str = "Plant Guard AI API"
    version: str = "0.2.0"
    environment: str = os.getenv("APP_ENV", "development")
    model_name: str = _model_name()
    checkpoint_path: Path = _path_setting(
        "PLANT_GUARD_CHECKPOINT", "models/best_model.pth"
    )
    class_mapping_path: Path = _path_setting(
        "PLANT_GUARD_CLASS_MAPPING", "datasets/metadata/class_mapping.json"
    )
    statistics_path: Path = _path_setting(
        "PLANT_GUARD_RGB_STATS", "datasets/metadata/rgb_stats.json"
    )
    database_url: str = os.getenv(
        "DATABASE_URL", "sqlite:///./data/plant_guard.db"
    )
    api_key: str | None = os.getenv("PLANT_GUARD_API_KEY")
    rate_limit_per_minute: int = int(
        os.getenv("PLANT_GUARD_RATE_LIMIT_PER_MINUTE", "60")
    )
    image_size: int = int(os.getenv("PLANT_GUARD_IMAGE_SIZE", "224"))
    max_upload_bytes: int = int(os.getenv("PLANT_GUARD_MAX_UPLOAD_BYTES", "10485760"))
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "PLANT_GUARD_CORS_ORIGINS", "http://localhost:3000"
        ).split(",")
        if origin.strip()
    )

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ValueError("PLANT_GUARD_IMAGE_SIZE must be positive")
        if self.max_upload_bytes <= 0:
            raise ValueError("PLANT_GUARD_MAX_UPLOAD_BYTES must be positive")
        if self.rate_limit_per_minute <= 0:
            raise ValueError("PLANT_GUARD_RATE_LIMIT_PER_MINUTE must be positive")
        if self.environment.lower() == "production":
            if not self.api_key or len(self.api_key) < 32:
                raise ValueError(
                    "Production requires a PLANT_GUARD_API_KEY of at least 32 characters"
                )
