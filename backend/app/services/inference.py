"""Shared model loading and thread-safe image inference service."""

from __future__ import annotations

import io
import base64
import json
import logging
import threading
import time
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from ai.configs.config import DEFAULT_MEAN, DEFAULT_STD
from ai.data.preprocessing import load_normalization_stats
from ai.explainability.gradcam import (
    GradCAM,
    TransformerTokenCAM,
    get_target_layer,
)
from ai.models import SUPPORTED_MODELS, build_model
from ai.utils.helpers import get_device, load_checkpoint
from ..core.config import Settings

logger = logging.getLogger(__name__)


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a supported safe image."""


class InferenceService:
    """Load one classifier and serve predictions across API requests."""

    ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
    MAX_IMAGE_PIXELS = 25_000_000

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.status = "not_loaded"
        self.load_error: str | None = None
        self.model: torch.nn.Module | None = None
        self.class_names: list[str] = []
        self.device = get_device()
        self.transform = None
        self.checkpoint_epoch: int | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.status == "ready" and self.model is not None

    def load(self) -> None:
        """Load the mapping, preprocessing, and checkpoint once at startup."""
        self.settings.validate()
        self.status = "loading"
        self.load_error = None
        if self.settings.model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {self.settings.model_name}")
        if not self.settings.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.settings.checkpoint_path}"
            )
        with self.settings.class_mapping_path.open("r", encoding="utf-8") as handle:
            mapping: dict[str, str] = json.load(handle)
        self.class_names = sorted(set(mapping.values()))
        if not self.class_names:
            raise ValueError("Class mapping contains no class names")
        if self.settings.statistics_path.is_file():
            mean, std = load_normalization_stats(self.settings.statistics_path)
        else:
            mean, std = DEFAULT_MEAN, DEFAULT_STD
            logger.warning(
                "Normalization statistics missing; using configured defaults"
            )
        resize_size = round(self.settings.image_size * 256 / 224)
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_size, antialias=True),
                transforms.CenterCrop(self.settings.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        model = build_model(
            self.settings.model_name,
            num_classes=len(self.class_names),
            pretrained=False,
        ).to(self.device)
        self.checkpoint_epoch = load_checkpoint(
            self.settings.checkpoint_path,
            model,
            map_location=self.device,
        )
        model.eval()
        self.model = model
        self.status = "ready"

    def fail(self, error: Exception) -> None:
        """Record startup failure without leaking filesystem details to clients."""
        self.status = "failed"
        self.load_error = type(error).__name__
        logger.exception("Plant Guard model failed to load")

    def predict_bytes(self, contents: bytes, top_k: int = 5) -> dict[str, Any]:
        """Validate an image and return the highest scoring classes."""
        if not self.ready or self.model is None or self.transform is None:
            raise RuntimeError("Inference model is not ready")
        if not contents:
            raise InvalidImageError("Uploaded image is empty")
        if len(contents) > self.settings.max_upload_bytes:
            raise InvalidImageError("Image exceeds the configured upload limit")
        image = self._decode_image(contents)

        width, height = image.size
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self._lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(input_tensor)
                probabilities = logits.softmax(dim=1)[0]
                count = min(max(1, int(top_k)), len(self.class_names))
                scores, indices = probabilities.topk(count)
                scores = scores.cpu()
                indices = indices.cpu()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_time_ms = (time.perf_counter() - start) * 1000
        candidates = [
            {
                "class_name": self.class_names[int(index)],
                "class_index": int(index),
                "confidence": float(score),
            }
            for score, index in zip(scores, indices)
        ]
        best = candidates[0]
        return {
            "model": self.settings.model_name,
            "checkpoint_epoch": self.checkpoint_epoch,
            "predicted_class": best["class_name"],
            "class_index": best["class_index"],
            "confidence": best["confidence"],
            "top_predictions": candidates,
            "image_width": width,
            "image_height": height,
            "inference_time_ms": inference_time_ms,
        }

    def _decode_image(self, contents: bytes) -> Image.Image:
        if not contents:
            raise InvalidImageError("Uploaded image is empty")
        if len(contents) > self.settings.max_upload_bytes:
            raise InvalidImageError("Image exceeds the configured upload limit")
        try:
            with Image.open(io.BytesIO(contents)) as source:
                if source.format not in self.ALLOWED_FORMATS:
                    raise InvalidImageError("Use a JPEG, PNG, or WEBP image")
                if source.width * source.height > self.MAX_IMAGE_PIXELS:
                    raise InvalidImageError(
                        "Image dimensions exceed the supported limit"
                    )
                return source.convert("RGB")
        except InvalidImageError:
            raise
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
            ValueError,
        ) as exc:
            raise InvalidImageError("Uploaded file is not a valid image") from exc

    def explain_bytes(
        self,
        contents: bytes,
        target_class: int | None = None,
        alpha: float = 0.4,
    ) -> dict[str, Any]:
        """Generate a class-specific CAM explanation and base64 PNG overlay."""
        if not self.ready or self.model is None or self.transform is None:
            raise RuntimeError("Inference model is not ready")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in the range [0, 1]")
        if target_class is not None and target_class >= len(self.class_names):
            raise ValueError(
                f"target_class must be less than {len(self.class_names)}"
            )
        image = self._decode_image(contents)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        target_layer = get_target_layer(self.model, self.settings.model_name)
        explainer_type = (
            TransformerTokenCAM
            if self.settings.model_name == "vit_b_16"
            else GradCAM
        )
        overlay_size = round(self.settings.image_size * 256 / 224)
        from torchvision import transforms

        overlay_source = transforms.Compose(
            [
                transforms.Resize(overlay_size, antialias=True),
                transforms.CenterCrop(self.settings.image_size),
            ]
        )(image)
        with self._lock:
            start = time.perf_counter()
            with explainer_type(self.model, target_layer) as explainer:
                heatmap = explainer.generate(input_tensor, target_class)[0]
                overlay = explainer.overlay_heatmap(heatmap, overlay_source, alpha)
            with torch.inference_mode():
                probabilities = self.model(input_tensor).softmax(dim=1)[0]
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_time_ms = (time.perf_counter() - start) * 1000
        predicted_index = int(probabilities.argmax().item())
        explained_index = predicted_index if target_class is None else target_class
        png_buffer = io.BytesIO()
        Image.fromarray(overlay, mode="RGB").save(png_buffer, format="PNG")
        return {
            "model": self.settings.model_name,
            "predicted_class": self.class_names[predicted_index],
            "confidence": float(probabilities[predicted_index].item()),
            "explained_class": self.class_names[explained_index],
            "explained_index": explained_index,
            "image_width": image.width,
            "image_height": image.height,
            "overlay_width": overlay.shape[1],
            "overlay_height": overlay.shape[0],
            "overlay_png_base64": base64.b64encode(png_buffer.getvalue()).decode(
                "ascii"
            ),
            "inference_time_ms": inference_time_ms,
        }
