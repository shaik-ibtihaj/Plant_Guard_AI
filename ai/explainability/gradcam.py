"""Grad-CAM generation and a command-line image explanation tool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

from ai.configs.config import DEFAULT_MEAN, DEFAULT_STD, DataConfig
from ai.data.preprocessing import build_eval_transform, load_normalization_stats
from ai.models import SUPPORTED_MODELS, build_model
from ai.utils.helpers import get_device, load_checkpoint


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the final convolutional or transformer feature layer."""
    if model_name == "baseline_cnn":
        return model.features[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "vit_b_16":
        # A patch token at the encoder's final output does not directly feed
        # the class head, so capture one block earlier while all tokens still
        # participate in self-attention.
        return model.encoder.layers[-2]
    raise ValueError(f"Unknown model '{model_name}'")


class GradCAM:
    """Compute a class-specific Grad-CAM map for a convolutional target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handle = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(self, _module, _inputs, output) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Grad-CAM target layer must produce (N, C, H, W) output")
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Return a normalized heatmap per image in the input batch.

        If ``target_class`` is omitted, each image's predicted class is explained.
        For an explicit class, that same class is explained for every batch item.
        """
        if input_tensor.ndim != 4:
            raise ValueError(
                "input_tensor must have shape (batch, channels, height, width)"
            )
        if target_class is not None and target_class < 0:
            raise ValueError("target_class must be non-negative")

        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None
        try:
            with torch.enable_grad():
                logits = self.model(input_tensor.requires_grad_(True))
                if logits.ndim != 2:
                    raise ValueError(
                        "model must return logits with shape (batch, classes)"
                    )
                classes = (
                    logits.argmax(dim=1)
                    if target_class is None
                    else torch.full(
                        (logits.shape[0],),
                        target_class,
                        dtype=torch.long,
                        device=logits.device,
                    )
                )
                if torch.any(classes >= logits.shape[1]):
                    raise ValueError("target_class is outside the model output range")
                logits.gather(1, classes[:, None]).sum().backward()

            if self.activations is None or self.gradients is None:
                raise RuntimeError(
                    "Grad-CAM hooks did not capture activations and gradients"
                )
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            maps = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
            maps = F.interpolate(
                maps,
                size=input_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            flat = maps.flatten(start_dim=1)
            minimum = flat.min(dim=1).values[:, None, None]
            maximum = flat.max(dim=1).values[:, None, None]
            maps = (maps - minimum) / (maximum - minimum).clamp_min(1e-8)
            return maps.detach().cpu().numpy().astype(np.float32)
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: Image.Image | np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Blend a JET heatmap over an RGB PIL image or H×W×3 array."""
        return overlay_heatmap(heatmap, original_image, alpha)

    def close(self) -> None:
        """Remove the model hook when this explainer is no longer needed."""
        self._handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class TransformerTokenCAM(GradCAM):
    """Gradient-weighted token attribution for torchvision Vision Transformers."""

    def _capture_activation(self, _module, _inputs, output) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise ValueError("Transformer target layer must produce (N, tokens, dim)")
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._capture_gradient)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Return a patch-grid attribution map for each image in a batch."""
        if input_tensor.ndim != 4:
            raise ValueError(
                "input_tensor must have shape (batch, channels, height, width)"
            )
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None
        try:
            with torch.enable_grad():
                logits = self.model(input_tensor.requires_grad_(True))
                if logits.ndim != 2:
                    raise ValueError(
                        "model must return logits with shape (batch, classes)"
                    )
                classes = (
                    logits.argmax(dim=1)
                    if target_class is None
                    else torch.full(
                        (logits.shape[0],),
                        target_class,
                        dtype=torch.long,
                        device=logits.device,
                    )
                )
                if target_class is not None and not 0 <= target_class < logits.shape[1]:
                    raise ValueError("target_class is outside the model output range")
                logits.gather(1, classes[:, None]).sum().backward()
            if self.activations is None or self.gradients is None:
                raise RuntimeError(
                    "Token CAM hooks did not capture activations and gradients"
                )
            # Torchvision ViT prepends one class token; the remaining tokens form
            # a square patch grid.
            token_count = self.activations.shape[1] - 1
            grid_size = int(token_count**0.5)
            if grid_size * grid_size != token_count:
                raise ValueError("Transformer patch tokens do not form a square grid")
            # Absolute gradient-times-activation keeps the map informative when
            # all patch contributions happen to be negative for a class.
            token_scores = (self.activations * self.gradients).sum(dim=-1).abs()
            maps = token_scores[:, 1:].reshape(-1, 1, grid_size, grid_size)
            maps = F.interpolate(
                maps,
                size=input_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            flat = maps.flatten(start_dim=1)
            minimum = flat.min(dim=1).values[:, None, None]
            maximum = flat.max(dim=1).values[:, None, None]
            maps = (maps - minimum) / (maximum - minimum).clamp_min(1e-8)
            return maps.detach().cpu().numpy().astype(np.float32)
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)


def overlay_heatmap(
    heatmap: np.ndarray,
    original_image: Image.Image | np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a JET heatmap over an RGB PIL image or H×W×3 array."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in the range [0, 1]")
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("heatmap must be a finite two-dimensional array")
    values = np.clip(values, 0.0, 1.0)
    if isinstance(original_image, Image.Image):
        image = np.asarray(original_image.convert("RGB"))
    else:
        image = np.asarray(original_image)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("original_image must have shape (height, width, 3 or 4)")
        image = image[:, :, :3]
    image = np.asarray(Image.fromarray(image.astype(np.uint8)).convert("RGB"))
    if values.shape != image.shape[:2]:
        resized = Image.fromarray(np.uint8(values * 255)).resize(
            (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR
        )
        values = np.asarray(resized, dtype=np.float32) / 255.0
    import matplotlib

    colors = matplotlib.colormaps["jet"](values)[..., :3]
    blended = (1 - alpha) * (image / 255.0) + alpha * colors
    return np.clip(blended * 255, 0, 255).astype(np.uint8)


def explain_image(
    model_name: str,
    checkpoint_path: Path,
    image_path: Path,
    output_path: Path,
    data_config: DataConfig = DataConfig(),
    target_class: Optional[int] = None,
    alpha: float = 0.4,
) -> dict:
    """Load a checkpoint, explain one image, and save an RGB overlay and JSON."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{model_name}'")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    mean, std = (
        load_normalization_stats(data_config.statistics_path)
        if data_config.statistics_path.is_file()
        else (DEFAULT_MEAN, DEFAULT_STD)
    )
    transform = build_eval_transform(data_config.image_size, mean, std)
    device = get_device()
    class_names = sorted(
        path.name
        for path in (data_config.data_dir / "train").iterdir()
        if path.is_dir()
    )
    if not class_names:
        raise FileNotFoundError(
            f"No training classes found in {data_config.data_dir / 'train'}"
        )
    model = build_model(model_name, len(class_names), pretrained=False).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    resize_size = round(data_config.image_size * 256 / 224)
    overlay_image = transforms.Compose(
        [
            transforms.Resize(resize_size, antialias=True),
            transforms.CenterCrop(data_config.image_size),
        ]
    )(original)
    input_tensor = transform(original).unsqueeze(0).to(device)
    layer = get_target_layer(model, model_name)
    explainer_type = TransformerTokenCAM if model_name == "vit_b_16" else GradCAM
    with explainer_type(model, layer) as explainer:
        heatmap = explainer.generate(input_tensor, target_class)[0]
        overlay = explainer.overlay_heatmap(heatmap, overlay_image, alpha)
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = logits.softmax(dim=1)[0]
            predicted_index = int(probabilities.argmax().item())
            explained_index = predicted_index if target_class is None else target_class
            result = {
                "model": model_name,
                "image": str(image_path),
                "predicted_class": class_names[predicted_index],
                "predicted_index": predicted_index,
                "confidence": float(probabilities[predicted_index].item()),
                "explained_class": class_names[explained_index],
                "explained_index": explained_index,
                "overlay": str(output_path),
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(output_path)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument(
        "--statistics", type=Path, default=Path("datasets/metadata/rgb_stats.json")
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--target-class", type=int)
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()
    result = explain_image(
        args.model,
        args.checkpoint,
        args.image,
        args.output,
        DataConfig(
            data_dir=args.data_dir,
            statistics_path=args.statistics,
            image_size=args.image_size,
        ),
        args.target_class,
        args.alpha,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
