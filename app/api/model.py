"""Load the API model with explicit container requirements and training metadata."""

import os
from pathlib import Path

import torch

from src.data.constants import CLASSES, DEFAULT_IMAGE_SIZE
from src.inference.predictor import FoodPredictor


def create_predictor() -> FoodPredictor:
    checkpoint = os.getenv("CHECKPOINT_PATH", "").strip()
    if os.getenv("REQUIRE_CHECKPOINT") == "1" and not checkpoint:
        raise ValueError("CHECKPOINT_PATH must point to a trained checkpoint")
    pretrained = os.getenv("USE_PRETRAINED", "0") == "1"
    image_size = None
    normalization = None
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("Expected a state-dictionary checkpoint")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Checkpoint metadata must be a dictionary")
        if metadata.get("classes", list(CLASSES)) != list(CLASSES):
            raise ValueError("Checkpoint classes differ from the canonical Food-11 order")
        if metadata.get("architecture", "resnet18") != "resnet18":
            raise ValueError("Only ResNet-18 checkpoints are supported")
        image_size = metadata.get("image_size")
        normalization = metadata.get("normalization")
        config = payload.get("config", {})
        if isinstance(config, dict):
            image_size = image_size if image_size is not None else config.get("image_size")
            if normalization is None and "pretrained" in config:
                normalization = "imagenet" if config["pretrained"] else "food11"
    if value := os.getenv("IMAGE_SIZE", "").strip():
        requested_size = int(value)
        if image_size is not None and requested_size != image_size:
            raise ValueError("IMAGE_SIZE conflicts with the checkpoint's training settings")
        image_size = requested_size
    if value := os.getenv("NORMALIZATION", "").strip():
        if normalization is not None and value != normalization:
            raise ValueError("NORMALIZATION conflicts with the checkpoint's training settings")
        normalization = value
    image_size = DEFAULT_IMAGE_SIZE if image_size is None else image_size
    normalization = normalization or ("imagenet" if pretrained else "food11")
    if type(image_size) is not int or image_size <= 0:
        raise ValueError("IMAGE_SIZE must be a positive integer")
    if normalization not in {"food11", "imagenet"}:
        raise ValueError("NORMALIZATION must be food11 or imagenet")
    kwargs = {
        "device": os.getenv("DEVICE") or None,
        "image_size": image_size,
        "use_imagenet_stats": normalization == "imagenet",
    }
    if pretrained and not checkpoint:
        from src.models import build_model

        return FoodPredictor(model=build_model(pretrained=True), **kwargs)
    return FoodPredictor(checkpoint_path=checkpoint or None, **kwargs)
