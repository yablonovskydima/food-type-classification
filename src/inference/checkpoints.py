from pathlib import Path
from typing import Any

import torch

from src.data.constants import CLASSES, DEFAULT_IMAGE_SIZE

DEFAULT_CHECKPOINT = Path("checkpoints") / "resnet18_scratch" / "best.pt"
NORMALIZATIONS = ("food11", "imagenet")


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} does not contain a state dictionary")

    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a dictionary")
    if metadata.get("architecture", "resnet18") != "resnet18":
        raise ValueError("Only ResNet-18 checkpoints are supported")
    if metadata.get("classes", list(CLASSES)) != list(CLASSES):
        raise ValueError("Checkpoint class order differs from the Food-11 classes")

    metadata = dict(metadata)
    config = checkpoint.get("config", {})  # last.pt keeps the training config instead
    if isinstance(config, dict):
        metadata.setdefault("image_size", config.get("image_size"))
        if metadata.get("normalization") is None and "pretrained" in config:
            metadata["normalization"] = "imagenet" if config["pretrained"] else "food11"
    metadata.setdefault("image_size", None)
    metadata.setdefault("normalization", None)

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    return state_dict, metadata


def resolve_preprocessing(
    metadata: dict[str, Any],
    image_size: int | None = None,
    normalization: str | None = None,
    *,
    require_metadata: bool = False,
) -> tuple[int, str]:
    saved_size, saved_normalization = metadata.get("image_size"), metadata.get("normalization")
    if image_size is not None and saved_size is not None and image_size != saved_size:
        raise ValueError(
            f"Image size {image_size} conflicts with the checkpoint's training size {saved_size}"
        )
    if (
        normalization is not None
        and saved_normalization is not None
        and normalization != saved_normalization
    ):
        raise ValueError(
            f"Normalization {normalization!r} conflicts with the checkpoint's "
            f"{saved_normalization!r}"
        )
    size = image_size if image_size is not None else saved_size
    norm = normalization if normalization is not None else saved_normalization
    if require_metadata and (size is None or norm is None):
        raise ValueError(
            "The checkpoint has no preprocessing metadata; provide the image size and "
            "normalization explicitly"
        )
    size = DEFAULT_IMAGE_SIZE if size is None else size
    norm = "food11" if norm is None else norm
    if type(size) is not int or size < 32:
        raise ValueError("Image size must be an integer >= 32")
    if norm not in NORMALIZATIONS:
        raise ValueError(f"Normalization must be one of {NORMALIZATIONS}")
    return size, norm
