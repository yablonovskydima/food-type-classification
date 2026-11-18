from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.data.constants import CLASSES, FOOD11_MEAN, FOOD11_STD, IMAGENET_MEAN, IMAGENET_STD
from src.data.transforms import get_eval_transforms
from src.inference.checkpoints import load_checkpoint, resolve_preprocessing
from src.models import build_model


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, 1, 3, 3, device="cuda")
            weight = torch.zeros(1, 1, 3, 3, device="cuda")
            torch.nn.functional.conv2d(test_tensor, weight)
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


class FoodPredictor:
    """Top-k food predictions from a trained checkpoint (or an already built model).

    Input size and normalisation are taken from the checkpoint metadata; explicit values are
    only needed for checkpoints without metadata and must not contradict it.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model: torch.nn.Module | None = None,
        device: str | torch.device | None = None,
        image_size: int | None = None,
        use_imagenet_stats: bool | None = None,
    ) -> None:
        if (checkpoint_path is None) == (model is None):
            raise ValueError("Provide exactly one of checkpoint_path or model")
        self.device = get_default_device() if device is None else torch.device(device)

        self.metadata: dict[str, Any] = {}
        if checkpoint_path is not None:
            state_dict, self.metadata = load_checkpoint(checkpoint_path, map_location=self.device)
            model = build_model(pretrained=False)
            model.load_state_dict(state_dict)
        self.model = model.to(self.device).eval()

        normalization = (
            None if use_imagenet_stats is None else ("imagenet" if use_imagenet_stats else "food11")
        )
        self.image_size, self.normalization = resolve_preprocessing(
            self.metadata, image_size, normalization
        )
        mean, std = (
            (IMAGENET_MEAN, IMAGENET_STD)
            if self.normalization == "imagenet"
            else (FOOD11_MEAN, FOOD11_STD)
        )
        self.transform = get_eval_transforms(image_size=self.image_size, mean=mean, std=std)
        self.classes = CLASSES

    def predict_image(
        self,
        image: Image.Image | str | Path,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if isinstance(image, str | Path):
            with Image.open(image) as source:
                pil_image = source.convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError(f"Expected Image.Image or path, got {type(image)}")

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=-1)[0]

        top_k = min(max(1, top_k), len(self.classes))
        top_probs, top_indices = torch.topk(probs, k=top_k)
        return [
            {"class": self.classes[idx], "probability": round(prob, 4), "index": idx}
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist(), strict=True)
        ]

    def predict_directory(
        self,
        directory_path: str | Path,
        top_k: int = 3,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
    ) -> dict[str, list[dict[str, Any]]]:
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        valid_exts = {ext.lower() for ext in extensions}
        return {
            file_path.name: self.predict_image(file_path, top_k=top_k)
            for file_path in sorted(dir_path.iterdir())
            if file_path.is_file() and file_path.suffix.lower() in valid_exts
        }
