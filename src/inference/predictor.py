from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.data.constants import (
    CLASSES,
    DEFAULT_IMAGE_SIZE,
    FOOD11_MEAN,
    FOOD11_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
)
from src.data.transforms import get_eval_transforms
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
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model: torch.nn.Module | None = None,
        device: str | torch.device | None = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
        use_imagenet_stats: bool = False,
    ) -> None:
        if device is None:
            self.device = get_default_device()
        else:
            self.device = torch.device(device)

        if model is not None:
            self.model = model
        elif checkpoint_path is not None:
            self.model = self._load_checkpoint(checkpoint_path)
        else:
            self.model = build_model(num_classes=NUM_CLASSES, pretrained=False)

        self.model.to(self.device)
        self.model.eval()

        mean = IMAGENET_MEAN if use_imagenet_stats else FOOD11_MEAN
        std = IMAGENET_STD if use_imagenet_stats else FOOD11_STD
        self.transform = get_eval_transforms(image_size=image_size, mean=mean, std=std)
        self.classes = CLASSES

    def _load_checkpoint(self, path: str | Path) -> torch.nn.Module:
        checkpoint_file = Path(path)
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=True)

        model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()

        model.load_state_dict(state_dict)
        return model

    def predict_image(
        self,
        image: Image.Image | str | Path,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if isinstance(image, str | Path):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError(f"Expected Image.Image or path, got {type(image)}")

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]

        top_k = min(max(1, top_k), len(self.classes))
        top_probs, top_indices = torch.topk(probs, k=top_k)

        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist(), strict=True):
            results.append({
                "class": self.classes[idx],
                "probability": round(prob, 4),
                "index": idx,
            })

        return results

    def predict_directory(
        self,
        directory_path: str | Path,
        top_k: int = 3,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
    ) -> dict[str, list[dict[str, Any]]]:
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        predictions: dict[str, list[dict[str, Any]]] = {}
        valid_exts = {ext.lower() for ext in extensions}

        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in valid_exts:
                predictions[file_path.name] = self.predict_image(file_path, top_k=top_k)

        return predictions
