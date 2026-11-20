from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.data.constants import CLASSES, FOOD11_MEAN, FOOD11_STD, IMAGENET_MEAN, IMAGENET_STD
from src.data.transforms import get_eval_transforms
from src.inference.checkpoints import load_checkpoint, resolve_preprocessing
from src.models import build_model


class FoodPredictor:
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
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

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
