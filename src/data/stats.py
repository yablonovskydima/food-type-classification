from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def compute_mean_std(
    image_dir: str | Path,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    files = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    total = np.zeros(3)
    total_sq = np.zeros(3)
    num_pixels = 0
    for path in files:
        with Image.open(path) as img:
            pixels = np.asarray(img.convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255.0
        total += pixels.sum(axis=0)
        total_sq += np.square(pixels).sum(axis=0)
        num_pixels += pixels.shape[0]

    mean = total / num_pixels
    std = np.sqrt(np.maximum(total_sq / num_pixels - np.square(mean), 0.0))
    return tuple(mean.round(4).tolist()), tuple(std.round(4).tolist())
