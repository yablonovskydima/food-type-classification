from pathlib import Path

import pytest
from PIL import Image

from src.data.stats import compute_mean_std


def test_mean_std_uses_all_pixels(tmp_path: Path) -> None:
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(tmp_path / "black.png")
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(tmp_path / "white.png")

    mean, std = compute_mean_std(tmp_path)

    assert mean == pytest.approx((0.5, 0.5, 0.5))
    assert std == pytest.approx((0.5, 0.5, 0.5))


def test_mean_std_weights_pixels_not_images(tmp_path: Path) -> None:
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(tmp_path / "small.png")
    Image.new("RGB", (3, 1), color=(0, 0, 0)).save(tmp_path / "large.png")

    mean, _ = compute_mean_std(tmp_path)

    assert mean == pytest.approx((0.25, 0.0, 0.0))


def test_mean_std_empty_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_mean_std(tmp_path)
