from pathlib import Path

import pytest
import torch

from app.api.model import create_predictor
from src.data.constants import CLASSES


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "CHECKPOINT_PATH",
        "REQUIRE_CHECKPOINT",
        "IMAGE_SIZE",
        "NORMALIZATION",
        "USE_PRETRAINED",
        "DEVICE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_container_requires_checkpoint(monkeypatch) -> None:
    monkeypatch.setenv("REQUIRE_CHECKPOINT", "1")
    with pytest.raises(ValueError, match="CHECKPOINT_PATH"):
        create_predictor()


@pytest.mark.parametrize(
    "payload,expected_size,expected_imagenet",
    [
        ({"metadata": {"image_size": 128, "normalization": "imagenet"}}, 128, True),
        ({"config": {"image_size": 224, "pretrained": False}}, 224, False),
        ({}, 224, False),
    ],
)
def test_checkpoint_preprocessing_is_used(
    tmp_path: Path, monkeypatch, payload, expected_size, expected_imagenet
) -> None:
    checkpoint = tmp_path / "model.pt"
    torch.save(payload, checkpoint)
    monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setattr("app.api.model.FoodPredictor", lambda **kwargs: kwargs)
    settings = create_predictor()
    assert settings["image_size"] == expected_size
    assert settings["use_imagenet_stats"] == expected_imagenet
    assert settings["device"] == "cpu"


@pytest.mark.parametrize("variable,value", [("IMAGE_SIZE", "224"), ("NORMALIZATION", "imagenet")])
def test_conflicting_settings_are_rejected(tmp_path: Path, monkeypatch, variable, value) -> None:
    checkpoint = tmp_path / "model.pt"
    torch.save({"metadata": {"image_size": 128, "normalization": "food11"}}, checkpoint)
    monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match="conflicts"):
        create_predictor()


def test_wrong_class_order_is_rejected(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pt"
    torch.save({"metadata": {"classes": list(reversed(CLASSES))}}, checkpoint)
    monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint))
    with pytest.raises(ValueError, match="Food-11 order"):
        create_predictor()
