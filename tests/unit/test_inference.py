import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.constants import CLASSES, NUM_CLASSES
from src.inference.cli import format_text_output, main, parse_args
from src.inference.predictor import FoodPredictor
from src.models import build_model


@pytest.fixture
def mock_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "sample_food.jpg"
    Image.new("RGB", (100, 100), color=(200, 100, 50)).save(img_path)
    return img_path


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    ckpt_path = tmp_path / "model_weights.pt"
    torch.save({"model_state_dict": build_model(num_classes=NUM_CLASSES).state_dict()}, ckpt_path)
    return ckpt_path


@pytest.fixture
def tagged_checkpoint(tmp_path: Path) -> Path:
    # Same layout as best.pt written by the training loop.
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": build_model().state_dict(),
            "metadata": {"image_size": 128, "normalization": "imagenet", "epoch": 7},
        },
        path,
    )
    return path


def test_predictor_initialization_with_checkpoint(mock_checkpoint: Path) -> None:
    predictor = FoodPredictor(checkpoint_path=mock_checkpoint, device="cpu")
    assert predictor.device.type == "cpu"
    assert len(predictor.classes) == NUM_CLASSES
    # Untagged checkpoints fall back to the defaults.
    assert (predictor.image_size, predictor.normalization) == (224, "food11")


def test_predictor_uses_checkpoint_preprocessing(tagged_checkpoint: Path) -> None:
    predictor = FoodPredictor(checkpoint_path=tagged_checkpoint, device="cpu")
    assert (predictor.image_size, predictor.normalization) == (128, "imagenet")
    assert predictor.metadata["epoch"] == 7
    assert predictor.transform(Image.new("RGB", (300, 200))).shape == (3, 128, 128)


def test_predictor_rejects_conflicting_preprocessing(tagged_checkpoint: Path) -> None:
    with pytest.raises(ValueError, match="conflicts"):
        FoodPredictor(checkpoint_path=tagged_checkpoint, device="cpu", image_size=224)
    with pytest.raises(ValueError, match="conflicts"):
        FoodPredictor(checkpoint_path=tagged_checkpoint, device="cpu", use_imagenet_stats=False)


def test_predictor_requires_a_model_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        FoodPredictor(device="cpu")


def test_predict_image_output_structure(mock_checkpoint: Path, mock_image: Path) -> None:
    predictor = FoodPredictor(checkpoint_path=mock_checkpoint, device="cpu")
    preds = predictor.predict_image(mock_image, top_k=3)

    assert len(preds) == 3
    for pred in preds:
        assert pred["class"] in CLASSES
        assert 0.0 <= pred["probability"] <= 1.0
        assert 0 <= pred["index"] < NUM_CLASSES
    probs = [p["probability"] for p in preds]
    assert probs == sorted(probs, reverse=True)


def test_predict_directory(mock_checkpoint: Path, tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for i in range(2):
        Image.new("RGB", (64, 64), color=(i * 20, 50, 100)).save(images / f"test_{i}.jpg")

    predictor = FoodPredictor(checkpoint_path=mock_checkpoint, device="cpu")
    dir_results = predictor.predict_directory(images, top_k=2)

    assert set(dir_results) == {"test_0.jpg", "test_1.jpg"}
    assert len(dir_results["test_0.jpg"]) == 2


def test_cli_argument_parsing() -> None:
    args = parse_args(["--image", "dish.jpg", "--top-k", "5", "--json"])
    assert args.image == Path("dish.jpg")
    assert args.top_k == 5
    assert args.json_output is True
    assert args.checkpoint == Path("checkpoints") / "resnet18_scratch" / "best.pt"


def test_format_text_output() -> None:
    predictions = [
        {"class": "rice", "probability": 0.852, "index": 7},
        {"class": "soup", "probability": 0.104, "index": 9},
    ]
    formatted = format_text_output("sample.jpg", predictions)
    assert "sample.jpg" in formatted
    assert "85.2%" in formatted
    assert "Rice" in formatted


def test_cli_execution_with_image(
    mock_checkpoint: Path, mock_image: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--image", str(mock_image), "--checkpoint", str(mock_checkpoint), "--device", "cpu"]
    assert main([*args, "--top-k", "2"]) == 0
    assert "sample_food.jpg" in capsys.readouterr().out


def test_cli_execution_with_json(
    mock_checkpoint: Path, mock_image: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--image", str(mock_image), "--checkpoint", str(mock_checkpoint), "--device", "cpu"]
    assert main([*args, "--top-k", "2", "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert len(parsed["sample_food.jpg"]) == 2


def test_cli_reports_missing_checkpoint(
    mock_image: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--image", str(mock_image), "--checkpoint", str(tmp_path / "missing.pt")]
    assert main(args) == 1
    assert "could not load the model" in capsys.readouterr().err
