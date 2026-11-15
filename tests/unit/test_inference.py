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
    img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    img.save(img_path)
    return img_path


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    ckpt_path = tmp_path / "model_weights.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    return ckpt_path


def test_predictor_initialization_with_checkpoint(mock_checkpoint: Path) -> None:
    predictor = FoodPredictor(checkpoint_path=mock_checkpoint, device="cpu")
    assert predictor.device.type == "cpu"
    assert len(predictor.classes) == NUM_CLASSES


def test_predict_image_output_structure(mock_image: Path) -> None:
    predictor = FoodPredictor(device="cpu")
    preds = predictor.predict_image(mock_image, top_k=3)

    assert len(preds) == 3
    for pred in preds:
        assert "class" in pred
        assert "probability" in pred
        assert "index" in pred
        assert pred["class"] in CLASSES
        assert 0.0 <= pred["probability"] <= 1.0
        assert 0 <= pred["index"] < NUM_CLASSES

    probs = [p["probability"] for p in preds]
    assert probs == sorted(probs, reverse=True)


def test_predict_directory(tmp_path: Path) -> None:
    for i in range(2):
        img = Image.new("RGB", (64, 64), color=(i * 20, 50, 100))
        img.save(tmp_path / f"test_{i}.jpg")

    predictor = FoodPredictor(device="cpu")
    dir_results = predictor.predict_directory(tmp_path, top_k=2)

    assert len(dir_results) == 2
    assert "test_0.jpg" in dir_results
    assert "test_1.jpg" in dir_results
    assert len(dir_results["test_0.jpg"]) == 2


def test_cli_argument_parsing() -> None:
    args = parse_args(["--image", "dish.jpg", "--top-k", "5", "--json"])
    assert args.image == Path("dish.jpg")
    assert args.top_k == 5
    assert args.json_output is True


def test_format_text_output() -> None:
    predictions = [
        {"class": "rice", "probability": 0.852, "index": 7},
        {"class": "soup", "probability": 0.104, "index": 9},
    ]
    formatted = format_text_output("sample.jpg", predictions)
    assert "sample.jpg" in formatted
    assert "85.2%" in formatted
    assert "Rice" in formatted


def test_cli_execution_with_image(mock_image: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--image", str(mock_image), "--top-k", "2", "--device", "cpu"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "sample_food.jpg" in captured.out


def test_cli_execution_with_json(mock_image: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--image", str(mock_image), "--top-k", "2", "--device", "cpu", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert "sample_food.jpg" in parsed
    assert len(parsed["sample_food.jpg"]) == 2
