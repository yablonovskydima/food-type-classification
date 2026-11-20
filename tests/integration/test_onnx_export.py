import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.constants import CLASSES, FOOD11_MEAN, FOOD11_STD
from src.data.transforms import get_eval_transforms
from src.inference.onnx_export import export_checkpoint, load_export_model
from src.inference.onnx_predictor import OnnxFoodPredictor
from src.models import build_model


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("onnx")
    checkpoint = root / "model.pt"
    with torch.random.fork_rng():
        torch.manual_seed(12)
        model = build_model()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "architecture": "resnet18",
                "classes": list(CLASSES),
                "image_size": 64,
                "normalization": "food11",
            },
        },
        checkpoint,
    )
    output = root / "export"
    report = export_checkpoint(checkpoint, output, warmup=1, iterations=2)
    return checkpoint, output, report


def test_real_export_dynamic_batch_parity_and_benchmark(exported) -> None:
    checkpoint, output, report = exported
    assert report["parity"]["passed"]
    assert [item["batch_size"] for item in report["parity"]["batches"]] == [1, 3]
    assert report["metadata"]["classes"] == list(CLASSES)
    assert len(report["metadata"]["checkpoint_sha256"]) == 64
    assert (output / "model.onnx").stat().st_size > 1_000_000
    assert set(path.name for path in output.iterdir()) == {"model.onnx", "export_report.json"}
    saved = json.loads((output / "export_report.json").read_text())
    assert saved == report
    for backend in ("pytorch", "onnxruntime"):
        assert report["benchmark"][backend]["mean_ms"] > 0
        assert report["benchmark"][backend]["images_per_second"] > 0
    with pytest.raises(FileExistsError):
        export_checkpoint(checkpoint, output)


@pytest.mark.parametrize(
    "dimensions,mode",
    [
        ((101, 33), "RGB"),
        ((33, 101), "RGB"),
        ((63, 48), "RGB"),
        ((49, 80), "L"),
        ((80, 49), "RGBA"),
        ((64, 64), "RGB"),
    ],
)
def test_pillow_numpy_preprocessing_matches_torchvision(exported, dimensions, mode) -> None:
    _, output, _ = exported
    random = np.random.default_rng(12)
    image = Image.fromarray(
        random.integers(0, 256, (dimensions[1], dimensions[0], 3), dtype=np.uint8)
    )
    image = image.convert(mode)
    predictor = OnnxFoodPredictor(output / "model.onnx")
    expected = get_eval_transforms(image_size=64)(image.convert("RGB")).numpy()[None]
    np.testing.assert_allclose(predictor.preprocess(image), expected, rtol=0, atol=5e-7)


def test_image_prediction_matches_pytorch(exported) -> None:
    checkpoint, output, _ = exported
    image = Image.new("RGB", (121, 81), (180, 110, 70))
    model, _ = load_export_model(checkpoint, None, None)
    with torch.inference_mode():
        expected = torch.softmax(model(get_eval_transforms(64)(image)[None]), dim=1)[0].numpy()
    predictor = OnnxFoodPredictor(output / "model.onnx")
    predictions = predictor.predict_image(image, top_k=3)
    assert [item["index"] for item in predictions] == np.argsort(-expected)[:3].tolist()
    for item in predictions:
        assert item["probability"] == pytest.approx(expected[item["index"]], abs=1e-4)


def test_standalone_inference_never_imports_torch(exported, tmp_path: Path) -> None:
    _, output, _ = exported
    photo = tmp_path / "dish.png"
    Image.new("RGB", (80, 65), "orange").save(photo)
    script = """
import importlib.abc
import sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "torchvision"}:
            raise RuntimeError("PyTorch must not be imported by ONNX inference")
sys.meta_path.insert(0, BlockTorch())
import runpy
sys.argv = ["predict.py", *sys.argv[1:]]
try:
    runpy.run_path("scripts/predict.py", run_name="__main__")
except SystemExit as exit_:
    assert exit_.code == 0, exit_.code
assert "torch" not in sys.modules
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "--model",
            str(output / "model.onnx"),
            "--image",
            str(photo),
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["dish.png"]) == 3


def test_conflicting_preprocessing_is_rejected(exported) -> None:
    checkpoint, _, _ = exported
    with pytest.raises(ValueError, match="conflicts with the checkpoint's training size"):
        load_export_model(checkpoint, 128, None)
    with pytest.raises(ValueError, match="Normalization 'imagenet' conflicts"):
        load_export_model(checkpoint, None, "imagenet")


def test_untagged_and_training_state_checkpoints(exported, tmp_path: Path) -> None:
    checkpoint, _, _ = exported
    weights = torch.load(checkpoint, weights_only=True)["model_state_dict"]
    raw_path = tmp_path / "raw.pt"
    torch.save(weights, raw_path)
    with pytest.raises(ValueError, match="no preprocessing metadata"):
        load_export_model(raw_path, None, None)
    _, metadata = load_export_model(raw_path, 128, "imagenet")
    assert metadata["normalization"] == "imagenet"
    state_path = tmp_path / "last.pt"
    torch.save(
        {"model_state_dict": weights, "config": {"image_size": 64, "pretrained": False}}, state_path
    )
    _, metadata = load_export_model(state_path, None, None)
    assert metadata["mean"] == list(FOOD11_MEAN)
    assert metadata["std"] == list(FOOD11_STD)


def test_failed_export_does_not_publish_or_change_threads(
    exported, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _, _ = exported
    threads = torch.get_num_threads()

    def fail(*args, **kwargs):
        raise RuntimeError("simulated exporter failure")

    monkeypatch.setattr(torch.onnx, "export", fail)
    destination = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="simulated"):
        export_checkpoint(checkpoint, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".onnx-*"))
    assert torch.get_num_threads() == threads


def test_invalid_runtime_inputs(exported) -> None:
    _, output, _ = exported
    predictor = OnnxFoodPredictor(output / "model.onnx")
    for invalid in (
        np.zeros((1, 3, 64, 64), dtype=np.float64),
        np.zeros((1, 3, 32, 32), dtype=np.float32),
        np.full((1, 3, 64, 64), np.nan, dtype=np.float32),
    ):
        with pytest.raises(ValueError, match="float32 NCHW"):
            predictor.predict_logits(invalid)


def test_export_cli_reports_missing_checkpoint(tmp_path: Path, capsys, load_script) -> None:
    main = load_script("export_onnx").main
    assert (
        main(
            ["--checkpoint", str(tmp_path / "missing.pt"), "--output-dir", str(tmp_path / "export")]
        )
        == 1
    )
    assert "Export failed" in capsys.readouterr().err
