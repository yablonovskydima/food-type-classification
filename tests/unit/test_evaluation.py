"""Synthetic checks: metric arithmetic, gradient attribution, and report integration."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.data.constants import CLASSES, NUM_CLASSES
from src.evaluation.cli import main, parse_args
from src.evaluation.gradcam import grad_cam
from src.evaluation.metrics import compute_metrics
from src.evaluation.pipeline import evaluate_checkpoint


def test_metrics_known_predictions() -> None:
    targets = np.array([0, 1, 2, 3])
    probabilities = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.2, 0.1],
            [0.4, 0.3, 0.2, 0.1],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    metrics = compute_metrics(targets, probabilities, ("a", "b", "c", "d"))
    assert metrics["accuracy"] == 0.5
    assert metrics["top3_accuracy"] == 0.75
    assert metrics["macro_f1"] == pytest.approx(0.375)
    assert metrics["per_class"]["a"]["precision"] == pytest.approx(1 / 3)
    assert metrics["per_class"]["b"]["recall"] == 1
    assert metrics["per_class"]["c"]["roc_auc_ovr"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"] == [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]


def test_missing_classes_have_null_auc_and_full_confusion_matrix() -> None:
    metrics = compute_metrics(
        np.array([0, 0]), np.array([[1, 0, 0], [0.5, 0.5, 0]]), ("a", "b", "c")
    )
    assert metrics["per_class"]["a"]["roc_auc_ovr"] is None
    assert metrics["per_class"]["c"]["roc_auc_ovr"] is None
    assert metrics["per_class"]["c"]["support"] == 0
    assert np.shape(metrics["confusion_matrix"]) == (3, 3)
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize(
    "targets, probabilities",
    [
        (np.array([], dtype=int), np.empty((0, 3))),
        (np.array([3]), np.array([[1, 0, 0]])),
        (np.array([0]), np.array([[np.nan, 0, 0]])),
        (np.array([0]), np.array([[0.1, 0.1, 0.1]])),
        (np.array([0]), np.array([[-1, 1, 1]])),
        (np.array([0]), np.array([[1, 0]])),
    ],
)
def test_metrics_reject_invalid_inputs(targets, probabilities) -> None:
    with pytest.raises(ValueError):
        compute_metrics(targets, probabilities, ("a", "b", "c"))


def test_gradcam_matches_known_spatial_activation_and_preserves_gradients() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 4, 1, bias=False),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 3, bias=False),
    )
    with torch.no_grad():
        model[0].weight.fill_(1)
        model[4].weight.fill_(1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
        parameter.requires_grad_(False)
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
    heatmap = grad_cam(model, image, model[1], class_index=0)
    assert torch.allclose(heatmap, image[0, 0] / 15)
    assert model.training
    assert all(torch.all(parameter.grad == 1) for parameter in model.parameters())
    assert not model[1]._forward_hooks
    with pytest.raises(ValueError, match="class index"):
        grad_cam(model, image, model[1], class_index=3)
    assert not model[1]._forward_hooks


def test_gradcam_zero_signal_is_finite() -> None:
    model = nn.Sequential(nn.Conv2d(3, 3, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    model.eval()
    heatmap = grad_cam(model, torch.zeros(1, 3, 8, 8), model[0])
    assert torch.isfinite(heatmap).all()
    assert heatmap.count_nonzero() == 0
    assert not model.training


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer4 = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, NUM_CLASSES)

    def forward(self, images):
        return self.fc(self.pool(self.layer4(images)).flatten(1))


def test_evaluation_cli_writes_complete_reports_without_other_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real checkpoint loading and ImageFolder traversal; no real images, GPU, or downloads.
    monkeypatch.setattr("src.inference.predictor.build_model", lambda **kwargs: TinyClassifier())
    data_dir = tmp_path / "data"
    for name in CLASSES:
        folder = data_dir / "validation" / name
        folder.mkdir(parents=True)
        Image.new("RGB", (40, 32), (180, 100, 70)).save(folder / "example.jpg")
    model = TinyClassifier()
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        model.fc.bias[0] = 5
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    output = tmp_path / "reports"
    assert (
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output),
                "--split",
                "validation",
                "--image-size",
                "32",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--gradcam-examples",
                "2",
            ]
        )
        == 0
    )
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["accuracy"] == pytest.approx(1 / NUM_CLASSES)
    assert metrics["num_samples"] == NUM_CLASSES
    assert metrics["run"]["split"] == "validation"
    assert len(metrics["run"]["checkpoint_sha256"]) == 64
    assert len(metrics["gradcam_examples"]) == 2
    assert all(item["true_class"] != "bread" for item in metrics["gradcam_examples"])
    with np.load(output / "predictions.npz") as predictions:
        assert predictions["probabilities"].shape == (NUM_CLASSES, NUM_CLASSES)
        assert np.array_equal(predictions["targets"], np.arange(NUM_CLASSES))
    with (output / "predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == NUM_CLASSES
    assert rows[1]["true_class"] == CLASSES[1]
    assert all(row["predicted_class"] == "bread" for row in rows)
    for plot in output.rglob("*.png"):
        with Image.open(plot) as image:
            image.verify()
    assert len(list(output.rglob("*.png"))) == 5
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(checkpoint, data_dir, output)


def test_cli_requires_checkpoint_and_reports_missing_file(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", str(tmp_path)])
    assert (
        main(
            ["--checkpoint", str(tmp_path / "missing.pt"), "--output-dir", str(tmp_path / "report")]
        )
        == 1
    )
    assert "Checkpoint file not found" in capsys.readouterr().err
