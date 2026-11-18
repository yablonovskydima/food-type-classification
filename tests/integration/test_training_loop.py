import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from src.data.constants import CLASSES
from src.inference.predictor import FoodPredictor
from src.training import trainer
from src.training.cli import main
from src.training.config import TrainConfig

IMAGES_PER_CLASS = 2


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "food11"
    for split in ("training", "validation", "evaluation"):
        for index, name in enumerate(CLASSES):
            folder = root / split / name
            folder.mkdir(parents=True)
            for i in range(IMAGES_PER_CLASS):
                Image.new("RGB", (40, 40), (index * 20, i * 50, 100)).save(folder / f"{i}.jpg")
    return root


@pytest.fixture
def config(tmp_path: Path, data_dir: Path) -> TrainConfig:
    return TrainConfig(
        experiment="tiny",
        data_dir=str(data_dir),
        image_size=32,
        batch_size=8,
        num_workers=0,
        epochs=3,
        lr=0.01,
        randaugment=False,
        early_stopping_patience=10,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "runs"),
    )


def test_fit_saves_checkpoints_history_and_tensorboard_logs(config: TrainConfig) -> None:
    summary = trainer.fit(config, device="cpu")

    run_dir = config.run_checkpoint_dir
    assert summary["epochs_completed"] == 3
    assert 1 <= summary["best_epoch"] <= 3
    assert (run_dir / "best.pt").is_file() and (run_dir / "last.pt").is_file()
    history = json.loads((run_dir / "history.json").read_text())
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert {"train_loss", "val_loss", "val_accuracy", "val_macro_f1", "lr"} <= history[0].keys()
    assert list(config.run_log_dir.glob("events.out.tfevents.*"))

    saved_summary = json.loads((run_dir / "summary.json").read_text())
    assert saved_summary["hardware"]["device"] == "cpu"
    assert saved_summary["training_seconds"] == pytest.approx(
        sum(row["seconds"] for row in history), abs=0.2
    )
    best_row = history[summary["best_epoch"] - 1]
    assert saved_summary["best_val_metrics"]["macro_f1"] == best_row["val_macro_f1"]

    best = torch.load(run_dir / "best.pt", weights_only=True)
    assert best["metadata"]["image_size"] == 32
    assert best["metadata"]["normalization"] == "food11"
    assert best["metadata"]["epoch"] == summary["best_epoch"]
    predictor = FoodPredictor(checkpoint_path=run_dir / "best.pt", device="cpu", image_size=32)
    assert len(predictor.predict_image(Image.new("RGB", (40, 40)), top_k=3)) == 3


def test_refuses_to_overwrite_existing_run(config: TrainConfig) -> None:
    trainer.fit(config, device="cpu")
    with pytest.raises(FileExistsError, match="--resume"):
        trainer.fit(config, device="cpu")


def test_resume_continues_an_interrupted_run(
    config: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = trainer.train_one_epoch
    calls = {"count": 0}

    def interrupt_on_second_epoch(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, "train_one_epoch", interrupt_on_second_epoch)
    with pytest.raises(KeyboardInterrupt):
        trainer.fit(config, device="cpu")
    monkeypatch.setattr(trainer, "train_one_epoch", original)

    last = config.run_checkpoint_dir / "last.pt"
    assert torch.load(last, weights_only=True)["epoch"] == 1

    summary = trainer.fit(config, device="cpu", resume=str(last))
    history = json.loads((config.run_checkpoint_dir / "history.json").read_text())
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert summary["epochs_completed"] == 3
    # Time of the epoch finished before the interruption is still counted.
    assert summary["training_seconds"] == pytest.approx(
        sum(row["seconds"] for row in history), abs=0.2
    )


def test_resume_rejects_changed_config(config: TrainConfig) -> None:
    trainer.fit(config, device="cpu")
    config.lr = 0.5
    with pytest.raises(ValueError, match="lr"):
        trainer.fit(config, device="cpu", resume=str(config.run_checkpoint_dir / "last.pt"))


def test_early_stopping(config: TrainConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    config.epochs = 10
    config.early_stopping_patience = 2
    monkeypatch.setattr(
        trainer,
        "evaluate",
        lambda *args, **kwargs: {"loss": 1.0, "accuracy": 0.5, "macro_f1": 0.5},
    )
    summary = trainer.fit(config, device="cpu")
    assert summary["stopped_early"]
    assert summary["epochs_completed"] == 3
    assert summary["best_epoch"] == 1


def test_disabled_early_stopping_runs_full_schedule(
    config: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.epochs = 5
    config.early_stopping_patience = None
    monkeypatch.setattr(
        trainer,
        "evaluate",
        lambda *args, **kwargs: {"loss": 1.0, "accuracy": 0.5, "macro_f1": 0.5},
    )
    summary = trainer.fit(config, device="cpu")
    assert not summary["stopped_early"]
    assert summary["epochs_completed"] == 5
    assert summary["best_epoch"] == 1


def test_weighted_sampler_strategy_trains(config: TrainConfig) -> None:
    config.imbalance = "weighted_sampler"
    config.epochs = 1
    assert trainer.fit(config, device="cpu")["epochs_completed"] == 1


def test_cli_smoke(config: TrainConfig, tmp_path: Path, capsys) -> None:
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    assert main(["--config", str(path), "--device", "cpu", "--epochs", "1"]) == 0
    assert "Best validation macro-F1" in capsys.readouterr().out
    assert main(["--config", str(path), "--device", "cpu"]) == 1
    assert "already contains a run" in capsys.readouterr().err
