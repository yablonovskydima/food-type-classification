import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from src.data.constants import CLASSES

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train.py"


def load_script():
    spec = importlib.util.spec_from_file_location("train_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def train_script():
    return load_script()


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    data_dir = tmp_path / "food11"
    for split in ("training", "validation", "evaluation"):
        for index, name in enumerate(CLASSES):
            folder = data_dir / split / name
            folder.mkdir(parents=True)
            for i in range(2):
                Image.new("RGB", (40, 40), (index * 20, i * 50, 90)).save(folder / f"{i}.jpg")
    config = {
        "experiment": "tiny",
        "data_dir": str(data_dir),
        "image_size": 32,
        "batch_size": 8,
        "num_workers": 0,
        "epochs": 2,
        "lr": 0.01,
        "randaugment": False,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "log_dir": str(tmp_path / "runs"),
    }
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def reports_in_tmp(train_script, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    reports = tmp_path / "reports"
    monkeypatch.setattr(
        train_script, "report_dir", lambda config: reports / config.experiment / "validation"
    )
    return reports


def test_pipeline_trains_and_evaluates(
    train_script, config_path: Path, reports_in_tmp: Path, capsys
) -> None:
    cwd = Path.cwd()
    assert train_script.main(["--config", str(config_path), "--allow-cpu"]) == 0
    assert Path.cwd() == cwd

    run_dir = config_path.parent / "checkpoints" / "tiny"
    assert (run_dir / "best.pt").is_file()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["epochs_completed"] == 2
    metrics = json.loads((reports_in_tmp / "tiny" / "validation" / "metrics.json").read_text())
    assert metrics["run"]["split"] == "validation"

    output = capsys.readouterr().out
    assert "=== Results ===" in output and "Training time" in output


def test_existing_run_requires_resume(
    train_script, config_path: Path, reports_in_tmp: Path, capsys
) -> None:
    assert train_script.main(["--config", str(config_path), "--allow-cpu", "--no-eval"]) == 0
    assert train_script.main(["--config", str(config_path), "--allow-cpu"]) == 1
    assert "--resume" in capsys.readouterr().err


def test_resume_without_previous_run_fails(train_script, config_path: Path, capsys) -> None:
    assert train_script.main(["--config", str(config_path), "--allow-cpu", "--resume"]) == 1
    assert "Nothing to resume" in capsys.readouterr().err


def test_missing_processed_data_is_explained(
    train_script, config_path: Path, tmp_path: Path, capsys
) -> None:
    config = yaml.safe_load(config_path.read_text())
    config["data_dir"] = str(tmp_path / "missing")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert train_script.main(["--config", str(config_path), "--allow-cpu"]) == 1
    assert "prepare_data.py" in capsys.readouterr().err


def test_cpu_requires_explicit_flag(
    train_script, config_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert train_script.main(["--config", str(config_path)]) == 1
    assert "--allow-cpu" in capsys.readouterr().err


def test_default_config_is_the_main_experiment(train_script) -> None:
    assert train_script.parse_args([]).config == Path("configs") / "resnet18_scratch.yaml"
