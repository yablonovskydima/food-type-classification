import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.data.constants import CLASSES
from src.training import tuning


@pytest.fixture
def tune(load_script, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = load_script("tune")
    for target in (tuning, module):
        monkeypatch.setattr(target, "TUNING_CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(tuning, "TUNING_LOG_DIR", tmp_path / "runs")
    monkeypatch.setattr(module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(module, "select_device", lambda allow_cpu: "cpu")
    return module


@pytest.fixture
def sweep_file(tmp_path: Path) -> Path:
    data_dir = tmp_path / "food11"
    for split in ("training", "validation"):
        for name in CLASSES:
            folder = data_dir / split / name
            folder.mkdir(parents=True)
            Image.new("RGB", (40, 40), "orange").save(folder / "0.jpg")
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"experiment": "base", "data_dir": str(data_dir)}))
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "base_config": str(base),
                "trials": {"baseline": {}, "fast": {"lr": 0.2}},
            }
        )
    )
    return sweep


def test_tune_writes_table_curves_and_best_config(tune, sweep_file: Path, tmp_path: Path) -> None:
    scores = {"demo_baseline": 0.6, "demo_fast": 0.7}

    def fake_fit(config, *, device, resume, limit_batches):
        run_dir = config.run_checkpoint_dir
        run_dir.mkdir(parents=True)
        f1 = scores[config.experiment]
        history = [
            {"epoch": epoch, "val_accuracy": f1, "val_macro_f1": f1, "val_loss": 1.0}
            for epoch in (1, 2)
        ]
        summary = {
            "best_epoch": 2,
            "best_val_macro_f1": f1,
            "best_val_metrics": {"accuracy": f1},
            "epochs_completed": 2,
            "training_seconds": 60.0,
            "hardware": {"device": "cpu"},
        }
        (run_dir / "history.json").write_text(json.dumps(history))
        (run_dir / "summary.json").write_text(json.dumps(summary))
        return summary

    tune.fit = fake_fit
    assert tune.main(["--sweep", str(sweep_file)]) == 0

    output = tmp_path / "reports" / "demo"
    assert (output / "val_curves.png").is_file()
    table = (output / "results.md").read_text(encoding="utf-8").splitlines()
    assert table[2].startswith("| fast | lr=0.2 | 0.700 |")
    best = yaml.safe_load((output / "best_config.yaml").read_text(encoding="utf-8"))
    assert best["lr"] == 0.2 and best["experiment"] == "base"


def test_tune_reports_missing_data(tune, sweep_file: Path, tmp_path: Path, capsys) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"experiment": "base", "data_dir": str(tmp_path / "none")}))
    assert tune.main(["--sweep", str(sweep_file)]) == 1
    assert "prepare_data.py" in capsys.readouterr().err
