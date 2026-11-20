import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from src.evaluation.compare import compare_runs, find_validation_report, load_run

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_runs.py"


def write_run(
    root: Path, experiment: str, accuracies: list[float], with_report: bool = False
) -> Path:
    run_dir = root / "checkpoints" / experiment
    run_dir.mkdir(parents=True)
    history = [
        {
            "epoch": epoch,
            "lr": 0.01,
            "seconds": 30.0,
            "train_loss": 1.0,
            "train_accuracy": accuracy,
            "train_macro_f1": accuracy,
            "val_loss": 2.0 - accuracy,
            "val_accuracy": accuracy,
            "val_macro_f1": accuracy - 0.01,
        }
        for epoch, accuracy in enumerate(accuracies, start=1)
    ]
    best_epoch = max(history, key=lambda row: row["val_macro_f1"])["epoch"]
    best = history[best_epoch - 1]
    (run_dir / "history.json").write_text(json.dumps(history))
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment": experiment,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best["val_macro_f1"],
                "best_val_metrics": {"accuracy": best["val_accuracy"]},
                "epochs_completed": len(history),
                "stopped_early": False,
                "training_seconds": 30.0 * len(history),
                "hardware": {"device": "cuda", "gpu": "Test GPU"},
            }
        )
    )
    checkpoint = run_dir / "best.pt"
    checkpoint.write_bytes(f"weights of {experiment}".encode())
    if with_report:
        report_dir = root / "reports" / experiment / "validation"
        report_dir.mkdir(parents=True)
        (report_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "top3_accuracy": 0.95,
                    "run": {
                        "split": "validation",
                        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    },
                }
            )
        )
    return run_dir


@pytest.fixture
def two_runs(tmp_path: Path) -> Path:
    write_run(tmp_path, "scratch", [0.3, 0.5, 0.7, 0.65], with_report=True)
    write_run(tmp_path, "pretrained", [0.8, 0.88, 0.9])
    return tmp_path


def test_compare_writes_curves_and_table(two_runs: Path) -> None:
    result = compare_runs(
        {"scratch": "From scratch", "pretrained": "Pretrained"},
        output_dir=two_runs / "out",
        checkpoint_dir=two_runs / "checkpoints",
        reports_dir=two_runs / "reports",
    )
    with Image.open(result["curves"]) as image:
        image.verify()

    rows = result["table"].splitlines()
    assert len(rows) == 4  # header, separator, two runs
    scratch, pretrained = rows[2], rows[3]
    assert scratch.startswith("| From scratch | Test GPU | 2.0 min | 4 (3) | 0.700 | 0.950 |")
    assert "| 0.690 |" in scratch
    assert pretrained.startswith("| Pretrained | Test GPU | 1.5 min | 3 (3) | 0.900 | — |")
    assert result["table_path"].read_text(encoding="utf-8").strip() == result["table"]


def test_validation_report_must_match_current_best_checkpoint(two_runs: Path) -> None:
    run_dir = two_runs / "checkpoints" / "scratch"
    reports = two_runs / "reports" / "scratch"
    assert find_validation_report(reports, run_dir / "best.pt")["top3_accuracy"] == 0.95

    (run_dir / "best.pt").write_bytes(b"retrained weights")  # report is now stale
    assert find_validation_report(reports, run_dir / "best.pt") is None


def test_missing_run_explains_how_to_train_it(two_runs: Path) -> None:
    with pytest.raises(FileNotFoundError, match="resnet18_pretrained.yaml"):
        load_run("resnet18_pretrained", "x", two_runs / "checkpoints", two_runs / "reports")


def test_needs_at_least_two_runs(two_runs: Path) -> None:
    with pytest.raises(ValueError, match="at least two"):
        compare_runs({"scratch": "x"}, checkpoint_dir=two_runs / "checkpoints")


def test_script_reports_missing_run(tmp_path: Path, capsys) -> None:
    spec = importlib.util.spec_from_file_location("compare_runs_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = ["--run", "missing_a", "--run", "missing_b", "--output-dir", str(tmp_path)]
    assert module.main(args) == 1
    assert "No finished run for 'missing_a'" in capsys.readouterr().err
