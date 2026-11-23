import json
from pathlib import Path

import pytest
import yaml

from src.training import tuning
from src.training.tuning import (
    TrialResult,
    best_config,
    best_trial,
    load_sweep,
    results_table,
    run_sweep,
    trial_configs,
)


@pytest.fixture
def sweep_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(tuning, "TUNING_CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(tuning, "TUNING_LOG_DIR", tmp_path / "runs")
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"experiment": "base", "epochs": 80, "lr": 0.1}))
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        yaml.safe_dump(
            {
                "name": "s",
                "base_config": str(base),
                "overrides": {"epochs": 3},
                "trials": {"baseline": None, "fast": {"lr": 0.2}, "mix": {"mixup_alpha": 0.2}},
            }
        )
    )
    return sweep


def fake_summary(macro_f1: float) -> dict:
    return {
        "best_val_macro_f1": macro_f1,
        "best_val_metrics": {"accuracy": macro_f1 - 0.01},
        "best_epoch": 2,
        "epochs_completed": 3,
        "training_seconds": 120.0,
    }


def test_trial_configs_apply_overrides_and_isolated_dirs(sweep_file: Path, tmp_path: Path) -> None:
    configs = trial_configs(load_sweep(sweep_file))
    assert list(configs) == ["baseline", "fast", "mix"]
    assert configs["fast"].experiment == "s_fast"
    assert configs["fast"].lr == 0.2 and configs["baseline"].lr == 0.1
    assert all(config.epochs == 3 for config in configs.values())
    assert configs["mix"].mixup_alpha == 0.2
    assert configs["baseline"].run_checkpoint_dir == tmp_path / "checkpoints" / "s_baseline"


@pytest.mark.parametrize(
    "trials, message",
    [({"bad": {"learning_rate": 1}}, "learning_rate"), ({"bad": {"lr": -1}}, "lr")],
)
def test_invalid_trials_fail_when_loading(sweep_file: Path, trials: dict, message: str) -> None:
    data = yaml.safe_load(sweep_file.read_text())
    data["trials"] = trials
    sweep_file.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=message):
        load_sweep(sweep_file)


def test_run_sweep_reuses_finished_and_resumes_interrupted(sweep_file: Path) -> None:
    sweep = load_sweep(sweep_file)
    configs = trial_configs(sweep)
    finished = configs["baseline"].run_checkpoint_dir
    finished.mkdir(parents=True)
    (finished / "summary.json").write_text(json.dumps(fake_summary(0.5)))
    interrupted = configs["fast"].run_checkpoint_dir
    interrupted.mkdir(parents=True)
    (interrupted / "last.pt").write_bytes(b"state")

    calls = []

    def fake_fit(config, *, device, resume, limit_batches):
        calls.append((config.experiment, resume))
        return fake_summary(0.7 if config.experiment == "s_fast" else 0.6)

    results = run_sweep(sweep, fake_fit, device="cpu")

    assert calls == [("s_fast", str(interrupted / "last.pt")), ("s_mix", None)]
    assert [result.trial for result in results] == ["baseline", "fast", "mix"]
    assert best_trial(results).trial == "fast"


def test_run_sweep_only_selected_trials(sweep_file: Path) -> None:
    sweep = load_sweep(sweep_file)
    ran = []
    run_sweep(
        sweep,
        lambda config, **kwargs: ran.append(config.experiment) or fake_summary(0.5),
        device="cpu",
        only=["mix"],
    )
    assert ran == ["s_mix"]
    with pytest.raises(ValueError, match="nope"):
        run_sweep(sweep, lambda *a, **k: None, device="cpu", only=["nope"])


def test_results_table_is_sorted_and_best_config_keeps_full_budget(sweep_file: Path) -> None:
    sweep = load_sweep(sweep_file)
    results = [
        TrialResult("baseline", {}, fake_summary(0.6)),
        TrialResult("fast", {"lr": 0.2}, fake_summary(0.8)),
    ]
    rows = results_table(results).splitlines()
    assert rows[2].startswith("| fast | lr=0.2 | 0.800 | 0.790 | 2/3 | 2.0 min |")
    assert rows[3].startswith("| baseline | — | 0.600 |")

    config = best_config(sweep, best_trial(results))
    assert config == {"experiment": "base", "epochs": 80, "lr": 0.2}
