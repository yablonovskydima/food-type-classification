from pathlib import Path

import pytest
import torch
import yaml

from src.models import build_model
from src.training.checkpoint import check_resume_compatible
from src.training.config import TrainConfig, load_config
from src.training.optim import build_optimizer, build_scheduler

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def write_config(path: Path, **values) -> Path:
    path.write_text(yaml.safe_dump({"experiment": "test_run", **values}), encoding="utf-8")
    return path


@pytest.mark.parametrize("name", ["resnet18_scratch.yaml", "resnet18_pretrained.yaml"])
def test_shipped_configs_load(name: str) -> None:
    config = load_config(CONFIG_DIR / name)
    assert config.experiment == Path(name).stem
    assert config.use_imagenet_stats == config.pretrained


def test_scratch_config_matches_project_decisions() -> None:
    config = load_config(CONFIG_DIR / "resnet18_scratch.yaml")
    assert not config.pretrained
    assert config.imbalance == "weighted_loss"
    assert config.label_smoothing == 0.1
    assert config.weight_decay == 5e-4


def test_overrides_and_none_values(tmp_path: Path) -> None:
    path = write_config(tmp_path / "c.yaml", epochs=10, num_workers=4)
    config = load_config(path, overrides={"epochs": 2, "num_workers": None})
    assert config.epochs == 2
    assert config.num_workers == 4


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path / "c.yaml", learning_rate=0.1)
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(path)


@pytest.mark.parametrize(
    "values",
    [
        {"optimizer": "rmsprop"},
        {"scheduler": "step"},
        {"imbalance": "oversample"},
        {"epochs": 0},
        {"lr": -1},
        {"scheduler": "cosine", "epochs": 3, "warmup_epochs": 3},
        {"label_smoothing": 1.0},
    ],
)
def test_invalid_values_are_rejected(values: dict) -> None:
    with pytest.raises(ValueError):
        TrainConfig(experiment="test_run", **values)


def test_experiment_must_be_a_folder_name() -> None:
    with pytest.raises(ValueError, match="folder name"):
        TrainConfig(experiment="../outside")


def _lrs(config: TrainConfig, steps_per_epoch: int) -> list[float]:
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch)
    lrs = []
    for _ in range(config.epochs * steps_per_epoch):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


def test_onecycle_warms_up_to_max_lr_and_anneals() -> None:
    config = TrainConfig(experiment="t", epochs=10, lr=0.1, scheduler="onecycle")
    lrs = _lrs(config, steps_per_epoch=10)
    assert max(lrs) == pytest.approx(0.1)
    assert lrs[0] < 0.01 and lrs[-1] < 1e-4
    assert lrs.index(max(lrs)) == pytest.approx(30, abs=1)  # pct_start = 0.3


def test_cosine_with_warmup() -> None:
    config = TrainConfig(
        experiment="t", epochs=10, lr=0.01, optimizer="adamw", scheduler="cosine", warmup_epochs=2
    )
    lrs = _lrs(config, steps_per_epoch=5)
    assert lrs[:10] == sorted(lrs[:10])  # linear warm-up over 2 epochs
    assert max(lrs) == pytest.approx(0.01)
    assert lrs[10:] == sorted(lrs[10:], reverse=True)
    assert lrs[-1] < 0.001


def test_optimizer_choice() -> None:
    model = build_model()
    sgd = build_optimizer(model, TrainConfig(experiment="t"))
    adamw = build_optimizer(model, TrainConfig(experiment="t", optimizer="adamw", lr=1e-3))
    assert isinstance(sgd, torch.optim.SGD) and sgd.defaults["nesterov"]
    assert isinstance(adamw, torch.optim.AdamW)


def test_resume_compatibility() -> None:
    saved = TrainConfig(experiment="t").to_dict()
    check_resume_compatible(saved, TrainConfig(experiment="t", num_workers=0))
    with pytest.raises(ValueError, match="lr"):
        check_resume_compatible(saved, TrainConfig(experiment="t", lr=0.5))
