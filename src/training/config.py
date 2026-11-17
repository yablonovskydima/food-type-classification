from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from src.data.class_balance import DEFAULT_IMBALANCE_STRATEGY, validate_strategy
from src.data.constants import DEFAULT_IMAGE_SIZE

OPTIMIZERS = ("sgd", "adamw")
SCHEDULERS = ("onecycle", "cosine")


@dataclass
class TrainConfig:
    experiment: str
    pretrained: bool = False

    data_dir: str = "data/processed/food11"
    image_size: int = DEFAULT_IMAGE_SIZE
    batch_size: int = 64
    num_workers: int = 4
    imbalance: str = DEFAULT_IMBALANCE_STRATEGY
    randaugment: bool = True
    random_erasing_p: float = 0.2

    epochs: int = 80
    optimizer: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler: str = "onecycle"
    warmup_epochs: int = 3
    label_smoothing: float = 0.1
    amp: bool = True
    grad_clip_norm: float | None = None
    early_stopping_patience: int = 15

    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"

    def __post_init__(self) -> None:
        validate_strategy(self.imbalance)
        if not self.experiment or Path(self.experiment).name != self.experiment:
            raise ValueError("experiment must be a plain folder name, e.g. 'resnet18_scratch'")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}")
        positive = ("image_size", "batch_size", "epochs", "lr", "early_stopping_patience")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0 or self.warmup_epochs < 0 or self.weight_decay < 0:
            raise ValueError("num_workers, warmup_epochs and weight_decay must be >= 0")
        if self.scheduler == "cosine" and self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be smaller than epochs")
        if not 0 <= self.label_smoothing < 1 or not 0 <= self.random_erasing_p <= 1:
            raise ValueError("label_smoothing must be in [0, 1) and random_erasing_p in [0, 1]")

    @property
    def use_imagenet_stats(self) -> bool:
        return self.pretrained

    @property
    def run_checkpoint_dir(self) -> Path:
        return Path(self.checkpoint_dir) / self.experiment

    @property
    def run_log_dir(self) -> Path:
        return Path(self.log_dir) / self.experiment

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    with open(path, encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError(f"{path} must contain a mapping of config values")
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})

    known = {field.name for field in fields(TrainConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {unknown}")
    return TrainConfig(**values)
