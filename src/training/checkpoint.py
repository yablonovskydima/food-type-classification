import os
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.data.constants import CLASSES
from src.training.config import TrainConfig

BEST_CHECKPOINT = "best.pt"
LAST_CHECKPOINT = "last.pt"

RESUMABLE_OVERRIDES = {"num_workers", "data_dir", "checkpoint_dir", "log_dir"}


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def model_metadata(config: TrainConfig, epoch: int, metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "architecture": "resnet18",
        "classes": list(CLASSES),
        "image_size": config.image_size,
        "normalization": "imagenet" if config.use_imagenet_stats else "food11",
        "pretrained": config.pretrained,
        "experiment": config.experiment,
        "epoch": epoch,
        "metrics": metrics,
    }


def save_best_model(
    path: Path, model: nn.Module, config: TrainConfig, epoch: int, metrics: dict[str, float]
) -> None:
    _atomic_save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": model_metadata(config, epoch, metrics),
        },
        path,
    )


def save_training_state(path: Path, **state: Any) -> None:
    _atomic_save(state, path)


def load_training_state(path: str | Path, map_location: torch.device | str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint to resume from not found: {path}")
    state = torch.load(path, map_location=map_location, weights_only=True)
    required = {"epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"{path} is not a training-state checkpoint (missing {sorted(missing)})")
    return state


def check_resume_compatible(saved_config: dict[str, Any], config: TrainConfig) -> None:
    current = config.to_dict()
    # Settings added after a run started count as their defaults.
    defaults = {field.name: field.default for field in fields(TrainConfig)}
    saved_config = {**defaults, **saved_config}
    changed = sorted(
        key
        for key in current.keys() | saved_config.keys()
        if key not in RESUMABLE_OVERRIDES and current.get(key) != saved_config.get(key)
    )
    if changed:
        raise ValueError(
            "Cannot resume: config differs from the interrupted run in "
            + ", ".join(f"{k} ({saved_config.get(k)!r} -> {current.get(k)!r})" for k in changed)
        )
