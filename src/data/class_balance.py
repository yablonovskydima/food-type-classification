from collections.abc import Sequence
from typing import Literal, get_args

import torch
from torch.utils.data import WeightedRandomSampler

from src.data.constants import NUM_CLASSES

ImbalanceStrategy = Literal["none", "weighted_loss", "weighted_sampler"]
IMBALANCE_STRATEGIES: tuple[str, ...] = get_args(ImbalanceStrategy)
DEFAULT_IMBALANCE_STRATEGY: ImbalanceStrategy = "weighted_loss"


def validate_strategy(strategy: str) -> ImbalanceStrategy:
    if strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(
            f"Unknown imbalance strategy {strategy!r}; expected one of {IMBALANCE_STRATEGIES}"
        )
    return strategy


def class_counts(targets: Sequence[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    counts = torch.bincount(torch.as_tensor(targets, dtype=torch.long), minlength=num_classes)
    if counts.numel() != num_classes:
        raise ValueError(f"Targets contain class indices >= {num_classes}")
    if (counts == 0).any():
        missing = (counts == 0).nonzero().flatten().tolist()
        raise ValueError(f"Classes without training samples: {missing}")
    return counts


def class_weights(targets: Sequence[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    inverse = 1.0 / class_counts(targets, num_classes).double()
    return (inverse * num_classes / inverse.sum()).float()


def weighted_sampler(
    targets: Sequence[int],
    num_classes: int = NUM_CLASSES,
    seed: int | None = None,
) -> WeightedRandomSampler:
    targets_tensor = torch.as_tensor(targets, dtype=torch.long)
    per_class = 1.0 / class_counts(targets_tensor, num_classes).double()
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return WeightedRandomSampler(
        weights=per_class[targets_tensor],
        num_samples=len(targets_tensor),
        replacement=True,
        generator=generator,
    )
