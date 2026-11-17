from collections.abc import Sequence

from torch import nn

from src.data.class_balance import class_weights, validate_strategy
from src.data.constants import NUM_CLASSES


def build_criterion(
    imbalance: str,
    train_targets: Sequence[int],
    label_smoothing: float = 0.1,
    num_classes: int = NUM_CLASSES,
) -> nn.CrossEntropyLoss:
    imbalance = validate_strategy(imbalance)
    weight = class_weights(train_targets, num_classes) if imbalance == "weighted_loss" else None
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
