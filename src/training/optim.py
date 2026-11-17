import math

from torch import nn
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, OneCycleLR

from src.training.config import TrainConfig


def build_optimizer(model: nn.Module, config: TrainConfig) -> Optimizer:
    if config.optimizer == "sgd":
        return SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.momentum > 0,
        )
    return AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def build_scheduler(optimizer: Optimizer, config: TrainConfig, steps_per_epoch: int) -> LRScheduler:
    total_steps = config.epochs * steps_per_epoch
    if config.scheduler == "onecycle":
        return OneCycleLR(optimizer, max_lr=config.lr, total_steps=total_steps, pct_start=0.3)

    warmup_steps = config.warmup_epochs * steps_per_epoch

    def warmup_cosine(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, warmup_cosine)
