"""DataLoader creation utilities for training and evaluation."""

from pathlib import Path

from torch.utils.data import DataLoader

from src.data.constants import (
    DEFAULT_IMAGE_SIZE,
    FOOD11_MEAN,
    FOOD11_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.data.dataset import get_dataset
from src.data.transforms import get_eval_transforms, get_train_transforms


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_imagenet_stats: bool = False,
    randaugment: bool = True,
    random_erasing_p: float = 0.2,
    drop_last_train: bool = True,
) -> dict[str, DataLoader]:
    """Create DataLoaders for all splits ('training', 'validation', 'evaluation').

    Args:
        data_dir: Root directory of processed dataset (containing training, validation, evaluation).
        batch_size: Batch size for all loaders.
        image_size: Target square image dimension (e.g., 128 or 224 px).
        num_workers: Number of subprocesses for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.
        use_imagenet_stats: Use ImageNet mean/std if True; Food-11 stats if False.
        randaugment: Whether to apply RandAugment in the training pipeline.
        random_erasing_p: Probability of RandomErasing in training (0 to disable).
        drop_last_train: Whether to drop the last incomplete training batch.

    Returns:
        Dict mapping split name ('training', 'validation', 'evaluation') to its DataLoader.
    """
    mean = IMAGENET_MEAN if use_imagenet_stats else FOOD11_MEAN
    std = IMAGENET_STD if use_imagenet_stats else FOOD11_STD

    train_transform = get_train_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        randaugment=randaugment,
        random_erasing_p=random_erasing_p,
    )
    eval_transform = get_eval_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
    )

    train_dataset = get_dataset(data_dir=data_dir, split="training", transform=train_transform)
    val_dataset = get_dataset(data_dir=data_dir, split="validation", transform=eval_transform)
    test_dataset = get_dataset(data_dir=data_dir, split="evaluation", transform=eval_transform)

    loaders = {
        "training": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last_train,
        ),
        "validation": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "evaluation": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }

    return loaders
