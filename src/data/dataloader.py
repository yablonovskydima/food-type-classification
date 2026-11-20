from pathlib import Path

from torch.utils.data import DataLoader

from src.data.class_balance import (
    DEFAULT_IMBALANCE_STRATEGY,
    validate_strategy,
    weighted_sampler,
)
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
    imbalance: str = DEFAULT_IMBALANCE_STRATEGY,
    seed: int | None = None,
) -> dict[str, DataLoader]:
    imbalance = validate_strategy(imbalance)
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

    sampler = (
        weighted_sampler(train_dataset.targets, seed=seed)
        if imbalance == "weighted_sampler"
        else None
    )

    loaders = {
        "training": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last_train,
            persistent_workers=num_workers > 0,
        ),
        "validation": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
        ),
    }

    return loaders
