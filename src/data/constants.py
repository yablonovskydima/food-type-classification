"""Constants for the Food-11 dataset pipeline."""

from typing import Final

CLASSES: Final[tuple[str, ...]] = (
    "bread",
    "dairy_product",
    "dessert",
    "egg",
    "fried_food",
    "meat",
    "noodles_pasta",
    "rice",
    "seafood",
    "soup",
    "vegetable_fruit",
)

NUM_CLASSES: Final[int] = len(CLASSES)

CLASS_TO_IDX: Final[dict[str, int]] = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS: Final[dict[int, str]] = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

# Exact RGB statistics computed on the deduplicated Food-11 training split
FOOD11_MEAN: Final[tuple[float, float, float]] = (0.5545, 0.4507, 0.3436)
FOOD11_STD: Final[tuple[float, float, float]] = (0.2299, 0.2402, 0.2387)

# ImageNet statistics for the reference pretrained model
IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE: Final[int] = 224
