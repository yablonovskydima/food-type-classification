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

FOOD11_MEAN: Final[tuple[float, float, float]] = (0.5503, 0.4493, 0.3437)
FOOD11_STD: Final[tuple[float, float, float]] = (0.2714, 0.2738, 0.2806)

IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE: Final[int] = 224
