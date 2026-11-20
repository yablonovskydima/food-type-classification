from collections.abc import Callable
from pathlib import Path

from torchvision.datasets import ImageFolder

from src.data.constants import CLASS_TO_IDX, CLASSES


class Food11Dataset(ImageFolder):
    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset root directory does not exist: {root_path}")

        super().__init__(
            root=str(root_path),
            transform=transform,
            target_transform=target_transform,
        )

    def find_classes(self, directory: str | Path) -> tuple[list[str], dict[str, int]]:
        found_classes, _ = super().find_classes(directory)
        if sorted(found_classes) != sorted(CLASSES):
            raise ValueError(
                f"Dataset classes at {directory} do not match expected Food-11 classes.\n"
                f"Found: {sorted(found_classes)}\n"
                f"Expected: {sorted(CLASSES)}"
            )
        return list(CLASSES), dict(CLASS_TO_IDX)


def get_dataset(
    data_dir: str | Path,
    split: str = "training",
    transform: Callable | None = None,
) -> Food11Dataset:
    return Food11Dataset(root=Path(data_dir) / split, transform=transform)
