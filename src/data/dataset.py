"""Food-11 dataset module."""

from collections.abc import Callable
from pathlib import Path

from torchvision.datasets import ImageFolder

from src.data.constants import CLASS_TO_IDX, CLASSES


class Food11Dataset(ImageFolder):
    """Dataset wrapper for a single split of the processed Food-11 dataset."""

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

        expected_classes = sorted(CLASSES)
        if sorted(self.classes) != expected_classes:
            raise ValueError(
                f"Dataset classes at {root_path} do not match expected Food-11 classes.\n"
                f"Found: {self.classes}\n"
                f"Expected: {expected_classes}"
            )

        # Enforce canonical index mapping
        self.class_to_idx = CLASS_TO_IDX


def get_dataset(
    data_dir: str | Path,
    split: str = "training",
    transform: Callable | None = None,
) -> Food11Dataset:
    """Get Food11Dataset for the given split.

    Args:
        data_dir: Path to the processed food11 root containing split subdirectories.
        split: Split name ('training', 'validation', or 'evaluation'/'test').
        transform: Optional transform to apply to images.

    Returns:
        Food11Dataset instance.
    """
    normalized_split = "evaluation" if split == "test" else split
    split_dir = Path(data_dir) / normalized_split
    return Food11Dataset(root=split_dir, transform=transform)
