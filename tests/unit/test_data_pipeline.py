from pathlib import Path

import pytest
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.data.constants import CLASSES, NUM_CLASSES
from src.data.dataloader import create_dataloaders
from src.data.dataset import Food11Dataset, get_dataset
from src.data.transforms import (
    get_eval_geometry_transforms,
    get_eval_transforms,
    get_train_transforms,
)


@pytest.fixture
def mock_image() -> Image.Image:
    return Image.new("RGB", (300, 400), color=(120, 80, 200))


@pytest.fixture
def mock_dataset_dir(tmp_path: Path) -> Path:
    splits = ("training", "validation", "evaluation")
    for split in splits:
        for cls_name in CLASSES:
            cls_dir = tmp_path / split / cls_name
            cls_dir.mkdir(parents=True)
            for i in range(2):
                img = Image.new("RGB", (32, 32), color=(i * 10, i * 20, i * 30))
                img.save(cls_dir / f"img_{i}.jpg")
    return tmp_path


def test_train_transforms_output_shape(mock_image: Image.Image) -> None:
    transform_224 = get_train_transforms(image_size=224)
    tensor_224 = transform_224(mock_image)
    assert isinstance(tensor_224, torch.Tensor)
    assert tensor_224.shape == (3, 224, 224)
    assert tensor_224.dtype == torch.float32

    transform_128 = get_train_transforms(image_size=128, randaugment=False, random_erasing_p=0.0)
    tensor_128 = transform_128(mock_image)
    assert tensor_128.shape == (3, 128, 128)


@pytest.mark.parametrize("image_size", [128, 224])
def test_eval_geometry_matches_eval_transforms(mock_image: Image.Image, image_size: int) -> None:
    mean, std = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    crop = get_eval_geometry_transforms(image_size=image_size)(mock_image)
    assert crop.size == (image_size, image_size)

    expected = get_eval_transforms(image_size=image_size, mean=mean, std=std)(mock_image)
    from_crop = TF.normalize(TF.to_tensor(crop), mean=mean, std=std)
    assert torch.equal(from_crop, expected)


def test_eval_transforms_deterministic(mock_image: Image.Image) -> None:
    transform = get_eval_transforms(image_size=224)
    out1 = transform(mock_image)
    out2 = transform(mock_image)
    assert torch.allclose(out1, out2)
    assert out1.shape == (3, 224, 224)


def test_dataset_classes_and_mapping(mock_dataset_dir: Path) -> None:
    train_dir = mock_dataset_dir / "training"
    dataset = Food11Dataset(root=train_dir)

    assert len(dataset.classes) == NUM_CLASSES
    assert tuple(sorted(dataset.classes)) == tuple(sorted(CLASSES))
    assert len(dataset) == NUM_CLASSES * 2

    img, target = dataset[0]
    assert isinstance(img, Image.Image)
    assert isinstance(target, int)
    assert 0 <= target < NUM_CLASSES


def test_dataset_labels_follow_canonical_order(
    mock_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reversed_mapping = {name: idx for idx, name in enumerate(reversed(CLASSES))}
    monkeypatch.setattr("src.data.dataset.CLASSES", tuple(reversed(CLASSES)))
    monkeypatch.setattr("src.data.dataset.CLASS_TO_IDX", reversed_mapping)

    dataset = Food11Dataset(root=mock_dataset_dir / "training")

    assert dataset.class_to_idx == reversed_mapping
    for path, label in dataset.samples:
        assert label == reversed_mapping[Path(path).parent.name]


def test_dataset_invalid_classes(tmp_path: Path) -> None:
    wrong_dir = tmp_path / "corrupted"
    (wrong_dir / "pizza").mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(wrong_dir / "pizza" / "0.jpg")

    with pytest.raises(ValueError, match="do not match expected Food-11 classes"):
        Food11Dataset(root=wrong_dir)


def test_get_dataset_split_helper(mock_dataset_dir: Path) -> None:
    eval_ds = get_dataset(mock_dataset_dir, split="evaluation")
    assert len(eval_ds) == NUM_CLASSES * 2


def test_create_dataloaders(mock_dataset_dir: Path) -> None:
    batch_size = 4
    image_size = 128
    loaders = create_dataloaders(
        data_dir=mock_dataset_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0,
        pin_memory=False,
    )

    assert set(loaders.keys()) == {"training", "validation"}

    for loader in loaders.values():
        batch_images, batch_targets = next(iter(loader))
        assert batch_images.shape == (batch_size, 3, image_size, image_size)
        assert batch_targets.shape == (batch_size,)
        assert batch_images.dtype == torch.float32
        assert batch_targets.dtype == torch.int64


def test_train_transforms_strength_options(mock_image: Image.Image) -> None:
    transform = get_train_transforms(
        image_size=64, randaugment_magnitude=20, crop_min_scale=0.3, random_erasing_p=0.5
    )
    crop = transform.transforms[0]
    assert crop.scale == (0.3, 1.0)
    assert any(getattr(step, "magnitude", None) == 20 for step in transform.transforms)
    assert transform(mock_image).shape == (3, 64, 64)
