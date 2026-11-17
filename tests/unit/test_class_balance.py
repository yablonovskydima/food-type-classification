from collections import Counter
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.class_balance import (
    DEFAULT_IMBALANCE_STRATEGY,
    class_counts,
    class_weights,
    weighted_sampler,
)
from src.data.constants import CLASSES, NUM_CLASSES
from src.data.dataloader import create_dataloaders
from src.training.losses import build_criterion

# Class index i gets i + 1 training images: class 0 is 11x rarer than class 10.
IMBALANCED_TARGETS = [index for index in range(NUM_CLASSES) for _ in range(index + 1)]


@pytest.fixture
def imbalanced_dataset_dir(tmp_path: Path) -> Path:
    for split in ("training", "validation", "evaluation"):
        for index, name in enumerate(CLASSES):
            folder = tmp_path / split / name
            folder.mkdir(parents=True)
            count = index + 1 if split == "training" else 1
            for i in range(count):
                Image.new("RGB", (16, 16), (i, index, 0)).save(folder / f"{i}.jpg")
    return tmp_path


def test_class_counts() -> None:
    assert class_counts(IMBALANCED_TARGETS).tolist() == list(range(1, NUM_CLASSES + 1))


@pytest.mark.parametrize(
    "targets, message",
    [([0, 1], "without training samples"), ([0, NUM_CLASSES], ">= 11")],
)
def test_class_counts_rejects_bad_targets(targets: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        class_counts(targets)


def test_class_weights_are_inverse_frequency_with_mean_one() -> None:
    weights = class_weights(IMBALANCED_TARGETS)
    assert weights.shape == (NUM_CLASSES,)
    assert weights.mean().item() == pytest.approx(1.0)
    # Rarest class (1 image) weighs 11x more than the most frequent one (11 images).
    assert (weights[0] / weights[-1]).item() == pytest.approx(11.0)
    assert torch.all(weights[:-1] > weights[1:])


def test_balanced_targets_get_equal_weights() -> None:
    weights = class_weights(list(range(NUM_CLASSES)) * 3)
    assert torch.allclose(weights, torch.ones(NUM_CLASSES))


def test_weighted_sampler_balances_classes_and_is_reproducible() -> None:
    targets = IMBALANCED_TARGETS * 50
    sampler = weighted_sampler(targets, seed=0)
    drawn = [targets[i] for i in sampler]
    assert len(drawn) == len(targets)

    frequencies = Counter(drawn)
    expected = len(drawn) / NUM_CLASSES
    assert all(abs(frequencies[c] - expected) < 0.25 * expected for c in range(NUM_CLASSES))
    assert drawn == [targets[i] for i in weighted_sampler(targets, seed=0)]


def test_default_strategy_is_weighted_loss() -> None:
    assert DEFAULT_IMBALANCE_STRATEGY == "weighted_loss"


@pytest.mark.parametrize("strategy", ["none", "weighted_loss"])
def test_dataloaders_shuffle_without_sampler(imbalanced_dataset_dir: Path, strategy: str) -> None:
    loaders = create_dataloaders(
        imbalanced_dataset_dir, batch_size=4, num_workers=0, pin_memory=False, imbalance=strategy
    )
    assert isinstance(loaders["training"].sampler, torch.utils.data.RandomSampler)


def test_dataloaders_use_weighted_sampler_only_for_training(imbalanced_dataset_dir: Path) -> None:
    loaders = create_dataloaders(
        imbalanced_dataset_dir,
        batch_size=4,
        image_size=32,
        num_workers=0,
        pin_memory=False,
        imbalance="weighted_sampler",
        seed=0,
    )
    assert isinstance(loaders["training"].sampler, torch.utils.data.WeightedRandomSampler)
    for split in ("validation", "evaluation"):
        assert isinstance(loaders[split].sampler, torch.utils.data.SequentialSampler)
    images, labels = next(iter(loaders["training"]))
    assert images.shape == (4, 3, 32, 32)


def test_dataloaders_reject_unknown_strategy(imbalanced_dataset_dir: Path) -> None:
    with pytest.raises(ValueError, match="Unknown imbalance strategy"):
        create_dataloaders(imbalanced_dataset_dir, num_workers=0, imbalance="oversample")


def test_criterion_weights_only_for_weighted_loss() -> None:
    weighted = build_criterion("weighted_loss", IMBALANCED_TARGETS)
    assert torch.allclose(weighted.weight, class_weights(IMBALANCED_TARGETS))
    assert weighted.label_smoothing == 0.1

    for strategy in ("weighted_sampler", "none"):
        assert build_criterion(strategy, IMBALANCED_TARGETS).weight is None


def test_weighted_loss_penalises_rare_class_errors_more() -> None:
    criterion = build_criterion("weighted_loss", IMBALANCED_TARGETS, label_smoothing=0.0)
    criterion.reduction = "none"
    logits = torch.zeros(2, NUM_CLASSES)
    logits[:, 5] = 3.0  # Both samples predicted as class 5: equally wrong, different true classes.
    losses = criterion(logits, torch.tensor([0, NUM_CLASSES - 1]))
    assert losses[0] > losses[1]
