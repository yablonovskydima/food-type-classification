import pytest
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.data.constants import NUM_CLASSES
from src.models import build_model, count_parameters, factory


@pytest.mark.parametrize("image_size", [128, 224])
def test_scratch_model_output_shape(image_size: int) -> None:
    model = build_model(pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, image_size, image_size))
    assert logits.shape == (2, NUM_CLASSES)


def test_custom_num_classes() -> None:
    model = build_model(num_classes=5)
    assert model.fc.out_features == 5


def test_resnet18_parameter_count() -> None:
    model = build_model()
    # ResNet-18 backbone (~11.18M) + 512 * 11 + 11 head parameters.
    assert count_parameters(model) == 11_176_512 + 512 * NUM_CLASSES + NUM_CLASSES


def test_count_parameters_trainable_only() -> None:
    model = build_model()
    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True
    assert count_parameters(model) == 512 * NUM_CLASSES
    assert count_parameters(model, trainable_only=False) > count_parameters(model)


def test_scratch_models_differ_between_calls() -> None:
    first, second = build_model(), build_model()
    assert not torch.equal(first.conv1.weight, second.conv1.weight)


def test_pretrained_requests_imagenet_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[object] = []

    def fake_resnet18(weights: object = None) -> nn.Module:
        requested.append(weights)
        return resnet18(weights=None)

    monkeypatch.setattr(factory, "resnet18", fake_resnet18)

    model = build_model(pretrained=True)
    assert requested == [ResNet18_Weights.IMAGENET1K_V1]
    assert model.fc.out_features == NUM_CLASSES

    build_model(pretrained=False)
    assert requested[-1] is None
