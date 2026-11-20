from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.data.constants import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = False) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
