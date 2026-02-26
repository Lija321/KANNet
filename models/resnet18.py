#Made with assitance of ChatGPT
from __future__ import annotations

import torch.nn as nn

from .common import build_preprocess


def get_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    import torchvision.models as models
    from torchvision.models import ResNet18_Weights

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_preprocess(pretrained: bool = True, img_size: int = 224):
    from torchvision.models import ResNet18_Weights

    weights = ResNet18_Weights.IMAGENET1K_V1
    return build_preprocess(weights=weights, pretrained=pretrained, img_size=img_size)
