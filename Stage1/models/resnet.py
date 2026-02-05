from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    )
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
