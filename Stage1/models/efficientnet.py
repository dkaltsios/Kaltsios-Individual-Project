from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_efficientnet_b0(num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
