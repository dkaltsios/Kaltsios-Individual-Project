"""
Model definitions for Stage1.
"""

from __future__ import annotations

from Stage1.models.efficientnet import build_efficientnet_b0
from Stage1.models.resnet import build_resnet50


def build_model(name: str, num_classes: int = 1, pretrained: bool = True):
    if name == "efficientnet_b0":
        return build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)
    if name == "resnet50":
        return build_resnet50(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model name: {name}")
