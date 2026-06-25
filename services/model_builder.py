from __future__ import annotations

import torch
import torch.nn as nn
import timm
from segmentation_models_pytorch import UnetPlusPlus


def build_classification_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
) -> nn.Module:
    """Build a classification model locally without downloading external weights."""
    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build classification model '{model_name}': {exc}"
        ) from exc
    return model


def build_segmentation_model(
    encoder_name: str = "resnet34",
    in_channels: int = 3,
    classes: int = 1,
    activation: str = None,
) -> nn.Module:
    """Build a U-Net++ segmentation model locally."""
    model = UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    return model
