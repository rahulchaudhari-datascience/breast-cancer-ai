from __future__ import annotations

from typing import Optional
import logging

import torch.nn as nn
import timm
from segmentation_models_pytorch import UnetPlusPlus

from config import PRETRAINED, ENCODER_WEIGHTS

logger = logging.getLogger(__name__)


def build_classification_model(
    model_name: str,
    num_classes: int,
    pretrained: Optional[bool] = None,
) -> nn.Module:
    """Create a classification model using `timm.create_model`.

    Args:
        model_name: timm model identifier (e.g. 'efficientnet_b0').
        num_classes: number of output classes.
        pretrained: whether to load pretrained ImageNet weights. If None, uses
            `config.PRETRAINED`.
    Returns:
        nn.Module instantiation of the model.
    """
    use_pretrained = PRETRAINED if pretrained is None else bool(pretrained)

    try:
        model = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=num_classes,
        )
        logger.info("Built classification model %s (pretrained=%s)", model_name, use_pretrained)
    except Exception as exc:
        logger.exception("Failed to build classification model '%s'", model_name)
        raise

    return model


def build_segmentation_model(
    encoder_name: str = "resnet34",
    in_channels: int = 3,
    classes: int = 1,
    activation: Optional[str] = None,
    encoder_weights: Optional[str] = None,
) -> nn.Module:
    """Create a U-Net++ segmentation model.

    Args:
        encoder_name: backbone encoder name supported by segmentation_models_pytorch.
        in_channels: input channels (usually 3)
        classes: number of output channels/classes for segmentation mask
        activation: activation for the final layer (e.g., None, 'sigmoid')
        encoder_weights: weights for the encoder (e.g. 'imagenet'). If None, uses
            `config.ENCODER_WEIGHTS`.
    """
    weights = ENCODER_WEIGHTS if encoder_weights is None else encoder_weights

    model = UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )

    logger.info("Built UnetPlusPlus encoder=%s weights=%s", encoder_name, weights)
    return model
