"""Factory helpers for building model backbones used across services."""

from __future__ import annotations

from typing import Optional
import logging

import torch.nn as nn
import timm

from config import PRETRAINED, CLASSIFICATION_MODEL_NAME

logger = logging.getLogger(__name__)


def build_classification_model(
    model_name: Optional[str] = None,
    num_classes: int = 2,
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
    model_name = model_name or CLASSIFICATION_MODEL_NAME
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
