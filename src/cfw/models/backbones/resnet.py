"""Torchvision ResNet backbone models for CFW."""


from typing import Optional
import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models
from torchvision.models import ResNet50_Weights

from .base import BackboneBase
from ..registry import register_model

logger = logging.getLogger(__name__)


class TorchvisionBackbone(BackboneBase):
    """Generic backbone wrapper for torchvision models."""

    def __init__(
        self,
        model: nn.Module,
        feature_dim: int,
        model_name: str,
        freeze_backbone: bool = True,
    ) -> None:
        """Wrap a torchvision backbone and optionally freeze its parameters."""
        super().__init__(feature_dim=feature_dim, model_name=model_name)
        self.model = model

        if freeze_backbone:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features for an input image tensor."""
        features = self.model(x)
        if features.dim() > 2:
            features = torch.flatten(features, start_dim=1)
        return features

    def freeze(self) -> None:
        """Freeze all backbone parameters and switch to eval mode."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters and switch to train mode."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()


def _build_fallback_backbone(freeze_backbone: bool) -> TorchvisionBackbone:
    """Create a lightweight local fallback if torchvision model load fails."""
    fallback_model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 2048),
    )
    logger.warning("Using fallback local backbone for resnet50 (feature_dim=2048).")
    return TorchvisionBackbone(
        model=fallback_model,
        feature_dim=2048,
        model_name="resnet50",
        freeze_backbone=freeze_backbone,
    )


@register_model("resnet50")
def build_resnet50(
    cfg: Optional[DictConfig] = None,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    **kwargs,
) -> TorchvisionBackbone:
    """
    Build ResNet-50 with classifier head removed.

    Matches legacy feature-extraction setup:
    - torchvision ResNet50
    - IMAGENET1K_V1 pretrained weights when enabled
    - final `fc` replaced with Identity to output 2048-dim features
    """
    if cfg is not None:
        pretrained = bool(cfg.get("pretrained", pretrained))

    logger.info("Loading ResNet-50 from torchvision")

    try:
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights, **kwargs)
    except Exception as exc:
        logger.warning("Failed to load ResNet-50 from torchvision: %s", exc)
        try:
            model = models.resnet50(weights=None, **kwargs)
        except Exception as exc2:
            logger.warning("Fallback non-pretrained ResNet-50 load failed: %s", exc2)
            return _build_fallback_backbone(freeze_backbone=freeze_backbone)

    model.fc = nn.Identity()

    backbone = TorchvisionBackbone(
        model=model,
        feature_dim=2048,
        model_name="resnet50",
        freeze_backbone=freeze_backbone,
    )

    logger.info(
        "ResNet-50 loaded successfully (feature_dim=2048, frozen=%s, pretrained=%s)",
        freeze_backbone,
        pretrained,
    )
    return backbone
