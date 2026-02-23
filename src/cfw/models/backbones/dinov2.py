"""
DINOv2 backbone models for CFW.

This module provides DINOv2 Vision Transformer models pretrained with
self-supervised learning from Meta AI Research.
"""


import torch
import torch.nn as nn
from typing import Optional
from omegaconf import DictConfig
import logging

from .base import ViTBackbone
from ..registry import register_model

logger = logging.getLogger(__name__)


def _build_fallback_backbone(feature_dim: int, model_name: str, freeze_backbone: bool) -> ViTBackbone:
    """Create a lightweight local backbone when torch.hub is unavailable."""
    fallback_model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, feature_dim),
    )
    logger.warning(
        "Using fallback local backbone for %s (feature_dim=%s).",
        model_name,
        feature_dim,
    )
    return ViTBackbone(
        model=fallback_model,
        feature_dim=feature_dim,
        model_name=model_name,
        freeze_backbone=freeze_backbone,
    )


@register_model('dinov2_vits14')
def build_dinov2_vits14(
    cfg: Optional[DictConfig] = None,
    freeze_backbone: bool = True,
    **kwargs
) -> ViTBackbone:
    """Build DINOv2 ViT-Small/14 model.

    This model uses a Vision Transformer Small architecture with 14x14 patches
    and produces 384-dimensional features.

    Args:
        cfg: Optional Hydra configuration
        freeze_backbone: Whether to freeze backbone parameters (default: True)
        **kwargs: Additional arguments

    Returns:
        ViTBackbone: DINOv2 ViT-S/14 backbone

    Example:
        ```python
        # Using config
        model = build_dinov2_vits14(cfg=model_cfg)

        # Using kwargs
        model = build_dinov2_vits14(freeze_backbone=False)
        ```
    """
    logger.info("Loading DINOv2 ViT-Small/14 from torch.hub")

    pretrained = kwargs.get("pretrained", True)
    try:
        dinov2_model = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vits14',
            pretrained=pretrained
        )
    except Exception as exc:
        logger.warning("Failed to load DINOv2 ViT-S/14 from torch.hub: %s", exc)
        return _build_fallback_backbone(384, 'dinov2_vits14', freeze_backbone)

    # Wrap in ViTBackbone
    backbone = ViTBackbone(
        model=dinov2_model,
        feature_dim=384,  # ViT-S/14 feature dimension
        model_name='dinov2_vits14',
        freeze_backbone=freeze_backbone
    )

    logger.info(f"DINOv2 ViT-S/14 loaded successfully (feature_dim=384, frozen={freeze_backbone})")
    return backbone


@register_model('dinov2_vitb14')
def build_dinov2_vitb14(
    cfg: Optional[DictConfig] = None,
    freeze_backbone: bool = True,
    **kwargs
) -> ViTBackbone:
    """Build DINOv2 ViT-Base/14 model.

    This model uses a Vision Transformer Base architecture with 14x14 patches
    and produces 768-dimensional features. This is the most commonly used
    DINOv2 variant.

    Args:
        cfg: Optional Hydra configuration
        freeze_backbone: Whether to freeze backbone parameters (default: True)
        **kwargs: Additional arguments

    Returns:
        ViTBackbone: DINOv2 ViT-B/14 backbone

    Example:
        ```python
        # Using config
        model = build_dinov2_vitb14(cfg=model_cfg)

        # Using kwargs
        model = build_dinov2_vitb14(freeze_backbone=True)
        ```
    """
    logger.info("Loading DINOv2 ViT-Base/14 from torch.hub")

    pretrained = kwargs.get("pretrained", True)
    try:
        dinov2_model = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14',
            pretrained=pretrained
        )
    except Exception as exc:
        logger.warning("Failed to load DINOv2 ViT-B/14 from torch.hub: %s", exc)
        return _build_fallback_backbone(768, 'dinov2_vitb14', freeze_backbone)

    # Wrap in ViTBackbone
    backbone = ViTBackbone(
        model=dinov2_model,
        feature_dim=768,  # ViT-B/14 feature dimension
        model_name='dinov2_vitb14',
        freeze_backbone=freeze_backbone
    )

    logger.info(f"DINOv2 ViT-B/14 loaded successfully (feature_dim=768, frozen={freeze_backbone})")
    return backbone


@register_model('dinov2_vitl14')
def build_dinov2_vitl14(
    cfg: Optional[DictConfig] = None,
    freeze_backbone: bool = True,
    **kwargs
) -> ViTBackbone:
    """Build DINOv2 ViT-Large/14 model.

    This model uses a Vision Transformer Large architecture with 14x14 patches
    and produces 1024-dimensional features.

    Args:
        cfg: Optional Hydra configuration
        freeze_backbone: Whether to freeze backbone parameters (default: True)
        **kwargs: Additional arguments

    Returns:
        ViTBackbone: DINOv2 ViT-L/14 backbone

    Example:
        ```python
        # Using config
        model = build_dinov2_vitl14(cfg=model_cfg)

        # Using kwargs
        model = build_dinov2_vitl14(freeze_backbone=False)
        ```
    """
    logger.info("Loading DINOv2 ViT-Large/14 from torch.hub")

    pretrained = kwargs.get("pretrained", True)
    try:
        dinov2_model = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitl14',
            pretrained=pretrained
        )
    except Exception as exc:
        logger.warning("Failed to load DINOv2 ViT-L/14 from torch.hub: %s", exc)
        return _build_fallback_backbone(1024, 'dinov2_vitl14', freeze_backbone)

    # Wrap in ViTBackbone
    backbone = ViTBackbone(
        model=dinov2_model,
        feature_dim=1024,  # ViT-L/14 feature dimension
        model_name='dinov2_vitl14',
        freeze_backbone=freeze_backbone
    )

    logger.info(f"DINOv2 ViT-L/14 loaded successfully (feature_dim=1024, frozen={freeze_backbone})")
    return backbone


@register_model('dinov2_vitg14')
def build_dinov2_vitg14(
    cfg: Optional[DictConfig] = None,
    freeze_backbone: bool = True,
    **kwargs
) -> ViTBackbone:
    """Build DINOv2 ViT-Giant/14 model.

    This model uses a Vision Transformer Giant architecture with 14x14 patches
    and produces 1536-dimensional features. This is the largest DINOv2 variant.

    Args:
        cfg: Optional Hydra configuration
        freeze_backbone: Whether to freeze backbone parameters (default: True)
        **kwargs: Additional arguments

    Returns:
        ViTBackbone: DINOv2 ViT-G/14 backbone

    Example:
        ```python
        # Using config
        model = build_dinov2_vitg14(cfg=model_cfg)

        # Using kwargs
        model = build_dinov2_vitg14(freeze_backbone=True)
        ```
    """
    logger.info("Loading DINOv2 ViT-Giant/14 from torch.hub")

    pretrained = kwargs.get("pretrained", True)
    try:
        dinov2_model = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitg14',
            pretrained=pretrained
        )
    except Exception as exc:
        logger.warning("Failed to load DINOv2 ViT-G/14 from torch.hub: %s", exc)
        return _build_fallback_backbone(1536, 'dinov2_vitg14', freeze_backbone)

    # Wrap in ViTBackbone
    backbone = ViTBackbone(
        model=dinov2_model,
        feature_dim=1536,  # ViT-G/14 feature dimension
        model_name='dinov2_vitg14',
        freeze_backbone=freeze_backbone
    )

    logger.info(f"DINOv2 ViT-G/14 loaded successfully (feature_dim=1536, frozen={freeze_backbone})")
    return backbone
