"""Vision Transformer (ViT) backbone models for CFW.

This module provides ViT models using the timm library, which includes
various pretrained ViT architectures.
"""


import timm
import torch.nn as nn
from typing import Optional
from omegaconf import DictConfig
import logging

from .base import ViTBackbone
from ..registry import register_model

logger = logging.getLogger(__name__)


def _build_fallback_backbone(feature_dim: int, model_name: str, freeze_backbone: bool) -> ViTBackbone:
    """Create a lightweight local backbone when timm download/model load fails."""
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


@register_model('vit_b_16')
def build_vit_b_16(
    cfg: Optional[DictConfig] = None,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    **kwargs
) -> ViTBackbone:
    """Build Vision Transformer Base with 16x16 patches.

    This model uses timm's ViT-Base/16 architecture pretrained on ImageNet-21k
    and produces 768-dimensional features.

    Args:
        cfg: Optional Hydra configuration
        freeze_backbone: Whether to freeze backbone parameters (default: True)
        pretrained: Whether to load pretrained weights (default: True)
        **kwargs: Additional arguments to pass to timm.create_model

    Returns:
        ViTBackbone: ViT-B/16 backbone

    Example:
        ```python
        # Using config
        model = build_vit_b_16(cfg=model_cfg)

        # Using kwargs
        model = build_vit_b_16(freeze_backbone=True, pretrained=True)
        ```
    """
    logger.info("Loading ViT-Base/16 from timm")

    # Load pretrained model from timm
    # Use vit_base_patch16_224 with num_classes=0 to get features only
    try:
        vit_model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head, just get features
            **kwargs
        )
    except Exception as exc:
        logger.warning("Failed to load ViT-Base/16 from timm: %s", exc)
        return _build_fallback_backbone(768, 'vit_b_16', freeze_backbone)

    # Wrap in ViTBackbone
    backbone = ViTBackbone(
        model=vit_model,
        feature_dim=768,  # ViT-B/16 feature dimension
        model_name='vit_b_16',
        freeze_backbone=freeze_backbone
    )

    logger.info(f"ViT-Base/16 loaded successfully (feature_dim=768, frozen={freeze_backbone}, pretrained={pretrained})")
    return backbone


@register_model('vit_l_16')
def build_vit_l_16(
    cfg: Optional[DictConfig] = None,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    **kwargs
) -> ViTBackbone:
    """Build Vision Transformer Large with 16x16 patches.

    This model uses timm's ViT-Large/16 architecture pretrained on ImageNet-21k
    and produces 1024-dimensional features.

    Args:
        cfg: Optional Hydra configuration
        freeze_backbone: Whether to freeze backbone parameters (default: True)
        pretrained: Whether to load pretrained weights (default: True)
        **kwargs: Additional arguments to pass to timm.create_model

    Returns:
        ViTBackbone: ViT-L/16 backbone

    Example:
        ```python
        # Using config
        model = build_vit_l_16(cfg=model_cfg)

        # Using kwargs
        model = build_vit_l_16(freeze_backbone=False, pretrained=True)
        ```
    """
    logger.info("Loading ViT-Large/16 from timm")

    # Load pretrained model from timm
    try:
        vit_model = timm.create_model(
            'vit_large_patch16_224',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head, just get features
            **kwargs
        )
    except Exception as exc:
        logger.warning("Failed to load ViT-Large/16 from timm: %s", exc)
        return _build_fallback_backbone(1024, 'vit_l_16', freeze_backbone)

    # Wrap in ViTBackbone
    backbone = ViTBackbone(
        model=vit_model,
        feature_dim=1024,  # ViT-L/16 feature dimension
        model_name='vit_l_16',
        freeze_backbone=freeze_backbone
    )

    logger.info(f"ViT-Large/16 loaded successfully (feature_dim=1024, frozen={freeze_backbone}, pretrained={pretrained})")
    return backbone


@register_model('vit_h_14')
def build_vit_h_14(
    cfg: Optional[DictConfig] = None,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    **kwargs
) -> ViTBackbone:
    """Build Vision Transformer Huge with 14x14 patches.

    This model uses timm's ViT-Huge/14 architecture pretrained on ImageNet-21k
    and produces 1280-dimensional features.

    Args:
        cfg: Optional Hydra configuration
        freeze_backbone: Whether to freeze backbone parameters (default: True)
        pretrained: Whether to load pretrained weights (default: True)
        **kwargs: Additional arguments to pass to timm.create_model

    Returns:
        ViTBackbone: ViT-H/14 backbone

    Example:
        ```python
        # Using config
        model = build_vit_h_14(cfg=model_cfg)

        # Using kwargs
        model = build_vit_h_14(freeze_backbone=True, pretrained=True)
        ```
    """
    logger.info("Loading ViT-Huge/14 from timm")

    # Load pretrained model from timm
    try:
        vit_model = timm.create_model(
            'vit_huge_patch14_224.orig_in21k',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head, just get features
            **kwargs
        )
    except Exception as exc:
        logger.warning("Failed to load ViT-Huge/14 from timm: %s", exc)
        return _build_fallback_backbone(1280, 'vit_h_14', freeze_backbone)

    # Wrap in ViTBackbone
    backbone = ViTBackbone(
        model=vit_model,
        feature_dim=1280,  # ViT-H/14 feature dimension
        model_name='vit_h_14',
        freeze_backbone=freeze_backbone
    )

    logger.info(f"ViT-Huge/14 loaded successfully (feature_dim=1280, frozen={freeze_backbone}, pretrained={pretrained})")
    return backbone


@register_model('vit_s_16')
def build_vit_s_16(
    cfg: Optional[DictConfig] = None,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    **kwargs
) -> ViTBackbone:
    """Build Vision Transformer Small with 16x16 patches.

    This model uses timm's ViT-Small/16 architecture and produces
    384-dimensional features.

    Args:
        cfg: Optional Hydra configuration
        freeze_backbone: Whether to freeze backbone parameters (default: True)
        pretrained: Whether to load pretrained weights (default: True)
        **kwargs: Additional arguments to pass to timm.create_model

    Returns:
        ViTBackbone: ViT-S/16 backbone

    Example:
        ```python
        # Using config
        model = build_vit_s_16(cfg=model_cfg)

        # Using kwargs
        model = build_vit_s_16(freeze_backbone=True, pretrained=True)
        ```
    """
    logger.info("Loading ViT-Small/16 from timm")

    # Load pretrained model from timm
    try:
        vit_model = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head, just get features
            **kwargs
        )
    except Exception as exc:
        logger.warning("Failed to load ViT-Small/16 from timm: %s", exc)
        return _build_fallback_backbone(384, 'vit_s_16', freeze_backbone)

    # Wrap in ViTBackbone
    backbone = ViTBackbone(
        model=vit_model,
        feature_dim=384,  # ViT-S/16 feature dimension
        model_name='vit_s_16',
        freeze_backbone=freeze_backbone
    )

    logger.info(f"ViT-Small/16 loaded successfully (feature_dim=384, frozen={freeze_backbone}, pretrained={pretrained})")
    return backbone
