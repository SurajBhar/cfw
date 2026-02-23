"""
Backbone models package for CFW.

This package contains backbone model implementations including:
- DINOv2: Self-supervised Vision Transformers from Meta AI
- ViT: Vision Transformers from timm library
- Base classes for consistent backbone interfaces
"""


from .base import BackboneBase, ViTBackbone
from . import dinov2
from . import resnet
from . import vit

__all__ = [
    'BackboneBase',
    'ViTBackbone',
    'dinov2',
    'resnet',
    'vit',
]
