"""
Models package for CFW.

This package provides a plug-and-play model registry system for easy
integration of different backbone architectures (DINOv2, ViT, etc.)
with classification heads (linear, MLP).

Public API:
-----------
Registry functions:
    - create_model: Factory function to create models from registry
    - register_model: Decorator to register new models
    - list_models: List all registered models

Builder functions:
    - build_classifier: Build complete model (backbone + head)
    - build_classifier_from_config: Build from Hydra config
    - get_model_info: Get model parameter information
    - print_model_summary: Print model summary

Backbones:
    - BackboneBase: Abstract base class for backbones
    - ViTBackbone: Base class for ViT-based backbones

Classification heads:
    - LinearClassifier: Simple linear classification head
    - LinearClassifierWithBackbone: Combined backbone + linear head
    - MLPClassifier: Multi-layer perceptron head
    - MLPClassifierWithBackbone: Combined backbone + MLP head

Example usage:
--------------
    ```python
    from cfw.models import create_model, build_classifier, list_models

    # List available models
    print(list_models())

    # Create a backbone only
    backbone = create_model('dinov2_vitb14', freeze_backbone=True)

    # Build complete classifier
    model = build_classifier(
        backbone_name='dinov2_vitb14',
        num_classes=2,
        head_type='linear'
    )

    # Build from Hydra config
    from omegaconf import DictConfig
    model = build_classifier_from_config(cfg)
    ```
"""


# Import all backbone builders to trigger registration
from .backbones import dinov2, resnet, vit  # noqa: F401

# Import registry functions
from .registry import (
    create_model,
    register_model,
    unregister_model,
    list_models,
    is_model_registered,
    get_model_builder,
    clear_registry,
)

# Import base classes
from .backbones.base import BackboneBase, ViTBackbone

# Import heads
from .heads.linear import (
    LinearClassifier,
    LinearClassifierWithBackbone,
)
from .heads.mlp import (
    MLPClassifier,
    MLPClassifierWithBackbone,
)

# Import builder functions
from .builders import (
    build_classifier,
    build_classifier_from_config,
    get_model_info,
    print_model_summary,
)

__all__ = [
    # Registry functions
    'create_model',
    'register_model',
    'unregister_model',
    'list_models',
    'is_model_registered',
    'get_model_builder',
    'clear_registry',
    # Base classes
    'BackboneBase',
    'ViTBackbone',
    # Heads
    'LinearClassifier',
    'LinearClassifierWithBackbone',
    'MLPClassifier',
    'MLPClassifierWithBackbone',
    # Builders
    'build_classifier',
    'build_classifier_from_config',
    'get_model_info',
    'print_model_summary',
]
