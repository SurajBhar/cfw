"""
Model building utilities for CFW.

This module provides high-level utility functions for building complete
models from configuration, including backbone + head combinations.
"""


import torch.nn as nn
from typing import Optional, List, Dict, Any
from omegaconf import DictConfig
import logging

from .registry import create_model as create_backbone
from .heads.linear import LinearClassifier, LinearClassifierWithBackbone
from .heads.mlp import MLPClassifier, MLPClassifierWithBackbone

logger = logging.getLogger(__name__)


def build_classifier(
    backbone_name: str,
    num_classes: int,
    head_type: str = 'linear',
    freeze_backbone: bool = True,
    dropout: float = 0.0,
    hidden_dims: Optional[List[int]] = None,
    use_batch_norm: bool = True,
    **kwargs
) -> nn.Module:
    """Build a complete classification model (backbone + head).

    This is a high-level function that creates a backbone model from the
    registry and attaches a classification head on top.

    Args:
        backbone_name: Name of backbone model (e.g., 'dinov2_vitb14', 'vit_b_16')
        num_classes: Number of output classes
        head_type: Type of classification head ('linear' or 'mlp')
        freeze_backbone: Whether to freeze backbone parameters (default: True)
        dropout: Dropout probability for classification head (default: 0.0)
        hidden_dims: Hidden layer dimensions for MLP head (only for head_type='mlp')
        use_batch_norm: Whether to use batch norm in MLP head (default: True)
        **kwargs: Additional arguments passed to backbone builder

    Returns:
        Complete classification model (backbone + head)

    Raises:
        ValueError: If head_type is not 'linear' or 'mlp'
        KeyError: If backbone_name is not registered

    Example:
        ```python
        # Linear classifier with DINOv2
        model = build_classifier(
            backbone_name='dinov2_vitb14',
            num_classes=2,
            head_type='linear',
            freeze_backbone=True
        )

        # MLP classifier with ViT
        model = build_classifier(
            backbone_name='vit_b_16',
            num_classes=34,
            head_type='mlp',
            hidden_dims=[512, 256],
            dropout=0.1
        )
        ```
    """
    logger.info(f"Building classifier: backbone={backbone_name}, head={head_type}, num_classes={num_classes}")

    if head_type not in {"linear", "mlp"}:
        raise ValueError(
            f"Unknown head type '{head_type}'. Must be 'linear' or 'mlp'."
        )

    # Backward-compatible alias used by legacy tests/configs.
    if hidden_dims is None and 'mlp_hidden_dims' in kwargs:
        hidden_dims = kwargs.pop('mlp_hidden_dims')

    # Create backbone
    try:
        backbone = create_backbone(
            name=backbone_name,
            freeze_backbone=freeze_backbone,
            **kwargs
        )
    except KeyError as exc:
        raise ValueError(f"Model '{backbone_name}' not found") from exc

    # Get feature dimension from backbone
    if hasattr(backbone, 'get_feature_dim'):
        num_features = backbone.get_feature_dim()
    elif hasattr(backbone, 'feature_dim'):
        num_features = backbone.feature_dim
    else:
        raise AttributeError(
            f"Backbone {backbone_name} does not have 'get_feature_dim()' method or 'feature_dim' attribute"
        )

    # Build classifier based on head type
    if head_type == 'linear':
        model = LinearClassifierWithBackbone(
            backbone=backbone,
            num_features=num_features,
            num_classes=num_classes,
            dropout=dropout
        )
    else:  # head_type == 'mlp'
        if hidden_dims is None:
            hidden_dims = [512]  # Default hidden dims
        model = MLPClassifierWithBackbone(
            backbone=backbone,
            num_features=num_features,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
    # Log model info
    num_params = model.get_num_parameters(trainable_only=False)
    num_trainable = model.get_num_parameters(trainable_only=True)
    logger.info(
        f"Model built successfully: "
        f"total_params={num_params:,}, trainable_params={num_trainable:,}"
    )

    return model


def build_classifier_from_config(cfg: DictConfig) -> nn.Module:
    """Build a classifier from Hydra configuration.

    This function extracts model parameters from a Hydra config and
    calls build_classifier to create the model.

    Expected config structure:
    ```yaml
    model:
      backbone: dinov2_vitb14
      num_classes: 2
      head_type: linear
      freeze_backbone: true
      dropout: 0.0
      # Optional for MLP head
      hidden_dims: [512, 256]
      use_batch_norm: true
    ```

    Args:
        cfg: Hydra configuration object

    Returns:
        Complete classification model

    Example:
        ```python
        import hydra
        from omegaconf import DictConfig

        @hydra.main(config_path="configs", config_name="config")
        def main(cfg: DictConfig):
            model = build_classifier_from_config(cfg)
            # Use model...

        if __name__ == "__main__":
            main()
        ```
    """
    # Extract model config
    model_cfg = cfg.model

    # Required parameters
    backbone_name = model_cfg.backbone
    num_classes = model_cfg.num_classes

    # Optional parameters with defaults
    head_type = model_cfg.get('head_type', 'linear')
    freeze_backbone = model_cfg.get('freeze_backbone', True)
    dropout = model_cfg.get('dropout', 0.0)

    # MLP-specific parameters
    kwargs: Dict[str, Any] = {}
    if head_type == 'mlp':
        if 'hidden_dims' in model_cfg:
            kwargs['hidden_dims'] = list(model_cfg.hidden_dims)
        kwargs['use_batch_norm'] = model_cfg.get('use_batch_norm', True)

    logger.info("Building classifier from config")

    return build_classifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        head_type=head_type,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        **kwargs
    )


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get information about a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary containing model information:
        - total_params: Total number of parameters
        - trainable_params: Number of trainable parameters
        - frozen_params: Number of frozen parameters
        - model_name: Model name (if available)

    Example:
        ```python
        model = build_classifier('dinov2_vitb14', num_classes=2)
        info = get_model_info(model)
        print(f"Total params: {info['total_params']:,}")
        print(f"Trainable params: {info['trainable_params']:,}")
        ```
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
    }

    # Try to get model name
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'get_model_name'):
        info['model_name'] = model.backbone.get_model_name()
    elif hasattr(model, 'get_model_name'):
        info['model_name'] = model.get_model_name()
    else:
        info['model_name'] = model.__class__.__name__

    return info


def print_model_summary(model: nn.Module) -> None:
    """Print a summary of the model architecture and parameters.

    Args:
        model: PyTorch model

    Example:
        ```python
        model = build_classifier('dinov2_vitb14', num_classes=2)
        print_model_summary(model)
        ```
    """
    info = get_model_info(model)

    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Model Name:        {info['model_name']}")
    print(f"Total Parameters:  {info['total_params']:,}")
    print(f"Trainable Params:  {info['trainable_params']:,}")
    print(f"Frozen Params:     {info['frozen_params']:,}")
    print("=" * 60)
