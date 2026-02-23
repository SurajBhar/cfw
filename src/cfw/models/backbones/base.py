"""
Base backbone interface for CFW models.

This module provides abstract base classes for backbone models to ensure
consistent interfaces across different architectures.
"""


from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn


class BackboneBase(ABC, nn.Module):
    """Abstract base class for backbone models.

    All backbone models should inherit from this class and implement the
    required methods to ensure a consistent interface.

    Attributes:
        feature_dim: Dimensionality of the output features
        model_name: Name of the backbone model
    """

    def __init__(self, feature_dim: int, model_name: str):
        """Initialize backbone base.

        Args:
            feature_dim: Dimensionality of output features
            model_name: Name of the model
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.model_name = model_name

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        pass

    @abstractmethod
    def freeze(self) -> None:
        """Freeze all parameters in the backbone.

        This is typically used when the backbone is used as a feature
        extractor with a frozen pretrained model.
        """
        pass

    @abstractmethod
    def unfreeze(self) -> None:
        """Unfreeze all parameters in the backbone.

        This allows fine-tuning of the backbone parameters.
        """
        pass

    def get_feature_dim(self) -> int:
        """Get the output feature dimensionality.

        Returns:
            Feature dimension
        """
        return self.feature_dim

    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            Model name
        """
        return self.model_name


class ViTBackbone(BackboneBase):
    """Base class for Vision Transformer backbones.

    Provides common functionality for ViT-based models including
    DINOv2 and standard ViT architectures.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_dim: int,
        model_name: str,
        freeze_backbone: bool = True
    ):
        """Initialize ViT backbone.

        Args:
            model: The underlying ViT model
            feature_dim: Dimensionality of output features
            model_name: Name of the model
            freeze_backbone: Whether to freeze the backbone by default
        """
        super().__init__(feature_dim, model_name)
        self.model = model

        if freeze_backbone:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ViT backbone.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        return self.model(x)

    def freeze(self) -> None:
        """Freeze all parameters in the backbone."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze(self) -> None:
        """Unfreeze all parameters in the backbone."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get the number of parameters in the model.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())
