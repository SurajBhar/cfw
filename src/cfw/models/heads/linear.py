"""Linear classification head for CFW models."""


import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LinearClassifier(nn.Module):
    """Linear classification head for frozen features.

    This is a simple linear layer that maps backbone features to class logits.
    It's typically used with frozen pretrained backbones for efficient
    linear probing or feature-based training.

    The linear layer is initialized with:
    - Weights: Normal distribution (mean=0.0, std=0.01)
    - Bias: Zeros

    Attributes:
        num_features: Input feature dimension from backbone
        num_classes: Number of output classes
        linear: Linear layer for classification
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        dropout: float = 0.0
    ):
        """Initialize linear classifier.

        Args:
            num_features: Input feature dimension from backbone
            num_classes: Number of output classes
            dropout: Dropout probability (default: 0.0, no dropout)

        Example:
            ```python
            # Binary classification with 768-dim features
            head = LinearClassifier(num_features=768, num_classes=2)

            # Multi-class with dropout
            head = LinearClassifier(num_features=768, num_classes=34, dropout=0.1)
            ```
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        # Optional dropout before classification
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Linear classification layer
        self.linear = nn.Linear(num_features, num_classes)

        # Initialize weights and bias
        self._init_weights()

        logger.debug(
            f"LinearClassifier initialized: "
            f"num_features={num_features}, num_classes={num_classes}, dropout={dropout}"
        )

    def _init_weights(self) -> None:
        """Initialize weights and bias.

        Weights are initialized from a normal distribution with mean=0.0 and
        std=0.01. Bias is initialized to zeros.
        """
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classifier.

        Args:
            x: Input features of shape (batch_size, num_features)

        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        x = self.dropout(x)
        return self.linear(x)

    def get_num_parameters(self) -> int:
        """Get the number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LinearClassifierWithBackbone(nn.Module):
    """Combined backbone + linear classifier model.

    This is a convenience wrapper that combines a backbone model with a
    linear classification head. It handles the forward pass through both
    components.

    Attributes:
        backbone: Feature extraction backbone (e.g., DINOv2, ViT)
        head: Linear classification head
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_features: int,
        num_classes: int,
        dropout: float = 0.0
    ):
        """Initialize combined model.

        Args:
            backbone: Feature extraction backbone
            num_features: Feature dimension from backbone
            num_classes: Number of output classes
            dropout: Dropout probability before classification (default: 0.0)

        Example:
            ```python
            from cfw.models import create_model

            # Create backbone
            backbone = create_model('dinov2_vitb14', freeze_backbone=True)

            # Create combined model
            model = LinearClassifierWithBackbone(
                backbone=backbone,
                num_features=768,
                num_classes=2
            )
            ```
        """
        super().__init__()

        self.backbone = backbone
        self.head = LinearClassifier(
            num_features=num_features,
            num_classes=num_classes,
            dropout=dropout
        )

        logger.info(
            f"LinearClassifierWithBackbone initialized: "
            f"backbone={backbone.get_model_name() if hasattr(backbone, 'get_model_name') else 'unknown'}, "
            f"num_features={num_features}, num_classes={num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and classifier.

        Args:
            x: Input images of shape (batch_size, channels, height, width)

        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Classify
        logits = self.head(features)

        return logits

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the backbone.

        This is useful when you want to train only the classification head.
        """
        if hasattr(self.backbone, 'freeze'):
            self.backbone.freeze()
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

        logger.info("Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters in the backbone.

        This enables fine-tuning of the backbone.
        """
        if hasattr(self.backbone, 'unfreeze'):
            self.backbone.unfreeze()
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

        logger.info("Backbone unfrozen")

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get the number of parameters in the model.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
