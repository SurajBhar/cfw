"""MLP classification head for CFW models."""


import torch
import torch.nn as nn
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class MLPClassifier(nn.Module):
    """Multi-layer perceptron classification head.

    This MLP head provides additional capacity compared to a simple linear
    classifier. It consists of one or more hidden layers with ReLU activation,
    batch normalization, and dropout for regularization.

    Architecture:
    - Input features
    - [Linear -> BatchNorm -> ReLU -> Dropout] x (num_hidden_layers)
    - Linear (output layer)

    Attributes:
        num_features: Input feature dimension from backbone
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dims: List[int] = [512],
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """Initialize MLP classifier.

        Args:
            num_features: Input feature dimension from backbone
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions (default: [512])
            dropout: Dropout probability (default: 0.1)
            use_batch_norm: Whether to use batch normalization (default: True)

        Example:
            ```python
            # Simple MLP with one hidden layer
            head = MLPClassifier(num_features=768, num_classes=2, hidden_dims=[512])

            # Deeper MLP with multiple hidden layers
            head = MLPClassifier(
                num_features=768,
                num_classes=34,
                hidden_dims=[1024, 512, 256],
                dropout=0.2
            )
            ```
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Build MLP layers
        layers = []
        in_dim = num_features

        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(in_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.ReLU(inplace=True))

            # Dropout
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

        logger.debug(
            f"MLPClassifier initialized: "
            f"num_features={num_features}, num_classes={num_classes}, "
            f"hidden_dims={hidden_dims}, dropout={dropout}, use_batch_norm={use_batch_norm}"
        )

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization for ReLU networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP classifier.

        Args:
            x: Input features of shape (batch_size, num_features)

        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        return self.mlp(x)

    def get_num_parameters(self) -> int:
        """Get the number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPClassifierWithBackbone(nn.Module):
    """Combined backbone + MLP classifier model.

    This is a convenience wrapper that combines a backbone model with an
    MLP classification head.

    Attributes:
        backbone: Feature extraction backbone (e.g., DINOv2, ViT)
        head: MLP classification head
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_features: int,
        num_classes: int,
        hidden_dims: List[int] = [512],
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """Initialize combined model.

        Args:
            backbone: Feature extraction backbone
            num_features: Feature dimension from backbone
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions (default: [512])
            dropout: Dropout probability (default: 0.1)
            use_batch_norm: Whether to use batch normalization (default: True)

        Example:
            ```python
            from cfw.models import create_model

            # Create backbone
            backbone = create_model('dinov2_vitb14', freeze_backbone=True)

            # Create combined model with MLP head
            model = MLPClassifierWithBackbone(
                backbone=backbone,
                num_features=768,
                num_classes=2,
                hidden_dims=[512, 256]
            )
            ```
        """
        super().__init__()

        self.backbone = backbone
        self.head = MLPClassifier(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

        logger.info(
            f"MLPClassifierWithBackbone initialized: "
            f"backbone={backbone.get_model_name() if hasattr(backbone, 'get_model_name') else 'unknown'}, "
            f"num_features={num_features}, num_classes={num_classes}, hidden_dims={hidden_dims}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and MLP classifier.

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
