"""
Loss functions for training classification models.

This module provides loss function wrappers and utilities including
cross-entropy with label smoothing, focal loss, and more.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss wrapper with optional label smoothing.

    This is a wrapper around torch.nn.functional.cross_entropy that provides
    additional features like label smoothing and a consistent interface.

    Args:
        weight: Manual rescaling weight for each class. Shape: (num_classes,)
            Default: None (equal weight for all classes)
        label_smoothing: Label smoothing factor in [0, 1]. Smooths the target
            distribution by mixing it with a uniform distribution.
            Default: 0.0 (no smoothing)
        reduction: Specifies the reduction to apply to the output:
            - 'mean': average loss over all samples
            - 'sum': sum loss over all samples
            - 'none': no reduction, returns loss per sample
            Default: 'mean'
        ignore_index: Target value that is ignored and doesn't contribute to
            the gradient. Default: -100

    Example:
        >>> criterion = CrossEntropyLoss(label_smoothing=0.1)
        >>> logits = torch.randn(32, 10)  # batch_size=32, num_classes=10
        >>> targets = torch.randint(0, 10, (32,))
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """Initialize cross-entropy loss options."""
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            input: Model output logits. Shape: (N, C) where N is batch size
                and C is number of classes.
            target: Ground truth class labels. Shape: (N,)

        Returns:
            Loss value (scalar if reduction='mean' or 'sum', tensor if 'none').
        """
        return F.cross_entropy(
            input=input,
            target=target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal loss applies a modulating term to the cross entropy loss to focus
    learning on hard misclassified examples. It is particularly useful for
    handling class imbalance.

    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)

    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples.
            Can be a scalar or a tensor of shape (num_classes,) for per-class weights.
            Default: 0.25
        gamma: Focusing parameter γ >= 0. Higher gamma increases the focus on
            hard examples. Default: 2.0
        reduction: Specifies the reduction: 'mean', 'sum', or 'none'.
            Default: 'mean'

    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(32, 10)
        >>> targets = torch.randint(0, 10, (32,))
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """Initialize focal-loss parameters."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            input: Model output logits. Shape: (N, C)
            target: Ground truth class labels. Shape: (N,)

        Returns:
            Loss value.
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(input, target, reduction='none')

        # Get probabilities
        p = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with per-sample weighting.

    This loss function accepts per-sample weights, useful for weighted sampling
    strategies like CFW where each sample has an associated weight.

    Args:
        reduction: Specifies the reduction: 'mean', 'sum', or 'none'.
            Default: 'mean'
        label_smoothing: Label smoothing factor. Default: 0.0

    Example:
        >>> criterion = WeightedCrossEntropyLoss()
        >>> logits = torch.randn(32, 10)
        >>> targets = torch.randint(0, 10, (32,))
        >>> weights = torch.rand(32)  # Per-sample weights
        >>> loss = criterion(logits, targets, weights)
    """

    def __init__(
        self,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """Initialize weighted cross-entropy options."""
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            input: Model output logits. Shape: (N, C)
            target: Ground truth class labels. Shape: (N,)
            weight: Per-sample weights. Shape: (N,). If None, uses equal weights.

        Returns:
            Loss value.
        """
        # Compute unreduced loss
        loss = F.cross_entropy(
            input,
            target,
            reduction='none',
            label_smoothing=self.label_smoothing
        )

        # Apply per-sample weights if provided
        if weight is not None:
            loss = loss * weight

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Create a loss function by name.

    Args:
        loss_name: Name of the loss function. Options:
            - 'cross_entropy': Standard cross-entropy loss
            - 'focal': Focal loss for class imbalance
            - 'weighted_ce': Weighted cross-entropy loss
        **kwargs: Additional arguments to pass to the loss function.

    Returns:
        Loss function module.

    Raises:
        ValueError: If loss_name is not recognized.

    Example:
        >>> criterion = get_loss_function('cross_entropy', label_smoothing=0.1)
        >>> criterion = get_loss_function('focal', alpha=0.25, gamma=2.0)
    """
    loss_name = loss_name.lower()

    if loss_name in ['cross_entropy', 'ce']:
        return CrossEntropyLoss(**kwargs)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    elif loss_name in ['weighted_ce', 'weighted_cross_entropy']:
        return WeightedCrossEntropyLoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Available options: cross_entropy, focal, weighted_ce"
        )


def compute_loss_with_metrics(
    loss_fn: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, dict]:
    """
    Compute loss and return additional metrics.

    This is a convenience function that computes the loss and also returns
    useful metrics like accuracy.

    Args:
        loss_fn: Loss function module.
        logits: Model output logits. Shape: (N, C)
        targets: Ground truth class labels. Shape: (N,)
        weights: Per-sample weights for weighted losses. Shape: (N,)

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict contains:
        - 'loss': loss value (detached)
        - 'accuracy': classification accuracy

    Example:
        >>> criterion = CrossEntropyLoss()
        >>> loss, metrics = compute_loss_with_metrics(
        ...     criterion, logits, targets
        ... )
        >>> print(f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")
    """
    # Compute loss
    if isinstance(loss_fn, WeightedCrossEntropyLoss) and weights is not None:
        loss = loss_fn(logits, targets, weights)
    else:
        loss = loss_fn(logits, targets)

    # Compute accuracy
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == targets).float().mean()

    # Prepare metrics
    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy.item()
    }

    return loss, metrics
