"""
Evaluation metrics for classification tasks.

This module provides metric computation functions including balanced accuracy,
per-class recall, confusion matrix utilities, and more.
"""


from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np


def _to_1d_long_tensor(values) -> torch.Tensor:
    """Convert numpy/list/tensor inputs to 1D torch.long tensor."""
    if isinstance(values, torch.Tensor):
        tensor = values.detach().clone()
    else:
        tensor = torch.as_tensor(values)

    if tensor.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {tuple(tensor.shape)}")

    if tensor.numel() == 0:
        raise ValueError("Input labels must be non-empty")

    return tensor.long()


def calculate_balanced_accuracy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: Optional[int] = None,
    epsilon: float = 1e-9
) -> float:
    """
    Calculate the balanced accuracy score.

    Balanced accuracy is the average of recall obtained on each class.
    It is particularly useful for imbalanced datasets where standard accuracy
    can be misleading.

    Formula: balanced_accuracy = (1 / num_classes) * Σ recall_i
    where recall_i = true_positives_i / (true_positives_i + false_negatives_i)

    Args:
        y_pred: Predicted class labels (not logits or probabilities).
            Shape: (N,) where N is the number of samples.
        y_true: True class labels.
            Shape: (N,) where N is the number of samples.
        num_classes: Number of classes in the dataset.
        epsilon: Small value to add to denominators to prevent division by zero.
            Default: 1e-9

    Returns:
        Balanced accuracy score as a float in the range [0, 1].
        Returns 0.0 if all classes have no samples (edge case).

    Example:
        >>> y_pred = torch.tensor([0, 1, 2, 0, 1])
        >>> y_true = torch.tensor([0, 1, 1, 0, 2])
        >>> calculate_balanced_accuracy(y_pred, y_true, num_classes=3)
        0.6111111111111112  # (recall_0 + recall_1 + recall_2) / 3

    Note:
        This implementation uses a confusion matrix approach which is more
        efficient than iterating through classes individually.
    """
    # Backward-compatible mode: when num_classes is omitted, treat args as
    # (y_true, y_pred) to match sklearn-like call sites used in tests.
    if num_classes is None:
        y_true_tensor = _to_1d_long_tensor(y_pred)
        y_pred_tensor = _to_1d_long_tensor(y_true)
        num_classes = int(
            max(
                int(torch.max(y_true_tensor).item()),
                int(torch.max(y_pred_tensor).item()),
            ) + 1
        )
    else:
        y_pred_tensor = _to_1d_long_tensor(y_pred)
        y_true_tensor = _to_1d_long_tensor(y_true)

    if y_true_tensor.numel() != y_pred_tensor.numel():
        raise ValueError(
            "y_true and y_pred must have the same number of samples"
        )

    # Ensure tensors are on the same device
    if y_pred_tensor.device != y_true_tensor.device:
        y_pred_tensor = y_pred_tensor.to(y_true_tensor.device)

    # Create confusion matrix
    # confusion_matrix[i, j] = number of samples with true label i and predicted label j
    confusion_matrix = torch.zeros(num_classes, num_classes, device=y_pred_tensor.device)
    for t, p in zip(y_true_tensor.view(-1), y_pred_tensor.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    # Calculate recall for each class
    # Recall = true_positives / (true_positives + false_negatives)
    # true_positives = diagonal elements of confusion matrix
    # (true_positives + false_negatives) = sum of elements in each row
    recall_per_class = torch.diag(confusion_matrix) / (confusion_matrix.sum(1) + epsilon)

    # Calculate balanced accuracy as mean of per-class recalls
    balanced_accuracy = recall_per_class.mean().item()

    return balanced_accuracy


def calculate_per_class_recall(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
    epsilon: float = 1e-9
) -> Dict[int, float]:
    """
    Calculate recall for each class individually.

    Args:
        y_pred: Predicted class labels (not logits or probabilities).
            Shape: (N,) where N is the number of samples.
        y_true: True class labels.
            Shape: (N,) where N is the number of samples.
        num_classes: Number of classes in the dataset.
        epsilon: Small value to add to denominators to prevent division by zero.
            Default: 1e-9

    Returns:
        Dictionary mapping class index to recall score.
        Keys are integers from 0 to (num_classes - 1).
        Values are floats in the range [0, 1].

    Example:
        >>> y_pred = torch.tensor([0, 1, 2, 0, 1])
        >>> y_true = torch.tensor([0, 1, 1, 0, 2])
        >>> calculate_per_class_recall(y_pred, y_true, num_classes=3)
        {0: 1.0, 1: 0.5, 2: 0.0}
    """
    # Ensure tensors are on the same device
    if y_pred.device != y_true.device:
        y_pred = y_pred.to(y_true.device)

    # Create confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, device=y_pred.device)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    # Calculate recall for each class
    recall_per_class = torch.diag(confusion_matrix) / (confusion_matrix.sum(1) + epsilon)

    # Convert to dictionary
    return {i: recall_per_class[i].item() for i in range(num_classes)}


def calculate_per_class_metrics(
    y_true,
    y_pred,
    num_classes: int,
    epsilon: float = 1e-9
) -> Dict[str, np.ndarray]:
    """Backward-compatible per-class metric API expected by tests."""
    y_true_tensor = _to_1d_long_tensor(y_true)
    y_pred_tensor = _to_1d_long_tensor(y_pred)

    if y_true_tensor.numel() != y_pred_tensor.numel():
        raise ValueError("y_true and y_pred must have the same number of samples")

    if y_pred_tensor.device != y_true_tensor.device:
        y_pred_tensor = y_pred_tensor.to(y_true_tensor.device)

    confusion_matrix = torch.zeros(num_classes, num_classes, device=y_pred_tensor.device)
    for t, p in zip(y_true_tensor.view(-1), y_pred_tensor.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    precision = torch.diag(confusion_matrix) / (confusion_matrix.sum(0) + epsilon)
    recall = torch.diag(confusion_matrix) / (confusion_matrix.sum(1) + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    return {
        "precision": precision.detach().cpu().numpy(),
        "recall": recall.detach().cpu().numpy(),
        "f1_score": f1_score.detach().cpu().numpy(),
    }


def calculate_confusion_matrix(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Calculate confusion matrix for classification results.

    Args:
        y_pred: Predicted class labels (not logits or probabilities).
            Shape: (N,) where N is the number of samples.
        y_true: True class labels.
            Shape: (N,) where N is the number of samples.
        num_classes: Number of classes in the dataset.

    Returns:
        Confusion matrix as a 2D tensor of shape (num_classes, num_classes).
        Element [i, j] represents the number of samples with true label i
        and predicted label j.

    Example:
        >>> y_pred = torch.tensor([0, 1, 2, 0, 1])
        >>> y_true = torch.tensor([0, 1, 1, 0, 2])
        >>> calculate_confusion_matrix(y_pred, y_true, num_classes=3)
        tensor([[2., 0., 0.],
                [0., 1., 1.],
                [0., 1., 0.]])
    """
    # Ensure tensors are on the same device
    if y_pred.device != y_true.device:
        y_pred = y_pred.to(y_true.device)

    # Create confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, device=y_pred.device)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix


def compute_confusion_matrix(
    y_true,
    y_pred,
    num_classes: int
) -> np.ndarray:
    """Backward-compatible confusion matrix API expected by tests."""
    y_true_tensor = _to_1d_long_tensor(y_true)
    y_pred_tensor = _to_1d_long_tensor(y_pred)

    if y_true_tensor.numel() != y_pred_tensor.numel():
        raise ValueError("y_true and y_pred must have the same number of samples")

    cm = calculate_confusion_matrix(y_pred_tensor, y_true_tensor, num_classes)
    return cm.detach().cpu().numpy()


def calculate_precision_recall_f1(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
    epsilon: float = 1e-9
) -> Dict[str, Dict[int, float]]:
    """
    Calculate precision, recall, and F1 score for each class.

    Args:
        y_pred: Predicted class labels (not logits or probabilities).
            Shape: (N,) where N is the number of samples.
        y_true: True class labels.
            Shape: (N,) where N is the number of samples.
        num_classes: Number of classes in the dataset.
        epsilon: Small value to add to denominators to prevent division by zero.
            Default: 1e-9

    Returns:
        Dictionary with keys 'precision', 'recall', 'f1', each mapping to
        a dictionary of class index to metric value.

    Example:
        >>> y_pred = torch.tensor([0, 1, 2, 0, 1])
        >>> y_true = torch.tensor([0, 1, 1, 0, 2])
        >>> metrics = calculate_precision_recall_f1(y_pred, y_true, num_classes=3)
        >>> metrics['precision'][0]
        1.0
        >>> metrics['recall'][1]
        0.5
    """
    # Ensure tensors are on the same device
    if y_pred.device != y_true.device:
        y_pred = y_pred.to(y_true.device)

    # Create confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, device=y_pred.device)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    # Calculate precision, recall, F1 for each class
    precision_per_class = torch.diag(confusion_matrix) / (confusion_matrix.sum(0) + epsilon)
    recall_per_class = torch.diag(confusion_matrix) / (confusion_matrix.sum(1) + epsilon)
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + epsilon)

    return {
        'precision': {i: precision_per_class[i].item() for i in range(num_classes)},
        'recall': {i: recall_per_class[i].item() for i in range(num_classes)},
        'f1': {i: f1_per_class[i].item() for i in range(num_classes)}
    }


def calculate_accuracy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor
) -> float:
    """
    Calculate standard classification accuracy.

    Args:
        y_pred: Predicted class labels (not logits or probabilities).
            Shape: (N,) where N is the number of samples.
        y_true: True class labels.
            Shape: (N,) where N is the number of samples.

    Returns:
        Accuracy as a float in the range [0, 1].

    Example:
        >>> y_pred = torch.tensor([0, 1, 2, 0, 1])
        >>> y_true = torch.tensor([0, 1, 1, 0, 2])
        >>> calculate_accuracy(y_pred, y_true)
        0.6  # 3 out of 5 predictions are correct
    """
    # Ensure tensors are on the same device
    if y_pred.device != y_true.device:
        y_pred = y_pred.to(y_true.device)

    correct = (y_pred == y_true).sum().item()
    total = y_true.size(0)

    return correct / total if total > 0 else 0.0


def calculate_top_k_accuracy(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    k: int = 5
) -> float:
    """
    Calculate top-k accuracy.

    Top-k accuracy checks if the true label is among the top k predictions.

    Args:
        logits: Model output logits or probabilities.
            Shape: (N, num_classes) where N is the number of samples.
        y_true: True class labels.
            Shape: (N,) where N is the number of samples.
        k: Number of top predictions to consider. Default: 5

    Returns:
        Top-k accuracy as a float in the range [0, 1].

    Example:
        >>> logits = torch.tensor([[0.1, 0.3, 0.6], [0.2, 0.5, 0.3]])
        >>> y_true = torch.tensor([1, 0])
        >>> calculate_top_k_accuracy(logits, y_true, k=2)
        1.0  # Both true labels are in top-2 predictions
    """
    # Ensure tensors are on the same device
    if logits.device != y_true.device:
        logits = logits.to(y_true.device)

    # Get top k predictions
    _, top_k_pred = logits.topk(k, dim=1)

    # Check if true label is in top k
    correct = top_k_pred.eq(y_true.view(-1, 1).expand_as(top_k_pred)).sum().item()
    total = y_true.size(0)

    return correct / total if total > 0 else 0.0


def format_metrics(
    metrics: Dict[str, float],
    prefix: str = ""
) -> str:
    """
    Format metrics dictionary as a readable string.

    Args:
        metrics: Dictionary mapping metric names to values.
        prefix: Optional prefix to add to each metric name. Default: ""

    Returns:
        Formatted string representation of metrics.

    Example:
        >>> metrics = {'loss': 0.5234, 'accuracy': 0.8567, 'bal_acc': 0.8123}
        >>> print(format_metrics(metrics, prefix="train_"))
        train_loss: 0.5234, train_accuracy: 0.8567, train_bal_acc: 0.8123
    """
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{prefix}{key}: {value:.4f}")
        else:
            formatted.append(f"{prefix}{key}: {value}")

    return ", ".join(formatted)
