"""
Custom samplers for CFW training.

This module provides sampler utilities for weighted random sampling
used in the CFW (Clustered Feature Weighting) approach, including
a DistributedWeightedSampler for CFW + DDP multi-GPU training.
"""


import math
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import WeightedRandomSampler, Sampler


def create_weighted_sampler(
    weights: List[float],
    num_samples: Optional[int] = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler from sample weights.

    Args:
        weights: List of weights for each sample in the dataset
        num_samples: Number of samples to draw per epoch
                    If None, defaults to len(weights) (default: None)
        replacement: If True, samples are drawn with replacement (default: True)

    Returns:
        WeightedRandomSampler configured with the given weights

    Example:
        >>> weights = [0.1, 0.5, 0.3, 0.7]  # Weights from CFW clustering
        >>> sampler = create_weighted_sampler(weights, num_samples=1000)
        >>> dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    Note:
        For CFW, weights are typically computed as:
        - 1 / cluster_size for clustered points
        - small constant (e.g., 0.001) for outliers
    """
    # Convert to tensor if not already
    if not isinstance(weights, torch.Tensor):
        weights_tensor = torch.tensor(weights, dtype=torch.float)
    else:
        weights_tensor = weights

    # Default num_samples to dataset length
    if num_samples is None:
        num_samples = len(weights)

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=num_samples,
        replacement=replacement,
    )

    return sampler


def validate_weights(weights: List[float]) -> bool:
    """
    Validate that weights are suitable for sampling.

    Args:
        weights: List of sample weights

    Returns:
        True if weights are valid

    Raises:
        ValueError: If weights are invalid (empty, negative, or all zero)

    Example:
        >>> weights = [0.1, 0.5, 0.3]
        >>> validate_weights(weights)
        True
    """
    if len(weights) == 0:
        raise ValueError("Weights list is empty")

    # Check for negative weights
    if any(w < 0 for w in weights):
        raise ValueError("Weights must be non-negative")

    # Check that at least some weights are positive
    if sum(weights) == 0:
        raise ValueError("All weights are zero - cannot sample")

    return True


def normalize_weights(weights: List[float]) -> List[float]:
    """
    Normalize weights to sum to 1.0.

    Args:
        weights: List of sample weights

    Returns:
        Normalized weights that sum to 1.0

    Example:
        >>> weights = [1.0, 2.0, 3.0]
        >>> normalized = normalize_weights(weights)
        >>> normalized
        [0.16666..., 0.33333..., 0.5]
        >>> sum(normalized)
        1.0
    """
    total = sum(weights)
    if total == 0:
        raise ValueError("Cannot normalize weights - sum is zero")

    return [w / total for w in weights]


def get_class_weights(labels: List[int], num_classes: int) -> List[float]:
    """
    Compute inverse class frequency weights for balanced sampling.

    This is useful for baseline (non-CFW) experiments with class imbalance.

    Args:
        labels: List of class labels for all samples
        num_classes: Total number of classes

    Returns:
        List of weights, one per sample (inverse of class frequency)

    Example:
        >>> labels = [0, 0, 0, 1, 1, 2]  # Class 0 appears 3 times
        >>> weights = get_class_weights(labels, num_classes=3)
        >>> # Class 0: weight = 1/3, Class 1: weight = 1/2, Class 2: weight = 1/1
    """
    # Count samples per class
    class_counts = [0] * num_classes
    for label in labels:
        class_counts[label] += 1

    # Compute weight per class (inverse frequency)
    total_samples = len(labels)
    class_weights = [
        total_samples / count if count > 0 else 0.0
        for count in class_counts
    ]

    # Assign weight to each sample based on its class
    sample_weights = [class_weights[label] for label in labels]

    return sample_weights


class DistributedWeightedSampler(Sampler[int]):
    """
    Distributed weighted random sampler for CFW + DDP multi-GPU training.

    Combines weighted sampling (from CFW clustering weights) with
    DDP-aware partitioning so each rank gets a disjoint shard.

    Supports:
      - replacement=True  : weighted oversampling (duplicates allowed)
      - replacement=False : weighted sampling without duplicates (global uniqueness)

    Key properties:
      - Deterministic across ranks via shared seed+epoch.
      - Produces disjoint index lists per rank by slicing a shared global draw.
      - Has set_epoch() for reproducibility across epochs.

    Args:
        weights: Per-sample weights (tensor or list).
        num_samples: Per-rank samples to draw. If None, auto-computed from
            dataset size and world size.
        replacement: Whether to sample with replacement.
        seed: Base random seed for deterministic sampling.
        drop_last: If True, all ranks get equal-length shards (recommended for DDP).
        num_replicas: Total number of DDP processes. Auto-detected if None.
        rank: This process's rank. Auto-detected if None.

    Example:
        >>> sampler = DistributedWeightedSampler(
        ...     weights=cfw_weights,
        ...     replacement=True,
        ...     drop_last=True,
        ...     seed=42,
        ... )
        >>> dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
        >>> for epoch in range(num_epochs):
        ...     sampler.set_epoch(epoch)
        ...     for batch in dataloader:
        ...         ...
    """

    def __init__(
        self,
        weights,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        """Initialize distributed weighted sampling parameters and shard sizes."""
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

        self.dataset_size = int(self.weights.numel())

        # Determine per-rank samples
        if num_samples is None:
            if self.drop_last:
                self.num_samples = self.dataset_size // self.num_replicas
            else:
                self.num_samples = int(math.ceil(self.dataset_size / self.num_replicas))
        else:
            self.num_samples = int(num_samples)

        # Determine global draw size
        if self.replacement:
            self.total_size = self.num_samples * self.num_replicas
        else:
            requested_total = self.num_samples * self.num_replicas
            if requested_total > self.dataset_size:
                if self.drop_last:
                    self.num_samples = self.dataset_size // self.num_replicas
                    self.total_size = self.num_samples * self.num_replicas
                else:
                    self.total_size = self.dataset_size
            else:
                self.total_size = requested_total

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling across ranks."""
        self.epoch = int(epoch)

    def __iter__(self):
        """Yield this rank's deterministic weighted sample indices for the current epoch."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Global draw (same on every rank due to shared seed)
        global_indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=g,
        ).tolist()

        # Shard by rank (disjoint slices via stride)
        rank_indices = global_indices[self.rank:self.total_size:self.num_replicas]

        return iter(rank_indices)

    def __len__(self) -> int:
        """Return the number of samples drawn on this rank per epoch."""
        return self.num_samples


def create_distributed_weighted_sampler(
    weights: List[float],
    num_samples: Optional[int] = None,
    replacement: bool = True,
    seed: int = 0,
    drop_last: bool = True,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
) -> DistributedWeightedSampler:
    """
    Create a DistributedWeightedSampler for CFW + DDP training.

    Args:
        weights: Per-sample weights from CFW clustering.
        num_samples: Per-rank samples per epoch. If None, auto-computed.
        replacement: Whether to sample with replacement.
        seed: Base random seed for deterministic sampling.
        drop_last: If True, all ranks get equal-length shards.
        num_replicas: Total number of DDP processes. Auto-detected if None.
        rank: This process's rank. Auto-detected if None.

    Returns:
        DistributedWeightedSampler configured for DDP.

    Example:
        >>> sampler = create_distributed_weighted_sampler(
        ...     weights=cfw_weights,
        ...     replacement=True,
        ...     seed=42,
        ... )
    """
    if not isinstance(weights, torch.Tensor):
        weights_tensor = torch.tensor(weights, dtype=torch.double)
    else:
        weights_tensor = weights.double()

    return DistributedWeightedSampler(
        weights=weights_tensor,
        num_samples=num_samples,
        replacement=replacement,
        seed=seed,
        drop_last=drop_last,
        num_replicas=num_replicas,
        rank=rank,
    )
