"""Weight computation strategies for CFW."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class WeightingConfig:
    """Configuration for sample-weight computation."""

    outlier_weight: float = 0.001
    max_outlier_cluster_size: int = 50
    weighting_strategy: str = "inverse_cluster_size"

    def __post_init__(self) -> None:
        """Validate numeric configuration values after dataclass initialization."""
        if self.outlier_weight <= 0:
            raise ValueError("outlier_weight must be positive")
        if self.max_outlier_cluster_size <= 0:
            raise ValueError("max_outlier_cluster_size must be positive")


class WeightComputer:
    """Compute sample weights from clustering labels."""

    def __init__(self, config: Optional[WeightingConfig] = None):
        """Initialize the weight computer with an optional weighting configuration."""
        self.config = config or WeightingConfig()

    def _to_numpy_labels(self, labels: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels)

        if labels.size == 0:
            raise ValueError("Cluster labels array is empty")

        return labels.astype(np.int64, copy=False)

    def _cluster_weight(self, cluster_size: int, strategy: str) -> float:
        if strategy == "inverse_cluster_size":
            return 1.0 / float(cluster_size)
        if strategy == "inverse_sqrt":
            return 1.0 / float(np.sqrt(cluster_size))
        if strategy == "uniform":
            return 1.0
        raise ValueError(f"Unknown weighting strategy: {strategy}")

    def compute_weights(
        self,
        labels: Union[np.ndarray, torch.Tensor],
        return_num_clusters: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, int]]:
        """Compute sample weights and outlier-updated labels."""
        labels_np = self._to_numpy_labels(labels)
        strategy = self.config.weighting_strategy

        if strategy not in {"inverse_cluster_size", "inverse_sqrt", "uniform"}:
            raise ValueError(f"Unknown weighting strategy: {strategy}")

        weights = np.zeros(labels_np.shape[0], dtype=np.float32)
        updated_labels = labels_np.copy()

        noise_label = -1
        non_outlier_labels = np.unique(labels_np[labels_np != noise_label])

        # Regular cluster weights
        for label in non_outlier_labels:
            idx = np.where(labels_np == label)[0]
            weights[idx] = self._cluster_weight(len(idx), strategy)

        # Outliers: assign fixed low weight and synthetic cluster ids
        outlier_idx = np.where(labels_np == noise_label)[0]
        if outlier_idx.size > 0:
            next_label = int(non_outlier_labels.max() + 1) if non_outlier_labels.size > 0 else 0
            chunk_size = int(self.config.max_outlier_cluster_size)

            for i in range(0, outlier_idx.size, chunk_size):
                chunk = outlier_idx[i : i + chunk_size]
                updated_labels[chunk] = next_label
                weights[chunk] = float(self.config.outlier_weight)
                next_label += 1

        total_clusters = int(len(np.unique(updated_labels)))

        if return_num_clusters:
            return weights, updated_labels, total_clusters
        return weights, updated_labels

    def get_weight_stats(self, weights: np.ndarray, labels: np.ndarray) -> dict:
        """Summarize computed weights."""
        unique_labels = np.unique(labels)

        weight_stats_per_cluster = {}
        for label in unique_labels:
            if label != -1:
                cluster_mask = labels == label
                cluster_weights = weights[cluster_mask]
                weight_stats_per_cluster[int(label)] = {
                    "mean_weight": float(np.mean(cluster_weights)),
                    "unique_weight": float(np.unique(cluster_weights)[0]),
                    "cluster_size": int(np.sum(cluster_mask)),
                }

        return {
            "mean_weight": float(np.mean(weights)),
            "std_weight": float(np.std(weights)),
            "min_weight": float(np.min(weights)),
            "max_weight": float(np.max(weights)),
            "n_unique_weights": len(np.unique(weights)),
            "per_cluster_stats": weight_stats_per_cluster,
        }

    def validate_weights(self, weights: np.ndarray) -> bool:
        """Validate computed weights."""
        if np.any(np.isnan(weights)):
            raise ValueError("Weights contain NaN values")
        if np.any(np.isinf(weights)):
            raise ValueError("Weights contain infinite values")
        if np.any(weights < 0):
            raise ValueError("Weights contain negative values")
        if np.any(weights == 0):
            raise ValueError("Weights contain zero values (should have minimal weight instead)")
        return True
