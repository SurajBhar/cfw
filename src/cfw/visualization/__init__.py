"""
Visualization utilities for CFW experiments.

This module provides plotting functions for:
- Class distribution analysis
- KL divergence comparison between baseline and CFW
- Confusion matrix visualization
- Training curves
- Feature clustering visualization (t-SNE/UMAP)
"""


from .plots import (
    plot_class_distribution,
    plot_kl_divergence_comparison,
    plot_confusion_matrix,
    plot_training_curves,
    plot_per_class_recall,
    plot_cumulative_unique_samples,
    plot_feature_clusters,
)

__all__ = [
    'plot_class_distribution',
    'plot_kl_divergence_comparison',
    'plot_confusion_matrix',
    'plot_training_curves',
    'plot_per_class_recall',
    'plot_cumulative_unique_samples',
    'plot_feature_clusters',
]
