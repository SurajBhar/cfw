"""
Plotting functions for CFW experiments.

This module provides visualization utilities for analyzing and debugging
CFW training, including class distributions, KL divergence comparisons,
confusion matrices, and training curves.
"""


from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


def plot_class_distribution(
    labels: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    color: str = 'steelblue',
) -> plt.Figure:
    """
    Plot class distribution as a bar chart.

    Args:
        labels: Array of class labels
        class_names: Names for each class (if None, uses indices)
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the figure (if None, displays interactively)
        ax: Matplotlib axes to plot on (if None, creates new figure)
        color: Bar color

    Returns:
        Matplotlib figure object

    Example:
        >>> labels = [0, 0, 0, 1, 1, 2]
        >>> plot_class_distribution(labels, class_names=['A', 'B', 'C'])
    """
    labels = np.asarray(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)

    if class_names is None:
        class_names = [f"Class {i}" for i in unique_labels]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bars = ax.bar(class_names, counts, color=color, edgecolor='black', alpha=0.8)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_kl_divergence_comparison(
    baseline_kl: List[float],
    cfw_kl: List[float],
    title: str = "KL Divergence Comparison: Baseline vs CFW",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare KL divergence between baseline and CFW dataloaders.

    Args:
        baseline_kl: KL divergence values for baseline dataloader per batch
        cfw_kl: KL divergence values for CFW dataloader per batch
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure object

    Example:
        >>> baseline_kl = [0.5, 0.4, 0.6, 0.55]
        >>> cfw_kl = [0.1, 0.08, 0.12, 0.09]
        >>> plot_kl_divergence_comparison(baseline_kl, cfw_kl)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Line plot
    ax1 = axes[0]
    batches = range(1, len(baseline_kl) + 1)
    ax1.plot(batches, baseline_kl, 'b-o', label='Baseline', alpha=0.7, markersize=4)
    ax1.plot(batches, cfw_kl, 'r-s', label='CFW', alpha=0.7, markersize=4)
    ax1.set_xlabel('Batch', fontsize=12)
    ax1.set_ylabel('KL Divergence', fontsize=12)
    ax1.set_title('KL Divergence per Batch', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot([baseline_kl, cfw_kl], labels=['Baseline', 'CFW'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    ax2.set_ylabel('KL Divergence', fontsize=12)
    ax2.set_title('KL Divergence Distribution', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics
    baseline_mean = np.mean(baseline_kl)
    cfw_mean = np.mean(cfw_kl)
    reduction = (baseline_mean - cfw_mean) / baseline_mean * 100
    fig.suptitle(f'{title}\nMean Baseline: {baseline_mean:.4f}, Mean CFW: {cfw_mean:.4f} ({reduction:.1f}% reduction)',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 8),
    cmap: str = 'Blues',
    normalize: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix (n_classes x n_classes)
        class_names: Names for each class
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        normalize: If True, normalize by row (true labels)
        save_path: Path to save figure

    Returns:
        Matplotlib figure object

    Example:
        >>> cm = np.array([[45, 5], [10, 40]])
        >>> plot_confusion_matrix(cm, ['Not Distracted', 'Distracted'])
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss',
                'train_accuracy', 'val_accuracy', 'train_balanced_accuracy', etc.
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure object

    Example:
        >>> history = {
        ...     'train_loss': [0.5, 0.3, 0.2],
        ...     'val_loss': [0.6, 0.4, 0.3],
        ...     'train_balanced_accuracy': [0.7, 0.8, 0.85],
        ...     'val_balanced_accuracy': [0.65, 0.75, 0.80],
        ... }
        >>> plot_training_curves(history)
    """
    # Determine which metrics are present
    has_loss = 'train_loss' in history or 'val_loss' in history
    has_accuracy = any(k for k in history.keys() if 'accuracy' in k.lower())

    if has_loss and has_accuracy:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    elif has_loss:
        fig, ax1 = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
        ax2 = None
    elif has_accuracy:
        fig, ax2 = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
        ax1 = None
    else:
        raise ValueError("history must contain loss or accuracy metrics")

    epochs = range(1, len(list(history.values())[0]) + 1)

    # Plot loss
    if ax1 is not None:
        if 'train_loss' in history:
            ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', alpha=0.7)
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', alpha=0.7)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot accuracy
    if ax2 is not None:
        for key in history:
            if 'accuracy' in key.lower() and 'train' in key.lower():
                ax2.plot(epochs, history[key], 'b-o', label=f'Train {key.split("_")[-1]}', alpha=0.7)
            elif 'accuracy' in key.lower() and 'val' in key.lower():
                ax2.plot(epochs, history[key], 'r-s', label=f'Val {key.split("_")[-1]}', alpha=0.7)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Accuracy', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_per_class_recall(
    recall_per_class: Dict[str, float],
    title: str = "Per-Class Recall",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot recall for each class as a horizontal bar chart.

    Args:
        recall_per_class: Dictionary mapping class names to recall values
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure object

    Example:
        >>> recall = {'distracted': 0.85, 'not_distracted': 0.92}
        >>> plot_per_class_recall(recall)
    """
    fig, ax = plt.subplots(figsize=figsize)

    classes = list(recall_per_class.keys())
    recalls = list(recall_per_class.values())

    # Color bars based on recall value
    colors = ['green' if r >= 0.8 else 'orange' if r >= 0.6 else 'red' for r in recalls]

    bars = ax.barh(classes, recalls, color=colors, edgecolor='black', alpha=0.8)

    # Add value labels
    for bar, recall in zip(bars, recalls):
        width = bar.get_width()
        ax.annotate(f'{recall:.2%}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=10)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Class', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_cumulative_unique_samples(
    baseline_unique: List[int],
    cfw_unique: List[int],
    title: str = "Cumulative Unique Samples per Epoch",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot cumulative unique samples seen over batches for baseline vs CFW.

    Args:
        baseline_unique: Cumulative unique samples for baseline per batch
        cfw_unique: Cumulative unique samples for CFW per batch
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure object

    Example:
        >>> baseline = [16, 32, 45, 58, 70]
        >>> cfw = [16, 30, 42, 55, 68]
        >>> plot_cumulative_unique_samples(baseline, cfw)
    """
    fig, ax = plt.subplots(figsize=figsize)

    batches = range(1, len(baseline_unique) + 1)
    ax.plot(batches, baseline_unique, 'b-o', label='Baseline', alpha=0.7, markersize=3)
    ax.plot(batches, cfw_unique, 'r-s', label='CFW', alpha=0.7, markersize=3)

    ax.set_xlabel('Batch', fontsize=12)
    ax.set_ylabel('Cumulative Unique Samples', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics
    baseline_total = baseline_unique[-1] if baseline_unique else 0
    cfw_total = cfw_unique[-1] if cfw_unique else 0
    ax.annotate(f'Baseline: {baseline_total}', xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', fontsize=10, color='blue')
    ax.annotate(f'CFW: {cfw_total}', xy=(0.98, 0.88), xycoords='axes fraction',
                ha='right', fontsize=10, color='red')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_feature_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    cluster_labels: Optional[np.ndarray] = None,
    method: str = 'tsne',
    class_names: Optional[List[str]] = None,
    title: str = "Feature Clustering",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    random_state: int = 42,
) -> plt.Figure:
    """
    Plot feature embeddings using t-SNE or UMAP dimensionality reduction.

    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Class labels
        cluster_labels: Optional cluster assignments from HDBSCAN (for comparison)
        method: Dimensionality reduction method ('tsne' or 'umap')
        class_names: Names for each class
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        random_state: Random seed for reproducibility

    Returns:
        Matplotlib figure object

    Example:
        >>> features = np.random.randn(100, 384)
        >>> labels = np.random.randint(0, 2, 100)
        >>> plot_feature_clusters(features, labels, method='tsne')
    """
    # Perform dimensionality reduction
    if method.lower() == 'tsne':
        try:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(features) - 1))
            features_2d = reducer.fit_transform(features)
        except ImportError:
            raise ImportError("scikit-learn is required for t-SNE. Install with: pip install scikit-learn")
    elif method.lower() == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=random_state)
            features_2d = reducer.fit_transform(features)
        except ImportError:
            raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'.")

    # Determine number of subplots
    n_plots = 2 if cluster_labels is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    # Plot by class labels
    ax1 = axes[0]
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if class_names else f"Class {label}"
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[colors[i]], label=name, alpha=0.6, s=20)

    ax1.set_xlabel(f'{method.upper()} 1', fontsize=11)
    ax1.set_ylabel(f'{method.upper()} 2', fontsize=11)
    ax1.set_title('By Class Label', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot by cluster labels (if provided)
    if cluster_labels is not None:
        ax2 = axes[1]
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters[unique_clusters >= 0])  # Exclude outliers (-1)

        # Create colormap for clusters
        cluster_colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

        for cluster in unique_clusters:
            mask = cluster_labels == cluster
            if cluster == -1:
                # Outliers in gray
                ax2.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c='gray', label='Outliers', alpha=0.4, s=15, marker='x')
            else:
                ax2.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=[cluster_colors[cluster % len(cluster_colors)]],
                           label=f'Cluster {cluster}', alpha=0.6, s=20)

        ax2.set_xlabel(f'{method.upper()} 1', fontsize=11)
        ax2.set_ylabel(f'{method.upper()} 2', fontsize=11)
        ax2.set_title(f'By HDBSCAN Cluster ({n_clusters} clusters)', fontsize=12)
        if len(unique_clusters) <= 10:
            ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig
