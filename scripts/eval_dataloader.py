#!/usr/bin/env python3
"""
Dataloader comparison and evaluation script.

This script compares baseline (Dataloader A) vs CFW (Dataloader B) in terms of:
- Class distribution per batch
- KL divergence to per-batch uniform target
- Unique samples per batch
- Cumulative unique samples per epoch
- Optional advanced diagnostics (ECDF, rolling KL, entropy, heatmaps, Lorenz/Gini)

Generates visualizations and statistics to demonstrate CFW effectiveness.

KL mode implemented here follows:
    KL(batch_distribution || uniform_distribution)

Uniform target for each batch is constructed from the *actual* batch size N and
class count C:
    - Each class gets floor(N / C) samples
    - Remaining samples (N % C) are distributed one-by-one to classes

This automatically handles any dataset size, class count, and last partial batch.

Usage:
    # Compare dataloaders for a specific dataset
    python scripts/eval_dataloader.py dataset=driveact_binary

    # Compare first 100 batches (optional; default is full epoch)
    python scripts/eval_dataloader.py dataset=driveact_binary num_batches=100

    # Save plots to specific directory
    python scripts/eval_dataloader.py dataset=driveact_binary output_dir=/path/to/output
"""


import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from scipy.stats import entropy

# Add src to path for imports
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from cfw.utils.logging import setup_logger
from cfw.utils.reproducibility import set_seeds
from cfw.utils.config_utils import validate_config
from cfw.data.dataloaders import create_dataloader


KL_DIRECTION = "KL(batch_distribution || uniform_distribution)"

# Plots that are disabled by default due to heavier rendering or larger output size.
COMPLEX_PLOTS = {
    "class_by_batch_heatmaps",
    "cfw_weight_vs_sampling_frequency",
    "sampling_concentration_lorenz",
}


def _extract_batch_labels_paths_and_weights(
    batch: Sequence[Any],
) -> tuple[np.ndarray, Optional[list[str]], Optional[np.ndarray]]:
    """
    Extract labels, optional image paths, and optional sample weights from a batch.

    Supports:
        - Baseline format: (images, labels)
        - Baseline+paths format: (images, labels, paths)
        - CFW format: (images, weights, labels, paths)
    """
    weights = None
    if len(batch) == 2:
        _, labels = batch
        paths = None
    elif len(batch) == 3:
        _, labels, paths = batch
    elif len(batch) == 4:
        _, weights, labels, paths = batch
    else:
        raise ValueError(f"Unexpected batch format with {len(batch)} elements")

    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    labels_np = labels_np.reshape(-1).astype(np.int64, copy=False)
    paths_list = [str(p) for p in paths] if paths is not None else None

    weights_np: Optional[np.ndarray] = None
    if weights is not None:
        if isinstance(weights, torch.Tensor):
            weights_np = weights.detach().cpu().numpy()
        else:
            weights_np = np.asarray(weights)
        weights_np = weights_np.reshape(-1).astype(np.float64, copy=False)

        if paths_list is not None and weights_np.shape[0] != len(paths_list):
            raise ValueError(
                "CFW batch has mismatched path and weight counts: "
                f"paths={len(paths_list)}, weights={weights_np.shape[0]}"
            )

    return labels_np, paths_list, weights_np


def _compute_class_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute per-class counts for a batch."""
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if labels.size == 0:
        raise ValueError("Batch has no labels; cannot compute class distribution")

    min_label = int(labels.min())
    max_label = int(labels.max())
    if min_label < 0 or max_label >= num_classes:
        raise ValueError(
            "Batch labels are out of expected range "
            f"[0, {num_classes - 1}]. Found min={min_label}, max={max_label}"
        )

    return np.bincount(labels, minlength=num_classes).astype(np.int64, copy=False)


def _normalize_counts(counts: np.ndarray) -> np.ndarray:
    """Convert class counts to a probability distribution."""
    total = int(np.sum(counts))
    if total <= 0:
        raise ValueError("Count vector sums to zero; cannot normalize")
    return counts.astype(np.float64) / float(total)


def build_uniform_class_counts(batch_size: int, num_classes: int) -> np.ndarray:
    """
    Build integer class counts for the most uniform batch possible.

    Example for N=1024 and C=10:
        - Base count: floor(1024/10) = 102 for each class
        - Remainder: 4 samples
        - Final counts: [103, 103, 103, 103, 102, 102, 102, 102, 102, 102]
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    base_count = batch_size // num_classes
    remainder = batch_size % num_classes

    counts = np.full(num_classes, base_count, dtype=np.int64)
    if remainder > 0:
        counts[:remainder] += 1
    return counts


def build_uniform_distribution(batch_size: int, num_classes: int) -> np.ndarray:
    """Build a per-batch uniform probability distribution from integer class counts."""
    uniform_counts = build_uniform_class_counts(batch_size=batch_size, num_classes=num_classes)
    return _normalize_counts(uniform_counts)


def compute_kl_divergence_batch_to_uniform(
    batch_distribution: np.ndarray,
    uniform_distribution: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute KL(batch_distribution || uniform_distribution).

    Args:
        batch_distribution: Batch class distribution P
        uniform_distribution: Uniform target distribution Q
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence value
    """
    p = batch_distribution + epsilon
    q = uniform_distribution + epsilon

    p = p / p.sum()
    q = q / q.sum()
    return float(entropy(p, q))


def _resolve_effective_num_batches(
    dataloader_len: int,
    requested_num_batches: Optional[int],
) -> int:
    """
    Resolve how many batches to analyze.

    Rules:
        - None or <= 0: analyze full epoch (all available batches)
        - > 0: analyze min(requested, len(dataloader))
    """
    if requested_num_batches is None:
        return dataloader_len

    requested = int(requested_num_batches)
    if requested <= 0:
        return dataloader_len
    return min(requested, dataloader_len)


def analyze_dataloader_uniform_kl(
    dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    requested_num_batches: Optional[int],
    logger,
) -> Dict:
    """
    Analyze a dataloader with per-batch uniform KL.

    Args:
        dataloader: DataLoader to analyze
        num_classes: Number of classes
        requested_num_batches: Optional max number of batches to analyze
        logger: Logger instance

    Returns:
        Dictionary with per-batch and aggregate statistics
    """
    batch_distributions = []
    batch_class_counts = []
    uniform_distributions = []
    kl_divergences = []
    batch_entropies = []
    normalized_batch_entropies = []
    batch_sizes = []
    unique_samples_per_batch = []
    cumulative_unique_samples = set()
    path_sampling_counts: Counter[str] = Counter()
    path_weight_map: Dict[str, float] = {}

    effective_num_batches = _resolve_effective_num_batches(
        dataloader_len=len(dataloader),
        requested_num_batches=requested_num_batches,
    )
    logger.info(f"Analyzing {effective_num_batches} batches...")

    iterator = tqdm(dataloader, desc="Analyzing batches", total=effective_num_batches)
    for batch_idx, batch in enumerate(iterator):
        if batch_idx >= effective_num_batches:
            break

        labels, paths, weights = _extract_batch_labels_paths_and_weights(batch=batch)
        current_batch_size = int(labels.shape[0])
        batch_sizes.append(current_batch_size)

        class_counts = _compute_class_counts(labels=labels, num_classes=num_classes)
        batch_dist = _normalize_counts(class_counts)
        uniform_dist = build_uniform_distribution(
            batch_size=current_batch_size,
            num_classes=num_classes,
        )

        batch_class_counts.append(class_counts)
        batch_distributions.append(batch_dist)
        uniform_distributions.append(uniform_dist)
        batch_entropy = float(entropy(batch_dist + 1e-10))
        batch_entropies.append(batch_entropy)
        max_entropy = float(np.log(num_classes)) if num_classes > 1 else 1.0
        normalized_batch_entropies.append(
            batch_entropy / max_entropy if max_entropy > 0 else float("nan")
        )

        kl_div = compute_kl_divergence_batch_to_uniform(
            batch_distribution=batch_dist,
            uniform_distribution=uniform_dist,
        )
        kl_divergences.append(kl_div)

        if paths is not None:
            unique_samples = set(paths)
            unique_samples_per_batch.append(len(unique_samples))
            cumulative_unique_samples.update(unique_samples)
            path_sampling_counts.update(paths)

            if weights is not None:
                for image_path, sample_weight in zip(paths, weights):
                    if image_path not in path_weight_map:
                        path_weight_map[image_path] = float(sample_weight)

    if kl_divergences:
        mean_kl = float(np.mean(kl_divergences))
        std_kl = float(np.std(kl_divergences))
    else:
        mean_kl = float("nan")
        std_kl = float("nan")

    batch_distributions_arr = (
        np.vstack(batch_distributions)
        if batch_distributions
        else np.empty((0, num_classes), dtype=np.float64)
    )
    uniform_distributions_arr = (
        np.vstack(uniform_distributions)
        if uniform_distributions
        else np.empty((0, num_classes), dtype=np.float64)
    )
    batch_class_counts_arr = (
        np.vstack(batch_class_counts)
        if batch_class_counts
        else np.empty((0, num_classes), dtype=np.int64)
    )

    results = {
        "batch_class_counts": batch_class_counts_arr,
        "batch_distributions": batch_distributions_arr,
        "uniform_distributions": uniform_distributions_arr,
        "batch_sizes": np.array(batch_sizes, dtype=np.int64),
        "kl_divergences": np.array(kl_divergences, dtype=np.float64),
        "batch_entropies": np.array(batch_entropies, dtype=np.float64),
        "normalized_batch_entropies": np.array(normalized_batch_entropies, dtype=np.float64),
        "mean_kl_divergence": mean_kl,
        "std_kl_divergence": std_kl,
        "num_batches_analyzed": len(kl_divergences),
        "unique_samples_per_batch": unique_samples_per_batch,
        "cumulative_unique_samples": (
            len(cumulative_unique_samples) if unique_samples_per_batch else 0
        ),
        "path_sampling_counts": dict(path_sampling_counts),
        "path_weight_map": path_weight_map,
        "kl_direction": KL_DIRECTION,
    }
    return results


def _resolve_plot_options(plot_cfg: Optional[Any]) -> Dict[str, Any]:
    """Resolve plot toggles and numeric options from config."""
    options: Dict[str, Any] = {
        "kl_trend_with_uniform": True,
        "kl_box_comparison": True,
        "mean_batch_distribution": True,
        "cumulative_unique_samples": True,
        "kl_histogram_overlay": True,
        "kl_ecdf": True,
        "kl_rolling_mean": True,
        "batch_entropy_trend": True,
        "class_by_batch_heatmaps": False,
        "per_class_uniform_deviation": True,
        "cfw_weight_vs_sampling_frequency": False,
        "sampling_concentration_lorenz": False,
        "enable_complex": False,
        "rolling_window": 10,
        "histogram_bins": 20,
    }

    if plot_cfg is None:
        return options

    if isinstance(plot_cfg, DictConfig):
        plot_cfg_dict = OmegaConf.to_container(plot_cfg, resolve=True)
    elif isinstance(plot_cfg, dict):
        plot_cfg_dict = plot_cfg
    else:
        return options

    if not isinstance(plot_cfg_dict, dict):
        return options

    if bool(plot_cfg_dict.get("enable_complex", False)):
        for complex_plot in COMPLEX_PLOTS:
            options[complex_plot] = True

    for key, default_value in options.items():
        if isinstance(default_value, bool) and key in plot_cfg_dict:
            options[key] = bool(plot_cfg_dict[key])

    if "rolling_window" in plot_cfg_dict:
        options["rolling_window"] = max(1, int(plot_cfg_dict["rolling_window"]))
    if "histogram_bins" in plot_cfg_dict:
        options["histogram_bins"] = max(5, int(plot_cfg_dict["histogram_bins"]))

    return options


def _compute_ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute ECDF x/y points for a 1-D array."""
    if values.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    x = np.sort(values.astype(np.float64))
    y = np.arange(1, x.size + 1, dtype=np.float64) / float(x.size)
    return x, y


def _compute_rolling_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute trailing rolling mean and corresponding x-axis batch indices."""
    if values.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    window = max(1, int(window))
    if values.size < window:
        return np.array([values.size], dtype=np.int64), np.array([float(values.mean())], dtype=np.float64)

    kernel = np.ones(window, dtype=np.float64) / float(window)
    rolling = np.convolve(values.astype(np.float64), kernel, mode="valid")
    x = np.arange(window, values.size + 1, dtype=np.int64)
    return x, rolling


def _compute_lorenz_curve_points(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Lorenz curve points for non-negative counts."""
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts >= 0]
    if counts.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    sorted_counts = np.sort(counts)
    cum_counts = np.cumsum(sorted_counts) / total
    cum_items = np.arange(1, sorted_counts.size + 1, dtype=np.float64) / float(sorted_counts.size)

    return (
        np.concatenate(([0.0], cum_items)),
        np.concatenate(([0.0], cum_counts)),
    )


def _compute_gini(counts: np.ndarray) -> float:
    """Compute Gini coefficient for non-negative sample-frequency counts."""
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts >= 0]
    if counts.size == 0:
        return float("nan")

    total = float(counts.sum())
    if total <= 0:
        return float("nan")

    sorted_counts = np.sort(counts)
    n = sorted_counts.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * np.sum(idx * sorted_counts) / (n * total)) - ((n + 1.0) / n)
    return float(gini)


def plot_comparison(
    baseline_results: Dict,
    cfw_results: Dict,
    output_dir: Path,
    logger,
    plot_cfg: Optional[Any] = None,
) -> None:
    """
    Create comparison plots for baseline vs CFW dataloaders.

    Args:
        baseline_results: Results from baseline dataloader analysis
        cfw_results: Results from CFW dataloader analysis
        output_dir: Directory to save plots
        logger: Logger instance
        plot_cfg: Optional dataloader_eval.plots config subtree
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    options = _resolve_plot_options(plot_cfg)
    enabled_plots = [name for name, enabled in options.items() if isinstance(enabled, bool) and enabled]
    logger.info(f"Enabled plots: {enabled_plots}")

    baseline_kl = baseline_results["kl_divergences"]
    cfw_kl = cfw_results["kl_divergences"]
    baseline_batches = np.arange(1, len(baseline_kl) + 1)
    cfw_batches = np.arange(1, len(cfw_kl) + 1)

    baseline_batch_distributions = baseline_results["batch_distributions"]
    cfw_batch_distributions = cfw_results["batch_distributions"]
    baseline_uniform_distributions = baseline_results["uniform_distributions"]
    cfw_uniform_distributions = cfw_results["uniform_distributions"]

    # Existing plots
    if options["kl_trend_with_uniform"]:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            baseline_batches,
            baseline_kl,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label="Baseline KL per batch",
        )
        ax.plot(
            cfw_batches,
            cfw_kl,
            marker="s",
            linewidth=1.8,
            markersize=4,
            label="CFW KL per batch",
        )
        ax.axhline(y=0.0, color="red", linestyle="--", linewidth=1.8, label="Ideal uniform KL = 0")
        ax.set_xlabel("Batch Number")
        ax.set_ylabel("KL Divergence")
        ax.set_title("KL Divergence vs Batch Number (Baseline vs CFW)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plot_path = output_dir / "kl_divergence_trend_with_uniform.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved KL trend plot to {plot_path}")

    if options["kl_box_comparison"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        data_to_plot = [baseline_kl, cfw_kl]
        ax.boxplot(data_to_plot, labels=["Baseline", "CFW"])
        ax.set_ylabel("KL Divergence")
        ax.set_title("KL(batch || uniform) per batch: Baseline vs CFW")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = output_dir / "kl_divergence_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved KL divergence box plot to {plot_path}")

    if options["mean_batch_distribution"]:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        mean_baseline_dist = baseline_batch_distributions.mean(axis=0)
        mean_uniform_baseline = baseline_uniform_distributions.mean(axis=0)
        axes[0].bar(range(len(mean_baseline_dist)), mean_baseline_dist, alpha=0.7, label="Batch Mean")
        axes[0].plot(
            range(len(mean_uniform_baseline)),
            mean_uniform_baseline,
            "r--",
            label="Uniform Target (Batch-Aware Mean)",
            linewidth=2,
        )
        axes[0].set_xlabel("Class")
        axes[0].set_ylabel("Proportion")
        axes[0].set_title("Baseline Dataloader: Mean Batch Distribution")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        mean_cfw_dist = cfw_batch_distributions.mean(axis=0)
        mean_uniform_cfw = cfw_uniform_distributions.mean(axis=0)
        axes[1].bar(range(len(mean_cfw_dist)), mean_cfw_dist, alpha=0.7, label="Batch Mean", color="green")
        axes[1].plot(
            range(len(mean_uniform_cfw)),
            mean_uniform_cfw,
            "r--",
            label="Uniform Target (Batch-Aware Mean)",
            linewidth=2,
        )
        axes[1].set_xlabel("Class")
        axes[1].set_ylabel("Proportion")
        axes[1].set_title("CFW Dataloader: Mean Batch Distribution")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "batch_distribution_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved mean batch-distribution plot to {plot_path}")

    if (
        options["cumulative_unique_samples"]
        and baseline_results["unique_samples_per_batch"]
        and cfw_results["unique_samples_per_batch"]
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            np.cumsum(baseline_results["unique_samples_per_batch"]),
            label="Baseline",
            linewidth=2,
        )
        ax.plot(
            np.cumsum(cfw_results["unique_samples_per_batch"]),
            label="CFW",
            linewidth=2,
        )
        ax.set_xlabel("Batch Number")
        ax.set_ylabel("Cumulative Unique Samples")
        ax.set_title("Cumulative Unique Samples: Baseline vs CFW")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = output_dir / "cumulative_unique_samples.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved cumulative unique samples plot to {plot_path}")

    # Suggested Plot 1: KL histogram overlay
    if options["kl_histogram_overlay"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = int(options["histogram_bins"])
        ax.hist(baseline_kl, bins=bins, alpha=0.55, density=True, label="Baseline")
        ax.hist(cfw_kl, bins=bins, alpha=0.55, density=True, label="CFW")
        ax.set_xlabel("KL Divergence")
        ax.set_ylabel("Density")
        ax.set_title("KL Divergence Distribution (Histogram Overlay)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = output_dir / "kl_divergence_histogram_overlay.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved KL histogram overlay plot to {plot_path}")

    # Suggested Plot 2: KL ECDF
    if options["kl_ecdf"]:
        baseline_ecdf_x, baseline_ecdf_y = _compute_ecdf(baseline_kl)
        cfw_ecdf_x, cfw_ecdf_y = _compute_ecdf(cfw_kl)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.step(baseline_ecdf_x, baseline_ecdf_y, where="post", label="Baseline", linewidth=2)
        ax.step(cfw_ecdf_x, cfw_ecdf_y, where="post", label="CFW", linewidth=2)
        ax.set_xlabel("KL Divergence")
        ax.set_ylabel("ECDF")
        ax.set_title("Empirical CDF of KL Divergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = output_dir / "kl_divergence_ecdf.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved KL ECDF plot to {plot_path}")

    # Suggested Plot 3: KL rolling mean trend
    if options["kl_rolling_mean"]:
        rolling_window = int(options["rolling_window"])
        baseline_roll_x, baseline_roll_y = _compute_rolling_mean(baseline_kl, rolling_window)
        cfw_roll_x, cfw_roll_y = _compute_rolling_mean(cfw_kl, rolling_window)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(baseline_batches, baseline_kl, color="tab:blue", alpha=0.25, label="Baseline KL (raw)")
        ax.plot(cfw_batches, cfw_kl, color="tab:orange", alpha=0.25, label="CFW KL (raw)")
        ax.plot(
            baseline_roll_x,
            baseline_roll_y,
            color="tab:blue",
            linewidth=2.2,
            label=f"Baseline rolling mean (w={rolling_window})",
        )
        ax.plot(
            cfw_roll_x,
            cfw_roll_y,
            color="tab:orange",
            linewidth=2.2,
            label=f"CFW rolling mean (w={rolling_window})",
        )
        ax.axhline(y=0.0, color="red", linestyle="--", linewidth=1.5, label="Ideal KL = 0")
        ax.set_xlabel("Batch Number")
        ax.set_ylabel("KL Divergence")
        ax.set_title("KL Divergence Rolling Mean Across Batches")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = output_dir / "kl_divergence_rolling_mean.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved KL rolling-mean plot to {plot_path}")

    # Suggested Plot 4: Batch entropy trend
    if options["batch_entropy_trend"]:
        baseline_entropy = baseline_results["normalized_batch_entropies"]
        cfw_entropy = cfw_results["normalized_batch_entropies"]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(np.arange(1, baseline_entropy.size + 1), baseline_entropy, label="Baseline", linewidth=1.8)
        ax.plot(np.arange(1, cfw_entropy.size + 1), cfw_entropy, label="CFW", linewidth=1.8)
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Ideal uniform entropy")
        ax.set_xlabel("Batch Number")
        ax.set_ylabel("Normalized Entropy (0-1)")
        ax.set_title("Batch Class-Entropy Trend")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = output_dir / "batch_entropy_trend.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved batch entropy trend plot to {plot_path}")

    # Suggested Plot 5 (complex): Class-by-batch heatmaps
    if options["class_by_batch_heatmaps"]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        vmax = float(
            max(
                np.max(baseline_batch_distributions),
                np.max(cfw_batch_distributions),
            )
        )
        baseline_img = axes[0].imshow(
            baseline_batch_distributions,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        axes[0].set_title("Baseline: Class Proportion by Batch")
        axes[0].set_xlabel("Class Index")
        axes[0].set_ylabel("Batch Number")

        cfw_img = axes[1].imshow(
            cfw_batch_distributions,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        axes[1].set_title("CFW: Class Proportion by Batch")
        axes[1].set_xlabel("Class Index")

        fig.colorbar(baseline_img, ax=axes[0], fraction=0.046, pad=0.04, label="Class proportion")
        fig.colorbar(cfw_img, ax=axes[1], fraction=0.046, pad=0.04, label="Class proportion")
        plt.tight_layout()
        plot_path = output_dir / "class_by_batch_heatmaps.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved class-by-batch heatmaps to {plot_path}")

    # Suggested Plot 6: Per-class deviation from uniform
    if options["per_class_uniform_deviation"]:
        baseline_dev = np.mean(
            np.abs(baseline_batch_distributions - baseline_uniform_distributions),
            axis=0,
        )
        cfw_dev = np.mean(
            np.abs(cfw_batch_distributions - cfw_uniform_distributions),
            axis=0,
        )
        class_indices = np.arange(baseline_dev.shape[0])
        width = 0.4

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(class_indices - width / 2, baseline_dev, width=width, label="Baseline")
        ax.bar(class_indices + width / 2, cfw_dev, width=width, label="CFW")
        ax.set_xlabel("Class Index")
        ax.set_ylabel("Mean |batch - uniform|")
        ax.set_title("Per-Class Deviation from Uniform Target")
        ax.set_xticks(class_indices)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plot_path = output_dir / "per_class_deviation_from_uniform.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved per-class deviation plot to {plot_path}")

    # Suggested Plot 7 (complex): CFW weight vs observed sampling frequency
    if options["cfw_weight_vs_sampling_frequency"]:
        cfw_sampling_counts = cfw_results.get("path_sampling_counts", {})
        cfw_weight_map = cfw_results.get("path_weight_map", {})
        shared_paths = [p for p in cfw_sampling_counts.keys() if p in cfw_weight_map]

        if len(shared_paths) < 2:
            logger.warning(
                "Skipping CFW weight-vs-frequency plot: insufficient shared path+weight samples."
            )
        else:
            weights = np.array([cfw_weight_map[p] for p in shared_paths], dtype=np.float64)
            frequencies = np.array([cfw_sampling_counts[p] for p in shared_paths], dtype=np.float64)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(weights, frequencies, alpha=0.35, s=16, color="tab:green", edgecolors="none")
            if np.all(weights > 0.0) and np.max(weights) / np.min(weights) > 50:
                ax.set_xscale("log")
            if np.all(frequencies > 0.0) and np.max(frequencies) / np.min(frequencies) > 50:
                ax.set_yscale("log")

            corr = float("nan")
            if np.std(weights) > 0 and np.std(frequencies) > 0:
                corr = float(np.corrcoef(weights, frequencies)[0, 1])
            corr_text = f"Pearson r={corr:.3f}" if np.isfinite(corr) else "Pearson r=N/A"

            ax.set_xlabel("CFW Sample Weight")
            ax.set_ylabel("Observed Sampling Frequency")
            ax.set_title(f"CFW Weight vs Sampling Frequency ({corr_text})")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = output_dir / "cfw_weight_vs_sampling_frequency.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved CFW weight-vs-frequency plot to {plot_path}")

    # Suggested Plot 8 (complex): Sampling concentration (Lorenz + Gini)
    if options["sampling_concentration_lorenz"]:
        baseline_counts = np.array(
            list(baseline_results.get("path_sampling_counts", {}).values()),
            dtype=np.float64,
        )
        cfw_counts = np.array(
            list(cfw_results.get("path_sampling_counts", {}).values()),
            dtype=np.float64,
        )

        baseline_lorenz_x, baseline_lorenz_y = _compute_lorenz_curve_points(baseline_counts)
        cfw_lorenz_x, cfw_lorenz_y = _compute_lorenz_curve_points(cfw_counts)

        if baseline_lorenz_x.size == 0 or cfw_lorenz_x.size == 0:
            logger.warning(
                "Skipping sampling concentration plot: missing path-level frequency data."
            )
        else:
            baseline_gini = _compute_gini(baseline_counts)
            cfw_gini = _compute_gini(cfw_counts)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                baseline_lorenz_x,
                baseline_lorenz_y,
                label=f"Baseline (Gini={baseline_gini:.3f})",
                linewidth=2,
            )
            ax.plot(
                cfw_lorenz_x,
                cfw_lorenz_y,
                label=f"CFW (Gini={cfw_gini:.3f})",
                linewidth=2,
            )
            ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect equality")
            ax.set_xlabel("Cumulative Share of Unique Samples")
            ax.set_ylabel("Cumulative Share of Draws")
            ax.set_title("Sampling Concentration (Lorenz Curves)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = output_dir / "sampling_concentration_lorenz_gini.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved sampling concentration plot to {plot_path}")


def save_statistics(
    baseline_results: Dict,
    cfw_results: Dict,
    output_dir: Path,
    logger,
) -> None:
    """
    Save comparison statistics to file.

    Args:
        baseline_results: Results from baseline dataloader analysis
        cfw_results: Results from CFW dataloader analysis
        output_dir: Directory to save statistics
        logger: Logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = output_dir / "dataloader_statistics.txt"

    with open(stats_file, "w") as f:
        f.write("Dataloader Comparison Statistics\n")
        f.write("=" * 80 + "\n\n")

        f.write("KL Mode: Uniform target per batch\n")
        f.write(f"KL Direction: {KL_DIRECTION}\n")
        f.write(
            "Uniform Target Construction: floor(N/C) samples per class + "
            "remainder distributed one-by-one\n\n"
        )

        f.write("Batch Summary:\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Baseline Batches Analyzed: "
            f"{baseline_results['num_batches_analyzed']} "
            f"(min={int(np.min(baseline_results['batch_sizes'])) if baseline_results['batch_sizes'].size else 'N/A'}, "
            f"max={int(np.max(baseline_results['batch_sizes'])) if baseline_results['batch_sizes'].size else 'N/A'})\n"
        )
        f.write(
            "CFW Batches Analyzed:      "
            f"{cfw_results['num_batches_analyzed']} "
            f"(min={int(np.min(cfw_results['batch_sizes'])) if cfw_results['batch_sizes'].size else 'N/A'}, "
            f"max={int(np.max(cfw_results['batch_sizes'])) if cfw_results['batch_sizes'].size else 'N/A'})\n"
        )
        f.write("\n")

        f.write("KL Divergence to Uniform Target:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Baseline Mean: {baseline_results['mean_kl_divergence']:.6f}\n")
        f.write(f"Baseline Std:  {baseline_results['std_kl_divergence']:.6f}\n")
        f.write(f"CFW Mean:      {cfw_results['mean_kl_divergence']:.6f}\n")
        f.write(f"CFW Std:       {cfw_results['std_kl_divergence']:.6f}\n")

        baseline_mean = baseline_results["mean_kl_divergence"]
        cfw_mean = cfw_results["mean_kl_divergence"]
        if np.isfinite(baseline_mean) and not np.isclose(baseline_mean, 0.0):
            improvement = (1 - cfw_mean / baseline_mean) * 100
            f.write(f"Improvement:   {improvement:.2f}%\n")
        else:
            f.write("Improvement:   N/A (baseline mean KL is zero or undefined)\n")
        f.write("\n")

        if baseline_results["cumulative_unique_samples"] > 0:
            f.write("Unique Samples:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Baseline Cumulative: {baseline_results['cumulative_unique_samples']}\n")
            f.write(f"CFW Cumulative:      {cfw_results['cumulative_unique_samples']}\n")
            f.write("\n")

    logger.info(f"Saved statistics to {stats_file}")


def create_train_dataloader_by_type(
    cfg: DictConfig,
    dataloader_type: str,
    shuffle: bool,
    drop_last: bool,
    return_paths: bool = False,
) -> torch.utils.data.DataLoader:
    """Create train dataloader by overriding cfg.dataloader.type temporarily."""
    previous_type = cfg.dataloader.type
    cfg.dataloader.type = dataloader_type
    try:
        return create_dataloader(
            cfg=cfg,
            split="train",
            shuffle=shuffle,
            drop_last=drop_last,
            return_paths=return_paths,
        )
    finally:
        cfg.dataloader.type = previous_type


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Run baseline-vs-CFW dataloader distribution diagnostics.

    Args:
        cfg: Hydra configuration object
    """
    print("=" * 80)
    print("Dataloader Comparison Script")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Validate configuration
    validate_config(cfg)

    # Set random seeds for reproducibility
    set_seeds(cfg.experiment.seed)

    # Get output directory
    if cfg.get("output_dir"):
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Create subdirectories
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="cfw_dataloader_eval",
        log_file=str(log_dir / "dataloader_eval.log"),
        level=cfg.get("log_level", "INFO")
    )

    logger.info("Starting dataloader comparison")
    logger.info(f"Dataset: {cfg.dataset.name}")
    logger.info(f"Output directory: {output_dir}")

    requested_num_batches = cfg.get("num_batches", None)
    if requested_num_batches is None:
        logger.info("num_batches not specified; analyzing full epoch for each dataloader.")
    else:
        logger.info(f"Requested num_batches={requested_num_batches}")

    dataloader_eval_cfg = cfg.get("dataloader_eval", {})
    return_paths_for_eval = bool(dataloader_eval_cfg.get("return_paths", True))
    plot_cfg = dataloader_eval_cfg.get("plots", None)
    logger.info(f"Path tracking for eval metrics enabled: {return_paths_for_eval}")

    # Create baseline dataloader
    logger.info("\n" + "=" * 80)
    logger.info("Creating BASELINE dataloader...")
    logger.info("=" * 80)
    baseline_loader = create_train_dataloader_by_type(
        cfg=cfg,
        dataloader_type="baseline",
        shuffle=True,
        drop_last=False,
        return_paths=return_paths_for_eval,
    )
    logger.info(f"Baseline loader created with {len(baseline_loader)} batches")

    # Create CFW dataloader
    logger.info("\n" + "=" * 80)
    logger.info("Creating CFW dataloader...")
    logger.info("=" * 80)
    cfw_loader = create_train_dataloader_by_type(
        cfg=cfg,
        dataloader_type="cfw",
        shuffle=True,
        drop_last=False,
        return_paths=return_paths_for_eval,
    )
    logger.info(f"CFW loader created with {len(cfw_loader)} batches")

    logger.info(f"KL mode: {KL_DIRECTION}")

    # Analyze baseline dataloader
    logger.info("\n" + "=" * 80)
    logger.info("Analyzing BASELINE dataloader...")
    logger.info("=" * 80)
    baseline_results = analyze_dataloader_uniform_kl(
        dataloader=baseline_loader,
        num_classes=cfg.dataset.num_classes,
        requested_num_batches=requested_num_batches,
        logger=logger,
    )
    logger.info(
        f"Baseline batches analyzed: {baseline_results['num_batches_analyzed']} "
        f"(batch_size min={int(np.min(baseline_results['batch_sizes'])) if baseline_results['batch_sizes'].size else 'N/A'}, "
        f"max={int(np.max(baseline_results['batch_sizes'])) if baseline_results['batch_sizes'].size else 'N/A'})"
    )
    logger.info(f"Baseline Mean KL Divergence: {baseline_results['mean_kl_divergence']:.6f}")
    logger.info(f"Baseline Std KL Divergence: {baseline_results['std_kl_divergence']:.6f}")

    # Analyze CFW dataloader
    logger.info("\n" + "=" * 80)
    logger.info("Analyzing CFW dataloader...")
    logger.info("=" * 80)
    cfw_results = analyze_dataloader_uniform_kl(
        dataloader=cfw_loader,
        num_classes=cfg.dataset.num_classes,
        requested_num_batches=requested_num_batches,
        logger=logger,
    )
    logger.info(
        f"CFW batches analyzed: {cfw_results['num_batches_analyzed']} "
        f"(batch_size min={int(np.min(cfw_results['batch_sizes'])) if cfw_results['batch_sizes'].size else 'N/A'}, "
        f"max={int(np.max(cfw_results['batch_sizes'])) if cfw_results['batch_sizes'].size else 'N/A'})"
    )
    logger.info(f"CFW Mean KL Divergence: {cfw_results['mean_kl_divergence']:.6f}")
    logger.info(f"CFW Std KL Divergence: {cfw_results['std_kl_divergence']:.6f}")

    if baseline_results["num_batches_analyzed"] == 0:
        raise RuntimeError(
            "Baseline dataloader produced zero analyzable batches. "
            "Check dataset paths, class labels, and batch settings."
        )
    if cfw_results["num_batches_analyzed"] == 0:
        raise RuntimeError(
            "CFW dataloader produced zero analyzable batches. "
            "Check feature files, labels, and batch settings."
        )

    # Compute improvement
    baseline_mean = baseline_results["mean_kl_divergence"]
    cfw_mean = cfw_results["mean_kl_divergence"]
    if np.isfinite(baseline_mean) and not np.isclose(baseline_mean, 0.0):
        improvement = (1 - cfw_mean / baseline_mean) * 100
        logger.info(f"\nKL Divergence Improvement: {improvement:.2f}%")
        improvement_msg = f"{improvement:.2f}%"
    else:
        logger.info("\nKL Divergence Improvement: N/A (baseline mean KL is zero or undefined)")
        improvement_msg = "N/A (baseline mean KL is zero or undefined)"

    # Generate plots
    logger.info("\n" + "=" * 80)
    logger.info("Generating comparison plots...")
    logger.info("=" * 80)
    plot_comparison(
        baseline_results=baseline_results,
        cfw_results=cfw_results,
        output_dir=plots_dir,
        logger=logger,
        plot_cfg=plot_cfg,
    )

    # Save statistics
    save_statistics(baseline_results, cfw_results, output_dir, logger)

    # Print summary
    print("\n" + "=" * 80)
    print("Dataloader Comparison Complete!")
    print("=" * 80)
    print(f"\nKL Direction: {KL_DIRECTION}")
    print(f"Baseline Batches Analyzed: {baseline_results['num_batches_analyzed']}")
    print(f"CFW Batches Analyzed:      {cfw_results['num_batches_analyzed']}")
    print(f"\nBaseline KL Divergence: {baseline_results['mean_kl_divergence']:.6f} ± {baseline_results['std_kl_divergence']:.6f}")
    print(f"CFW KL Divergence:      {cfw_results['mean_kl_divergence']:.6f} ± {cfw_results['std_kl_divergence']:.6f}")
    print(f"Improvement:            {improvement_msg}")
    print(f"\nPlots saved to: {plots_dir}")
    print(f"Statistics saved to: {output_dir / 'dataloader_statistics.txt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
