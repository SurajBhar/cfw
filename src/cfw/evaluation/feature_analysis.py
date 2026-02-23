"""
Feature quality analysis for CFW.

This module provides notebook-aligned, metrics-only feature analysis utilities.
"""


import pickle
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier


def _json_float(value: float) -> Optional[float]:
    """Convert numeric values to JSON-safe floats (NaN/inf -> None)."""
    v = float(value)
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def _l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize feature rows."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return features / norms


def _validate_metric(metric: str) -> str:
    """Validate and normalize metric names."""
    normalized = str(metric).strip().lower()
    if normalized not in {"euclidean", "cosine"}:
        raise ValueError(
            f"Unsupported metric '{metric}'. Expected one of: euclidean, cosine."
        )
    return normalized


def _nearest_centroid_predict_with_metric(
    features: np.ndarray,
    centroids: np.ndarray,
    classes: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """Predict labels by nearest centroid with configurable distance metric."""
    resolved_metric = _validate_metric(metric)
    if resolved_metric == "euclidean":
        x2 = np.sum(features * features, axis=1, keepdims=True)  # (N, 1)
        c2 = np.sum(centroids * centroids, axis=1, keepdims=False)[None, :]  # (1, C)
        d2 = x2 + c2 - 2.0 * (features @ centroids.T)  # (N, C)
        pred_idx = np.argmin(d2, axis=1)
        return classes[pred_idx]

    # Cosine mode: nearest by maximum cosine similarity.
    x_unit = _l2_normalize(features)
    c_unit = _l2_normalize(centroids)
    sims = x_unit @ c_unit.T
    pred_idx = np.argmax(sims, axis=1)
    return classes[pred_idx]


def _nearest_centroid_accuracy_in_sample(
    features: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    classes: np.ndarray,
    metric: str = "euclidean",
) -> Optional[float]:
    """Nearest-centroid accuracy evaluated in-sample (optimistic)."""
    try:
        pred_labels = _nearest_centroid_predict_with_metric(
            features=features,
            centroids=centroids,
            classes=classes,
            metric=metric,
        )
        return _json_float(np.mean(pred_labels == labels))
    except Exception:
        return None


def _nearest_centroid_accuracy_holdout(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    metric: str = "euclidean",
) -> Optional[float]:
    """
    Nearest-centroid accuracy on stratified holdout split.

    Centroids are fit only on the train split and evaluated on the held-out split.
    """
    try:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=float(test_size),
            random_state=int(seed),
        )
        (train_idx, test_idx), = splitter.split(features, labels)
        x_train = features[train_idx]
        y_train = labels[train_idx]
        x_test = features[test_idx]
        y_test = labels[test_idx]

        classes = np.unique(y_train)
        centroids = np.stack(
            [x_train[y_train == c].mean(axis=0) for c in classes],
            axis=0,
        )
        pred = _nearest_centroid_predict_with_metric(
            features=x_test,
            centroids=centroids,
            classes=classes,
            metric=metric,
        )
        return _json_float(np.mean(pred == y_test))
    except Exception:
        return None


def _knn_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    seed: int = 42,
    metric: str = "euclidean",
    test_size: float = 0.2,
) -> Optional[float]:
    """
    Non-parametric kNN accuracy on a stratified holdout split.

    Returns None if split/class constraints are insufficient.
    """
    try:
        resolved_metric = _validate_metric(metric)
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=float(test_size),
            random_state=int(seed),
        )
        (train_idx, test_idx), = splitter.split(features, labels)
        k_eff = max(1, min(int(k), int(len(train_idx))))
        clf = KNeighborsClassifier(n_neighbors=k_eff, metric=resolved_metric)
        clf.fit(features[train_idx], labels[train_idx])
        return _json_float(clf.score(features[test_idx], labels[test_idx]))
    except Exception:
        return None


def _silhouette_subsample(
    features: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 5000,
    seed: int = 42,
) -> Optional[float]:
    """Silhouette score (subsampled for large datasets)."""
    try:
        classes, counts = np.unique(labels, return_counts=True)
        if len(classes) < 2 or int(np.min(counts)) <= 1:
            return None

        n_samples = int(features.shape[0])
        if n_samples > int(max_samples):
            rs = np.random.RandomState(seed)
            idx = rs.choice(n_samples, size=int(max_samples), replace=False)
            return _json_float(
                silhouette_score(features[idx], labels[idx], metric="euclidean")
            )

        return _json_float(silhouette_score(features, labels, metric="euclidean"))
    except Exception:
        return None


def _sample_pair_distances(
    features: np.ndarray,
    labels: np.ndarray,
    n_pairs: int = 30000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample random pairwise distances and split into intra/inter-class sets."""
    n_samples = int(features.shape[0])
    if n_samples < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    rs = np.random.RandomState(seed)
    a = rs.randint(0, n_samples, size=int(n_pairs))
    b = rs.randint(0, n_samples, size=int(n_pairs))

    # Ensure a != b for each sampled pair.
    mask = a == b
    while mask.any():
        b[mask] = rs.randint(0, n_samples, size=int(mask.sum()))
        mask = a == b

    diff = features[a] - features[b]
    d = np.linalg.norm(diff, axis=1)
    same = (labels[a] == labels[b])
    return d[same], d[~same]


def compute_metrics_only_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    knn_k: int = 10,
    silhouette_max_samples: int = 5000,
    pair_samples: int = 30000,
    seed: int = 42,
    large_scale_mode: bool = False,
    enable_knn: Optional[bool] = None,
    enable_silhouette: Optional[bool] = None,
    normalize: str = "none",
    knn_metric: str = "euclidean",
    nearest_centroid_mode: str = "holdout",
    nearest_centroid_metric: str = "euclidean",
    nearest_centroid_test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    Compute notebook-aligned metrics-only separability analysis.

    This implements the core metrics from:
    `variance_analysis/vit_embedding_separability_audit_bal_sf.ipynb`
    without any plotting steps.
    """
    # For very large datasets, float32 reduces memory pressure significantly.
    compute_dtype = np.float32 if large_scale_mode else np.float64
    features = features.astype(compute_dtype, copy=False)
    labels = labels.astype(np.int64)

    normalize_mode = str(normalize).strip().lower()
    if normalize_mode not in {"none", "l2"}:
        raise ValueError("normalize must be one of: none, l2")

    if normalize_mode == "l2":
        features = _l2_normalize(features)

    knn_metric = _validate_metric(knn_metric)
    nearest_centroid_metric = _validate_metric(nearest_centroid_metric)

    nc_mode = str(nearest_centroid_mode).strip().lower()
    if nc_mode not in {"holdout", "in_sample"}:
        raise ValueError("nearest_centroid_mode must be one of: holdout, in_sample")
    if not (0.0 < float(nearest_centroid_test_size) < 1.0):
        raise ValueError("nearest_centroid_test_size must be between 0 and 1")

    if enable_knn is None:
        enable_knn = not large_scale_mode
    if enable_silhouette is None:
        enable_silhouette = not large_scale_mode

    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes for separability metrics.")

    n_samples, embedding_dim = features.shape
    class_indices = {c: np.where(labels == c)[0] for c in classes}
    class_counts = {int(c): int(len(class_indices[c])) for c in classes}
    if any(v <= 0 for v in class_counts.values()):
        raise ValueError("Each class must contain at least one sample.")

    # Centroids
    mu = features.mean(axis=0)
    centroids = np.stack([features[class_indices[c]].mean(axis=0) for c in classes], axis=0)

    # Intra scatter
    intra_msd = np.zeros(len(classes), dtype=np.float64)
    intra_md = np.zeros(len(classes), dtype=np.float64)
    for k, c in enumerate(classes):
        x_c = features[class_indices[c]]
        dif = x_c - centroids[k]
        d2 = np.sum(dif * dif, axis=1)
        intra_msd[k] = np.mean(d2)
        intra_md[k] = np.mean(np.sqrt(np.maximum(d2, 0.0)))

    intra_macro_msd = float(np.mean(intra_msd))
    intra_micro_msd = float(
        np.sum([class_counts[int(c)] * intra_msd[k] for k, c in enumerate(classes)]) / n_samples
    )
    intra_macro_md = float(np.mean(intra_md))
    intra_micro_md = float(
        np.sum([class_counts[int(c)] * intra_md[k] for k, c in enumerate(classes)]) / n_samples
    )

    # Inter scatter
    inter_vals = np.sum((centroids - mu) ** 2, axis=1)
    inter_macro = float(np.mean(inter_vals))
    inter_weighted = float(
        np.sum([class_counts[int(c)] * inter_vals[k] for k, c in enumerate(classes)]) / n_samples
    )

    # Centroid distances
    dist_mat = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :],
        axis=2,
    )
    upper_idx = np.triu_indices(len(classes), k=1)
    upper_tri = dist_mat[upper_idx]
    mean_center_dist = float(np.mean(upper_tri))
    min_center_dist = float(np.min(upper_tri))

    # DBI (custom notebook variant)
    denom = dist_mat + 1e-12
    r = (intra_md[:, None] + intra_md[None, :]) / denom
    np.fill_diagonal(r, -np.inf)
    r_i = np.max(r, axis=1)
    dbi_custom = float(np.mean(r_i))

    # DBI (sklearn cross-check)
    dbi_sklearn = float(davies_bouldin_score(features, labels))

    fisher_ratio = float(inter_weighted / (intra_micro_msd + 1e-12))

    # Metrics-only sanity checks.
    if nc_mode == "holdout":
        nearest_centroid_acc = _nearest_centroid_accuracy_holdout(
            features=features,
            labels=labels,
            test_size=float(nearest_centroid_test_size),
            seed=int(seed),
            metric=nearest_centroid_metric,
        )
    else:
        nearest_centroid_acc = _nearest_centroid_accuracy_in_sample(
            features=features,
            labels=labels,
            centroids=centroids,
            classes=classes,
            metric=nearest_centroid_metric,
        )

    knn_acc = None
    if enable_knn:
        knn_acc = _knn_accuracy(
            features=features,
            labels=labels,
            k=int(knn_k),
            seed=int(seed),
            metric=knn_metric,
            test_size=float(nearest_centroid_test_size),
        )

    silhouette = None
    if enable_silhouette:
        silhouette = _silhouette_subsample(
            features=features,
            labels=labels,
            max_samples=int(silhouette_max_samples),
            seed=int(seed),
        )

    intra_d, inter_d = _sample_pair_distances(
        features=features,
        labels=labels,
        n_pairs=int(pair_samples),
        seed=int(seed),
    )
    if len(inter_d) > 0 and len(intra_d) > 0:
        median_inter_distance = _json_float(np.median(inter_d))
        frac_intra_gt_median_inter = _json_float(np.mean(intra_d > np.median(inter_d)))
    else:
        median_inter_distance = None
        frac_intra_gt_median_inter = None

    # Keep legacy top-level aliases for broad compatibility with existing code paths.
    return {
        "analysis_variant": "metrics_only_notebook_aligned",
        "analysis_config": {
            "large_scale_mode": bool(large_scale_mode),
            "compute_dtype": str(np.dtype(compute_dtype)),
            "normalize": normalize_mode,
            "enable_knn": bool(enable_knn),
            "enable_silhouette": bool(enable_silhouette),
            "knn_k": int(knn_k),
            "knn_metric": knn_metric,
            "nearest_centroid_mode": nc_mode,
            "nearest_centroid_metric": nearest_centroid_metric,
            "nearest_centroid_test_size": float(nearest_centroid_test_size),
            "silhouette_max_samples": int(silhouette_max_samples),
            "pair_samples": int(pair_samples),
            "seed": int(seed),
        },
        "metric_notes": {
            "intra_msd": "mean squared Euclidean distance to class centroid (sum over dims)",
            "inter_scatter": "scatter from global mean (sum over dims)",
            "dbi_custom": "mean(max_j((S_i + S_j) / M_ij)) with S_i=mean Euclidean distance to centroid",
            "compat_aliases": {
                "avg_intra_class_variance_macro": "mapped to intra_macro_msd",
                "inter_class_variance": "mapped to inter_macro",
                "mean_center_distance": "mapped to mean_center_distance",
                "davies_bouldin_index": "mapped to davies_bouldin_index_sklearn",
            },
        },
        "num_samples": int(n_samples),
        "embedding_dim": int(embedding_dim),
        "num_classes": int(len(classes)),
        "class_counts": class_counts,
        "intra_macro_msd": _json_float(intra_macro_msd),
        "intra_micro_msd": _json_float(intra_micro_msd),
        "intra_macro_mean_distance": _json_float(intra_macro_md),
        "intra_micro_mean_distance": _json_float(intra_micro_md),
        "inter_macro": _json_float(inter_macro),
        "inter_weighted": _json_float(inter_weighted),
        "mean_center_distance": _json_float(mean_center_dist),
        "min_center_distance": _json_float(min_center_dist),
        "davies_bouldin_index_custom": _json_float(dbi_custom),
        "davies_bouldin_index_sklearn": _json_float(dbi_sklearn),
        "fisher_ratio": _json_float(fisher_ratio),
        "sanity_checks": {
            "nearest_centroid_accuracy": nearest_centroid_acc,
            f"knn_accuracy_k{int(knn_k)}": knn_acc,
            "silhouette": silhouette,
            "nearest_centroid_mode": nc_mode,
            "nearest_centroid_metric": nearest_centroid_metric,
            "knn_metric": knn_metric,
        },
        "distance_overlap": {
            "median_inter_distance": median_inter_distance,
            "fraction_intra_gt_median_inter": frac_intra_gt_median_inter,
            "num_intra_pairs": int(len(intra_d)),
            "num_inter_pairs": int(len(inter_d)),
        },
        "per_class": {
            "class_ids": [int(c) for c in classes],
            "counts": [int(class_counts[int(c)]) for c in classes],
            "intra_msd": [float(v) for v in intra_msd],
            "intra_mean_distance": [float(v) for v in intra_md],
        },
        # Backward-compatible aliases expected by some existing consumers.
        "avg_intra_class_variance_macro": _json_float(intra_macro_msd),
        "inter_class_variance": _json_float(inter_macro),
        "davies_bouldin_index": _json_float(dbi_sklearn),
        "intra_class_variance_per_class": {
            int(c): float(intra_msd[k]) for k, c in enumerate(classes)
        },
    }


def compute_feature_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Compute notebook-aligned metrics-only feature analysis.

    This is a compatibility wrapper around `compute_metrics_only_analysis()`
    so existing callers can keep using the same API.
    """
    return compute_metrics_only_analysis(features, labels, **kwargs)


def validate_feature_quality(
    metrics: Dict[str, Any],
    dbi_threshold: float = 2.0,
    intra_var_threshold: float = 1.0,
) -> Dict[str, Any]:
    """
    Validate feature quality against thresholds.

    Args:
        metrics: Metrics from compute_feature_metrics()
        dbi_threshold: Maximum acceptable DBI (lower is better).
                      Default 2.0 is reasonable for most cases.
        intra_var_threshold: Maximum acceptable intra-class variance.
                            Default 1.0 (adjust based on your features).

    Returns:
        Dictionary with validation results and warnings

    Example:
        >>> metrics = compute_feature_metrics(features, labels)
        >>> validation = validate_feature_quality(metrics)
        >>> if not validation['is_valid']:
        ...     for warning in validation['warnings']:
        ...         print(f"WARNING: {warning}")
    """
    warnings = []
    analysis_variant = metrics.get("analysis_variant", "")

    intra_value = float(metrics['avg_intra_class_variance_macro'])
    inter_value = float(metrics['inter_class_variance'])

    # Notebook-aligned metrics use sum-over-dim scatter terms, so normalize by
    # embedding dimensionality for threshold checks to preserve historical behavior.
    if analysis_variant == "metrics_only_notebook_aligned":
        emb_dim = max(1.0, float(metrics.get("embedding_dim", 1)))
        intra_value = intra_value / emb_dim
        inter_value = inter_value / emb_dim

    if metrics['davies_bouldin_index'] > dbi_threshold:
        warnings.append(
            f"High Davies-Bouldin Index ({metrics['davies_bouldin_index']:.4f} > {dbi_threshold}) "
            "- features may not cluster well. Consider using different backbone or feature extraction."
        )

    if intra_value > intra_var_threshold:
        warnings.append(
            f"High intra-class variance ({intra_value:.4f} > {intra_var_threshold}) "
            "- classes may not be compact. This could affect clustering quality."
        )

    # Additional check: inter-class variance should be reasonable
    if inter_value < 0.01:
        warnings.append(
            f"Very low inter-class variance ({inter_value:.4f}) "
            "- classes may be too similar. Feature extraction might not be discriminative enough."
        )

    return {
        "is_valid": len(warnings) == 0,
        "warnings": warnings,
        "metrics": metrics,
    }


def analyze_feature_file(
    feature_path: Path,
    label_path: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Analyze features from pickle files.

    Args:
        feature_path: Path to features.pkl
        label_path: Path to labels.pkl
        output_path: Optional path to save metrics JSON

    Returns:
        Computed metrics dictionary

    Example:
        >>> from pathlib import Path
        >>> feature_path = Path("outputs/features/train/features.pkl")
        >>> label_path = Path("outputs/features/train/labels.pkl")
        >>> metrics = analyze_feature_file(feature_path, label_path)
        >>> print(f"DBI: {metrics['davies_bouldin_index']:.4f}")
    """
    import json

    # Load features and labels
    with open(feature_path, 'rb') as f:
        features = pickle.load(f)
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)

    # Compute metrics
    metrics = compute_feature_metrics(features, labels)

    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    return metrics


def print_feature_metrics(metrics: Dict[str, Any]) -> None:
    """
    Pretty print feature metrics.

    Args:
        metrics: Metrics from compute_feature_metrics()

    Example:
        >>> metrics = compute_feature_metrics(features, labels)
        >>> print_feature_metrics(metrics)
    """
    print("\n" + "=" * 80)
    print("Feature Quality Metrics")
    print("=" * 80)
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"Embedding dimension: {metrics['embedding_dim']}")
    print(f"Number of classes: {metrics['num_classes']}")
    print(f"\nClass distribution:")
    for cls, count in sorted(metrics['class_counts'].items()):
        print(f"  Class {cls}: {count} samples")
    print(f"\nCluster Quality Metrics:")
    print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f} (lower is better)")
    print(f"  Avg intra-class variance: {metrics['avg_intra_class_variance_macro']:.4f} (lower is better)")
    print(f"  Inter-class variance: {metrics['inter_class_variance']:.4f} (higher is better)")
    print(f"  Mean center distance: {metrics['mean_center_distance']:.4f} (higher is better)")
    if 'davies_bouldin_index_custom' in metrics:
        print(f"  Davies-Bouldin Index (custom): {metrics['davies_bouldin_index_custom']:.4f}")
    if 'fisher_ratio' in metrics:
        print(f"  Fisher ratio: {metrics['fisher_ratio']:.4f} (higher is better)")
    if 'sanity_checks' in metrics:
        sanity = metrics['sanity_checks']
        print(f"  Nearest-centroid acc: {sanity.get('nearest_centroid_accuracy')}")
        for key, val in sanity.items():
            if key.startswith("knn_accuracy_k"):
                print(f"  {key}: {val}")
        print(f"  Silhouette: {sanity.get('silhouette')}")
    print(f"\nPer-class intra-class variance:")
    for cls, var in sorted(metrics['intra_class_variance_per_class'].items()):
        print(f"  Class {cls}: {var:.4f}")
    print("=" * 80)


def compare_feature_sets(
    features_list: list[tuple[str, np.ndarray, np.ndarray]],
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple feature sets (e.g., different models or splits).

    Args:
        features_list: List of (name, features, labels) tuples

    Returns:
        Dictionary mapping name to metrics

    Example:
        >>> dinov2_features = (dinov2_emb, labels)
        >>> vit_features = (vit_emb, labels)
        >>> comparison = compare_feature_sets([
        ...     ("DINOv2", *dinov2_features),
        ...     ("ViT", *vit_features),
        ... ])
        >>> for name, metrics in comparison.items():
        ...     print(f"{name} DBI: {metrics['davies_bouldin_index']:.4f}")
    """
    results = {}

    for name, features, labels in features_list:
        metrics = compute_feature_metrics(features, labels)
        results[name] = metrics

    return results


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python feature_analysis.py <features.pkl> <labels.pkl> [output.json]")
        sys.exit(1)

    feature_path = Path(sys.argv[1])
    label_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    print(f"Analyzing features from {feature_path} and {label_path}...")

    metrics = analyze_feature_file(feature_path, label_path, output_path)
    print_feature_metrics(metrics)

    # Validate quality
    validation = validate_feature_quality(metrics)
    if not validation['is_valid']:
        print("\nWARNINGS:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    else:
        print("\n✓ Feature quality validation passed")

    if output_path:
        print(f"\nMetrics saved to {output_path}")
