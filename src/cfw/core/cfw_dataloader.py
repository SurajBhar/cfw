"""
Clustered Feature Weighting (CFW) Dataloader.

This module implements the main CFW dataloader that:
1. Loads pre-extracted features
2. Performs clustering in batches
3. Computes sample weights
4. Creates a weighted dataloader for training

The CFW dataloader is the core innovation that addresses class imbalance
by clustering features in latent space and weighting samples inversely
to their cluster size.
"""


from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
import os
import json
import pickle
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..data.datasets import WeightedImageDataset

from .clustering import FeatureClusterer, ClusteringConfig
from .weighting import WeightComputer, WeightingConfig


logger = logging.getLogger(__name__)


class CFWDataLoaderBuilder:
    """
    Builder for Clustered Feature Weighting DataLoader.

    Processes pre-extracted features through clustering and weighting to create
    a dataloader with balanced sampling properties.

    The CFW algorithm works as follows:
    1. Load features extracted from a pre-trained model (DINOv2, ViT, etc.)
    2. Process features in batches (default: 1024 samples per batch)
    3. For each batch:
       a. Compute cosine distance matrix
       b. Apply HDBSCAN clustering
       c. Assign weights: 1/cluster_size for regular, 0.001 for outliers
    4. Create WeightedRandomSampler with computed weights
    5. Build DataLoader with weighted sampling

    Example:
        >>> from cfw.core.cfw_dataloader import CFWDataLoaderBuilder
        >>> from cfw.core.clustering import ClusteringConfig
        >>> from cfw.core.weighting import WeightingConfig
        >>>
        >>> clustering_cfg = ClusteringConfig(min_cluster_size=25)
        >>> weighting_cfg = WeightingConfig(outlier_weight=0.001)
        >>>
        >>> builder = CFWDataLoaderBuilder(
        ...     feature_file_path='features.pkl',
        ...     label_file_path='labels.pkl',
        ...     img_path_file_path='paths.pkl',
        ...     data_dir='/path/to/images',
        ...     clustering_config=clustering_cfg,
        ...     weighting_config=weighting_cfg,
        ... )
        >>>
        >>> dataloader = builder.build(batch_size=64)
    """

    def __init__(
        self,
        feature_file_path: str,
        label_file_path: str,
        img_path_file_path: str,
        data_dir: str = None,
        clustering_config: ClusteringConfig = None,
        weighting_config: WeightingConfig = None,
        weights_table_file_path: Optional[str] = None,
        save_generated_weights_file_path: Optional[str] = None,
    ):
        """Initialize CFW DataLoader builder.

        Args:
            feature_file_path: Path to pickle file with pre-extracted features
            label_file_path: Path to pickle file with ground truth labels
            img_path_file_path: Path to pickle file with image file paths
            data_dir: Base directory for resolving relative image paths
            clustering_config: Configuration for HDBSCAN clustering (uses defaults if None)
            weighting_config: Configuration for weight computation (uses defaults if None)
            weights_table_file_path: Optional precomputed weights table to load
            save_generated_weights_file_path: Optional .pkl path to persist runtime weights
        """
        self.feature_file_path = feature_file_path
        self.label_file_path = label_file_path
        self.img_path_file_path = img_path_file_path
        self.data_dir = data_dir
        self.weights_table_file_path = weights_table_file_path
        self.save_generated_weights_file_path = save_generated_weights_file_path

        # Use default configs if not provided
        self.clustering_config = clustering_config if clustering_config else ClusteringConfig()
        self.weighting_config = weighting_config if weighting_config else WeightingConfig()

        # Initialize clusterer and weight computer
        self.clusterer = FeatureClusterer(self.clustering_config)
        self.weight_computer = WeightComputer(self.weighting_config)

        # Storage for loaded data
        self.features = None
        self.labels = None
        self.img_paths = None

        # Storage for computed results
        self.all_weights = None
        self.all_cluster_labels = None
        self.clustering_stats = None
        self.feature_extraction_batch_size = None

    def _should_prefix_data_dir(self, path_str: str) -> bool:
        """Return True when a relative path should be prefixed with data_dir."""
        if self.data_dir is None:
            return False

        norm_input = os.path.normpath(path_str)
        norm_data_dir = os.path.normpath(str(self.data_dir))

        # Path already rooted at split dir (e.g. data/.../train/c1/img.jpg).
        if norm_input == norm_data_dir:
            return False
        if norm_input.startswith(norm_data_dir + os.sep):
            return False

        return True

    def _canonicalize_path(self, image_path: str) -> str:
        """Normalize path keys so features/image-path/weights alignment is stable."""
        path_str = str(image_path)
        path_obj = Path(path_str)
        if not path_obj.is_absolute() and self._should_prefix_data_dir(path_str):
            path_obj = Path(self.data_dir) / path_obj
        return os.path.normpath(str(path_obj))

    def _log_input_file_provenance(self) -> None:
        """Log feature bundle input file provenance."""
        for label, file_path in (
            ("feature_file", self.feature_file_path),
            ("label_file", self.label_file_path),
            ("img_path_file", self.img_path_file_path),
        ):
            path_obj = Path(file_path)
            if not path_obj.is_file():
                logger.error("Missing required %s: %s", label, path_obj)
                raise FileNotFoundError(f"Required file not found: {path_obj}")
            size_mb = path_obj.stat().st_size / (1024 * 1024)
            logger.info("Using %s=%s (%.2f MB)", label, path_obj, size_mb)

    def _load_feature_extraction_metadata(self) -> None:
        """Load optional metadata produced during feature extraction."""
        feature_file_dir = Path(str(self.feature_file_path)).parent
        metadata_path = feature_file_dir / "feature_extraction_metadata.json"
        if not metadata_path.is_file():
            logger.info(
                "feature_extraction_metadata.json not found next to feature file; "
                "cannot auto-verify extraction batch size."
            )
            return

        try:
            with open(metadata_path, "r", encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
        except Exception as exc:
            logger.warning(
                "Failed to parse feature extraction metadata from %s: %s",
                metadata_path,
                exc,
            )
            return

        extraction_batch_size = metadata.get("batch_size")
        if extraction_batch_size is None:
            logger.info(
                "feature_extraction_metadata.json present but batch_size is missing."
            )
            return

        self.feature_extraction_batch_size = int(extraction_batch_size)
        logger.info(
            "Loaded feature extraction metadata: batch_size=%d, shuffle=%s, drop_last=%s",
            self.feature_extraction_batch_size,
            metadata.get("shuffle"),
            metadata.get("drop_last"),
        )

    def _validate_loaded_alignment(self) -> None:
        """Validate that loaded feature, label, and image-path arrays are aligned."""
        n_samples = len(self.features)
        if len(self.labels) != n_samples:
            raise ValueError(
                f"Feature and label counts don't match: "
                f"{n_samples} features vs {len(self.labels)} labels"
            )

        if len(self.img_paths) != n_samples:
            raise ValueError(
                f"Feature and image path counts don't match: "
                f"{n_samples} features vs {len(self.img_paths)} paths"
            )

    def _align_precomputed_weights(
        self,
        table_paths: List[str],
        table_weights: np.ndarray,
        table_labels: Optional[np.ndarray] = None,
        table_cluster_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align precomputed table rows to the loaded feature order."""
        expected_count = len(self.img_paths)
        if len(table_paths) != expected_count:
            raise ValueError(
                "Precomputed weights table length mismatch: "
                f"table_paths={len(table_paths)} vs loaded_paths={expected_count}"
            )
        if len(table_weights) != expected_count:
            raise ValueError(
                "Precomputed weights table length mismatch: "
                f"table_weights={len(table_weights)} vs loaded_paths={expected_count}"
            )
        if table_cluster_labels is None:
            table_cluster_labels = np.full(expected_count, -1, dtype=np.int64)
        if len(table_cluster_labels) != expected_count:
            raise ValueError(
                "Precomputed cluster_labels length mismatch: "
                f"cluster_labels={len(table_cluster_labels)} vs loaded_paths={expected_count}"
            )

        table_paths_canon = [self._canonicalize_path(p) for p in table_paths]
        loaded_paths_canon = [self._canonicalize_path(p) for p in self.img_paths]

        labels_available = table_labels is not None
        if labels_available and len(table_labels) != expected_count:
            raise ValueError(
                "Precomputed labels length mismatch: "
                f"table_labels={len(table_labels)} vs loaded_paths={expected_count}"
            )

        if table_paths_canon == loaded_paths_canon:
            if labels_available:
                loaded_labels = np.asarray(self.labels).astype(np.int64, copy=False)
                table_labels_np = np.asarray(table_labels).astype(np.int64, copy=False)
                if not np.array_equal(loaded_labels, table_labels_np):
                    raise ValueError(
                        "Precomputed labels do not match loaded labels even though path order matches."
                    )
            return (
                np.asarray(table_weights).astype(np.float32, copy=False),
                np.asarray(table_cluster_labels).astype(np.int64, copy=False),
            )

        index_map: Dict[Tuple[str, Optional[int]], Deque[int]] = defaultdict(deque)
        if labels_available:
            table_labels_np = np.asarray(table_labels).astype(np.int64, copy=False)
            for idx, (path_key, label) in enumerate(zip(table_paths_canon, table_labels_np)):
                index_map[(path_key, int(label))].append(idx)
        else:
            for idx, path_key in enumerate(table_paths_canon):
                index_map[(path_key, None)].append(idx)

        aligned_indices: List[int] = []
        loaded_labels = np.asarray(self.labels).astype(np.int64, copy=False)
        for i, path_key in enumerate(loaded_paths_canon):
            key = (path_key, int(loaded_labels[i])) if labels_available else (path_key, None)
            if not index_map[key]:
                raise ValueError(
                    "Could not align precomputed weights to loaded features. "
                    f"Missing entry for path={self.img_paths[i]} label={int(loaded_labels[i])}."
                )
            aligned_indices.append(index_map[key].popleft())

        remaining = sum(len(v) for v in index_map.values())
        if remaining != 0:
            raise ValueError(
                "Precomputed weights table has unused rows after alignment. "
                f"unused_rows={remaining}"
            )

        aligned_weights = np.asarray(table_weights, dtype=np.float32)[aligned_indices]
        aligned_clusters = np.asarray(table_cluster_labels, dtype=np.int64)[aligned_indices]
        logger.info(
            "Aligned precomputed weights to loaded feature order using %s matching.",
            "path+label" if labels_available else "path-only",
        )
        return aligned_weights, aligned_clusters

    def _load_precomputed_weights_table(self) -> bool:
        """Load optional precomputed weights table and align to feature order."""
        if not self.weights_table_file_path:
            return False

        table_path = Path(str(self.weights_table_file_path))
        if not table_path.is_file():
            raise FileNotFoundError(f"Precomputed weights table file not found: {table_path}")

        logger.info("Loading precomputed CFW weights table from: %s", table_path)
        with open(table_path, "rb") as table_file:
            table_payload = pickle.load(table_file)

        if not isinstance(table_payload, dict):
            raise ValueError(
                "Precomputed weights table must be a dictionary containing "
                "image_paths, labels (optional), weights, and cluster_labels (optional)."
            )

        table_paths = table_payload.get("image_paths")
        table_weights = table_payload.get("weights")
        table_labels = table_payload.get("labels")
        table_cluster_labels = table_payload.get("cluster_labels")

        if table_paths is None or table_weights is None:
            raise ValueError(
                "Precomputed weights table missing required keys: image_paths and/or weights."
            )

        aligned_weights, aligned_clusters = self._align_precomputed_weights(
            table_paths=[str(p) for p in table_paths],
            table_weights=np.asarray(table_weights),
            table_labels=None if table_labels is None else np.asarray(table_labels),
            table_cluster_labels=None if table_cluster_labels is None else np.asarray(table_cluster_labels),
        )

        self.weight_computer.validate_weights(aligned_weights)
        self.all_weights = aligned_weights
        self.all_cluster_labels = aligned_clusters
        self.clustering_stats = [
            {
                "n_samples": int(len(self.features)),
                "source": "precomputed_weights_table",
                "weights_table_file": str(table_path),
            }
        ]

        logger.info(
            "Using precomputed weights table: samples=%d, min_weight=%.6f, max_weight=%.6f",
            len(self.all_weights),
            float(np.min(self.all_weights)),
            float(np.max(self.all_weights)),
        )
        return True

    def _persist_weights_table(self, source: str) -> None:
        """Persist aligned/generated CFW weights table if an output path is configured."""
        if not self.save_generated_weights_file_path:
            logger.info("No weights table save path configured; keeping CFW weights in memory only.")
            return

        output_path = Path(str(self.save_generated_weights_file_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as output_file:
            pickle.dump(
                {
                    "image_paths": [str(p) for p in self.img_paths],
                    "labels": np.asarray(self.labels).astype(np.int64, copy=False),
                    "cluster_labels": np.asarray(self.all_cluster_labels).astype(np.int64, copy=False),
                    "weights": np.asarray(self.all_weights).astype(np.float32, copy=False),
                    "source": source,
                },
                output_file,
            )
        logger.info("Saved CFW weights table (%s) to: %s", source, output_path)

    def load_data(self) -> None:
        """Load features, labels, and image paths from pickle files.

        Raises:
            FileNotFoundError: If any of the pickle files don't exist
            ValueError: If loaded data has inconsistent shapes
        """
        logger.info("Loading pre-extracted CFW feature bundle...")
        self._log_input_file_provenance()
        self._load_feature_extraction_metadata()

        try:
            with open(self.feature_file_path, 'rb') as f:
                self.features = pickle.load(f)

            with open(self.label_file_path, 'rb') as f:
                self.labels = pickle.load(f)

            with open(self.img_path_file_path, 'rb') as f:
                self.img_paths = pickle.load(f)

        except FileNotFoundError as e:
            logger.error(f"Failed to load data files: {e}")
            raise

        # Convert to numpy arrays if needed
        if isinstance(self.features, torch.Tensor):
            self.features = self.features.numpy()
        elif not isinstance(self.features, np.ndarray):
            self.features = np.asarray(self.features)
        if isinstance(self.labels, torch.Tensor):
            self.labels = self.labels.numpy()
        elif not isinstance(self.labels, np.ndarray):
            self.labels = np.asarray(self.labels)

        n_samples = len(self.features)
        if len(self.img_paths) == 0:
            raise ValueError("Image path file is empty; cannot build CFW dataloader.")

        # Handle nested image paths (flatten if needed)
        if isinstance(self.img_paths[0], list):
            self.img_paths = [path for sublist in self.img_paths for path in sublist]

        # Resolve relative paths if data_dir is provided
        if self.data_dir:
            resolved_paths = []
            for path in self.img_paths:
                path_str = str(path)
                if not os.path.isabs(path_str) and self._should_prefix_data_dir(path_str):
                    resolved_path = os.path.join(self.data_dir, path_str)
                else:
                    resolved_path = path_str
                resolved_paths.append(resolved_path)
            self.img_paths = resolved_paths

        self._validate_loaded_alignment()

        logger.info(
            "Loaded aligned feature bundle: samples=%d, feature_dim=%d",
            n_samples,
            int(self.features.shape[1]),
        )

    def _load_features(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load features, labels, and image paths from pickle files.

        This is an alias for load_data() that returns the loaded data
        for testing and external use.

        Returns:
            Tuple of (features, labels, image_paths)

        Raises:
            FileNotFoundError: If any of the pickle files don't exist
            ValueError: If loaded data has inconsistent shapes
        """
        self.load_data()
        return self.features, self.labels, self.img_paths

    def _process_batch(
        self,
        batch_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Process a batch of features: cluster and compute weights.

        Args:
            batch_features: Feature batch of shape (batch_size, feature_dim)

        Returns:
            Tuple containing:
                - weights: Sample weights for this batch
                - cluster_labels: Cluster labels for this batch
                - stats: Dictionary with clustering statistics
        """
        # Cluster features
        cluster_labels, clusterer_obj = self.clusterer.cluster(batch_features)

        # Compute weights from cluster assignments
        weights, updated_labels, n_clusters = self.weight_computer.compute_weights(
            cluster_labels,
            return_num_clusters=True,
        )

        # Validate weights
        self.weight_computer.validate_weights(weights)

        # Gather statistics
        cluster_stats = self.clusterer.get_cluster_stats(cluster_labels)
        weight_stats = self.weight_computer.get_weight_stats(weights, updated_labels)

        stats = {
            'n_samples': len(batch_features),
            'clustering': cluster_stats,
            'weighting': weight_stats
        }

        return weights, updated_labels, stats

    def batch_process_features(self) -> None:
        """Process all features in batches for memory efficiency.

        Divides features into batches, clusters each batch separately,
        and aggregates results.
        """
        logger.info(
            "Processing features for clustering in chunks of %d samples...",
            self.clustering_batch_size,
        )

        n_samples = len(self.features)
        n_batches = (n_samples + self.clustering_batch_size - 1) // self.clustering_batch_size

        all_weights_list = []
        all_cluster_labels_list = []
        all_stats = []

        for i in range(n_batches):
            start_idx = i * self.clustering_batch_size
            end_idx = min((i + 1) * self.clustering_batch_size, n_samples)

            batch_features = self.features[start_idx:end_idx]

            # Process batch
            weights, cluster_labels, stats = self._process_batch(batch_features)

            all_weights_list.append(weights)
            all_cluster_labels_list.append(cluster_labels)
            all_stats.append(stats)

            logger.info(
                f"Batch {i+1}/{n_batches}: "
                f"{stats['clustering']['n_clusters']} clusters, "
                f"{stats['clustering']['n_outliers']} outliers"
            )

        # Concatenate all batches
        self.all_weights = np.concatenate(all_weights_list)
        self.all_cluster_labels = np.concatenate(all_cluster_labels_list)
        self.clustering_stats = all_stats

        logger.info(
            f"Clustering complete: {len(self.all_weights)} samples processed"
        )

    def create_dataset(self, transform: Optional[Any] = None) -> WeightedImageDataset:
        """Create WeightedImageDataset with computed weights.

        Args:
            transform: Optional transform to apply to images

        Returns:
            WeightedImageDataset instance ready for DataLoader
        """
        if transform is None:
            # Ensure default DataLoader collation works even when callers don't
            # pass explicit transforms.
            from ..data.transforms import make_no_aug_transform
            transform = make_no_aug_transform(resize_size=224)

        # WeightedImageDataset expects nested lists, so wrap in single-item lists
        dataset = WeightedImageDataset(
            image_paths_list=[self.img_paths],
            weights_list=[self.all_weights.tolist()],
            labels_list=[self.labels.tolist()],
            data_dir=self.data_dir,
            transform=transform,
        )
        return dataset

    def _aggregate_stats(self, clustering_batch_size: int, batch_size: int) -> dict:
        """Aggregate clustering statistics.

        Args:
            clustering_batch_size: Batch size used for clustering
            batch_size: Batch size for training dataloader

        Returns:
            Dictionary with aggregated statistics
        """
        return {
            'total_samples': len(self.all_weights),
            'feature_dim': self.features.shape[1],
            'n_batches': len(self.clustering_stats),
            'clustering_batch_size': clustering_batch_size,
            'training_batch_size': batch_size,
            'batch_stats': self.clustering_stats,
            'overall': {
                'mean_weight': float(np.mean(self.all_weights)),
                'std_weight': float(np.std(self.all_weights)),
                'min_weight': float(np.min(self.all_weights)),
                'max_weight': float(np.max(self.all_weights)),
                'n_unique_clusters': len(np.unique(self.all_cluster_labels))
            }
        }

    def build(
        self,
        batch_size: int = 1024,
        num_workers: int = 4,
        transform: Optional[Any] = None,
        clustering_batch_size: int = 1024,
        pin_memory: bool = True,
        drop_last: bool = False,
        shuffle: bool = True,
        prefetch_factor: Optional[int] = 2,
        distributed: bool = False,
        distributed_seed: int = 0,
    ) -> DataLoader:
        """Build the CFW dataloader.

        Complete pipeline:
        1. Load pre-extracted features
        2. Process features in batches (clustering + weighting)
        3. Create dataset and weighted sampler
        4. Build DataLoader

        Args:
            batch_size: Batch size for training dataloader
            num_workers: Number of worker threads for dataloader
            transform: Optional transform to apply to images
            clustering_batch_size: Batch size for clustering chunks (must equal batch_size)
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
            shuffle: Whether to use weighted random sampling (True) or sequential (False)
            prefetch_factor: Number of batches to prefetch per worker
            distributed: If True, use DistributedWeightedSampler for DDP training
            distributed_seed: Base random seed for DistributedWeightedSampler

        Returns:
            PyTorch DataLoader with weighted sampling
        """
        batch_size = int(batch_size)
        clustering_batch_size = int(clustering_batch_size)
        if clustering_batch_size != batch_size:
            raise ValueError(
                "CFW coupled-batch policy violation: "
                f"training batch_size={batch_size} but "
                f"clustering_batch_size={clustering_batch_size}. "
                "Set both values to the same number."
            )

        # Step 1: Load data
        self.load_data()
        if self.feature_extraction_batch_size is not None:
            if int(self.feature_extraction_batch_size) != batch_size:
                raise ValueError(
                    "CFW coupled-batch policy violation: "
                    f"feature_extraction batch_size={self.feature_extraction_batch_size}, "
                    f"training/clustering batch_size={batch_size}. "
                    "Re-run feature extraction with matching dataloader.batch_size."
                )
            logger.info(
                "Verified coupled-batch policy from metadata: "
                "feature_extraction_batch_size=%d, training_batch_size=%d, "
                "clustering_batch_size=%d",
                self.feature_extraction_batch_size,
                batch_size,
                clustering_batch_size,
            )
        else:
            raise ValueError(
                "CFW coupled-batch policy cannot be verified because "
                "feature_extraction_metadata.json is missing next to the feature files. "
                "Re-run scripts/extract_features.py with the intended dataloader.batch_size."
            )

        # Step 2: Load precomputed weights (if provided) or run on-the-fly clustering
        used_precomputed_weights = self._load_precomputed_weights_table()
        if not used_precomputed_weights:
            self._clustering_batch_size = clustering_batch_size
            n_samples = len(self.features)
            n_batches = (n_samples + clustering_batch_size - 1) // clustering_batch_size

            logger.info(
                "CFW batch settings: training_batch_size=%d, clustering_batch_size=%d",
                batch_size,
                clustering_batch_size,
            )
            logger.info(
                "CFW coupled-batch policy active: "
                "feature extraction, clustering, and training use the same batch size."
            )
            logger.info("Generating CFW weights on-the-fly from features (no precomputed table provided).")
            logger.info(
                "Clustering config: min_cluster_size=%s, min_samples=%s, metric=%s, "
                "selection_method=%s, epsilon=%s, allow_single_cluster=%s",
                self.clustering_config.min_cluster_size,
                self.clustering_config.min_samples,
                self.clustering_config.metric,
                self.clustering_config.cluster_selection_method,
                self.clustering_config.cluster_selection_epsilon,
                self.clustering_config.allow_single_cluster,
            )
            logger.info(
                "Weighting config: strategy=%s, outlier_weight=%s, max_outlier_cluster_size=%s",
                self.weighting_config.weighting_strategy,
                self.weighting_config.outlier_weight,
                self.weighting_config.max_outlier_cluster_size,
            )
            logger.info(
                "Processing features for clustering in chunks of %d samples...",
                clustering_batch_size,
            )

            all_weights_list = []
            all_cluster_labels_list = []
            all_stats = []

            for i in range(n_batches):
                start_idx = i * clustering_batch_size
                end_idx = min((i + 1) * clustering_batch_size, n_samples)

                batch_features = self.features[start_idx:end_idx]

                # Process batch
                weights, cluster_labels, stats = self._process_batch(batch_features)

                all_weights_list.append(weights)
                all_cluster_labels_list.append(cluster_labels)
                all_stats.append(stats)

                logger.info(
                    f"Batch {i+1}/{n_batches}: "
                    f"{stats['clustering']['n_clusters']} clusters, "
                    f"{stats['clustering']['n_outliers']} outliers"
                )

            # Concatenate all batches
            self.all_weights = np.concatenate(all_weights_list)
            self.all_cluster_labels = np.concatenate(all_cluster_labels_list)
            self.clustering_stats = all_stats
            logger.info(f"Clustering complete: {len(self.all_weights)} samples processed")
            self._persist_weights_table(source="generated_on_the_fly")
        else:
            logger.info("Skipping clustering because precomputed CFW weights table is being used.")
            self._persist_weights_table(source="precomputed_and_aligned")

        logger.info(
            "CFW bundle in use: samples=%d, labels=%d, image_paths=%d, weights=%d",
            len(self.features),
            len(self.labels),
            len(self.img_paths),
            len(self.all_weights),
        )
        logger.info(
            "CFW sample preview[0]: path=%s label=%s weight=%.6f",
            self.img_paths[0] if len(self.img_paths) > 0 else "n/a",
            int(self.labels[0]) if len(self.labels) > 0 else -1,
            float(self.all_weights[0]) if len(self.all_weights) > 0 else -1.0,
        )

        # Step 3: Create dataset
        dataset = self.create_dataset(transform=transform)

        # Step 4: Create sampler and DataLoader
        if shuffle:
            if distributed:
                from ..data.samplers import create_distributed_weighted_sampler
                sampler = create_distributed_weighted_sampler(
                    weights=self.all_weights.tolist(),
                    replacement=True,
                    seed=distributed_seed,
                    drop_last=drop_last,
                )
                logger.info("Using DistributedWeightedSampler for CFW + DDP")
            else:
                sampler = WeightedRandomSampler(
                    weights=torch.from_numpy(self.all_weights).double(),
                    num_samples=len(self.all_weights),
                    replacement=True,
                )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            )
        else:
            # Sequential iteration without shuffling
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            )

        # Store stats for later retrieval
        self._last_stats = self._aggregate_stats(clustering_batch_size, batch_size)

        logger.info(
            f"CFW DataLoader built successfully: "
            f"{self._last_stats['total_samples']} samples, "
            f"{self._last_stats['overall']['n_unique_clusters']} clusters"
        )

        return dataloader

    def build_dataloader(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        transform: Optional[Any] = None,
        clustering_batch_size: int = 1024,
        pin_memory: bool = True,
        drop_last: bool = False,
        device: str = 'cpu',
        distributed: bool = False,
        distributed_seed: int = 0,
    ) -> DataLoader:
        """Build the CFW dataloader (alias for build()).

        This method is provided for API compatibility with dataloaders.py.

        Args:
            batch_size: Batch size for training dataloader
            num_workers: Number of worker threads for dataloader
            transform: Optional transform to apply to images
            clustering_batch_size: Batch size for processing features during clustering
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
            device: Device for computation (currently unused, kept for API compatibility)
            distributed: If True, use DistributedWeightedSampler for DDP training
            distributed_seed: Base random seed for DistributedWeightedSampler

        Returns:
            PyTorch DataLoader with weighted sampling
        """
        return self.build(
            batch_size=batch_size,
            num_workers=num_workers,
            transform=transform,
            clustering_batch_size=clustering_batch_size,
            pin_memory=pin_memory,
            drop_last=drop_last,
            shuffle=True,  # CFW always uses weighted random sampling
            distributed=distributed,
            distributed_seed=distributed_seed,
        )

    def get_stats(self) -> dict:
        """Get statistics from the last build() call.

        Returns:
            Dictionary with aggregated statistics, or None if build() hasn't been called
        """
        return getattr(self, '_last_stats', None)

    def batch_process_features(self, clustering_batch_size: int = 1024) -> None:
        """Process all features in batches for memory efficiency.

        Divides features into batches, clusters each batch separately,
        and aggregates results. This is called internally by build().

        Args:
            clustering_batch_size: Batch size for processing features
        """
        logger.info(
            "Processing features for clustering in chunks of %d samples...",
            clustering_batch_size,
        )

        n_samples = len(self.features)
        n_batches = (n_samples + clustering_batch_size - 1) // clustering_batch_size

        all_weights_list = []
        all_cluster_labels_list = []
        all_stats = []

        for i in range(n_batches):
            start_idx = i * clustering_batch_size
            end_idx = min((i + 1) * clustering_batch_size, n_samples)

            batch_features = self.features[start_idx:end_idx]

            # Process batch
            weights, cluster_labels, stats = self._process_batch(batch_features)

            all_weights_list.append(weights)
            all_cluster_labels_list.append(cluster_labels)
            all_stats.append(stats)

            logger.info(
                f"Batch {i+1}/{n_batches}: "
                f"{stats['clustering']['n_clusters']} clusters, "
                f"{stats['clustering']['n_outliers']} outliers"
            )

        # Concatenate all batches
        self.all_weights = np.concatenate(all_weights_list)
        self.all_cluster_labels = np.concatenate(all_cluster_labels_list)
        self.clustering_stats = all_stats

        logger.info(
            f"Clustering complete: {len(self.all_weights)} samples processed"
        )


def create_cfw_dataloader(
    feature_file_path: str,
    label_file_path: str,
    img_path_file_path: str,
    data_dir: str = None,
    clustering_config: ClusteringConfig = None,
    weighting_config: WeightingConfig = None,
    weights_table_file_path: Optional[str] = None,
    save_generated_weights_file_path: Optional[str] = None,
    batch_size: int = 64,
    clustering_batch_size: int = 1024,
    transform: Optional[Any] = None,
    num_workers: int = 4,
    prefetch_factor: Optional[int] = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> Tuple[DataLoader, dict]:
    """
    Create a CFW dataloader.

    Args:
        feature_file_path: Path to features pickle file
        label_file_path: Path to labels pickle file
        img_path_file_path: Path to image paths pickle file
        data_dir: Base directory for resolving relative image paths
        clustering_config: Clustering configuration (uses defaults if None)
        weighting_config: Weighting configuration (uses defaults if None)
        weights_table_file_path: Optional precomputed weights table (.pkl) path
        save_generated_weights_file_path: Optional output .pkl path to persist weights
        batch_size: Training batch size
        clustering_batch_size: Batch size for clustering
        transform: Image transforms
        num_workers: DataLoader workers
        prefetch_factor: Prefetch factor
        pin_memory: Pin memory for GPU
        drop_last: Whether to drop last incomplete batch

    Returns:
        Tuple of (dataloader, statistics)
    """
    builder = CFWDataLoaderBuilder(
        feature_file_path=feature_file_path,
        label_file_path=label_file_path,
        img_path_file_path=img_path_file_path,
        data_dir=data_dir,
        clustering_config=clustering_config,
        weighting_config=weighting_config,
        weights_table_file_path=weights_table_file_path,
        save_generated_weights_file_path=save_generated_weights_file_path,
    )

    dataloader = builder.build(
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform,
        clustering_batch_size=clustering_batch_size,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
    )

    return dataloader, builder.get_stats()
