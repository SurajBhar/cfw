#!/usr/bin/env python3
"""
Feature extraction script for CFW (Clustered Feature Weighting).

This script extracts features from a pre-trained model (DINOv2 or ViT) and saves them
along with labels and image paths. These features are required for CFW dataloader
which performs clustering and weighted sampling.

Outputs:
    - features.pkl: Feature embeddings (numpy array)
    - labels.pkl: Class labels (numpy array)
    - image_paths.pkl: Image file paths (list)

Usage:
    # Extract features for all splits
    python scripts/extract_features.py model=dinov2_vitb14 dataset=driveact_binary

    # Extract features for specific split
    python scripts/extract_features.py model=dinov2_vitb14 dataset=driveact_binary split=train

    # With custom output directory
    python scripts/extract_features.py model=dinov2_vitb14 dataset=driveact_binary output_dir=/path/to/output

    # Force randomized stage-1 sampling order
    python scripts/extract_features.py model=dinov2_vitb14 dataset=driveact_binary \
        feature_extraction.shuffle=true
"""


import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import pickle
import json

import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add src to path for imports
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from cfw.utils.logging import setup_logger
from cfw.utils.reproducibility import set_seeds
from cfw.utils.config_utils import validate_config
from cfw.data.dataloaders import create_dataloader
from cfw.models.builders import build_classifier_from_config
from cfw.evaluation.feature_analysis import (
    compute_metrics_only_analysis,
)


def _resolve_azureml_output_root() -> Optional[Path]:
    """Resolve Azure ML output mount path when present."""
    def _is_resolved_path(value: str) -> bool:
        stripped = value.strip()
        if not stripped:
            return False
        if "${{" in stripped or "${" in stripped:
            return False
        return True

    for key in ("AZUREML_OUTPUTS_DIR", "AZUREML_RUN_OUTPUT_PATH", "OUTPUTS_DIR"):
        value = os.environ.get(key)
        if value and _is_resolved_path(value):
            return Path(value)

    for key, value in os.environ.items():
        if key.startswith("AZUREML_OUTPUT_") and value and _is_resolved_path(value):
            return Path(value)

    return None


def _resolve_device(cfg: DictConfig) -> torch.device:
    """Resolve device from trainer.device with legacy trainer.gpu_id fallback."""
    device_value = str(cfg.trainer.get("device", cfg.trainer.get("gpu_id", "0"))).lower()
    if device_value == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    if device_value.startswith("cuda:"):
        return torch.device(device_value)

    return torch.device(f"cuda:{device_value}")


def _resolve_dataset_image_paths(dataloader: torch.utils.data.DataLoader) -> Optional[List[str]]:
    """
    Resolve deterministic image path order from dataset metadata.

    This is used when batches do not include paths (baseline dataloader).
    """
    dataset = dataloader.dataset

    if hasattr(dataset, "paths"):
        return [str(p) for p in dataset.paths]

    if hasattr(dataset, "samples"):
        return [str(sample[0]) for sample in dataset.samples]

    if hasattr(dataset, "image_paths"):
        return [str(p) for p in dataset.image_paths]

    return None


def extract_features_from_dataloader(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    logger,
    fallback_paths: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features from a dataloader using a pre-trained model.

    Args:
        model: Pre-trained model (backbone only, no classifier head)
        dataloader: DataLoader to extract features from
        device: Device to run model on
        logger: Logger instance

    Returns:
        Tuple of (features, labels, image_paths)
    """
    model.eval()
    all_features = []
    all_labels = []
    all_paths = []
    path_cursor = 0

    logger.info(f"Extracting features from {len(dataloader)} batches...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Handle baseline and CFW batch formats:
            # - baseline without paths: (images, labels)
            # - baseline with paths: (images, labels, paths)
            # - CFW: (images, weights, labels, paths)
            if len(batch) == 2:
                images, labels = batch
                paths = None
            elif len(batch) == 3:
                images, labels, paths = batch
            elif len(batch) == 4:
                images, _, labels, paths = batch
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            # Move to device
            images = images.to(device)

            # Extract features (use backbone only, not classifier head)
            if hasattr(model, 'backbone'):
                features = model.backbone(images)
            elif hasattr(model, 'model') and hasattr(model.model, 'backbone'):
                features = model.model.backbone(images)
            else:
                # Assume the model itself is the backbone
                features = model(images)

            # Flatten features if needed (e.g., from [B, C, 1, 1] to [B, C])
            if features.dim() > 2:
                features = features.view(features.size(0), -1)

            # Store results
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
            batch_size = int(features.size(0))

            if paths is not None:
                batch_paths = [str(p) for p in paths]
                all_paths.extend(batch_paths)
                path_cursor += len(batch_paths)
                continue

            # Baseline dataloaders do not include paths in the batch.
            if fallback_paths is not None:
                batch_paths = fallback_paths[path_cursor:path_cursor + batch_size]
                if len(batch_paths) != batch_size:
                    raise ValueError(
                        "Could not align fallback image paths with extracted batches. "
                        f"cursor={path_cursor}, batch_size={batch_size}, "
                        f"available={len(fallback_paths)}"
                    )
                all_paths.extend(batch_paths)
                path_cursor += batch_size
            else:
                # Keep output shape-valid even if dataset metadata is unavailable.
                batch_paths = [
                    f"sample_{idx:08d}"
                    for idx in range(path_cursor, path_cursor + batch_size)
                ]
                all_paths.extend(batch_paths)
                path_cursor += batch_size

    # Concatenate all batches
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if len(all_paths) != len(labels):
        raise ValueError(
            f"Mismatch between labels ({len(labels)}) and image paths ({len(all_paths)})"
        )

    logger.info(f"Extracted features shape: {features.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Number of image paths: {len(all_paths) if all_paths else 'N/A'}")

    return features, labels, all_paths


def save_features(
    features: np.ndarray,
    labels: np.ndarray,
    image_paths: List[str],
    output_dir: Path,
    split: str,
    logger
) -> None:
    """
    Save extracted features, labels, and image paths to pickle files.

    Args:
        features: Feature embeddings
        labels: Class labels
        image_paths: Image file paths
        output_dir: Output directory
        split: Data split name (train/val/test)
        logger: Logger instance
    """
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    features_path = split_dir / "features.pkl"
    with open(features_path, "wb") as f:
        pickle.dump(features, f)
    logger.info(f"Saved features to {features_path}")

    # Save labels
    labels_path = split_dir / "labels.pkl"
    with open(labels_path, "wb") as f:
        pickle.dump(labels, f)
    logger.info(f"Saved labels to {labels_path}")

    # Save image paths (always required for CFW dataloader)
    paths_path = split_dir / "image_paths.pkl"
    with open(paths_path, "wb") as f:
        pickle.dump(image_paths, f)
    logger.info(f"Saved image paths to {paths_path}")


def save_feature_extraction_metadata(
    output_dir: Path,
    split: str,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_samples: int,
    logger,
) -> None:
    """Save feature extraction settings used to produce per-split artifacts."""
    split_dir = output_dir / split
    metadata_path = split_dir / "feature_extraction_metadata.json"
    metadata = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "drop_last": bool(drop_last),
        "num_samples": int(num_samples),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved feature extraction metadata to {metadata_path}")


def get_backbone_only(model: torch.nn.Module) -> torch.nn.Module:
    """
    Extract backbone from a model (remove classifier head).

    Args:
        model: Full model (backbone + classifier head)

    Returns:
        Backbone only
    """
    if hasattr(model, 'backbone'):
        return model.backbone
    elif hasattr(model, 'model') and hasattr(model.model, 'backbone'):
        return model.model.backbone
    else:
        # Assume the model is already a backbone
        return model


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Run feature extraction for configured dataset splits.

    Args:
        cfg: Hydra configuration object
    """
    print("=" * 80)
    print("CFW Feature Extraction Script")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Determine which splits to process
    if cfg.get("split"):
        splits = [str(cfg.split).lower()]
        validate_config(
            cfg,
            require_all_splits=False,
            required_splits=splits,
        )
    else:
        splits = ["train", "val", "test"]
        validate_config(cfg)

    # Set random seeds for reproducibility
    set_seeds(cfg.experiment.seed)

    # Get output directory
    if cfg.get("output_dir"):
        output_dir = Path(cfg.output_dir)
    elif _resolve_azureml_output_root() is not None:
        output_dir = _resolve_azureml_output_root() / cfg.experiment.name / "feature_extraction"
    else:
        # Use Hydra output directory
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Create features directory
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="cfw_feature_extraction",
        log_file=str(log_dir / "feature_extraction.log"),
        level=cfg.get("log_level", "INFO")
    )

    logger.info("Starting feature extraction")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Dataset: {cfg.dataset.name}")
    logger.info(f"Output directory: {features_dir}")

    # Feature extraction dataloader behavior (Stage-1 loading for CFW pipeline)
    feature_extraction_cfg = cfg.get("feature_extraction", {})
    feature_loader_shuffle = bool(
        feature_extraction_cfg.get(
            "shuffle",
            cfg.dataloader.get("shuffle", True),
        )
    )
    feature_loader_drop_last = bool(
        feature_extraction_cfg.get(
            "drop_last",
            False,
        )
    )
    logger.info(
        "Feature extraction loader settings: "
        f"shuffle={feature_loader_shuffle}, drop_last={feature_loader_drop_last}"
    )

    # Setup device
    device = _resolve_device(cfg)
    logger.info(f"Using device: {device}")

    # Build model
    logger.info(f"Building model: {cfg.model.name}")
    model = build_classifier_from_config(cfg)

    # Extract backbone only (remove classifier head)
    backbone = get_backbone_only(model)
    backbone = backbone.to(device)

    num_params = sum(p.numel() for p in backbone.parameters())
    logger.info(f"Backbone parameters: {num_params:,}")

    logger.info(f"Processing splits: {splits}")

    # Process each split
    for split in splits:
        logger.info(f"\nProcessing {split} split...")

        # Create dataloader (use baseline dataloader for feature extraction)
        # Temporarily override dataloader config to use baseline
        original_dataloader_type = cfg.dataloader.type
        cfg.dataloader.type = "baseline"

        dataloader = create_dataloader(
            cfg=cfg,
            split=split,
            shuffle=feature_loader_shuffle,
            drop_last=feature_loader_drop_last,
            return_paths=True,
        )

        # Restore original dataloader type
        cfg.dataloader.type = original_dataloader_type

        logger.info(f"{split.capitalize()} batches: {len(dataloader)}")

        fallback_paths = _resolve_dataset_image_paths(dataloader)
        if fallback_paths is None:
            logger.warning(
                "Dataset does not expose image path metadata; using generated "
                "placeholder paths for image_paths.pkl."
            )
        else:
            logger.info(f"Resolved {len(fallback_paths)} dataset image paths for {split} split")

        # Extract features
        features, labels, image_paths = extract_features_from_dataloader(
            model=backbone,
            dataloader=dataloader,
            device=device,
            logger=logger,
            fallback_paths=fallback_paths,
        )

        # Save features
        save_features(
            features=features,
            labels=labels,
            image_paths=image_paths,
            output_dir=features_dir,
            split=split,
            logger=logger
        )
        save_feature_extraction_metadata(
            output_dir=features_dir,
            split=split,
            batch_size=int(cfg.dataloader.batch_size),
            shuffle=feature_loader_shuffle,
            drop_last=feature_loader_drop_last,
            num_samples=int(len(labels)),
            logger=logger,
        )

        # Compute notebook-aligned metrics-only feature analysis.
        logger.info(f"Computing metrics-only feature analysis for {split} split...")
        try:
            analysis_cfg = cfg.get("feature_analysis", {})
            large_scale_mode = bool(analysis_cfg.get("large_scale_mode", False))
            enable_knn = analysis_cfg.get("enable_knn", None)
            enable_silhouette = analysis_cfg.get("enable_silhouette", None)
            knn_k = int(analysis_cfg.get("knn_k", 10))
            silhouette_max_samples = int(analysis_cfg.get("silhouette_max_samples", 5000))
            pair_samples = int(analysis_cfg.get("pair_samples", 30000))
            normalize = str(analysis_cfg.get("normalize", "none"))
            knn_metric = str(analysis_cfg.get("knn_metric", "euclidean"))
            nearest_centroid_mode = str(
                analysis_cfg.get("nearest_centroid_mode", "holdout")
            )
            nearest_centroid_metric = str(
                analysis_cfg.get("nearest_centroid_metric", "euclidean")
            )
            nearest_centroid_test_size = float(
                analysis_cfg.get("nearest_centroid_test_size", 0.2)
            )

            metrics = compute_metrics_only_analysis(
                features=features,
                labels=labels,
                seed=int(cfg.experiment.seed),
                large_scale_mode=large_scale_mode,
                enable_knn=enable_knn,
                enable_silhouette=enable_silhouette,
                knn_k=knn_k,
                silhouette_max_samples=silhouette_max_samples,
                pair_samples=pair_samples,
                normalize=normalize,
                knn_metric=knn_metric,
                nearest_centroid_mode=nearest_centroid_mode,
                nearest_centroid_metric=nearest_centroid_metric,
                nearest_centroid_test_size=nearest_centroid_test_size,
            )

            def _fmt_metric(value) -> str:
                if value is None:
                    return "n/a"
                return f"{float(value):.4f}"

            # Log key metrics
            logger.info("Feature Analysis Metrics (notebook-aligned):")
            logger.info(
                f"  DBI (sklearn): {_fmt_metric(metrics.get('davies_bouldin_index_sklearn'))}"
            )
            logger.info(
                f"  DBI (custom): {_fmt_metric(metrics.get('davies_bouldin_index_custom'))}"
            )
            logger.info(
                f"  Intra macro MSD: {_fmt_metric(metrics.get('intra_macro_msd'))}"
            )
            logger.info(
                f"  Inter weighted scatter: {_fmt_metric(metrics.get('inter_weighted'))}"
            )
            logger.info(
                f"  Mean center distance: {_fmt_metric(metrics.get('mean_center_distance'))}"
            )
            logger.info(
                f"  Fisher ratio: {_fmt_metric(metrics.get('fisher_ratio'))}"
            )
            logger.info(
                (
                    "  Analysis mode: large_scale=%s, dtype=%s, normalize=%s, "
                    "nearest_centroid=%s(%s), knn=%s(%s), silhouette=%s"
                ),
                metrics.get("analysis_config", {}).get("large_scale_mode"),
                metrics.get("analysis_config", {}).get("compute_dtype"),
                metrics.get("analysis_config", {}).get("normalize"),
                metrics.get("analysis_config", {}).get("nearest_centroid_mode"),
                metrics.get("analysis_config", {}).get("nearest_centroid_metric"),
                metrics.get("analysis_config", {}).get("enable_knn"),
                metrics.get("analysis_config", {}).get("knn_metric"),
                metrics.get("analysis_config", {}).get("enable_silhouette"),
            )
            sanity = metrics.get("sanity_checks", {})
            logger.info(
                "  Sanity checks: nearest_centroid=%s, knn=%s, silhouette=%s",
                _fmt_metric(sanity.get("nearest_centroid_accuracy")),
                _fmt_metric(next(
                    (v for k, v in sanity.items() if k.startswith("knn_accuracy_k")),
                    None,
                )),
                _fmt_metric(sanity.get("silhouette")),
            )

            # Save metrics analysis JSON
            split_dir = features_dir / split
            metrics_path = split_dir / "feature_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved metrics-only analysis to {metrics_path}")

        except Exception as e:
            logger.error(f"Failed to compute feature metrics: {e}")
            logger.warning("Continuing without feature quality analysis")

        logger.info(f"Completed {split} split")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Feature Extraction Complete!")
    logger.info("=" * 80)
    logger.info(f"Features saved to: {features_dir}")
    logger.info("\nExtracted files:")
    for split in splits:
        split_dir = features_dir / split
        logger.info(f"\n{split.upper()}:")
        logger.info(f"  - {split_dir / 'features.pkl'}")
        logger.info(f"  - {split_dir / 'labels.pkl'}")
        logger.info(f"  - {split_dir / 'image_paths.pkl'}")
        logger.info(f"  - {split_dir / 'feature_extraction_metadata.json'}")
        if (split_dir / "feature_metrics.json").exists():
            logger.info(f"  - {split_dir / 'feature_metrics.json'}")

    logger.info("\n" + "=" * 80)
    logger.info("You can now use these features for CFW dataloader")
    logger.info("Update your config with:")
    logger.info(f"  dataloader.cfw.feature_dir: {features_dir}")
    logger.info("=" * 80)

    print("\n" + "=" * 80)
    print("Feature Extraction Complete!")
    print("=" * 80)
    print(f"\nFeatures saved to: {features_dir}")
    print(f"\nTo use these features, update your config with:")
    print(f"  dataloader.cfw.feature_dir: {features_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
