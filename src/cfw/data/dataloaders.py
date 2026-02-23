"""
Dataloader factory for CFW training.

This module provides factory functions to create dataloaders for:
- Baseline approach: Standard random sampling
- CFW approach: Clustered Feature Weighting with weighted sampling
"""

import os
from typing import Optional, Tuple, Any, Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig

from .datasets import ImageFolderCustom, BinaryClassificationDataset, WeightedImageDataset
from .transforms import (
    make_classification_train_transform,
    make_classification_eval_transform,
    make_no_aug_transform,
)
from .samplers import create_weighted_sampler
from ..core.cfw_dataloader import CFWDataLoaderBuilder
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _as_list_or_default(value: Optional[Sequence[float]], default: Sequence[float]) -> Sequence[float]:
    """Convert config sequences to plain lists with fallback defaults."""
    if value is None:
        return list(default)
    return list(value)


def _is_resolved_path(value: Optional[str]) -> bool:
    """Return True when a value is a concrete path and not a template token."""
    if value is None:
        return False
    stripped = str(value).strip()
    if not stripped:
        return False
    if "${{" in stripped or "${" in stripped:
        return False
    return True


def _resolve_runtime_artifacts_root(cfg: DictConfig) -> Optional[Path]:
    """Resolve a writable artifacts root for the current run."""
    env_keys = ("AZUREML_OUTPUTS_DIR", "AZUREML_RUN_OUTPUT_PATH", "OUTPUTS_DIR")
    azure_root: Optional[Path] = None
    for key in env_keys:
        value = os.environ.get(key)
        if _is_resolved_path(value):
            azure_root = Path(str(value))
            break

    if azure_root is None:
        for key, value in os.environ.items():
            if key.startswith("AZUREML_OUTPUT_") and _is_resolved_path(value):
                azure_root = Path(str(value))
                break

    if azure_root is not None:
        experiment_name = str(cfg.experiment.get("name", "cfw_experiment"))
        return azure_root / experiment_name

    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            runtime_dir = HydraConfig.get().runtime.output_dir
            if _is_resolved_path(runtime_dir):
                return Path(str(runtime_dir))
    except Exception:
        return None

    return None


def _default_weights_save_path(cfg: DictConfig, split: str) -> Optional[str]:
    """Build a default path for persisting generated CFW weights tables."""
    artifacts_root = _resolve_runtime_artifacts_root(cfg)
    if artifacts_root is None:
        return None
    return str(artifacts_root / "artifacts" / "cfw_weights" / f"{split}_weights_table.pkl")


def _build_transform_from_config(cfg: DictConfig, split: str):
    """
    Build transforms from either nested (train/eval) or flat augmentation config.

    Supports both:
    - New schema: augmentation.train.*, augmentation.eval.*
    - Legacy schema: augmentation.enabled, augmentation.crop_size, ...
    """
    is_train = (split == "train")
    aug_cfg = cfg.augmentation
    dataset_cfg = cfg.dataset

    default_mean = _as_list_or_default(dataset_cfg.get("mean", None), [0.485, 0.456, 0.406])
    default_std = _as_list_or_default(dataset_cfg.get("std", None), [0.229, 0.224, 0.225])
    dataset_image_size = int(dataset_cfg.get("image_size", 224))

    # Nested schema
    if "train" in aug_cfg or "eval" in aug_cfg:
        train_cfg = aug_cfg.get("train", {})
        eval_cfg = aug_cfg.get("eval", {})

        if is_train:
            rrc_cfg = train_cfg.get("random_resized_crop", {})
            if isinstance(rrc_cfg, bool):
                rrc_enabled = bool(rrc_cfg)
                crop_size = dataset_image_size
            else:
                rrc_enabled = bool(rrc_cfg.get("enabled", True))
                crop_size = int(rrc_cfg.get("size", dataset_image_size))

            hflip_cfg = train_cfg.get("horizontal_flip", {})
            if isinstance(hflip_cfg, bool):
                hflip_prob = 0.5 if hflip_cfg else 0.0
            else:
                hflip_prob = float(hflip_cfg.get("p", 0.5 if hflip_cfg.get("enabled", False) else 0.0))
                if not hflip_cfg.get("enabled", False):
                    hflip_prob = 0.0

            vflip_cfg = train_cfg.get("vertical_flip", {})
            if isinstance(vflip_cfg, bool):
                vflip_prob = 0.5 if vflip_cfg else 0.0
            else:
                vflip_prob = float(vflip_cfg.get("p", 0.0))
                if not vflip_cfg.get("enabled", False):
                    vflip_prob = 0.0

            cj_cfg = train_cfg.get("color_jitter", {})
            if isinstance(cj_cfg, bool):
                color_jitter_enabled = bool(cj_cfg)
                jitter_brightness = 0.4
                jitter_contrast = 0.4
                jitter_saturation = 0.4
                jitter_hue = 0.1
                color_jitter_p = 1.0 if color_jitter_enabled else 0.0
            else:
                color_jitter_enabled = bool(cj_cfg.get("enabled", False))
                color_jitter_p = float(cj_cfg.get("p", 1.0))
                jitter_brightness = float(cj_cfg.get("brightness", 0.4))
                jitter_contrast = float(cj_cfg.get("contrast", 0.4))
                jitter_saturation = float(cj_cfg.get("saturation", 0.4))
                jitter_hue = float(cj_cfg.get("hue", 0.1))

            gb_cfg = train_cfg.get("gaussian_blur", {})
            if isinstance(gb_cfg, bool):
                gaussian_blur_enabled = bool(gb_cfg)
                blur_prob = 0.5 if gaussian_blur_enabled else 0.0
            else:
                gaussian_blur_enabled = bool(gb_cfg.get("enabled", False))
                blur_prob = float(gb_cfg.get("p", 0.5))
                if not gaussian_blur_enabled:
                    blur_prob = 0.0

            normalize_cfg = train_cfg.get("normalize", {})
            mean = _as_list_or_default(normalize_cfg.get("mean", None), default_mean)
            std = _as_list_or_default(normalize_cfg.get("std", None), default_std)

            resize_size = train_cfg.get("resize", None)

            has_augmentation = (
                rrc_enabled
                or hflip_prob > 0.0
                or vflip_prob > 0.0
                or (color_jitter_enabled and color_jitter_p > 0.0)
                or (gaussian_blur_enabled and blur_prob > 0.0)
            )

            if has_augmentation:
                # Note: the transform factory supports color-jitter as on/off, not
                # stochastic probability. If p>0, we enable it.
                return make_classification_train_transform(
                    resize_size=resize_size,
                    crop_size=crop_size,
                    hflip_prob=hflip_prob,
                    vflip_prob=vflip_prob,
                    color_jitter=(color_jitter_enabled and color_jitter_p > 0.0),
                    jitter_brightness=jitter_brightness,
                    jitter_contrast=jitter_contrast,
                    jitter_saturation=jitter_saturation,
                    jitter_hue=jitter_hue,
                    gaussian_blur=(gaussian_blur_enabled and blur_prob > 0.0),
                    blur_prob=blur_prob,
                    mean=mean,
                    std=std,
                )

            no_aug_resize = int(train_cfg.get("resize", dataset_image_size))
            return make_no_aug_transform(
                resize_size=no_aug_resize,
                mean=mean,
                std=std,
            )

        normalize_cfg = eval_cfg.get("normalize", {})
        mean = _as_list_or_default(normalize_cfg.get("mean", None), default_mean)
        std = _as_list_or_default(normalize_cfg.get("std", None), default_std)
        resize_size = int(eval_cfg.get("resize", aug_cfg.get("resize_size", 256)))
        crop_size = int(eval_cfg.get("center_crop", aug_cfg.get("crop_size", dataset_image_size)))
        return make_classification_eval_transform(
            resize_size=resize_size,
            crop_size=crop_size,
            mean=mean,
            std=std,
        )

    # Flat schema (legacy)
    if bool(aug_cfg.get("enabled", False)) and is_train:
        return make_classification_train_transform(
            crop_size=aug_cfg.get("crop_size", dataset_image_size),
            hflip_prob=aug_cfg.get("hflip_prob", 0.5),
            vflip_prob=aug_cfg.get("vflip_prob", 0.0),
            color_jitter=aug_cfg.get("color_jitter", False),
            gaussian_blur=aug_cfg.get("gaussian_blur", False),
            mean=_as_list_or_default(aug_cfg.get("mean", None), default_mean),
            std=_as_list_or_default(aug_cfg.get("std", None), default_std),
        )

    if bool(aug_cfg.get("no_aug", False)):
        return make_no_aug_transform(
            resize_size=aug_cfg.get("resize_size", dataset_image_size),
            mean=_as_list_or_default(aug_cfg.get("mean", None), default_mean),
            std=_as_list_or_default(aug_cfg.get("std", None), default_std),
        )

    return make_classification_eval_transform(
        resize_size=aug_cfg.get("resize_size", 256),
        crop_size=aug_cfg.get("crop_size", dataset_image_size),
        mean=_as_list_or_default(aug_cfg.get("mean", None), default_mean),
        std=_as_list_or_default(aug_cfg.get("std", None), default_std),
    )


def create_baseline_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    transform: Optional[Any] = None,
    is_binary: bool = False,
    non_distracted_classes: Optional[set] = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    return_paths: bool = False,
    distributed: bool = False,
    distributed_seed: int = 0,
) -> DataLoader:
    """
    Create baseline dataloader with standard random sampling.

    This is "Dataloader A" from the paper - standard PyTorch dataloader
    without CFW weighting.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        transform: Transform to apply (if None, uses default eval transform)
        is_binary: If True, use BinaryClassificationDataset
        non_distracted_classes: Set of classes to map to "non_distracted" (only for binary)
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch
        return_paths: Whether baseline dataset should also return image paths
        distributed: If True, use DistributedSampler to shard dataset across ranks
        distributed_seed: Base seed for DistributedSampler shuffling

    Returns:
        DataLoader with standard random sampling

    Example:
        >>> train_loader = create_baseline_dataloader(
        ...     data_dir="/path/to/train",
        ...     batch_size=32,
        ...     is_binary=True,
        ...     shuffle=True
        ... )
    """
    # Use default transform if none provided
    if transform is None:
        transform = make_classification_eval_transform()

    # Create dataset
    if is_binary:
        dataset = BinaryClassificationDataset(
            root_dir=data_dir,
            non_distracted_classes=non_distracted_classes,
            transform=transform,
            return_paths=return_paths,
        )
        logger.info(
            f"Created binary dataset with {len(dataset)} samples, "
            f"distribution: {dataset.count_samples_per_class()}"
        )
    else:
        dataset = ImageFolderCustom(
            root_dir=data_dir,
            transform=transform,
            return_paths=return_paths,
        )
        logger.info(
            f"Created dataset with {len(dataset)} samples, "
            f"{len(dataset.classes)} classes"
        )

    # In DDP training, shard baseline train data across ranks.
    sampler = None
    dataloader_shuffle = shuffle
    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=int(distributed_seed),
        )
        dataloader_shuffle = False
        logger.info(
            "Using DistributedSampler for baseline dataloader "
            f"(dataset_size={len(dataset)}, drop_last={drop_last}, shuffle={shuffle})"
        )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader


def create_cfw_dataloader(
    feature_file_path: str,
    label_file_path: str,
    img_path_file_path: str,
    batch_size: int,
    data_dir: Optional[str] = None,
    num_workers: int = 4,
    transform: Optional[Any] = None,
    clustering_config: Optional[DictConfig] = None,
    weighting_config: Optional[DictConfig] = None,
    clustering_batch_size: int = 1024,
    weights_table_file_path: Optional[str] = None,
    save_generated_weights_file_path: Optional[str] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    device: str = 'cpu',
    distributed: bool = False,
    distributed_seed: int = 0,
) -> DataLoader:
    """
    Create CFW dataloader with clustered feature weighting.

    This is "Dataloader B" from the paper - uses pre-extracted features,
    performs clustering, computes weights, and creates weighted sampler.

    Args:
        feature_file_path: Path to features pickle file
        label_file_path: Path to labels pickle file
        img_path_file_path: Path to image paths pickle file
        data_dir: Dataset split directory used to resolve relative image paths
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        transform: Transform to apply (if None, uses default eval transform)
        clustering_config: Configuration for HDBSCAN clustering
        weighting_config: Configuration for weight computation
        clustering_batch_size: Batch size for processing features during clustering
        weights_table_file_path: Optional precomputed CFW weights table (.pkl)
        save_generated_weights_file_path: Optional output path to persist generated/aligned weights
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch
        device: Device for clustering computation ('cpu' or 'cuda')
        distributed: If True, use DistributedWeightedSampler for DDP training
        distributed_seed: Base random seed for DistributedWeightedSampler

    Returns:
        DataLoader with CFW weighted sampling

    Example:
        >>> cfw_loader = create_cfw_dataloader(
        ...     feature_file_path="/path/to/features.pkl",
        ...     label_file_path="/path/to/labels.pkl",
        ...     img_path_file_path="/path/to/paths.pkl",
        ...     batch_size=32,
        ...     clustering_batch_size=1024
        ... )
    """
    # Use default transform if none provided
    if transform is None:
        transform = make_classification_eval_transform()

    logger.info("Building CFW dataloader...")
    logger.info("CFW feature_file=%s", feature_file_path)
    logger.info("CFW label_file=%s", label_file_path)
    logger.info("CFW img_path_file=%s", img_path_file_path)
    logger.info("CFW data_dir=%s", data_dir if data_dir is not None else "None")
    if weights_table_file_path:
        logger.info("CFW weights_table_file=%s (precomputed weights mode)", weights_table_file_path)
    else:
        logger.info("CFW weights_table_file not provided; weights will be generated on-the-fly.")
    if save_generated_weights_file_path:
        logger.info("CFW save_generated_weights_file=%s", save_generated_weights_file_path)
    else:
        logger.info("CFW save_generated_weights_file not provided; weights will remain in-memory.")

    # Import clustering and weighting configs
    from ..core.clustering import ClusteringConfig
    from ..core.weighting import WeightingConfig

    # Convert OmegaConf to config objects if needed
    if clustering_config is not None and not isinstance(clustering_config, ClusteringConfig):
        clustering_config = ClusteringConfig(
            min_cluster_size=clustering_config.get('min_cluster_size', 25),
            min_samples=clustering_config.get('min_samples', 1),
            cluster_selection_epsilon=clustering_config.get('cluster_selection_epsilon', 0.0),
            cluster_selection_method=clustering_config.get('cluster_selection_method', 'eom'),
            allow_single_cluster=clustering_config.get('allow_single_cluster', False),
            metric=clustering_config.get('metric', 'cosine'),
        )

    if weighting_config is not None and not isinstance(weighting_config, WeightingConfig):
        weighting_config = WeightingConfig(
            outlier_weight=weighting_config.get('outlier_weight', 0.001),
            max_outlier_cluster_size=weighting_config.get('max_outlier_cluster_size', 50),
        )

    # Create CFW dataloader builder
    cfw_builder = CFWDataLoaderBuilder(
        feature_file_path=feature_file_path,
        label_file_path=label_file_path,
        img_path_file_path=img_path_file_path,
        data_dir=data_dir,
        clustering_config=clustering_config,
        weighting_config=weighting_config,
        weights_table_file_path=weights_table_file_path,
        save_generated_weights_file_path=save_generated_weights_file_path,
    )

    # Build dataloader
    dataloader = cfw_builder.build(
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform,
        clustering_batch_size=clustering_batch_size,
        pin_memory=pin_memory,
        drop_last=drop_last,
        distributed=distributed,
        distributed_seed=distributed_seed,
    )

    logger.info(f"CFW dataloader created with {len(dataloader.dataset)} samples")

    return dataloader


def create_dataloader_from_config(
    cfg: DictConfig,
    split: str = 'train',
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    return_paths: bool = False,
) -> DataLoader:
    """
    Create dataloader from Hydra config.

    This is the main factory function that routes to baseline or CFW
    based on the config.

    Args:
        cfg: Hydra configuration
        split: Data split ('train', 'val', or 'test')
        shuffle: Override shuffle behavior (only for baseline dataloader).
                 If None, defaults to True for train split, False otherwise.
                 Ignored for CFW dataloader (always uses weighted sampling).
        drop_last: Override drop_last behavior.
                   If None, defaults to True for train split, False otherwise.
        return_paths: If True, baseline dataloader returns paths with each sample.
                     Ignored for CFW dataloader (already returns paths).

    Returns:
        DataLoader (baseline or CFW based on config)

    Example:
        >>> # In your training script with Hydra config
        >>> train_loader = create_dataloader_from_config(cfg, split='train')
        >>> val_loader = create_dataloader_from_config(cfg, split='val')
        >>> # With explicit shuffle and drop_last
        >>> test_loader = create_dataloader_from_config(
        ...     cfg, split='test', shuffle=False, drop_last=False
        ... )
    """
    # Get dataloader type from config
    dataloader_type = cfg.dataloader.type.lower()

    # Get split-specific config
    split_cfg = getattr(cfg.dataset, f'{split}_dir', None)
    if split_cfg is None:
        raise ValueError(f"Config missing '{split}_dir' for dataset")

    # Determine if training split (for defaults below)
    is_train = (split == 'train')

    # Determine defaults for shuffle and drop_last if not provided
    if shuffle is None:
        shuffle = is_train
    if drop_last is None:
        drop_last = bool(cfg.dataloader.get('drop_last', is_train))

    # Create transform (supports both nested and flat augmentation schemas)
    transform = _build_transform_from_config(cfg, split)

    # Baseline distributed routing is used for:
    # - baseline dataloader type (all splits)
    # - CFW train-only fallback val/test loaders.
    from ..utils.distributed import is_distributed as _is_distributed
    runtime_distributed = _is_distributed()
    baseline_cfg = cfg.dataloader.get('baseline', {})
    baseline_use_distributed_sampler = baseline_cfg.get('use_distributed_sampler', True)
    baseline_distributed_eval = baseline_cfg.get('distributed_eval', False)

    def _baseline_distributed_for_split(target_split: str) -> bool:
        if not runtime_distributed or not baseline_use_distributed_sampler:
            return False
        if target_split == 'train':
            return True
        return bool(baseline_distributed_eval)

    def _create_baseline_loader(distributed_for_split: bool) -> DataLoader:
        return create_baseline_dataloader(
            data_dir=split_cfg,
            batch_size=cfg.dataloader.batch_size,
            num_workers=cfg.dataloader.num_workers,
            transform=transform,
            is_binary=cfg.dataset.get('is_binary', False),
            non_distracted_classes=set(cfg.dataset.get('non_distracted_classes', [])),
            shuffle=shuffle,
            pin_memory=cfg.dataloader.get('pin_memory', True),
            drop_last=drop_last,
            return_paths=return_paths,
            distributed=distributed_for_split,
            distributed_seed=cfg.experiment.get('seed', 0),
        )

    # Route to appropriate factory
    if dataloader_type == 'baseline':
        logger.info(f"Creating baseline dataloader for {split} split")
        distributed_for_split = _baseline_distributed_for_split(split)

        if is_train and runtime_distributed and not baseline_use_distributed_sampler:
            logger.warning(
                "Distributed training detected for baseline dataloader, but "
                "dataloader.baseline.use_distributed_sampler is False. "
                "Each rank will iterate full dataset."
            )
        elif (
            split != 'train'
            and runtime_distributed
            and baseline_distributed_eval
            and not baseline_use_distributed_sampler
        ):
            logger.warning(
                "dataloader.baseline.distributed_eval is True, but "
                "dataloader.baseline.use_distributed_sampler is False. "
                "Evaluation will run without distributed sharding."
            )

        return _create_baseline_loader(distributed_for_split=distributed_for_split)

    elif dataloader_type == 'cfw':
        logger.info(f"Creating CFW dataloader for {split} split")
        cfw_cfg = cfg.dataloader.cfw
        cfw_train_only = cfw_cfg.get('cfw_train_only', cfw_cfg.get('train_only', True))

        if split != 'train' and cfw_train_only:
            logger.info(
                "CFW train-only mode enabled: using baseline dataloader for "
                f"{split} split."
            )
            distributed_for_split = _baseline_distributed_for_split(split)
            if (
                runtime_distributed
                and baseline_distributed_eval
                and not baseline_use_distributed_sampler
            ):
                logger.warning(
                    "CFW train-only baseline eval requested with distributed_eval=True, but "
                    "dataloader.baseline.use_distributed_sampler is False. "
                    "Evaluation will run without distributed sharding."
                )
            return _create_baseline_loader(distributed_for_split=distributed_for_split)

        # Warn if shuffle=False is explicitly requested for training
        # (CFW always uses weighted random sampling regardless)
        if not shuffle and is_train:
            logger.warning(
                "shuffle=False ignored for CFW dataloader. "
                "WeightedRandomSampler always uses random sampling."
            )

        # Get feature file paths for this split
        feature_path = cfw_cfg.get(f'{split}_feature_file', None)
        label_path = cfw_cfg.get(f'{split}_label_file', None)
        img_path = cfw_cfg.get(f'{split}_img_path_file', None)
        train_batch_size = int(cfg.dataloader.batch_size)
        configured_clustering_batch_size = int(
            cfw_cfg.get('clustering_batch_size', train_batch_size)
        )
        if configured_clustering_batch_size != train_batch_size:
            raise ValueError(
                "CFW coupled-batch policy violation: "
                f"dataloader.batch_size={train_batch_size} but "
                f"dataloader.cfw.clustering_batch_size={configured_clustering_batch_size}. "
                "Set both to the same value."
            )
        weights_table_path = cfw_cfg.get(
            f"{split}_weights_table_file",
            cfw_cfg.get("weights_table_file", None),
        )
        save_weights_path = cfw_cfg.get(
            f"{split}_save_generated_weights_file",
            cfw_cfg.get("save_generated_weights_file", None),
        )
        if not save_weights_path:
            auto_weights_path = _default_weights_save_path(cfg, split)
            if auto_weights_path is not None:
                save_weights_path = auto_weights_path
                logger.info(
                    "No explicit CFW weights save path configured; "
                    "auto-persisting to %s",
                    save_weights_path,
                )
            else:
                logger.warning(
                    "No explicit CFW weights save path configured and runtime artifacts root "
                    "could not be resolved; generated CFW weights will remain in memory."
                )

        if not all([feature_path, label_path, img_path]):
            raise ValueError(
                f"CFW dataloader requires feature, label, and img_path files "
                f"for {split} split. Check config, or set "
                "dataloader.cfw.cfw_train_only=true to use baseline val/test loaders."
            )

        # Detect distributed mode for CFW + DDP
        use_distributed_sampler = cfw_cfg.get('use_distributed_sampler', False)
        from ..utils.distributed import is_distributed as _is_distributed
        distributed = use_distributed_sampler and _is_distributed()

        if _is_distributed() and not use_distributed_sampler:
            logger.warning(
                "Distributed training detected but use_distributed_sampler is False. "
                "Set dataloader.cfw.use_distributed_sampler=true to enable CFW+DDP."
            )

        trainer_device = cfg.trainer.get('device', cfg.trainer.get('gpu_id', 'cpu'))
        trainer_device = str(trainer_device)
        cfw_device = 'cpu'
        if trainer_device != 'cpu' and torch.cuda.is_available():
            cfw_device = 'cuda'

        return create_cfw_dataloader(
            feature_file_path=feature_path,
            label_file_path=label_path,
            img_path_file_path=img_path,
            data_dir=split_cfg,
            batch_size=cfg.dataloader.batch_size,
            num_workers=cfg.dataloader.num_workers,
            transform=transform,
            clustering_config=cfw_cfg.get('clustering', None),
            weighting_config=cfw_cfg.get('weighting', None),
            clustering_batch_size=train_batch_size,
            weights_table_file_path=weights_table_path,
            save_generated_weights_file_path=save_weights_path,
            pin_memory=cfg.dataloader.get('pin_memory', True),
            drop_last=drop_last,
            device=cfw_device,
            distributed=distributed,
            distributed_seed=cfg.experiment.get('seed', 0),
        )

    else:
        raise ValueError(
            f"Unknown dataloader type: {dataloader_type}. "
            f"Must be 'baseline' or 'cfw'"
        )


def create_train_val_test_dataloaders(
    cfg: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders from config.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        >>> train_loader, val_loader, test_loader = create_train_val_test_dataloaders(cfg)
    """
    train_loader = create_dataloader_from_config(cfg, split='train')
    val_loader = create_dataloader_from_config(cfg, split='val')
    test_loader = create_dataloader_from_config(cfg, split='test')

    logger.info(
        f"Created dataloaders - Train: {len(train_loader.dataset)} samples, "
        f"Val: {len(val_loader.dataset)} samples, "
        f"Test: {len(test_loader.dataset)} samples"
    )

    return train_loader, val_loader, test_loader


# Alias for convenience - matches usage in train.py and extract_features.py
create_dataloader = create_dataloader_from_config
