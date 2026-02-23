"""Configuration utilities for Hydra.

This module provides utilities for loading, validating, and managing
Hydra configurations for CFW experiments.
"""


import os
from pathlib import Path
from typing import Any, Optional, List

from omegaconf import DictConfig, OmegaConf

from .logging import get_logger

logger = get_logger(__name__)


def validate_config(
    cfg: DictConfig,
    require_all_splits: bool = True,
    required_splits: Optional[List[str]] = None,
) -> None:
    """
    Validate Hydra configuration for required fields.

    Args:
        cfg: Hydra configuration to validate

    Raises:
        ValueError: If required fields are missing or invalid

    Example:
        >>> validate_config(cfg)  # Raises ValueError if invalid
    """
    # Required top-level keys
    required_keys = ['dataset', 'dataloader', 'experiment']

    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Config missing required key: {key}")

    # Validate dataset split directories.
    if require_all_splits:
        splits_to_validate = ['train', 'val', 'test']
    else:
        splits_to_validate = required_splits or ['train']
        if isinstance(splits_to_validate, str):
            splits_to_validate = [splits_to_validate]
        splits_to_validate = [str(s).lower() for s in splits_to_validate]

    for split in splits_to_validate:
        split_key = f'{split}_dir'
        if split_key not in cfg.dataset:
            raise ValueError(f"Config missing 'dataset.{split_key}'")

    # Validate dataloader config
    if 'type' not in cfg.dataloader:
        raise ValueError("Config missing 'dataloader.type'")
    if 'batch_size' not in cfg.dataloader:
        raise ValueError("Config missing 'dataloader.batch_size'")

    # If CFW dataloader, check for feature files
    if cfg.dataloader.type.lower() == 'cfw':
        if 'cfw' not in cfg.dataloader:
            raise ValueError("CFW dataloader requires 'dataloader.cfw' config")

        cfw_train_only = cfg.dataloader.cfw.get(
            'cfw_train_only',
            cfg.dataloader.cfw.get('train_only', True)
        )
        cfw_required_splits = ['train'] if cfw_train_only else ['train', 'val', 'test']

        for split in cfw_required_splits:
            feature_key = f'{split}_feature_file'
            label_key = f'{split}_label_file'
            img_path_key = f'{split}_img_path_file'

            if feature_key not in cfg.dataloader.cfw:
                raise ValueError(f"CFW config missing '{feature_key}'")
            if label_key not in cfg.dataloader.cfw:
                raise ValueError(f"CFW config missing '{label_key}'")
            if img_path_key not in cfg.dataloader.cfw:
                raise ValueError(f"CFW config missing '{img_path_key}'")

            if not cfg.dataloader.cfw.get(feature_key):
                raise ValueError(f"CFW config '{feature_key}' must be set")
            if not cfg.dataloader.cfw.get(label_key):
                raise ValueError(f"CFW config '{label_key}' must be set")
            if not cfg.dataloader.cfw.get(img_path_key):
                raise ValueError(f"CFW config '{img_path_key}' must be set")

    # Check CFW + DDP compatibility
    if cfg.dataloader.type.lower() == 'cfw':
        # Check if distributed training is enabled
        is_distributed = False
        if 'trainer' in cfg:
            is_distributed = cfg.trainer.get('distributed', False)
            if cfg.trainer.get('mode', '').lower() in ['ddp', 'distributed']:
                is_distributed = True
            # Also check for DDP-specific settings
            if cfg.trainer.get('use_ddp', False):
                is_distributed = True
            if cfg.trainer.get('backend') in ['nccl', 'gloo', 'mpi']:
                is_distributed = True

        if is_distributed:
            use_distributed_sampler = cfg.dataloader.cfw.get(
                'use_distributed_sampler', False
            )
            if use_distributed_sampler:
                logger.info(
                    "CFW + DDP enabled with DistributedWeightedSampler. "
                    "Weighted sampling will be partitioned across ranks."
                )
            else:
                raise ValueError(
                    "CFW dataloader requires DistributedWeightedSampler for DDP training. "
                    "WeightedRandomSampler cannot be used with DistributedSampler.\n"
                    "Options:\n"
                    "  1. Enable distributed sampler: Set dataloader.cfw.use_distributed_sampler=true\n"
                    "  2. Use single GPU: Set trainer.distributed=False\n"
                    "  3. Use baseline dataloader with DDP: Set dataloader.type='baseline'"
                )

    logger.info("Configuration validation passed")


def validate_paths(cfg: DictConfig) -> None:
    """
    Validate that all file paths in config exist.

    Args:
        cfg: Hydra configuration

    Raises:
        FileNotFoundError: If any required path doesn't exist

    Example:
        >>> validate_paths(cfg)  # Raises FileNotFoundError if paths invalid
    """
    # Check dataset directories
    for split in ['train', 'val', 'test']:
        dir_key = f'{split}_dir'
        if dir_key in cfg.dataset:
            path = cfg.dataset[dir_key]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dataset path not found: {path}")

    # Check CFW feature files if applicable
    if cfg.dataloader.type.lower() == 'cfw':
        for split in ['train', 'val', 'test']:
            feature_file = cfg.dataloader.cfw.get(f'{split}_feature_file')
            label_file = cfg.dataloader.cfw.get(f'{split}_label_file')
            img_path_file = cfg.dataloader.cfw.get(f'{split}_img_path_file')

            for file_path in [feature_file, label_file, img_path_file]:
                if file_path and not os.path.exists(file_path):
                    raise FileNotFoundError(f"Feature file not found: {file_path}")

    logger.info("All paths validated successfully")


def print_config(
    cfg: DictConfig,
    resolve: bool = True,
    save_to_file: Optional[str] = None,
) -> None:
    """
    Pretty print Hydra configuration.

    Args:
        cfg: Hydra configuration to print
        resolve: Whether to resolve interpolations (default: True)
        save_to_file: Optional file path to save config (default: None)

    Example:
        >>> print_config(cfg, save_to_file="config.yaml")
    """
    config_str = OmegaConf.to_yaml(cfg, resolve=resolve)

    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info("=" * 80)
    logger.info(config_str)
    logger.info("=" * 80)

    if save_to_file:
        Path(save_to_file).parent.mkdir(parents=True, exist_ok=True)
        with open(save_to_file, 'w') as f:
            f.write(config_str)
        logger.info(f"Configuration saved to: {save_to_file}")


def get_output_dir(cfg: DictConfig) -> Path:
    """
    Get output directory for experiment from config.

    Args:
        cfg: Hydra configuration

    Returns:
        Path to output directory

    Example:
        >>> output_dir = get_output_dir(cfg)
        >>> checkpoint_path = output_dir / "checkpoints" / "model.pth"
    """
    # Hydra automatically creates output directory
    # Access via hydra.run.dir or manually construct
    if 'hydra' in cfg and 'run' in cfg.hydra and 'dir' in cfg.hydra.run:
        return Path(cfg.hydra.run.dir)

    # Fallback: construct from experiment name
    base_dir = cfg.get('output_dir', 'outputs')
    exp_name = cfg.experiment.get('name', 'default')
    return Path(base_dir) / exp_name


def create_experiment_name(cfg: DictConfig) -> str:
    """
    Create experiment name from config.

    Args:
        cfg: Hydra configuration

    Returns:
        Experiment name string

    Example:
        >>> exp_name = create_experiment_name(cfg)
        >>> # Returns something like: "driveact_binary_dinov2_vitb14_cfw"
    """
    # Use explicit name if provided
    if 'experiment' in cfg and 'name' in cfg.experiment:
        return cfg.experiment.name

    # Otherwise, construct from components
    parts = []

    # Dataset name
    if 'dataset' in cfg and 'name' in cfg.dataset:
        parts.append(cfg.dataset.name)

    # Model name
    if 'model' in cfg and 'name' in cfg.model:
        parts.append(cfg.model.name)

    # Dataloader type
    if 'dataloader' in cfg and 'type' in cfg.dataloader:
        parts.append(cfg.dataloader.type)

    # Join with underscores
    exp_name = '_'.join(parts) if parts else 'experiment'

    return exp_name


def merge_configs(base_cfg: DictConfig, override_cfg: DictConfig) -> DictConfig:
    """
    Merge two configurations with override taking precedence.

    Args:
        base_cfg: Base configuration
        override_cfg: Override configuration

    Returns:
        Merged configuration

    Example:
        >>> merged = merge_configs(base_cfg, experiment_cfg)
    """
    return OmegaConf.merge(base_cfg, override_cfg)


def resolve_interpolations(cfg: DictConfig) -> DictConfig:
    """
    Resolve all interpolations in config.

    Args:
        cfg: Configuration with possible interpolations

    Returns:
        Resolved configuration

    Example:
        >>> cfg.experiment.name = "${dataset.name}_${model.name}"
        >>> resolved_cfg = resolve_interpolations(cfg)
        >>> # Now cfg.experiment.name is "driveact_binary_dinov2_vitb14"
    """
    OmegaConf.resolve(cfg)
    return cfg


def get_config_value(
    cfg: DictConfig,
    key: str,
    default: Any = None,
) -> Any:
    """
    Safely get config value with default fallback.

    Args:
        cfg: Configuration
        key: Dot-separated key path (e.g., "trainer.num_epochs")
        default: Default value if key not found

    Returns:
        Config value or default

    Example:
        >>> num_epochs = get_config_value(cfg, "trainer.num_epochs", default=100)
    """
    try:
        # Split key into parts
        parts = key.split('.')
        value = cfg
        for part in parts:
            value = value[part]
        return value
    except (KeyError, AttributeError):
        return default


def save_config(cfg: DictConfig, output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        output_path: Path to save config file

    Example:
        >>> save_config(cfg, "outputs/experiment_1/config.yaml")
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f, resolve=True)
    logger.info(f"Configuration saved to: {output_path}")


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Loaded configuration

    Example:
        >>> cfg = load_config("configs/experiment/cfw_dinov2_binary.yaml")
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    logger.info(f"Configuration loaded from: {config_path}")
    return cfg
