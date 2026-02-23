"""Utility functions for Bayesian optimization.

Responsibilities:
- Define Bayesian/BOHB search spaces and trainable integration logic.
- Coordinate distributed hyperparameter optimization experiments.
- Persist best configurations for downstream final training runs.
"""


import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from omegaconf import DictConfig, OmegaConf


def _register_omegaconf_resolvers() -> None:
    """
    Register resolvers needed when configs are loaded outside Hydra runtime.

    Ray trial workers load YAML configs directly via OmegaConf, so Hydra's
    runtime resolver registration (e.g., `${now:...}`) may be missing there.
    """
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver(
            "now",
            lambda fmt="%Y-%m-%d_%H-%M-%S": datetime.now().strftime(fmt),
        )


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logging for Bayesian optimization runs.

    Args:
        log_dir: Directory for log files. If None, logs to console only.
        log_level: Logging level (default: logging.INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("cfw.bayesian")
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "bayesian_optimization.log")
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def save_best_config(config: Dict[str, Any], path: str) -> None:
    """
    Save best configuration to a YAML file.

    Args:
        config: Best hyperparameter configuration dictionary
        path: Path to save the configuration file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert to OmegaConf and save
    cfg = OmegaConf.create(_to_primitive(config))
    OmegaConf.save(cfg, path)


def save_best_config_json(config: Dict[str, Any], path: str) -> None:
    """
    Save best configuration to a JSON file.

    Args:
        config: Best hyperparameter configuration dictionary
        path: Path to save the JSON file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        json.dump(_to_primitive(config), f, indent=2)


def load_base_config(config_path: str) -> DictConfig:
    """
    Load base Hydra configuration from file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        DictConfig object with loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _register_omegaconf_resolvers()
    return OmegaConf.load(config_path)


def _to_primitive(value: Any) -> Any:
    """
    Convert sampled trial values to plain Python primitives.

    BOHB/ConfigSpace can emit NumPy scalar types (e.g., numpy.str_, numpy.float64),
    which OmegaConf can reject when assigning into typed nodes.
    """
    # Handle NumPy-like scalars first (e.g., numpy.str_, numpy.float64).
    # Some of these are subclasses of builtins, so normalize them before
    # generic isinstance checks.
    item_fn = getattr(value, "item", None)
    if callable(item_fn):
        try:
            item_value = item_fn()
        except Exception:
            item_value = value
        value = item_value

    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (list, tuple)):
        return [_to_primitive(v) for v in value]

    if isinstance(value, dict):
        return {
            str(_to_primitive(k)): _to_primitive(v)
            for k, v in value.items()
        }

    return str(value)


def merge_configs(base_cfg: DictConfig, trial_cfg: Dict[str, Any]) -> DictConfig:
    """
    Merge base configuration with trial-specific hyperparameters.

    This function takes a base Hydra configuration and merges it with
    hyperparameters sampled for a specific trial during optimization.

    Args:
        base_cfg: Base Hydra configuration
        trial_cfg: Trial-specific hyperparameters from search space

    Returns:
        Merged DictConfig with trial hyperparameters applied

    Example:
        >>> base_cfg = OmegaConf.create({'optimizer': {'lr': 0.001}})
        >>> trial_cfg = {'learning_rate': 0.0001}
        >>> merged = merge_configs(base_cfg, trial_cfg)
    """
    # Create a deep copy of base config without eagerly resolving interpolations.
    # Eager resolution can fail in Ray workers when Hydra-only resolvers are absent.
    merged = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    safe_trial_cfg = {k: _to_primitive(v) for k, v in trial_cfg.items()}

    def _update(path: str, value: Any) -> None:
        """Update merged config with a Python-primitive value."""
        OmegaConf.update(merged, path, _to_primitive(value))

    def _normalize_optimizer_config_for_name(merged_cfg: DictConfig, optimizer_name: str) -> None:
        """
        Remove optimizer-specific keys that are invalid for the selected optimizer.

        Trial BOHB runs can switch optimizer families (e.g., SGD <-> ADAMW).
        Base configs often include family-specific kwargs (betas vs momentum).
        This normalization avoids passing incompatible kwargs into torch optimizers.
        """
        if 'optimizer' not in merged_cfg:
            return

        opt_cfg = merged_cfg.optimizer
        normalized = str(optimizer_name).upper()

        if normalized == "SGD":
            for key in ("betas", "eps", "amsgrad"):
                if key in opt_cfg:
                    del opt_cfg[key]
            if 'momentum' not in opt_cfg:
                OmegaConf.update(merged_cfg, 'optimizer.momentum', 0.9)
            if 'dampening' not in opt_cfg:
                OmegaConf.update(merged_cfg, 'optimizer.dampening', 0.0)
            if 'nesterov' not in opt_cfg:
                OmegaConf.update(merged_cfg, 'optimizer.nesterov', False)
            return

        if normalized in {"ADAM", "ADAMW"}:
            for key in ("dampening", "nesterov"):
                if key in opt_cfg:
                    del opt_cfg[key]
            if 'betas' not in opt_cfg:
                OmegaConf.update(merged_cfg, 'optimizer.betas', [0.9, 0.999])
            if 'eps' not in opt_cfg:
                OmegaConf.update(merged_cfg, 'optimizer.eps', 1e-8)
            if 'amsgrad' not in opt_cfg:
                OmegaConf.update(merged_cfg, 'optimizer.amsgrad', False)

    # Map trial hyperparameters to config structure
    if 'learning_rate' in safe_trial_cfg:
        learning_rate = float(safe_trial_cfg['learning_rate'])
        _update('optimizer.lr', learning_rate)
        _update('scheduler.start_lr', learning_rate)

    if 'weight_decay' in safe_trial_cfg:
        _update('optimizer.weight_decay', float(safe_trial_cfg['weight_decay']))

    if 'initial_lr' in safe_trial_cfg:
        initial_lr = float(safe_trial_cfg['initial_lr'])
        _update('optimizer.lr', initial_lr)
        _update('scheduler.start_lr', initial_lr)

    if 'end_lr' in safe_trial_cfg:
        _update('scheduler.end_lr', float(safe_trial_cfg['end_lr']))

    if 'optimizer' in safe_trial_cfg:
        optimizer_name = str(safe_trial_cfg['optimizer']).upper()
        _update('optimizer.name', optimizer_name)
        _normalize_optimizer_config_for_name(merged, optimizer_name)

    if 'scheduler' in safe_trial_cfg:
        _update('scheduler.name', str(safe_trial_cfg['scheduler']))

    if 'batch_size' in safe_trial_cfg:
        _update('dataloader.batch_size', int(safe_trial_cfg['batch_size']))

    if 'dropout' in safe_trial_cfg:
        _update('model.dropout', float(safe_trial_cfg['dropout']))

    return merged


def validate_search_space_config(cfg: DictConfig) -> bool:
    """
    Validate that the configuration has all required fields for search space.

    Args:
        cfg: Hydra configuration to validate

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If required fields are missing
    """
    required_sections = ['optimization']

    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required config section: {section}")

    opt_cfg = cfg.optimization

    # Check for search_space configuration
    if 'search_space' not in opt_cfg:
        raise ValueError("Missing 'search_space' in optimization config")

    return True


def format_trial_result(
    trial_id: str,
    config: Dict[str, Any],
    metrics: Dict[str, float]
) -> str:
    """
    Format trial result for logging.

    Args:
        trial_id: Unique trial identifier
        config: Trial hyperparameter configuration
        metrics: Trial metrics (e.g., val_balanced_accuracy)

    Returns:
        Formatted string for logging
    """
    config_str = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in config.items()
                           if not k.startswith('_'))

    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())

    return f"Trial {trial_id}: Config=[{config_str}] | Metrics=[{metrics_str}]"


def get_resource_config(cfg: DictConfig) -> Dict[str, Any]:
    """
    Extract Ray resource configuration from Hydra config.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary with Ray resource settings
    """
    ray_cfg = cfg.optimization.get('ray', {})

    return {
        'num_cpus': ray_cfg.get('num_cpus', os.cpu_count()),
        'num_gpus': ray_cfg.get('num_gpus', 1),
        'resources_per_trial': {
            'cpu': ray_cfg.get('resources_per_trial', {}).get('cpu', 4),
            'gpu': ray_cfg.get('resources_per_trial', {}).get('gpu', 1),
        }
    }
