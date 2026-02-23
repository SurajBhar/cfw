"""BOHB (Bayesian Optimization HyperBand) orchestration for CFW.

Responsibilities:
- Define Bayesian/BOHB search spaces and trainable integration logic.
- Coordinate distributed hyperparameter optimization experiments.
- Persist best configurations for downstream final training runs.
"""


import os
import glob
import logging
from typing import Optional, Dict, Any

import ray
import torch
import torch.nn as nn
from ray import tune
from ray.tune import RunConfig, CheckpointConfig
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from omegaconf import DictConfig, OmegaConf

from cfw.models.builders import build_classifier_from_config
from cfw.data.dataloaders import create_dataloader_from_config
from cfw.evaluation.metrics import calculate_balanced_accuracy

from .trainable import CFWTrainable
from .search_space import create_search_space, create_default_search_space, print_search_space
from .utils import (
    setup_logging,
    save_best_config,
    save_best_config_json,
    get_resource_config,
    merge_configs,
    load_base_config,
)

logger = logging.getLogger(__name__)


def run_bayesian_optimization(cfg: DictConfig) -> tune.ResultGrid:
    """
    Run BOHB optimization using CFW training infrastructure.

    This is the main entry point for Bayesian hyperparameter optimization.
    It sets up Ray Tune with BOHB scheduler and searcher, then runs
    optimization using the CFWTrainable.

    Args:
        cfg: Hydra configuration containing:
            - optimization.search_space: Hyperparameter bounds
            - optimization.bohb: BOHB scheduler settings
            - optimization.tuner: Ray Tuner settings
            - optimization.ray: Ray cluster settings
            - optimization.storage: Result storage settings

    Returns:
        Ray Tune ResultGrid containing all trial results

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.load('configs/config.yaml')
        >>> results = run_bayesian_optimization(cfg)
        >>> best_result = results.get_best_result()
    """
    # Setup logging
    log_dir = cfg.optimization.storage.get('path', './ray_results')
    setup_logging(log_dir)

    logger.info("=" * 60)
    logger.info("Starting BOHB Bayesian Optimization")
    logger.info("=" * 60)

    # Initialize Ray
    _init_ray(cfg)

    # Create search space
    if 'search_space' in cfg.optimization:
        config_space = create_search_space(cfg)
    else:
        logger.warning("No search_space in config, using default search space")
        config_space = create_default_search_space()

    logger.info("Search space configuration:")
    print_search_space(config_space)

    # Create BOHB scheduler
    scheduler = _create_scheduler(cfg)

    # Create BOHB searcher
    searcher = _create_searcher(config_space, cfg)

    # Get tuner settings
    tuner_cfg = cfg.optimization.tuner
    metric = tuner_cfg.get('metric', 'val_balanced_accuracy')
    mode = tuner_cfg.get('mode', 'max')
    num_samples = tuner_cfg.get('num_samples', 40)
    max_concurrent = tuner_cfg.get('max_concurrent', 4)
    bohb_cfg = cfg.optimization.get('bohb', {})
    max_t = int(bohb_cfg.get('max_t', 50))

    # Get resource settings
    resource_cfg = get_resource_config(cfg)

    # Prepare trainable config (passed to each trial)
    # Need to resolve base_config_path to absolute path
    base_config_path = _get_base_config_path(cfg)

    trainable_config = {
        "base_config_path": base_config_path,
        "_max_iterations": max_t,
    }

    logger.info(f"Base config path: {base_config_path}")
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Max concurrent trials: {max_concurrent}")
    logger.info(f"Hard stop cap (training_iteration): {max_t}")
    logger.info(f"Optimization metric: {metric} ({mode})")

    # Configure storage
    storage_cfg = cfg.optimization.storage
    storage_path = storage_cfg.get('path', './ray_results')
    experiment_name = storage_cfg.get('name', 'cfw_bohb_experiment')
    checkpoints_to_keep = cfg.optimization.get('checkpoints_to_keep', 3)
    checkpoint_frequency = cfg.optimization.get('checkpoint_frequency', None)
    run_verbose = cfg.optimization.get('verbose', None)

    if checkpoint_frequency not in (None, 0, False):
        logger.info(
            "Periodic checkpointing is disabled for BOHB; ignoring "
            "optimization.checkpoint_frequency=%s",
            checkpoint_frequency,
        )
    if run_verbose is not None:
        logger.info(
            "RunConfig(verbose) is deprecated in this Ray version; ignoring "
            "optimization.verbose=%s",
            run_verbose,
        )

    # Create Tuner
    tuner = tune.Tuner(
        tune.with_resources(
            CFWTrainable,
            resources={
                "cpu": resource_cfg['resources_per_trial']['cpu'],
                "gpu": resource_cfg['resources_per_trial']['gpu'],
            }
        ),
        param_space=trainable_config,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=searcher,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
            metric=metric,
            mode=mode,
        ),
        run_config=RunConfig(
            name=experiment_name,
            storage_path=storage_path,
            stop={"training_iteration": max_t},
            checkpoint_config=CheckpointConfig(
                num_to_keep=checkpoints_to_keep,
            ),
        ),
    )

    logger.info("Starting optimization...")

    # Run optimization
    results = tuner.fit()

    logger.info("=" * 60)
    logger.info("Optimization Complete")
    logger.info("=" * 60)

    # Log best result
    best_result = results.get_best_result()
    if best_result:
        _log_best_result(best_result, metric, storage_path)
        if bool(cfg.optimization.get("evaluate_test_on_best", False)):
            _evaluate_best_result_on_test(
                best_result=best_result,
                storage_path=storage_path,
            )

    return results


def _init_ray(cfg: DictConfig) -> None:
    """
    Initialize Ray cluster.

    Args:
        cfg: Hydra configuration
    """
    if ray.is_initialized():
        logger.info("Ray already initialized")
        return

    ray_cfg = cfg.optimization.get('ray', {})
    mode = str(ray_cfg.get('mode', 'local')).lower()

    if mode == 'cluster':
        address = ray_cfg.get('address', 'auto')
        namespace = ray_cfg.get('namespace', 'cfw-ray')

        logger.info(
            f"Initializing Ray in cluster mode (address={address}, namespace={namespace})"
        )
        ray.init(
            address=address,
            namespace=namespace,
            ignore_reinit_error=True,
        )
        return

    if mode != 'local':
        raise ValueError(
            f"Unsupported Ray mode '{mode}'. Expected 'local' or 'cluster'."
        )

    num_cpus = ray_cfg.get('num_cpus', None)  # None = auto-detect
    num_gpus = ray_cfg.get('num_gpus', None)  # None = auto-detect

    logger.info(f"Initializing Ray in local mode (cpus={num_cpus}, gpus={num_gpus})")

    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        ignore_reinit_error=True,
    )


def _create_scheduler(cfg: DictConfig) -> HyperBandForBOHB:
    """
    Create HyperBand scheduler for BOHB.

    Args:
        cfg: Hydra configuration

    Returns:
        Configured HyperBandForBOHB scheduler
    """
    bohb_cfg = cfg.optimization.get('bohb', {})

    max_t = bohb_cfg.get('max_t', 50)
    reduction_factor = bohb_cfg.get('reduction_factor', 4)
    stop_last_trials = bohb_cfg.get('stop_last_trials', False)

    logger.info(f"BOHB Scheduler: max_t={max_t}, reduction_factor={reduction_factor}")

    return HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=max_t,
        reduction_factor=reduction_factor,
        stop_last_trials=stop_last_trials,
    )


def _create_searcher(
    config_space,
    cfg: DictConfig
) -> TuneBOHB:
    """
    Create BOHB searcher.

    Args:
        config_space: ConfigSpace defining hyperparameter search space
        cfg: Hydra configuration

    Returns:
        Configured TuneBOHB searcher
    """
    tuner_cfg = cfg.optimization.tuner
    metric = tuner_cfg.get('metric', 'val_balanced_accuracy')
    mode = tuner_cfg.get('mode', 'max')

    return TuneBOHB(
        space=config_space,
        metric=metric,
        mode=mode,
    )


def _get_base_config_path(cfg: DictConfig) -> str:
    """
    Get absolute path to base configuration file.

    Args:
        cfg: Hydra configuration

    Returns:
        Absolute path to base config file
    """
    # Check if explicitly specified
    if 'base_config_path' in cfg.optimization:
        path = cfg.optimization.base_config_path
        if os.path.isabs(path):
            return path
        # Make relative path absolute
        return os.path.abspath(path)

    # Default: save current config to temp file and use that
    temp_config_path = os.path.join(
        cfg.optimization.storage.get('path', './ray_results'),
        'base_config.yaml'
    )
    os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
    OmegaConf.save(cfg, temp_config_path)

    return os.path.abspath(temp_config_path)


def _log_best_result(
    best_result: tune.Result,
    metric: str,
    storage_path: str
) -> None:
    """
    Log and save best result from optimization.

    Args:
        best_result: Best trial result from Ray Tune
        metric: Name of optimization metric
        storage_path: Path to save results
    """
    best_config = best_result.config
    best_metrics = best_result.metrics

    logger.info("Best Configuration:")
    for k, v in best_config.items():
        if not k.startswith('_') and k != 'base_config_path':
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.6g}")
            else:
                logger.info(f"  {k}: {v}")

    metric_value = best_metrics.get(metric, None)
    if metric_value is None:
        logger.info(f"Best {metric}: N/A")
    else:
        logger.info(f"Best {metric}: {metric_value:.4f}")

    if 'val_loss' in best_metrics:
        logger.info(f"Best val_loss: {best_metrics['val_loss']:.4f}")

    # Save best config
    save_path = os.path.join(storage_path, "best_config.yaml")
    try:
        save_best_config(best_config, save_path)
        logger.info(f"Best config saved to: {save_path}")
    except Exception as exc:
        logger.warning("Failed to save best config YAML: %s", exc)
        json_save_path = os.path.join(storage_path, "best_config.json")
        save_best_config_json(best_config, json_save_path)
        logger.info(f"Best config saved to JSON fallback: {json_save_path}")


def _resolve_checkpoint_path(best_result: tune.Result) -> Optional[str]:
    """Resolve a checkpoint.pt path from a Ray best_result object."""
    checkpoint = getattr(best_result, "checkpoint", None)
    if checkpoint is None:
        logger.warning(
            "No checkpoint found for best trial; skipping one-time test evaluation."
        )
        return None

    try:
        checkpoint_dir = checkpoint.to_directory()
    except Exception as exc:
        logger.warning(
            "Failed to materialize best trial checkpoint directory: %s",
            exc,
        )
        return None

    default_checkpoint = os.path.join(checkpoint_dir, "checkpoint.pt")
    if os.path.exists(default_checkpoint):
        return default_checkpoint

    matches = glob.glob(
        os.path.join(checkpoint_dir, "**", "checkpoint.pt"),
        recursive=True,
    )
    if matches:
        return matches[0]

    logger.warning(
        "No checkpoint.pt found under best trial checkpoint dir: %s",
        checkpoint_dir,
    )
    return None


def _evaluate_best_result_on_test(
    best_result: tune.Result,
    storage_path: str,
) -> Optional[Dict[str, float]]:
    """
    Run one-time test evaluation for the best BOHB trial.

    This evaluates the best trial checkpoint exactly once on the test split
    after BOHB finishes. It is not used for trial selection.
    """
    checkpoint_path = _resolve_checkpoint_path(best_result)
    if checkpoint_path is None:
        return None

    base_config_path = best_result.config.get("base_config_path")
    if not base_config_path:
        logger.warning(
            "Best trial config missing base_config_path; "
            "skipping one-time test evaluation."
        )
        return None

    logger.info("Starting one-time test evaluation for best trial...")
    logger.info("Using checkpoint: %s", checkpoint_path)

    base_cfg = load_base_config(base_config_path)
    eval_cfg = merge_configs(base_cfg, best_result.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier_from_config(eval_cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = create_dataloader_from_config(eval_cfg, split="test")

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = CFWTrainable._parse_batch(batch)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.append(predicted)
            all_targets.append(targets)

    if total == 0:
        logger.warning("Test loader produced zero samples; skipping metric computation.")
        return None

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    num_classes = int(eval_cfg.model.num_classes)

    test_metrics = {
        "test_loss": total_loss / len(test_loader),
        "test_accuracy": 100.0 * correct / total,
        "test_balanced_accuracy": calculate_balanced_accuracy(
            all_preds,
            all_targets,
            num_classes,
        ),
    }

    logger.info("Best-trial one-time test metrics:")
    logger.info("  test_loss: %.6f", test_metrics["test_loss"])
    logger.info("  test_accuracy: %.4f%%", test_metrics["test_accuracy"])
    logger.info(
        "  test_balanced_accuracy: %.6f",
        test_metrics["test_balanced_accuracy"],
    )

    try:
        best_result.metrics.update(test_metrics)
    except Exception:
        # Result metrics may be immutable depending on Ray internals.
        pass

    save_path = os.path.join(storage_path, "best_trial_test_metrics.yaml")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(test_metrics), f=save_path)
    logger.info("Best-trial test metrics saved to: %s", save_path)

    return test_metrics


def get_best_config(results: tune.ResultGrid) -> Dict[str, Any]:
    """
    Extract best configuration from optimization results.

    Args:
        results: Ray Tune ResultGrid

    Returns:
        Dictionary containing best hyperparameters
    """
    best_result = results.get_best_result()
    return best_result.config if best_result else {}


def get_all_trial_results(results: tune.ResultGrid) -> list:
    """
    Extract all trial results for analysis.

    Args:
        results: Ray Tune ResultGrid

    Returns:
        List of dictionaries containing trial configs and metrics
    """
    trial_results = []

    for result in results:
        trial_data = {
            'config': {k: v for k, v in result.config.items()
                      if not k.startswith('_') and k != 'base_config_path'},
            'metrics': result.metrics,
            'trial_id': result.metrics.get('trial_id', 'unknown'),
        }
        trial_results.append(trial_data)

    return trial_results
