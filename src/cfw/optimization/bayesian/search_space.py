"""Search space definitions for Bayesian optimization.

Responsibilities:
- Define Bayesian/BOHB search spaces and trainable integration logic.
- Coordinate distributed hyperparameter optimization experiments.
- Persist best configurations for downstream final training runs.
"""


import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from typing import Optional, List, Dict, Any
from omegaconf import DictConfig


def create_search_space(cfg: DictConfig) -> CS.ConfigurationSpace:
    """
    Create ConfigSpace from Hydra configuration.

    This function reads search space bounds from the configuration and
    creates a ConfigSpace object for BOHB optimization.

    Args:
        cfg: Hydra configuration containing optimization.search_space settings

    Returns:
        ConfigSpace object with hyperparameters defined according to config

    Example:
        >>> cfg = OmegaConf.load('configs/optimization/bayesian.yaml')
        >>> config_space = create_search_space(cfg)
    """
    config_space = CS.ConfigurationSpace()
    search_space_cfg = cfg.optimization.search_space

    # Learning rate (log scale)
    if 'learning_rate' in search_space_cfg:
        lr_cfg = search_space_cfg.learning_rate
        config_space.add(CSH.UniformFloatHyperparameter(
            name="learning_rate",
            lower=lr_cfg.get('lower', 1e-5),
            upper=lr_cfg.get('upper', 5e-2),
            log=lr_cfg.get('log', True)
        ))

    # Weight decay (log scale)
    if 'weight_decay' in search_space_cfg:
        wd_cfg = search_space_cfg.weight_decay
        config_space.add(CSH.UniformFloatHyperparameter(
            name="weight_decay",
            lower=wd_cfg.get('lower', 1e-6),
            upper=wd_cfg.get('upper', 1e-3),
            log=wd_cfg.get('log', True)
        ))

    # Initial learning rate for linear interpolation schedules (log scale)
    if 'initial_lr' in search_space_cfg:
        initial_lr_cfg = search_space_cfg.initial_lr
        config_space.add(CSH.UniformFloatHyperparameter(
            name="initial_lr",
            lower=initial_lr_cfg.get('lower', 1e-5),
            upper=initial_lr_cfg.get('upper', 5e-2),
            log=initial_lr_cfg.get('log', True)
        ))

    # End learning rate for linear interpolation schedules (log scale)
    if 'end_lr' in search_space_cfg:
        end_lr_cfg = search_space_cfg.end_lr
        config_space.add(CSH.UniformFloatHyperparameter(
            name="end_lr",
            lower=end_lr_cfg.get('lower', 1e-7),
            upper=end_lr_cfg.get('upper', 1e-3),
            log=end_lr_cfg.get('log', True)
        ))

    # Optimizer (categorical)
    if 'optimizer' in search_space_cfg:
        opt_cfg = search_space_cfg.optimizer
        choices = list(opt_cfg.get('choices', ['ADAM']))
        config_space.add(CSH.CategoricalHyperparameter(
            name="optimizer",
            choices=choices
        ))

    # Scheduler (categorical)
    if 'scheduler' in search_space_cfg:
        sched_cfg = search_space_cfg.scheduler
        choices = list(sched_cfg.get('choices', ['CosineAnnealingLR']))
        config_space.add(CSH.CategoricalHyperparameter(
            name="scheduler",
            choices=choices
        ))

    # Batch size (categorical or uniform)
    if 'batch_size' in search_space_cfg:
        bs_cfg = search_space_cfg.batch_size
        if 'choices' in bs_cfg:
            config_space.add(CSH.CategoricalHyperparameter(
                name="batch_size",
                choices=list(bs_cfg.choices)
            ))
        else:
            config_space.add(CSH.UniformIntegerHyperparameter(
                name="batch_size",
                lower=bs_cfg.get('lower', 16),
                upper=bs_cfg.get('upper', 64)
            ))

    # Dropout
    if 'dropout' in search_space_cfg:
        dropout_cfg = search_space_cfg.dropout
        config_space.add(CSH.UniformFloatHyperparameter(
            name="dropout",
            lower=dropout_cfg.get('lower', 0.0),
            upper=dropout_cfg.get('upper', 0.5)
        ))

    return config_space


def create_default_search_space() -> CS.ConfigurationSpace:
    """
    Create a default search space without configuration.

    This provides a sensible default search space for CFW optimization
    focusing on learning rate and weight decay.

    Returns:
        ConfigSpace with default hyperparameter ranges
    """
    config_space = CS.ConfigurationSpace()

    # Learning rate (log scale)
    config_space.add(CSH.UniformFloatHyperparameter(
        name="learning_rate",
        lower=1e-5,
        upper=5e-2,
        log=True
    ))

    # Weight decay (log scale)
    config_space.add(CSH.UniformFloatHyperparameter(
        name="weight_decay",
        lower=1e-6,
        upper=1e-3,
        log=True
    ))

    # Optimizer
    config_space.add(CSH.CategoricalHyperparameter(
        name="optimizer",
        choices=["ADAM", "SGD"]
    ))

    return config_space


def create_narrow_search_space(
    lr_lower: float = 1e-4,
    lr_upper: float = 1e-2,
    wd_lower: float = 1e-5,
    wd_upper: float = 1e-4,
    optimizer_choices: Optional[List[str]] = None
) -> CS.ConfigurationSpace:
    """
    Create a narrow search space with custom bounds.

    This is useful for fine-tuning when you have a good estimate
    of the optimal hyperparameter region.

    Args:
        lr_lower: Lower bound for learning rate
        lr_upper: Upper bound for learning rate
        wd_lower: Lower bound for weight decay
        wd_upper: Upper bound for weight decay
        optimizer_choices: List of optimizer names to search over

    Returns:
        ConfigSpace with specified bounds
    """
    if optimizer_choices is None:
        optimizer_choices = ["ADAM"]

    config_space = CS.ConfigurationSpace()

    config_space.add(CSH.UniformFloatHyperparameter(
        name="learning_rate",
        lower=lr_lower,
        upper=lr_upper,
        log=True
    ))

    config_space.add(CSH.UniformFloatHyperparameter(
        name="weight_decay",
        lower=wd_lower,
        upper=wd_upper,
        log=True
    ))

    if len(optimizer_choices) > 1:
        config_space.add(CSH.CategoricalHyperparameter(
            name="optimizer",
            choices=optimizer_choices
        ))

    return config_space


def create_cfw_search_space(cfg: DictConfig) -> CS.ConfigurationSpace:
    """
    Create search space including CFW-specific hyperparameters.

    This extends the base search space with CFW clustering parameters
    like min_cluster_size and outlier_weight.

    Args:
        cfg: Hydra configuration

    Returns:
        ConfigSpace with CFW-specific hyperparameters
    """
    # Start with base search space
    config_space = create_search_space(cfg)
    search_space_cfg = cfg.optimization.search_space

    # CFW-specific: min_cluster_size
    if 'min_cluster_size' in search_space_cfg:
        mcs_cfg = search_space_cfg.min_cluster_size
        config_space.add(CSH.UniformIntegerHyperparameter(
            name="min_cluster_size",
            lower=mcs_cfg.get('lower', 10),
            upper=mcs_cfg.get('upper', 50)
        ))

    # CFW-specific: outlier_weight
    if 'outlier_weight' in search_space_cfg:
        ow_cfg = search_space_cfg.outlier_weight
        config_space.add(CSH.UniformFloatHyperparameter(
            name="outlier_weight",
            lower=ow_cfg.get('lower', 0.0001),
            upper=ow_cfg.get('upper', 0.01),
            log=ow_cfg.get('log', True)
        ))

    return config_space


def get_search_space_info(config_space: CS.ConfigurationSpace) -> Dict[str, Any]:
    """
    Get information about a search space.

    Args:
        config_space: ConfigSpace object

    Returns:
        Dictionary containing search space information
    """
    info = {
        'num_hyperparameters': len(config_space.get_hyperparameters()),
        'hyperparameters': {}
    }

    for hp in config_space.get_hyperparameters():
        hp_info = {
            'type': hp.__class__.__name__,
        }

        if isinstance(hp, CSH.UniformFloatHyperparameter):
            hp_info['lower'] = hp.lower
            hp_info['upper'] = hp.upper
            hp_info['log'] = hp.log
        elif isinstance(hp, CSH.UniformIntegerHyperparameter):
            hp_info['lower'] = hp.lower
            hp_info['upper'] = hp.upper
        elif isinstance(hp, CSH.CategoricalHyperparameter):
            hp_info['choices'] = list(hp.choices)

        info['hyperparameters'][hp.name] = hp_info

    return info


def print_search_space(config_space: CS.ConfigurationSpace) -> None:
    """
    Print a human-readable summary of the search space.

    Args:
        config_space: ConfigSpace object to print
    """
    print("=" * 60)
    print("Search Space Configuration")
    print("=" * 60)

    for hp in config_space.get_hyperparameters():
        if isinstance(hp, CSH.UniformFloatHyperparameter):
            scale = "log" if hp.log else "linear"
            print(f"  {hp.name}: [{hp.lower:.2e}, {hp.upper:.2e}] ({scale} scale)")
        elif isinstance(hp, CSH.UniformIntegerHyperparameter):
            print(f"  {hp.name}: [{hp.lower}, {hp.upper}] (integer)")
        elif isinstance(hp, CSH.CategoricalHyperparameter):
            print(f"  {hp.name}: {list(hp.choices)} (categorical)")
        else:
            print(f"  {hp.name}: {hp}")

    print("=" * 60)
