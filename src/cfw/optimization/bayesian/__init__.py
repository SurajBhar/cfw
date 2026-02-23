"""
Bayesian Optimization module for CFW hyperparameter tuning.

This module provides BOHB (Bayesian Optimization HyperBand) integration
with the CFW training pipeline using Ray Tune.

Components:
    - CFWTrainable: Ray Trainable wrapper for CFW training
    - create_search_space: ConfigSpace creation from Hydra config
    - run_bayesian_optimization: Main entry point for BOHB optimization
    - get_best_config: Extract best configuration from results

Example usage:
    >>> from cfw.optimization.bayesian import run_bayesian_optimization
    >>> from omegaconf import OmegaConf
    >>>
    >>> cfg = OmegaConf.load('configs/config.yaml')
    >>> results = run_bayesian_optimization(cfg)
    >>> best_result = results.get_best_result()
    >>> print(f"Best config: {best_result.config}")
"""


from .trainable import CFWTrainable, create_trainable_with_resources
from .search_space import (
    create_search_space,
    create_default_search_space,
    create_narrow_search_space,
    create_cfw_search_space,
    get_search_space_info,
    print_search_space,
)
from .bohb_optimizer import (
    run_bayesian_optimization,
    get_best_config,
    get_all_trial_results,
)
from .utils import (
    setup_logging,
    save_best_config,
    load_base_config,
    merge_configs,
)

__all__ = [
    # Trainable
    'CFWTrainable',
    'create_trainable_with_resources',
    # Search space
    'create_search_space',
    'create_default_search_space',
    'create_narrow_search_space',
    'create_cfw_search_space',
    'get_search_space_info',
    'print_search_space',
    # BOHB optimizer
    'run_bayesian_optimization',
    'get_best_config',
    'get_all_trial_results',
    # Utilities
    'setup_logging',
    'save_best_config',
    'load_base_config',
    'merge_configs',
]
