"""
Optimization module for creating optimizers, schedulers, and hyperparameter optimization.

This module provides factory functions for creating optimizers and schedulers
from Hydra configurations, consolidating logic that was previously duplicated
across multiple trainer files. It also includes Bayesian optimization capabilities.

Public API:
    Optimizer Factory:
    - create_optimizer: Create optimizer from config
    - get_optimizer_info: Get optimizer information

    Scheduler Factory:
    - create_scheduler: Create learning rate scheduler from config
    - get_scheduler_info: Get scheduler information
    - compute_total_lr_schedule: Compute LR schedule for visualization

    Bayesian Optimization (submodule):
    - bayesian.run_bayesian_optimization: Run BOHB hyperparameter optimization
    - bayesian.create_search_space: Create ConfigSpace from config
    - bayesian.CFWTrainable: Ray Trainable wrapper for CFW training

Example:
    >>> from cfw.optimization import create_optimizer, create_scheduler
    >>> from omegaconf import OmegaConf
    >>>
    >>> # Create optimizer
    >>> opt_config = OmegaConf.create({'name': 'ADAM', 'lr': 0.001, 'weight_decay': 0.0})
    >>> optimizer = create_optimizer(model.parameters(), opt_config)
    >>>
    >>> # Create scheduler
    >>> sched_config = OmegaConf.create({'name': 'CosineAnnealingLR', 'T_max': 100})
    >>> scheduler = create_scheduler(optimizer, sched_config, num_epochs=100, initial_lr=0.001)
    >>>
    >>> # Bayesian Optimization
    >>> from cfw.optimization.bayesian import run_bayesian_optimization
    >>> results = run_bayesian_optimization(cfg)
"""


from .optimizers import (
    create_optimizer,
    get_optimizer_info,
)

from .schedulers import (
    create_scheduler,
    get_scheduler_info,
    compute_total_lr_schedule,
)

# Bayesian optimization is available as a submodule
# Import with: from cfw.optimization.bayesian import run_bayesian_optimization

__all__ = [
    # Optimizer factory
    'create_optimizer',
    'get_optimizer_info',
    # Scheduler factory
    'create_scheduler',
    'get_scheduler_info',
    'compute_total_lr_schedule',
]
