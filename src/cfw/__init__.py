"""
CFW (Clustered Feature Weighting) - A modern implementation.

This package provides a production-ready implementation of the Clustered Feature
Weighting method for handling imbalanced datasets in deep learning.

Modules:
    core: CFW algorithm (clustering, weighting, dataloader builder)
    data: Dataset classes, transforms, and dataloader factories
    models: Model registry and architectures (DINOv2, ViT, etc.)
    optimization: Optimizer and scheduler factories
    utils: Logging, reproducibility, config utilities

Example:
    >>> from cfw.models import build_classifier_from_config
    >>> from cfw.data import create_dataloader
    >>> from cfw.optimization import create_optimizer, create_scheduler
    >>>
    >>> # Build model
    >>> model = build_classifier_from_config(model_config)
    >>>
    >>> # Create dataloader (baseline or CFW)
    >>> train_loader = create_dataloader(cfg, 'train')
    >>>
    >>> # Create optimizer and scheduler
    >>> optimizer = create_optimizer(model.parameters(), optimizer_config)
    >>> scheduler = create_scheduler(optimizer, scheduler_config, num_epochs, lr)
"""


__version__ = '0.1.0'

# Core modules are imported in subpackages
# Users should import from specific modules:
# from cfw.core import FeatureClusterer, WeightComputer, CFWDataLoaderBuilder
# from cfw.data import create_dataloader, create_baseline_dataloader, create_cfw_dataloader
# from cfw.models import build_classifier, build_classifier_from_config
# from cfw.optimization import create_optimizer, create_scheduler
# from cfw.utils import setup_logging, set_seed

__all__ = ['__version__']
