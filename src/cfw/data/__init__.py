"""
Data pipeline module for CFW.

This module provides:
- Dataset classes for different data formats
- Data transforms and augmentation
- Dataloader factories for baseline and CFW approaches
- Custom samplers for weighted sampling
"""


from .datasets import (
    ImageFolderCustom,
    BinaryClassificationDataset,
    WeightedImageDataset,
)
from .transforms import (
    GaussianBlur,
    MaybeToTensor,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    make_normalize_transform,
    make_classification_train_transform,
    make_classification_eval_transform,
    make_no_aug_transform,
)
from .samplers import (
    create_weighted_sampler,
    create_distributed_weighted_sampler,
    DistributedWeightedSampler,
    validate_weights,
    normalize_weights,
    get_class_weights,
)

__all__ = [
    # Datasets
    'ImageFolderCustom',
    'BinaryClassificationDataset',
    'WeightedImageDataset',
    # Transforms
    'GaussianBlur',
    'MaybeToTensor',
    'IMAGENET_DEFAULT_MEAN',
    'IMAGENET_DEFAULT_STD',
    'make_normalize_transform',
    'make_classification_train_transform',
    'make_classification_eval_transform',
    'make_no_aug_transform',
    # Samplers
    'create_weighted_sampler',
    'create_distributed_weighted_sampler',
    'DistributedWeightedSampler',
    'validate_weights',
    'normalize_weights',
    'get_class_weights',
    # Dataloaders
    'create_baseline_dataloader',
    'create_cfw_dataloader',
    'create_dataloader_from_config',
    'create_train_val_test_dataloaders',
]


def __getattr__(name):
    # Lazy-load dataloader helpers to avoid circular imports with core.cfw_dataloader.
    if name in {
        'create_baseline_dataloader',
        'create_cfw_dataloader',
        'create_dataloader_from_config',
        'create_train_val_test_dataloaders',
    }:
        from . import dataloaders as _dataloaders
        return getattr(_dataloaders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
