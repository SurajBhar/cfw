"""
Training module for CFW.

This module provides training infrastructure including trainers for single-GPU
and multi-GPU (DDP) training, loss functions, callbacks, and utilities.
"""


from .trainer import create_trainer, create_trainer_from_config
from .trainer_single_gpu import SingleGPUTrainer
from .trainer_ddp import DDPTrainer

from .losses import (
    CrossEntropyLoss,
    FocalLoss,
    WeightedCrossEntropyLoss,
    get_loss_function,
    compute_loss_with_metrics
)

from .callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    LearningRateLoggerCallback,
    MetricsLoggerCallback,
    CallbackList
)


__all__ = [
    # Trainers
    'create_trainer',
    'create_trainer_from_config',
    'SingleGPUTrainer',
    'DDPTrainer',
    # Loss functions
    'CrossEntropyLoss',
    'FocalLoss',
    'WeightedCrossEntropyLoss',
    'get_loss_function',
    'compute_loss_with_metrics',
    # Callbacks
    'Callback',
    'CheckpointCallback',
    'EarlyStoppingCallback',
    'LearningRateLoggerCallback',
    'MetricsLoggerCallback',
    'CallbackList',
]
