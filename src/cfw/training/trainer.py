"""Trainer factory for automatic trainer selection.

This module provides the create_trainer() factory function that automatically
selects between SingleGPUTrainer and DDPTrainer based on the training environment.
"""


import logging
from typing import Optional, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from .trainer_single_gpu import SingleGPUTrainer
from .trainer_ddp import DDPTrainer
from .callbacks import Callback
from ..utils.distributed import is_distributed


logger = logging.getLogger(__name__)


def create_trainer(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    loss_fn: Optional[nn.Module] = None,
    num_classes: int = 2,
    device: Optional[str] = None,
    log_dir: Optional[str] = None,
    experiment_name: str = "experiment",
    callbacks: Optional[list[Callback]] = None,
    gradient_clip_max_norm: Optional[float] = None,
    use_amp: bool = False,
    **kwargs
) -> Union[SingleGPUTrainer, DDPTrainer]:
    """
    Create the appropriate trainer based on environment.

    Automatically detects if running in distributed mode (via torch.distributed)
    and creates either a SingleGPUTrainer or DDPTrainer accordingly.

    Args:
        model: Model to train.
        train_dataloader: Training data loader.
        val_dataloader: Validation data loader.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler. Optional.
        loss_fn: Loss function. If None, uses CrossEntropyLoss. Optional.
        num_classes: Number of classes in the dataset. Default: 2
        device: Device to use for single-GPU training (e.g., 'cuda:0', 'cpu').
            Ignored if in distributed mode. Optional.
        log_dir: Directory for TensorBoard logs. Optional.
        experiment_name: Name of the experiment. Default: "experiment"
        callbacks: List of callback instances. Optional.
        gradient_clip_max_norm: Maximum gradient norm for clipping. Optional.
        use_amp: Enable automatic mixed precision. Default: False
        **kwargs: Additional arguments passed to the trainer.

    Returns:
        SingleGPUTrainer if not in distributed mode, DDPTrainer otherwise.

    Example:
        Single-GPU usage:
        >>> trainer = create_trainer(
        ...     model=model,
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     num_classes=10,
        ...     device='cuda:0'
        ... )
        >>> trainer.train(num_epochs=50)

        Distributed usage (launched with torchrun):
        >>> # torchrun --nproc_per_node=4 train_script.py
        >>> def main():
        ...     # Distributed environment is automatically detected
        ...     trainer = create_trainer(
        ...         model=model,
        ...         train_dataloader=train_loader,
        ...         val_dataloader=val_loader,
        ...         optimizer=optimizer,
        ...         num_classes=10
        ...     )
        ...     trainer.train(num_epochs=50)

    Note:
        - For distributed training, ensure torch.distributed.init_process_group()
          is called before creating the trainer (or use setup_ddp() utility)
        - Train and validation dataloaders should use DistributedSampler for DDP
        - Callbacks will only run on rank 0 in distributed mode
    """
    if is_distributed():
        # Distributed mode detected - create DDPTrainer
        from ..utils.distributed import get_rank, get_world_size

        rank = get_rank()
        world_size = get_world_size()

        logger.info(
            f"Distributed mode detected. Creating DDPTrainer "
            f"(rank={rank}, world_size={world_size})"
        )

        # Extract DDP-specific kwargs
        find_unused_parameters = kwargs.pop('find_unused_parameters', False)

        # Set default gradient clipping for DDP if not specified
        if gradient_clip_max_norm is None:
            gradient_clip_max_norm = 1.0

        trainer = DDPTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            num_classes=num_classes,
            rank=rank,
            world_size=world_size,
            log_dir=log_dir,
            experiment_name=experiment_name,
            callbacks=callbacks,
            gradient_clip_max_norm=gradient_clip_max_norm,
            find_unused_parameters=find_unused_parameters,
            use_amp=use_amp,
        )
    else:
        # Single-GPU mode - create SingleGPUTrainer
        logger.info("Single-GPU mode detected. Creating SingleGPUTrainer")

        trainer = SingleGPUTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            num_classes=num_classes,
            device=device,
            log_dir=log_dir,
            experiment_name=experiment_name,
            callbacks=callbacks,
            gradient_clip_max_norm=gradient_clip_max_norm,
            use_amp=use_amp,
        )

    return trainer


def create_trainer_from_config(cfg, model, train_loader, val_loader, optimizer, scheduler=None):
    """
    Create trainer from a Hydra configuration object.

    This is a convenience function for creating trainers from Hydra configs
    without manually passing all arguments.

    Args:
        cfg: Hydra configuration object (DictConfig).
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler. Optional.

    Returns:
        Trainer instance (SingleGPUTrainer or DDPTrainer).

    Example:
        >>> @hydra.main(config_path="../configs", config_name="config")
        >>> def main(cfg):
        ...     model = build_classifier_from_config(cfg.model)
        ...     train_loader = create_dataloader(cfg.dataloader)
        ...     optimizer = create_optimizer(cfg.optimizer, model)
        ...
        ...     trainer = create_trainer_from_config(
        ...         cfg, model, train_loader, val_loader, optimizer
        ...     )
        ...     trainer.train(num_epochs=cfg.trainer.num_epochs)
    """
    from .losses import get_loss_function
    from .callbacks import (
        CheckpointCallback,
        EarlyStoppingCallback,
        LearningRateLoggerCallback,
        MetricsLoggerCallback,
        MLflowLoggerCallback,
    )
    from torch.utils.tensorboard import SummaryWriter

    # Get training config
    trainer_cfg = cfg.trainer
    dataset_cfg = cfg.dataset

    # Create loss function
    loss_fn = get_loss_function(
        trainer_cfg.get('loss', 'cross_entropy'),
        label_smoothing=trainer_cfg.get('label_smoothing', 0.0)
    )

    # Setup logging directory
    log_dir = trainer_cfg.get('log_dir', None)
    experiment_name = cfg.experiment.get('name', 'experiment')

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    # Create callbacks
    callbacks = []
    monitor_mode = trainer_cfg.get('monitor_mode', trainer_cfg.get('mode', 'min'))
    if monitor_mode not in ['min', 'max']:
        monitor_mode = 'min'

    # Checkpoint callback
    if trainer_cfg.get('save_checkpoints', True):
        checkpoint_callback = CheckpointCallback(
            checkpoint_dir=trainer_cfg.get('checkpoint_dir', './checkpoints'),
            experiment_name=experiment_name,
            save_every=trainer_cfg.get('save_every', 1),
            monitor=trainer_cfg.get('monitor', 'val_loss'),
            mode=monitor_mode
        )
        callbacks.append(checkpoint_callback)

    # Early stopping callback
    early_stopping_cfg = trainer_cfg.get('early_stopping', False)
    if hasattr(early_stopping_cfg, 'get'):
        early_stopping_enabled = early_stopping_cfg.get('enabled', True)
        early_stopping_patience = early_stopping_cfg.get(
            'patience',
            trainer_cfg.get('patience', 10)
        )
        early_stopping_min_delta = early_stopping_cfg.get(
            'min_delta',
            trainer_cfg.get('min_delta', 0.0)
        )
    else:
        early_stopping_enabled = bool(early_stopping_cfg)
        early_stopping_patience = trainer_cfg.get('patience', 10)
        early_stopping_min_delta = trainer_cfg.get('min_delta', 0.0)

    if early_stopping_enabled:
        early_stopping_callback = EarlyStoppingCallback(
            monitor=trainer_cfg.get('monitor', 'val_loss'),
            patience=early_stopping_patience,
            mode=monitor_mode,
            min_delta=early_stopping_min_delta,
        )
        callbacks.append(early_stopping_callback)

    # Learning rate logger
    if writer is not None:
        lr_logger = LearningRateLoggerCallback(writer=writer)
        callbacks.append(lr_logger)

        # Metrics logger
        metrics_logger = MetricsLoggerCallback(writer=writer)
        callbacks.append(metrics_logger)

    # Optional MLflow logger callback
    runtime_mlflow_cfg = cfg.get('runtime', {}).get('mlflow', {})
    mlflow_enabled = trainer_cfg.get(
        'mlflow_enabled',
        runtime_mlflow_cfg.get('enabled', False)
    )
    if mlflow_enabled:
        mlflow_callback = MLflowLoggerCallback(
            experiment_name=runtime_mlflow_cfg.get('experiment_name', experiment_name),
            run_name=experiment_name,
        )
        callbacks.append(mlflow_callback)

    # Create trainer
    device = trainer_cfg.get('device', trainer_cfg.get('gpu_id', None))
    if device is not None:
        if isinstance(device, int):
            device = f"cuda:{device}"
        elif isinstance(device, str):
            normalized_device = device.strip()
            if normalized_device.isdigit():
                device = f"cuda:{normalized_device}"
            elif normalized_device.lower().startswith("gpu:"):
                gpu_idx = normalized_device.split(":", 1)[1]
                if gpu_idx.isdigit():
                    device = f"cuda:{gpu_idx}"

    gradient_clip_max_norm = trainer_cfg.get(
        'gradient_clip_max_norm',
        trainer_cfg.get('gradient_clip_value', None)
    )
    use_amp = bool(trainer_cfg.get('use_amp', False))
    find_unused_parameters = bool(trainer_cfg.get('find_unused_parameters', False))

    trainer = create_trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        num_classes=dataset_cfg.num_classes,
        device=device,
        log_dir=log_dir,
        experiment_name=experiment_name,
        callbacks=callbacks if callbacks else None,
        gradient_clip_max_norm=gradient_clip_max_norm,
        use_amp=use_amp,
        find_unused_parameters=find_unused_parameters,
    )

    return trainer


__all__ = [
    'create_trainer',
    'create_trainer_from_config',
    'SingleGPUTrainer',
    'DDPTrainer'
]
