"""
Training callbacks for managing training workflows.

This module provides callback classes for common training tasks like
checkpointing, early stopping, and learning rate logging.
"""


import logging
from typing import Dict, Optional, Any
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

try:
    import mlflow
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None


logger = logging.getLogger(__name__)


class Callback:
    """
    Base class for all callbacks.

    Callbacks are hooks that get called at specific points during training.
    They allow modular extension of training logic without modifying the trainer.
    """

    def on_train_begin(self, trainer: Any):
        """Run logic at the beginning of training."""
        pass

    def on_train_end(self, trainer: Any):
        """Run logic at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, trainer: Any):
        """Run logic at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Run logic at the end of each epoch."""
        pass

    def on_batch_begin(self, batch_idx: int, trainer: Any):
        """Run logic at the beginning of each batch."""
        pass

    def on_batch_end(self, batch_idx: int, batch_metrics: Dict[str, float], trainer: Any):
        """Run logic at the end of each batch."""
        pass


class CheckpointCallback(Callback):
    """
    Callback for saving model checkpoints during training.

    This callback saves checkpoints at specified intervals and keeps track of
    the best model based on a monitored metric.

    Args:
        checkpoint_dir: Directory where checkpoints will be saved.
        experiment_name: Name of the experiment for checkpoint naming.
        save_every: Save checkpoint every N epochs. Default: 1
        monitor: Metric to monitor for best model. Examples: 'val_loss', 'val_acc'
            Default: 'val_loss'
        mode: 'min' to save when monitor decreases, 'max' when it increases.
            Default: 'min'
        save_best_only: If True, only saves the best checkpoint. Default: False
        verbose: Whether to print checkpoint save messages. Default: True

    Example:
        >>> callback = CheckpointCallback(
        ...     checkpoint_dir="./checkpoints",
        ...     experiment_name="exp1",
        ...     save_every=5,
        ...     monitor='val_loss',
        ...     mode='min'
        ... )
        >>> # In trainer: callback.on_epoch_end(epoch, metrics, trainer)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        experiment_name: str,
        save_every: int = 1,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = False,
        verbose: bool = True
    ):
        """Initialize checkpoint-saving behavior."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.save_every = save_every
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track best metric
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """
        Save checkpoint at the end of epoch if criteria are met.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metrics from this epoch.
            trainer: Trainer instance (must have model, optimizer, scheduler attributes).
        """
        # Check if should save this epoch
        if (epoch + 1) % self.save_every != 0 and not self.save_best_only:
            return

        # Get current metric value
        if self.monitor not in metrics:
            logger.warning(
                f"Monitored metric '{self.monitor}' not found in metrics. "
                f"Available: {list(metrics.keys())}"
            )
            return

        current_metric = metrics[self.monitor]

        # Check if this is the best model
        is_best = False
        if self.mode == 'min':
            is_best = current_metric < self.best_metric
        else:
            is_best = current_metric > self.best_metric

        # Update best metric if improved
        if is_best:
            self.best_metric = current_metric
            self.best_epoch = epoch

        # Save checkpoint if conditions met
        if not self.save_best_only or is_best:
            self._save_checkpoint(epoch, trainer, metrics, is_best)

    def _save_checkpoint(
        self,
        epoch: int,
        trainer: Any,
        metrics: Dict[str, float],
        is_best: bool
    ):
        """Save one checkpoint file to disk."""
        # Handle DDP wrapped models
        model = trainer.model
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'metrics': metrics,
        }

        # Add scheduler if available
        if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()

        # Determine filename
        if is_best:
            checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
        else:
            checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_epoch_{epoch}.pth"

        # Save
        torch.save(checkpoint, checkpoint_path)

        if self.verbose:
            msg = f"Saved checkpoint at epoch {epoch}: {checkpoint_path}"
            if is_best:
                msg += f" (NEW BEST: {self.monitor}={self.best_metric:.4f})"
            logger.info(msg)


class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping when monitored metric stops improving.

    Early stopping prevents overfitting by stopping training when the validation
    metric hasn't improved for a specified number of epochs.

    Args:
        monitor: Metric to monitor. Examples: 'val_loss', 'val_acc'
            Default: 'val_loss'
        patience: Number of epochs with no improvement after which training stops.
            Default: 10
        mode: 'min' to stop when monitor stops decreasing, 'max' when stops increasing.
            Default: 'min'
        min_delta: Minimum change in monitored metric to qualify as improvement.
            Default: 0.0
        verbose: Whether to print early stopping messages. Default: True
        restore_best_weights: Whether to restore model weights from best epoch.
            Default: True

    Example:
        >>> callback = EarlyStoppingCallback(
        ...     monitor='val_loss',
        ...     patience=10,
        ...     mode='min',
        ...     verbose=True
        ... )
        >>> # In trainer: callback.on_epoch_end(epoch, metrics, trainer)
        >>> if callback.should_stop:
        ...     break
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        min_delta: float = 0.0,
        verbose: bool = True,
        restore_best_weights: bool = True
    ):
        """Initialize early-stopping state and thresholds."""
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights

        # Internal state
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        self.best_weights = None

    def on_train_begin(self, trainer: Any):
        """Reset state at the beginning of training."""
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.best_epoch = -1
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        self.best_weights = None

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """
        Check if should stop training at the end of epoch.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metrics from this epoch.
            trainer: Trainer instance.
        """
        # Get current metric
        if self.monitor not in metrics:
            logger.warning(
                f"Monitored metric '{self.monitor}' not found. "
                f"Early stopping disabled."
            )
            return

        current_metric = metrics[self.monitor]

        # Check if improved
        if self.mode == 'min':
            improved = current_metric < (self.best_metric - self.min_delta)
        else:
            improved = current_metric > (self.best_metric + self.min_delta)

        if improved:
            # Improvement found
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.wait = 0

            # Save best weights if requested
            if self.restore_best_weights:
                if hasattr(trainer.model, 'module'):
                    self.best_weights = trainer.model.module.state_dict().copy()
                else:
                    self.best_weights = trainer.model.state_dict().copy()

            if self.verbose:
                logger.info(
                    f"Epoch {epoch}: {self.monitor} improved to {current_metric:.4f}"
                )
        else:
            # No improvement
            self.wait += 1

            if self.verbose:
                logger.info(
                    f"Epoch {epoch}: {self.monitor} did not improve. "
                    f"Patience: {self.wait}/{self.patience}"
                )

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True

                if self.verbose:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best {self.monitor}: {self.best_metric:.4f} at epoch {self.best_epoch}"
                    )

    def on_train_end(self, trainer: Any):
        """Restore best weights if training stopped early."""
        if self.stopped_epoch > 0 and self.verbose:
            logger.info(
                f"Training stopped at epoch {self.stopped_epoch}. "
                f"Best epoch: {self.best_epoch}"
            )

        if self.restore_best_weights and self.best_weights is not None:
            if hasattr(trainer.model, 'module'):
                trainer.model.module.load_state_dict(self.best_weights)
            else:
                trainer.model.load_state_dict(self.best_weights)

            if self.verbose:
                logger.info("Restored best model weights")


class LearningRateLoggerCallback(Callback):
    """
    Callback for logging learning rate during training.

    This callback logs the current learning rate to TensorBoard and/or logger
    at each epoch.

    Args:
        writer: TensorBoard SummaryWriter instance. Optional.
        log_to_console: Whether to log to console via logger. Default: True
        param_group_idx: Which optimizer parameter group to log. Default: 0

    Example:
        >>> writer = SummaryWriter(log_dir="./logs")
        >>> callback = LearningRateLoggerCallback(writer=writer)
        >>> # In trainer: callback.on_epoch_end(epoch, metrics, trainer)
    """

    def __init__(
        self,
        writer: Optional[SummaryWriter] = None,
        log_to_console: bool = True,
        param_group_idx: int = 0
    ):
        """Initialize learning-rate logging targets."""
        self.writer = writer
        self.log_to_console = log_to_console
        self.param_group_idx = param_group_idx

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """
        Log learning rate at the end of epoch.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metrics (not used, for consistency).
            trainer: Trainer instance (must have optimizer attribute).
        """
        # Get current learning rate
        lr = trainer.optimizer.param_groups[self.param_group_idx]['lr']

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Learning_Rate', lr, epoch)

        # Log to console
        if self.log_to_console:
            logger.info(f"Epoch {epoch}: Learning rate = {lr:.6f}")


class MetricsLoggerCallback(Callback):
    """
    Callback for logging metrics to TensorBoard and console.

    This callback logs training and validation metrics at the end of each epoch.

    Args:
        writer: TensorBoard SummaryWriter instance.
        log_to_console: Whether to log to console via logger. Default: True

    Example:
        >>> writer = SummaryWriter(log_dir="./logs")
        >>> callback = MetricsLoggerCallback(writer=writer)
        >>> # In trainer: callback.on_epoch_end(epoch, metrics, trainer)
    """

    def __init__(
        self,
        writer: SummaryWriter,
        log_to_console: bool = True
    ):
        """Initialize metric logging sinks."""
        self.writer = writer
        self.log_to_console = log_to_console

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """
        Log metrics at the end of epoch.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metrics to log.
            trainer: Trainer instance (not used, for consistency).
        """
        # Log to TensorBoard
        for metric_name, metric_value in metrics.items():
            # Determine tag based on metric name
            if metric_name.startswith('train_'):
                tag = f"Train/{metric_name[6:]}"  # Remove 'train_' prefix
            elif metric_name.startswith('val_'):
                tag = f"Validation/{metric_name[4:]}"  # Remove 'val_' prefix
            else:
                tag = metric_name

            self.writer.add_scalar(tag, metric_value, epoch)

        # Log to console
        if self.log_to_console:
            metrics_str = ", ".join(
                f"{k}: {v:.4f}" for k, v in metrics.items()
            )
            logger.info(f"Epoch {epoch}: {metrics_str}")


class MLflowLoggerCallback(Callback):
    """
    Callback for logging key training metrics to MLflow.

    Logs a stable metric contract for Azure ML sweeps and experiment tracking:
    - train_loss
    - val_loss
    - val_accuracy
    - val_balanced_accuracy
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """Initialize optional MLflow tracking settings."""
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.enabled = mlflow is not None
        self.started_run = False

        if not self.enabled:
            logger.warning(
                "MLflowLoggerCallback enabled, but mlflow is not installed. "
                "Skipping MLflow logging."
            )

    def on_train_begin(self, trainer: Any):
        """Start or attach to an MLflow run and log base parameters."""
        if not self.enabled:
            return

        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

        if mlflow.active_run() is None:
            mlflow.start_run(run_name=self.run_name or trainer.experiment_name)
            self.started_run = True

        # Record basic run metadata once.
        mlflow.log_params(
            {
                "experiment_name": trainer.experiment_name,
                "num_classes": trainer.num_classes,
            }
        )

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Log tracked training and validation metrics to MLflow."""
        if not self.enabled:
            return

        tracked = {}
        for key in ("train_loss", "val_loss", "val_accuracy", "val_balanced_accuracy"):
            if key in metrics:
                tracked[key] = float(metrics[key])

        if tracked:
            mlflow.log_metrics(tracked, step=epoch)

    def on_train_end(self, trainer: Any):
        """Leave run management to the caller after training ends."""
        # Keep run open so caller can log final test metrics/artifacts.
        pass


class CallbackList:
    """
    Container for managing multiple callbacks.

    This class aggregates multiple callbacks and calls them in sequence.

    Args:
        callbacks: List of callback instances.

    Example:
        >>> callbacks = CallbackList([
        ...     CheckpointCallback(...),
        ...     EarlyStoppingCallback(...),
        ...     LearningRateLoggerCallback(...)
        ... ])
        >>> callbacks.on_epoch_end(epoch, metrics, trainer)
    """

    def __init__(self, callbacks: list[Callback]):
        """Store callback instances in execution order."""
        self.callbacks = callbacks

    def on_train_begin(self, trainer: Any):
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer: Any):
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(self, epoch: int, trainer: Any):
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, trainer)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, trainer)

    def on_batch_begin(self, batch_idx: int, trainer: Any):
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch_idx, trainer)

    def on_batch_end(self, batch_idx: int, batch_metrics: Dict[str, float], trainer: Any):
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, batch_metrics, trainer)
