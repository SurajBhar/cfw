"""
Single-GPU trainer for classification models.

This module provides the SingleGPUTrainer class for training models on a single GPU.
Supports both baseline (2-element batch) and CFW (4-element batch) dataloader formats.

DDPTrainer (in trainer_ddp.py) extends this class, overriding hook methods to add
distributed capabilities without duplicating the core training logic.
"""


import time
import logging
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ..evaluation.metrics import (
    calculate_balanced_accuracy,
    calculate_accuracy,
    calculate_per_class_recall
)
from ..utils.time_utils import format_time
from .callbacks import CallbackList, Callback


logger = logging.getLogger(__name__)


class SingleGPUTrainer:
    """
    Trainer for single-GPU training with support for baseline and CFW dataloaders.

    This trainer handles:
    - Training and validation loops
    - Both baseline (2-element) and CFW (4-element) batch formats
    - Metric computation (accuracy, balanced accuracy, per-class recall)
    - TensorBoard logging
    - Checkpoint saving via callbacks
    - Progress bars with tqdm

    DDPTrainer extends this class by overriding hook methods (prefixed with _)
    to add distributed training capabilities.

    Args:
        model: Model to train.
        train_dataloader: Training data loader.
        val_dataloader: Validation data loader.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler. Optional.
        loss_fn: Loss function. If None, uses CrossEntropyLoss.
        num_classes: Number of classes in the dataset.
        device: Device to train on ('cuda:0', 'cuda:1', 'cpu', etc.).
            Default: 'cuda' if available, else 'cpu'
        log_dir: Directory for TensorBoard logs. Optional.
        experiment_name: Name of the experiment for logging.
        callbacks: List of callback instances. Optional.
        gradient_clip_max_norm: Maximum gradient norm for clipping.
            If None, no clipping is applied. Default: None

    Example:
        >>> trainer = SingleGPUTrainer(
        ...     model=model,
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     num_classes=10,
        ...     device='cuda:0',
        ...     experiment_name='exp1'
        ... )
        >>> trainer.train(num_epochs=50)
    """

    def __init__(
        self,
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
    ):
        """Initialize a single-device trainer with dataloaders, optimizer, and logging hooks."""
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Move model to device
        self.model = model.to(self.device)

        # Store training components
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.experiment_name = experiment_name
        self.gradient_clip_max_norm = gradient_clip_max_norm
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        if use_amp and not self.use_amp:
            logger.warning("AMP requested but CUDA device is not in use. AMP disabled.")
        self.scaler = self._create_grad_scaler(enabled=True) if self.use_amp else None

        # Loss function
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

        # TensorBoard writer
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        # Callbacks
        if callbacks is not None:
            self.callbacks = CallbackList(callbacks)
        else:
            self.callbacks = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        logger.info(f"SingleGPUTrainer initialized on device: {self.device}")
        logger.info(f"AMP enabled: {self.use_amp}")

    @staticmethod
    def _create_grad_scaler(enabled: bool):
        """Create a GradScaler compatible with the installed PyTorch version."""
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                return torch.amp.GradScaler("cuda", enabled=enabled)
            except TypeError:
                return torch.amp.GradScaler(enabled=enabled)
        return torch.cuda.amp.GradScaler(enabled=enabled)

    def _autocast_context(self):
        """Return autocast context when AMP is enabled, otherwise a no-op context."""
        if not self.use_amp:
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        return torch.cuda.amp.autocast(dtype=torch.float16)

    # ------------------------------------------------------------------
    # Hook methods — override in DDPTrainer for distributed behavior
    # ------------------------------------------------------------------

    def _parse_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse a batch into (images, labels), handling both formats.

        Supports:
        - Baseline format: 2-element (images, labels)
        - CFW format: 4-element (images, weights, labels, paths)
        """
        if len(batch) == 2:
            images, labels = batch
        elif len(batch) == 4:
            images, _, labels, _ = batch
        else:
            raise ValueError(
                f"Unexpected batch format with {len(batch)} elements. "
                f"Expected 2 (baseline) or 4 (CFW) elements."
            )
        return images.to(self.device), labels.to(self.device)

    def _init_epoch_accumulators(self):
        """Initialize loss/sample accumulators for an epoch.

        Override in DDPTrainer to use GPU tensors for all_reduce.
        """
        return 0.0, 0

    def _accumulate_loss(self, running_loss, loss_value, batch_size):
        """Accumulate loss value. Override in DDPTrainer for GPU tensor ops."""
        return running_loss + loss_value * batch_size

    def _accumulate_samples(self, num_samples, batch_size):
        """Accumulate sample count. Override in DDPTrainer for GPU tensor ops."""
        return num_samples + batch_size

    def _collect_predictions(self, predictions, labels, all_predictions, all_labels):
        """Collect predictions and labels. Override in DDPTrainer to keep on GPU."""
        all_predictions.append(predictions.detach().cpu())
        all_labels.append(labels.detach().cpu())

    def _compute_epoch_metrics(
        self, running_loss, num_samples, all_predictions, all_labels, prefix: str
    ) -> Dict[str, float]:
        """Compute metrics from accumulated predictions.

        Override in DDPTrainer to gather from all ranks before computing.

        Args:
            running_loss: Accumulated loss (scalar or tensor).
            num_samples: Total sample count (scalar or tensor).
            all_predictions: List of prediction tensors.
            all_labels: List of label tensors.
            prefix: Metric key prefix ('train' or 'val').

        Returns:
            Dictionary of metrics.
        """
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        avg_loss = running_loss / num_samples

        accuracy = calculate_accuracy(all_predictions, all_labels)
        balanced_accuracy = calculate_balanced_accuracy(
            all_predictions, all_labels, self.num_classes
        )

        metrics = {
            f'{prefix}_loss': avg_loss,
            f'{prefix}_accuracy': accuracy,
            f'{prefix}_balanced_accuracy': balanced_accuracy,
        }

        if prefix == 'val':
            per_class_recall = calculate_per_class_recall(
                all_predictions, all_labels, self.num_classes
            )
            for class_idx, recall in per_class_recall.items():
                metrics[f'val_recall_class_{class_idx}'] = recall

        return metrics

    def _create_progress_bar(self, dataloader, desc: str):
        """Create a progress bar. Override in DDPTrainer for rank-0 only."""
        return tqdm(dataloader, desc=desc, leave=False)

    def _update_progress_bar(self, progress_bar, postfix: dict):
        """Update progress bar postfix. Override in DDPTrainer for rank guard."""
        progress_bar.set_postfix(postfix)

    def _on_epoch_start(self, epoch: int):
        """Run hook logic at the start of each epoch.

        Override in DDPTrainer for sampler.set_epoch().
        """
        pass

    def _on_epoch_end_logging(self, epoch: int, all_metrics: Dict[str, float]):
        """Log metrics and invoke callbacks at epoch end.

        Override in DDPTrainer for rank-0 guard and barrier.
        """
        # Log to TensorBoard
        if self.writer is not None:
            for metric_name, metric_value in all_metrics.items():
                if metric_name.startswith('train_'):
                    tag = f"Train/{metric_name[6:]}"
                elif metric_name.startswith('val_'):
                    tag = f"Validation/{metric_name[4:]}"
                else:
                    tag = metric_name
                self.writer.add_scalar(tag, metric_value, epoch)

            if self.scheduler is not None:
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', lr, epoch)

        # Call epoch end callbacks
        if self.callbacks is not None:
            self.callbacks.on_epoch_end(epoch, all_metrics, self)

    def _on_training_end(self, total_start_time: float):
        """Cleanup at end of training.

        Override in DDPTrainer for rank-0 guard and barrier.
        """
        if self.callbacks is not None:
            self.callbacks.on_train_end(self)

        if self.writer is not None:
            self.writer.close()

        total_time = time.time() - total_start_time
        logger.info(f"Training completed. Total time: {format_time(total_time)}")

    # ------------------------------------------------------------------
    # Public training methods
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        """
        Execute one training epoch.

        Handles both baseline (2-element) and CFW (4-element) batch formats:
        - Baseline format: (images, labels)
        - CFW format: (images, weights, labels, paths)

        Returns:
            Dictionary of training metrics including:
            - 'train_loss': Average training loss
            - 'train_accuracy': Training accuracy
            - 'train_balanced_accuracy': Training balanced accuracy
        """
        start_time = time.time()

        self.model.train()

        running_loss, num_samples = self._init_epoch_accumulators()
        all_predictions = []
        all_labels = []

        progress_bar = self._create_progress_bar(
            self.train_dataloader, f"Epoch {self.current_epoch} [Train]"
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Call batch begin callbacks
            if self.callbacks is not None:
                self.callbacks.on_batch_begin(batch_idx, self)

            images, labels = self._parse_batch(batch)

            # Forward + backward + optimize
            self.optimizer.zero_grad()
            with self._autocast_context():
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()

                if self.gradient_clip_max_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.gradient_clip_max_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                if self.gradient_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.gradient_clip_max_norm
                    )

                self.optimizer.step()

            # Accumulate metrics
            running_loss = self._accumulate_loss(running_loss, loss.item(), images.size(0))
            num_samples = self._accumulate_samples(num_samples, images.size(0))

            predictions = logits.argmax(dim=1)
            self._collect_predictions(predictions, labels, all_predictions, all_labels)

            self._update_progress_bar(progress_bar, {'loss': loss.item()})

            # Call batch end callbacks
            if self.callbacks is not None:
                batch_metrics = {'batch_loss': loss.item()}
                self.callbacks.on_batch_end(batch_idx, batch_metrics, self)

            self.global_step += 1

        if self.scheduler is not None:
            self.scheduler.step()

        metrics = self._compute_epoch_metrics(
            running_loss, num_samples, all_predictions, all_labels, 'train'
        )
        metrics['train_time'] = time.time() - start_time

        logger.info(
            f"Epoch {self.current_epoch} [Train]: "
            f"Loss={metrics.get('train_loss', 0):.4f}, "
            f"Acc={metrics.get('train_accuracy', 0):.4f}, "
            f"BalAcc={metrics.get('train_balanced_accuracy', 0):.4f}, "
            f"Time={format_time(metrics['train_time'])}"
        )

        return metrics

    def validate_epoch(self) -> Dict[str, float]:
        """
        Execute one validation epoch.

        Handles both baseline (2-element) and CFW (4-element) batch formats.

        Returns:
            Dictionary of validation metrics including:
            - 'val_loss': Average validation loss
            - 'val_accuracy': Validation accuracy
            - 'val_balanced_accuracy': Validation balanced accuracy
        """
        start_time = time.time()

        self.model.eval()

        running_loss, num_samples = self._init_epoch_accumulators()
        all_predictions = []
        all_labels = []

        progress_bar = self._create_progress_bar(
            self.val_dataloader, f"Epoch {self.current_epoch} [Val]"
        )

        with torch.no_grad():
            for batch in progress_bar:
                images, labels = self._parse_batch(batch)

                with self._autocast_context():
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)

                running_loss = self._accumulate_loss(running_loss, loss.item(), images.size(0))
                num_samples = self._accumulate_samples(num_samples, images.size(0))

                predictions = logits.argmax(dim=1)
                self._collect_predictions(predictions, labels, all_predictions, all_labels)

                self._update_progress_bar(progress_bar, {'loss': loss.item()})

        metrics = self._compute_epoch_metrics(
            running_loss, num_samples, all_predictions, all_labels, 'val'
        )
        metrics['val_time'] = time.time() - start_time

        logger.info(
            f"Epoch {self.current_epoch} [Val]: "
            f"Loss={metrics.get('val_loss', 0):.4f}, "
            f"Acc={metrics.get('val_accuracy', 0):.4f}, "
            f"BalAcc={metrics.get('val_balanced_accuracy', 0):.4f}, "
            f"Time={format_time(metrics['val_time'])}"
        )

        return metrics

    def train(
        self,
        num_epochs: int,
        start_epoch: int = 0
    ):
        """
        Execute full training loop for specified number of epochs.

        Args:
            num_epochs: Total number of epochs to train for.
            start_epoch: Starting epoch number (for resuming). Default: 0

        Example:
            >>> trainer.train(num_epochs=50)
            >>> # To resume from epoch 25:
            >>> trainer.train(num_epochs=50, start_epoch=25)
        """
        total_start_time = time.time()

        # Call training begin callbacks
        if self.callbacks is not None:
            self.callbacks.on_train_begin(self)

        try:
            for epoch in range(start_epoch, num_epochs):
                self.current_epoch = epoch

                # Hook for DDP sampler.set_epoch() etc.
                self._on_epoch_start(epoch)

                # Call epoch begin callbacks
                if self.callbacks is not None:
                    self.callbacks.on_epoch_begin(epoch, self)

                # Training step
                train_metrics = self.train_epoch()

                # Validation step
                val_metrics = self.validate_epoch()

                # Combine metrics
                all_metrics = {**train_metrics, **val_metrics}

                # Logging and callbacks (gated on rank-0 in DDP)
                self._on_epoch_end_logging(epoch, all_metrics)

                # Check for early stopping
                if self.callbacks is not None:
                    for callback in self.callbacks.callbacks:
                        if hasattr(callback, 'should_stop') and callback.should_stop:
                            logger.info(f"Early stopping triggered at epoch {epoch}")
                            break

        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

        finally:
            self._on_training_end(total_start_time)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on a given dataloader.

        This is useful for testing on a separate test set after training.

        Args:
            dataloader: Data loader to evaluate on.

        Returns:
            Dictionary of evaluation metrics.

        Example:
            >>> test_metrics = trainer.evaluate(test_dataloader)
            >>> print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        """
        self.model.eval()

        running_loss = 0.0
        num_samples = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                images, labels = self._parse_batch(batch)

                with self._autocast_context():
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)

                running_loss += loss.item() * images.size(0)
                num_samples += images.size(0)

                predictions = logits.argmax(dim=1)
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

        avg_loss = running_loss / num_samples
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        accuracy = calculate_accuracy(all_predictions, all_labels)
        balanced_accuracy = calculate_balanced_accuracy(
            all_predictions, all_labels, self.num_classes
        )
        per_class_recall = calculate_per_class_recall(
            all_predictions, all_labels, self.num_classes
        )

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy
        }

        for class_idx, recall in per_class_recall.items():
            metrics[f'recall_class_{class_idx}'] = recall

        logger.info(
            f"Evaluation: Loss={avg_loss:.4f}, Acc={accuracy:.4f}, "
            f"BalAcc={balanced_accuracy:.4f}"
        )

        return metrics
