"""Checkpoint management utilities for saving and loading model states.

This module provides the CheckpointManager class for handling model checkpoints,
optimizer states, and training metadata. Supports both single-GPU and DDP training.
"""


import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoint saving and loading with DDP support.

    This class handles:
    - Saving model, optimizer, and scheduler states
    - Loading checkpoints for resuming training
    - Proper handling of DDP wrapped models (model.module.state_dict())
    - Tracking best model based on metrics
    - Automatic checkpoint directory creation

    Attributes:
        checkpoint_dir: Directory where checkpoints are saved.
        experiment_name: Name of the experiment for checkpoint naming.
        save_every: Save checkpoint every N epochs. Default: 1
        keep_last_n: Number of recent checkpoints to keep. None = keep all.

    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir="./checkpoints",
        ...     experiment_name="exp1",
        ...     save_every=5
        ... )
        >>> manager.save_checkpoint(
        ...     epoch=10,
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     metrics={'val_loss': 0.5, 'val_acc': 0.85}
        ... )
        >>> checkpoint = manager.load_checkpoint(checkpoint_path)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        experiment_name: str,
        save_every: int = 1,
        keep_last_n: Optional[int] = None
    ):
        """
        Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory where checkpoints will be saved.
            experiment_name: Name of the experiment for checkpoint naming.
            save_every: Save checkpoint every N epochs. Default: 1
            keep_last_n: Number of recent checkpoints to keep. If None, keeps all.
                Default: None
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.save_every = save_every
        self.keep_last_n = keep_last_n

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """
        Save a training checkpoint.

        Handles DDP wrapped models automatically by checking for 'module' attribute.

        Args:
            epoch: Current training epoch.
            model: Model to save (can be DDP wrapped).
            optimizer: Optimizer to save.
            scheduler: Learning rate scheduler to save. Optional.
            metrics: Dictionary of metrics to save (e.g., {'val_loss': 0.5}).
            extra_state: Additional state to save. Optional.
            is_best: Whether this is the best checkpoint so far. Default: False

        Returns:
            Path to the saved checkpoint file.

        Example:
            >>> path = manager.save_checkpoint(
            ...     epoch=10,
            ...     model=model,
            ...     optimizer=optimizer,
            ...     metrics={'val_loss': 0.5}
            ... )
            >>> print(f"Saved checkpoint: {path}")
        """
        # Check if should save this epoch
        if (epoch + 1) % self.save_every != 0 and not is_best:
            return ""

        # Handle DDP wrapped models
        if hasattr(model, 'module'):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        # Prepare checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
        }

        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Add metrics if provided
        if metrics is not None:
            checkpoint['metrics'] = metrics

        # Add extra state if provided
        if extra_state is not None:
            checkpoint['extra_state'] = extra_state

        # Determine checkpoint filename
        if is_best:
            checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
        else:
            checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_epoch_{epoch}.pth"

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint at epoch {epoch}: {checkpoint_path}")

        # Clean up old checkpoints if keep_last_n is set
        if self.keep_last_n is not None and not is_best:
            self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore model/optimizer/scheduler states.

        Args:
            checkpoint_path: Path to the checkpoint file.
            model: Model to load state into (can be DDP wrapped).
            optimizer: Optimizer to load state into. Optional.
            scheduler: Scheduler to load state into. Optional.
            map_location: Device to map tensors to (e.g., 'cpu', 'cuda:0').
                If None, uses current device. Default: None

        Returns:
            Dictionary containing checkpoint metadata (epoch, metrics, extra_state).

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            RuntimeError: If checkpoint loading fails.

        Example:
            >>> metadata = manager.load_checkpoint(
            ...     checkpoint_path="./checkpoints/exp1_epoch_10.pth",
            ...     model=model,
            ...     optimizer=optimizer
            ... )
            >>> start_epoch = metadata['epoch'] + 1
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Load model state
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Extract metadata
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'extra_state': checkpoint.get('extra_state', {})
        }

        logger.info(f"Successfully loaded checkpoint from epoch {metadata['epoch']}")

        return metadata

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the most recent checkpoint.

        Returns:
            Path to the latest checkpoint file, or None if no checkpoints exist.

        Example:
            >>> latest = manager.get_latest_checkpoint()
            >>> if latest:
            ...     manager.load_checkpoint(latest, model, optimizer)
        """
        # Find all checkpoint files
        checkpoints = list(self.checkpoint_dir.glob(f"{self.experiment_name}_epoch_*.pth"))

        if not checkpoints:
            return None

        # Sort by epoch number (extract from filename)
        checkpoints.sort(key=lambda p: int(p.stem.split('_epoch_')[-1]))

        return str(checkpoints[-1])

    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get the path to the best checkpoint.

        Returns:
            Path to the best checkpoint file, or None if it doesn't exist.

        Example:
            >>> best = manager.get_best_checkpoint()
            >>> if best:
            ...     manager.load_checkpoint(best, model)
        """
        best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"

        if best_path.exists():
            return str(best_path)

        return None

    def _cleanup_old_checkpoints(self):
        """
        Remove old checkpoints, keeping only the last N.

        This method is called automatically when keep_last_n is set.
        """
        # Find all checkpoint files (excluding best)
        checkpoints = list(self.checkpoint_dir.glob(f"{self.experiment_name}_epoch_*.pth"))

        if len(checkpoints) <= self.keep_last_n:
            return

        # Sort by epoch number
        checkpoints.sort(key=lambda p: int(p.stem.split('_epoch_')[-1]))

        # Remove old checkpoints
        for checkpoint in checkpoints[:-self.keep_last_n]:
            checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {checkpoint}")

    def list_checkpoints(self) -> list[str]:
        """
        List all available checkpoints for this experiment.

        Returns:
            List of checkpoint file paths, sorted by epoch.

        Example:
            >>> checkpoints = manager.list_checkpoints()
            >>> for ckpt in checkpoints:
            ...     print(ckpt)
        """
        checkpoints = list(self.checkpoint_dir.glob(f"{self.experiment_name}_*.pth"))
        checkpoints.sort()

        return [str(p) for p in checkpoints]


def save_checkpoint(
    checkpoint_path: str,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    metrics: Optional[Dict[str, float]] = None
):
    """
    Standalone function to save a checkpoint.

    This is a convenience function for simple checkpoint saving without
    using the CheckpointManager class.

    Args:
        checkpoint_path: Path where checkpoint will be saved.
        epoch: Current training epoch.
        model: Model to save (can be DDP wrapped).
        optimizer: Optimizer to save.
        scheduler: Learning rate scheduler to save. Optional.
        metrics: Dictionary of metrics to save. Optional.

    Example:
        >>> save_checkpoint(
        ...     checkpoint_path="./checkpoints/model_epoch_10.pth",
        ...     epoch=10,
        ...     model=model,
        ...     optimizer=optimizer,
        ...     metrics={'val_loss': 0.5}
        ... )
    """
    # Handle DDP wrapped models
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if metrics is not None:
        checkpoint['metrics'] = metrics

    # Save
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Standalone function to load a checkpoint.

    This is a convenience function for simple checkpoint loading without
    using the CheckpointManager class.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load state into (can be DDP wrapped).
        optimizer: Optimizer to load state into. Optional.
        scheduler: Scheduler to load state into. Optional.
        map_location: Device to map tensors to. Default: None

    Returns:
        Dictionary containing checkpoint metadata.

    Example:
        >>> metadata = load_checkpoint(
        ...     checkpoint_path="./checkpoints/model_epoch_10.pth",
        ...     model=model,
        ...     optimizer=optimizer
        ... )
        >>> start_epoch = metadata['epoch'] + 1
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Load model state
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'extra_state': checkpoint.get('extra_state', {})
    }
