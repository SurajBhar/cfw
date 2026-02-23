"""
DDP (Distributed Data Parallel) trainer for multi-GPU training.

This module provides the DDPTrainer class, which extends SingleGPUTrainer
with distributed training capabilities using PyTorch's DistributedDataParallel.

DDPTrainer overrides hook methods from SingleGPUTrainer to add:
- DDP model wrapping
- Distributed metric gathering (all_reduce, all_gather)
- Rank-0 gated logging, callbacks, and checkpointing
- Epoch synchronization via barriers
"""


import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..evaluation.metrics import (
    calculate_balanced_accuracy,
    calculate_accuracy,
    calculate_per_class_recall
)
from ..utils.distributed import is_main_process, barrier
from .callbacks import Callback
from .trainer_single_gpu import SingleGPUTrainer


logger = logging.getLogger(__name__)


class DDPTrainer(SingleGPUTrainer):
    """
    Trainer for multi-GPU distributed training using DistributedDataParallel.

    Extends SingleGPUTrainer by overriding hook methods to add distributed
    capabilities. All core training logic (forward/backward pass, optimizer
    step, metric computation functions) is inherited from SingleGPUTrainer.

    This ensures metric consistency between single-GPU and multi-GPU training:
    both use the same calculate_accuracy, calculate_balanced_accuracy, and
    calculate_per_class_recall functions. DDPTrainer gathers predictions from
    all ranks before computing metrics, so results are identical regardless
    of GPU count.

    Args:
        model: Model to train (will be wrapped with DDP).
        train_dataloader: Training data loader (should use DistributedSampler
            or DistributedWeightedSampler).
        val_dataloader: Validation data loader.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler. Optional.
        loss_fn: Loss function. If None, uses CrossEntropyLoss.
        num_classes: Number of classes in the dataset.
        rank: Current process rank (0 to world_size-1).
        world_size: Total number of processes.
        log_dir: Directory for TensorBoard logs (only used on rank 0). Optional.
        experiment_name: Name of the experiment for logging.
        callbacks: List of callback instances. Optional.
        gradient_clip_max_norm: Maximum gradient norm for clipping.
            Default: 1.0
        find_unused_parameters: Whether to find unused parameters in DDP.
            Default: False

    Example:
        >>> # Typically called from a distributed launcher like torchrun
        >>> def train_worker(rank, world_size):
        ...     setup_ddp(rank, world_size)
        ...     trainer = DDPTrainer(
        ...         model=model,
        ...         train_dataloader=train_loader,
        ...         val_dataloader=val_loader,
        ...         optimizer=optimizer,
        ...         num_classes=10,
        ...         rank=rank,
        ...         world_size=world_size,
        ...     )
        ...     trainer.train(num_epochs=50)
        ...     cleanup_ddp()
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
        rank: int = 0,
        world_size: int = 1,
        log_dir: Optional[str] = None,
        experiment_name: str = "experiment",
        callbacks: Optional[list[Callback]] = None,
        gradient_clip_max_norm: float = 1.0,
        find_unused_parameters: bool = False,
        use_amp: bool = False,
    ):
        """Initialize a DDP trainer instance and wrap the model in `DistributedDataParallel`."""
        self.rank = rank
        self.world_size = world_size

        # Use local rank for device placement on multi-node jobs.
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)

        # Gate callbacks and writer on rank 0
        effective_callbacks = callbacks if is_main_process() else None
        effective_log_dir = log_dir if is_main_process() else None

        # Call base class init (moves model to device, sets up everything)
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            num_classes=num_classes,
            device=device,
            log_dir=effective_log_dir,
            experiment_name=experiment_name,
            callbacks=effective_callbacks,
            gradient_clip_max_norm=gradient_clip_max_norm,
            use_amp=use_amp,
        )

        # Wrap model with DDP after base __init__ moves it to device
        self.model = DDP(
            self.model,
            device_ids=[local_rank],
            find_unused_parameters=find_unused_parameters
        )

        if is_main_process():
            logger.info(
                f"DDPTrainer initialized on rank {rank}/{world_size} "
                f"(device: {self.device})"
            )

    # ------------------------------------------------------------------
    # Override hook methods for distributed behavior
    # ------------------------------------------------------------------

    def _init_epoch_accumulators(self):
        """Use GPU tensors for distributed all_reduce."""
        return (
            torch.tensor(0.0, device=self.device),
            torch.tensor(0, device=self.device),
        )

    def _accumulate_loss(self, running_loss, loss_value, batch_size):
        """Accumulate on GPU tensor."""
        running_loss += loss_value * batch_size
        return running_loss

    def _accumulate_samples(self, num_samples, batch_size):
        """Accumulate on GPU tensor."""
        num_samples += batch_size
        return num_samples

    def _collect_predictions(self, predictions, labels, all_predictions, all_labels):
        """Keep predictions on GPU for later all_gather."""
        all_predictions.append(predictions.detach())
        all_labels.append(labels.detach())

    def _compute_epoch_metrics(
        self, running_loss, num_samples, all_predictions, all_labels, prefix: str
    ) -> Dict[str, float]:
        """Gather metrics from all ranks, then compute on rank 0.

        Uses dist.all_reduce for loss/samples and dist.all_gather for
        predictions/labels to ensure globally accurate metrics.
        """
        # Reduce loss and sample counts across all ranks
        running_loss = running_loss.float()
        num_samples = num_samples.float()
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)

        # Gather predictions and labels from all ranks
        all_predictions_cat = torch.cat(all_predictions)
        all_labels_cat = torch.cat(all_labels)

        gathered_predictions = [
            torch.zeros_like(all_predictions_cat) for _ in range(self.world_size)
        ]
        gathered_labels = [
            torch.zeros_like(all_labels_cat) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_predictions, all_predictions_cat)
        dist.all_gather(gathered_labels, all_labels_cat)

        if is_main_process():
            # Compute metrics on rank 0 using the same functions as SingleGPUTrainer
            all_preds = torch.cat(gathered_predictions)
            all_labs = torch.cat(gathered_labels)

            avg_loss = running_loss.item() / num_samples.item()
            accuracy = calculate_accuracy(all_preds, all_labs)
            balanced_accuracy = calculate_balanced_accuracy(
                all_preds, all_labs, self.num_classes
            )

            metrics = {
                f'{prefix}_loss': avg_loss,
                f'{prefix}_accuracy': accuracy,
                f'{prefix}_balanced_accuracy': balanced_accuracy,
            }

            if prefix == 'val':
                per_class_recall = calculate_per_class_recall(
                    all_preds, all_labs, self.num_classes
                )
                for class_idx, recall in per_class_recall.items():
                    metrics[f'val_recall_class_{class_idx}'] = recall

            return metrics
        else:
            return {}

    def _create_progress_bar(self, dataloader, desc: str):
        """Only show progress bar on rank 0."""
        if is_main_process():
            return tqdm(dataloader, desc=desc, leave=False)
        return dataloader

    def _update_progress_bar(self, progress_bar, postfix: dict):
        """Only update progress bar on rank 0."""
        if is_main_process() and hasattr(progress_bar, 'set_postfix'):
            progress_bar.set_postfix(postfix)

    def _on_epoch_start(self, epoch: int):
        """Set epoch on distributed samplers for reproducibility."""
        if hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(epoch)
        elif is_main_process() and epoch == 0:
            logger.warning(
                "Train sampler does not have set_epoch() method. "
                "This may cause issues with distributed training reproducibility. "
                "Use DistributedSampler or DistributedWeightedSampler instead."
            )

        if hasattr(self.val_dataloader.sampler, 'set_epoch'):
            self.val_dataloader.sampler.set_epoch(epoch)

    def _on_epoch_end_logging(self, epoch: int, all_metrics: Dict[str, float]):
        """Log and invoke callbacks on rank 0 only, then synchronize."""
        if is_main_process():
            super()._on_epoch_end_logging(epoch, all_metrics)
        barrier()

    def _on_training_end(self, total_start_time: float):
        """Cleanup on rank 0 only, then synchronize."""
        if is_main_process():
            super()._on_training_end(total_start_time)
        barrier()

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on a dataloader with distributed metric aggregation.

        All ranks must call this method when running in distributed mode.
        Loss is reduced across ranks; predictions/labels are gathered and
        metrics are computed on rank 0.
        """
        self.model.eval()

        running_loss = torch.tensor(0.0, device=self.device)
        num_samples = torch.tensor(0.0, device=self.device)
        local_predictions: list[torch.Tensor] = []
        local_labels: list[torch.Tensor] = []

        progress_bar = self._create_progress_bar(dataloader, desc="Evaluating")
        with torch.no_grad():
            for batch in progress_bar:
                images, labels = self._parse_batch(batch)

                with self._autocast_context():
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)

                batch_size = float(images.size(0))
                running_loss += loss.detach() * batch_size
                num_samples += batch_size

                predictions = logits.argmax(dim=1)
                local_predictions.append(predictions.detach().cpu())
                local_labels.append(labels.detach().cpu())

                self._update_progress_bar(
                    progress_bar, {'eval_loss': f'{loss.item():.4f}'}
                )

        if hasattr(progress_bar, 'close'):
            progress_bar.close()

        # Reduce loss/sample counts across all ranks.
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)

        local_predictions_cat = (
            torch.cat(local_predictions)
            if local_predictions
            else torch.empty(0, dtype=torch.long)
        )
        local_labels_cat = (
            torch.cat(local_labels)
            if local_labels
            else torch.empty(0, dtype=torch.long)
        )

        # all_gather_object supports variable-length tensors per rank.
        gathered_predictions: list[Optional[torch.Tensor]] = [None] * self.world_size
        gathered_labels: list[Optional[torch.Tensor]] = [None] * self.world_size
        dist.all_gather_object(gathered_predictions, local_predictions_cat)
        dist.all_gather_object(gathered_labels, local_labels_cat)

        if not is_main_process():
            return {}

        total_samples = float(num_samples.item())
        if total_samples <= 0:
            logger.warning("Evaluation skipped metric computation: no samples found.")
            return {
                'loss': 0.0,
                'accuracy': 0.0,
                'balanced_accuracy': 0.0,
            }

        all_predictions = torch.cat([pred for pred in gathered_predictions if pred is not None])
        all_labels = torch.cat([lab for lab in gathered_labels if lab is not None])

        avg_loss = float(running_loss.item() / total_samples)
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
            'balanced_accuracy': balanced_accuracy,
        }
        for class_idx, recall in per_class_recall.items():
            metrics[f'recall_class_{class_idx}'] = recall

        logger.info(
            f"Evaluation: Loss={avg_loss:.4f}, Acc={accuracy:.4f}, "
            f"BalAcc={balanced_accuracy:.4f}"
        )

        return metrics
