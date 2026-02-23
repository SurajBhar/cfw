"""
Ray Trainable wrapper for CFW training pipeline.

This module provides a Ray Tune Trainable class that wraps the existing
CFW training infrastructure, enabling hyperparameter optimization with
Ray Tune's scheduling algorithms (BOHB, ASHA, etc.).
"""


import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from ray import tune
from omegaconf import DictConfig, OmegaConf

from cfw.models.builders import build_classifier_from_config
from cfw.data.dataloaders import create_dataloader_from_config
from cfw.optimization.optimizers import create_optimizer
from cfw.optimization.schedulers import create_scheduler
from cfw.evaluation.metrics import calculate_balanced_accuracy

from .utils import merge_configs, load_base_config


class CFWTrainable(tune.Trainable):
    """
    Ray Trainable wrapper for CFW training pipeline.

    This class wraps the existing CFW training infrastructure
    to work with Ray Tune for hyperparameter optimization.

    The Trainable receives hyperparameters from Ray Tune's search algorithm,
    merges them with a base configuration, and runs training epochs.
    Metrics are reported back to Ray Tune for scheduling decisions.

    Expected config keys:
        - base_config_path (str): Path to base Hydra config YAML file
        - learning_rate (float, optional): Learning rate override
        - weight_decay (float, optional): Weight decay override
        - optimizer (str, optional): Optimizer name override
        - scheduler (str, optional): Scheduler name override
        - batch_size (int, optional): Batch size override
        - dropout (float, optional): Dropout rate override

    Reported metrics:
        - train_loss: Training loss for the epoch
        - train_accuracy: Training accuracy (%)
        - val_loss: Validation loss
        - val_accuracy: Validation accuracy (%)
        - val_balanced_accuracy: Validation balanced accuracy (primary metric)
    """

    def setup(self, config: Dict[str, Any]) -> None:
        """
        Initialize training components from Ray config.

        This method is called once when the trial starts. It:
        1. Loads the base configuration from file
        2. Merges trial-specific hyperparameters
        3. Creates model, dataloaders, optimizer, and scheduler

        Args:
            config: Dictionary containing hyperparameters and base_config_path
        """
        # Load and merge configurations
        base_config_path = config.get('base_config_path')
        if base_config_path is None:
            raise ValueError("config must contain 'base_config_path'")

        base_cfg = load_base_config(base_config_path)
        self.cfg = merge_configs(base_cfg, config)

        # Store hyperparameters for logging
        self.trial_config = {
            k: v for k, v in config.items()
            if k != 'base_config_path' and not str(k).startswith('_')
        }

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model
        self.model = build_classifier_from_config(self.cfg)
        self.model = self.model.to(self.device)

        # Create dataloaders
        self.train_loader = create_dataloader_from_config(self.cfg, split="train")
        self.val_loader = create_dataloader_from_config(self.cfg, split="val")

        # Get number of classes from dataset
        self.num_classes = self.cfg.model.num_classes

        # Create optimizer with trial hyperparameters
        # The merge_configs already updated cfg.optimizer with trial values
        self.optimizer = create_optimizer(
            self.model.parameters(),
            self.cfg.optimizer
        )

        # Create scheduler
        scheduler_start_lr = self.cfg.scheduler.get('start_lr', self.cfg.optimizer.lr)
        scheduler_end_lr = self.cfg.scheduler.get('end_lr', None)
        self.scheduler = create_scheduler(
            self.optimizer,
            self.cfg.scheduler,
            num_epochs=self.cfg.trainer.num_epochs,
            initial_lr=scheduler_start_lr,
            end_lr=scheduler_end_lr
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Epoch counter
        self.current_epoch = 0
        self.max_iterations = int(config.get("_max_iterations", 0) or 0)

    def step(self) -> Dict[str, Any]:
        """
        Execute one training epoch and return metrics.

        This method is called by Ray Tune's scheduler to advance training.
        It runs one epoch of training, validates, and returns metrics
        that Ray Tune uses for scheduling decisions.

        Returns:
            Dictionary containing training and validation metrics
        """
        # Train one epoch
        train_metrics = self._train_epoch()

        # Validate
        val_metrics = self._validate()

        # Step scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        self.current_epoch += 1

        result: Dict[str, Any] = {
            "training_iteration": self.current_epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        }
        if self.max_iterations > 0 and self.current_epoch >= self.max_iterations:
            # Defensive cap in case scheduler/stop config does not terminate the trial.
            result["done"] = True

        return result

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """
        Save model and optimizer state for checkpointing.

        Args:
            checkpoint_dir: Directory to save checkpoint

        Returns:
            Checkpoint directory path expected by Ray Trainable API
        """
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "trial_config": self.trial_config,
        }, checkpoint_path)
        # Ray expects class Trainables to return either None or the same
        # checkpoint_dir argument passed into save_checkpoint.
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Restore training state from checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint
        """
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.current_epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def _train_epoch(self) -> Dict[str, float]:
        """
        Run single training epoch.

        Returns:
            Dictionary with 'loss' and 'accuracy' metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(self.train_loader):
            inputs, targets = self._parse_batch(batch)
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

    def _validate(self) -> Dict[str, float]:
        """
        Run validation epoch.

        Returns:
            Dictionary with 'loss', 'accuracy', and 'balanced_accuracy' metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = self._parse_batch(batch)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)

                all_preds.append(predicted)
                all_targets.append(targets)

        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        correct = all_preds.eq(all_targets).sum().item()
        accuracy = 100.0 * correct / all_targets.size(0)
        balanced_acc = calculate_balanced_accuracy(
            all_preds, all_targets, self.num_classes
        )

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
        }

    @staticmethod
    def _parse_batch(batch):
        """Support both baseline and CFW dataloader batch formats."""
        if len(batch) == 2:
            return batch
        if len(batch) == 4:
            inputs, _, targets, _ = batch
            return inputs, targets

        raise ValueError(
            f"Unsupported batch format with {len(batch)} elements. "
            "Expected 2 (baseline) or 4 (CFW)."
        )


def create_trainable_with_resources(
    cpu: int = 4,
    gpu: int = 1
) -> type:
    """
    Create a CFWTrainable class with resource specifications.

    This is a convenience function for creating a Trainable with
    specific resource requirements for Ray Tune.

    Args:
        cpu: Number of CPUs per trial
        gpu: Number of GPUs per trial

    Returns:
        CFWTrainable class wrapped with resource specifications
    """
    return tune.with_resources(
        CFWTrainable,
        resources={"cpu": cpu, "gpu": gpu}
    )
