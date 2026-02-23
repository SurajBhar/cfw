"""
Model evaluation utilities for computing metrics and generating reports.

This module provides the Evaluator class for comprehensive model evaluation
including metrics computation, confusion matrix generation, and results saving.
"""


import logging
from typing import Dict, Optional, Any, List
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from .metrics import (
    calculate_balanced_accuracy,
    calculate_accuracy,
    calculate_per_class_recall,
    calculate_confusion_matrix,
    calculate_precision_recall_f1
)


logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for comprehensive model evaluation.

    This class handles:
    - Evaluation on test/validation sets
    - Metrics computation (accuracy, balanced accuracy, per-class metrics)
    - Confusion matrix generation
    - Results saving to disk
    - Support for both baseline and CFW dataloader formats

    Args:
        model: Model to evaluate.
        num_classes: Number of classes in the dataset.
        device: Device to evaluate on ('cuda', 'cpu', etc.).
            Default: 'cuda' if available, else 'cpu'
        loss_fn: Loss function for computing test loss. Optional.

    Example:
        >>> evaluator = Evaluator(
        ...     model=model,
        ...     num_classes=10,
        ...     device='cuda'
        ... )
        >>> results = evaluator.evaluate(test_loader)
        >>> evaluator.save_results(results, save_dir="./results")
        >>> print(f"Test accuracy: {results['accuracy']:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        device: Optional[str] = None,
        loss_fn: Optional[nn.Module] = None
    ):
        """Initialize evaluator state and move the model to the selected device."""
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Move model to device
        self.model = model.to(self.device)
        self.num_classes = num_classes

        # Loss function
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

        logger.info(f"Evaluator initialized on device: {self.device}")

    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        r"""
        Evaluate model on a dataloader.

        Args:
            dataloader: Data loader to evaluate on.
            return_predictions: Whether to return predictions and labels.
                Default: False

        Returns:
            Dictionary containing:
            - 'loss': Average loss
            - 'accuracy': Classification accuracy
            - 'balanced_accuracy': Balanced accuracy
            - 'per_class_recall': Dictionary of per-class recall
            - 'per_class_precision': Dictionary of per-class precision
            - 'per_class_f1': Dictionary of per-class F1 score
            - 'confusion_matrix': Confusion matrix as numpy array
            - 'predictions': Predictions (if return_predictions=True)
            - 'labels': Ground truth labels (if return_predictions=True)

        Example:
            >>> results = evaluator.evaluate(test_loader)
            >>> print(f"Test accuracy: {results['accuracy']:.4f}")
            >>> print(f"Confusion matrix:\n{results['confusion_matrix']}")
        """
        # Set model to evaluation mode
        self.model.eval()

        # Initialize metrics
        running_loss = 0.0
        num_samples = 0
        all_predictions = []
        all_labels = []

        # Evaluation loop
        logger.info("Running evaluation...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                # Parse batch (handle both formats)
                if len(batch) == 2:
                    images, labels = batch
                elif len(batch) == 4:
                    images, _, labels, _ = batch
                else:
                    raise ValueError(
                        f"Unexpected batch format with {len(batch)} elements"
                    )

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits = self.model(images)

                # Compute loss
                loss = self.loss_fn(logits, labels)
                running_loss += loss.item() * images.size(0)
                num_samples += images.size(0)

                # Get predictions
                predictions = logits.argmax(dim=1)
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        # Compute loss
        avg_loss = running_loss / num_samples

        # Compute accuracy metrics
        accuracy = calculate_accuracy(all_predictions, all_labels)
        balanced_accuracy = calculate_balanced_accuracy(
            all_predictions,
            all_labels,
            self.num_classes
        )

        # Compute per-class metrics
        per_class_metrics = calculate_precision_recall_f1(
            all_predictions,
            all_labels,
            self.num_classes
        )

        # Compute confusion matrix
        confusion_matrix = calculate_confusion_matrix(
            all_predictions,
            all_labels,
            self.num_classes
        )

        # Prepare results
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'per_class_recall': per_class_metrics['recall'],
            'per_class_precision': per_class_metrics['precision'],
            'per_class_f1': per_class_metrics['f1'],
            'confusion_matrix': confusion_matrix.cpu().numpy(),
            'num_samples': num_samples
        }

        # Add predictions and labels if requested
        if return_predictions:
            results['predictions'] = all_predictions.numpy()
            results['labels'] = all_labels.numpy()

        # Log summary
        logger.info(
            f"Evaluation complete: "
            f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}, "
            f"BalAcc={balanced_accuracy:.4f}"
        )

        return results

    def evaluate_with_class_names(
        self,
        dataloader: DataLoader,
        class_names: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate model and include class names in results.

        Args:
            dataloader: Data loader to evaluate on.
            class_names: List of class names (must match num_classes).

        Returns:
            Results dictionary with class names included.

        Example:
            >>> class_names = ['dog', 'cat', 'bird']
            >>> results = evaluator.evaluate_with_class_names(
            ...     test_loader, class_names
            ... )
            >>> print(results['per_class_metrics_named'])
        """
        if len(class_names) != self.num_classes:
            raise ValueError(
                f"Number of class names ({len(class_names)}) does not match "
                f"num_classes ({self.num_classes})"
            )

        # Run evaluation
        results = self.evaluate(dataloader)

        # Add class names to per-class metrics
        results['class_names'] = class_names

        results['per_class_metrics_named'] = {
            'recall': {
                class_names[i]: results['per_class_recall'][i]
                for i in range(self.num_classes)
            },
            'precision': {
                class_names[i]: results['per_class_precision'][i]
                for i in range(self.num_classes)
            },
            'f1': {
                class_names[i]: results['per_class_f1'][i]
                for i in range(self.num_classes)
            }
        }

        return results

    def save_results(
        self,
        results: Dict[str, Any],
        save_dir: str,
        experiment_name: str = "evaluation"
    ):
        """
        Save evaluation results to disk.

        Saves:
        - metrics.json: All metrics in JSON format
        - confusion_matrix.npy: Confusion matrix as numpy array
        - predictions.npy: Predictions (if present in results)
        - labels.npy: Labels (if present in results)

        Args:
            results: Results dictionary from evaluate().
            save_dir: Directory to save results.
            experiment_name: Name for result files. Default: "evaluation"

        Example:
            >>> results = evaluator.evaluate(test_loader)
            >>> evaluator.save_results(
            ...     results,
            ...     save_dir="./results",
            ...     experiment_name="exp1_test"
            ... )
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Prepare metrics for JSON (exclude non-serializable items)
        metrics_to_save = {
            k: v for k, v in results.items()
            if k not in ['confusion_matrix', 'predictions', 'labels']
        }

        # Save metrics as JSON
        metrics_file = save_path / f"{experiment_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        logger.info(f"Saved metrics to: {metrics_file}")

        # Save confusion matrix
        if 'confusion_matrix' in results:
            cm_file = save_path / f"{experiment_name}_confusion_matrix.npy"
            np.save(cm_file, results['confusion_matrix'])
            logger.info(f"Saved confusion matrix to: {cm_file}")

        # Save predictions and labels if present
        if 'predictions' in results:
            pred_file = save_path / f"{experiment_name}_predictions.npy"
            np.save(pred_file, results['predictions'])
            logger.info(f"Saved predictions to: {pred_file}")

        if 'labels' in results:
            labels_file = save_path / f"{experiment_name}_labels.npy"
            np.save(labels_file, results['labels'])
            logger.info(f"Saved labels to: {labels_file}")

    def print_results(self, results: Dict[str, Any], class_names: Optional[List[str]] = None):
        """
        Print evaluation results in a formatted way.

        Args:
            results: Results dictionary from evaluate().
            class_names: Optional list of class names for better readability.

        Example:
            >>> results = evaluator.evaluate(test_loader)
            >>> evaluator.print_results(results, class_names=['dog', 'cat'])
        """
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        # Overall metrics
        print(f"\nOverall Metrics:")
        print(f"  Loss:                {results['loss']:.4f}")
        print(f"  Accuracy:            {results['accuracy']:.4f}")
        print(f"  Balanced Accuracy:   {results['balanced_accuracy']:.4f}")
        print(f"  Num Samples:         {results['num_samples']}")

        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 60)

        for i in range(self.num_classes):
            if class_names is not None and i < len(class_names):
                class_label = class_names[i]
            else:
                class_label = f"Class {i}"

            precision = results['per_class_precision'][i]
            recall = results['per_class_recall'][i]
            f1 = results['per_class_f1'][i]

            print(f"{class_label:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")

        print("=" * 60 + "\n")

    def compare_models(
        self,
        models: Dict[str, nn.Module],
        dataloader: DataLoader
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models on the same dataset.

        Args:
            models: Dictionary mapping model names to model instances.
            dataloader: Data loader to evaluate on.

        Returns:
            Dictionary mapping model names to their evaluation results.

        Example:
            >>> models = {
            ...     'baseline': baseline_model,
            ...     'cfw': cfw_model
            ... }
            >>> comparison = evaluator.compare_models(models, test_loader)
            >>> for name, results in comparison.items():
            ...     print(f"{name}: {results['accuracy']:.4f}")
        """
        results_dict = {}

        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")

            # Temporarily replace model
            original_model = self.model
            self.model = model.to(self.device)

            # Evaluate
            results = self.evaluate(dataloader)
            results_dict[model_name] = results

            # Restore original model
            self.model = original_model

        return results_dict
