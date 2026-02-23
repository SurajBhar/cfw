"""
Testing utilities for model evaluation on test sets.

This module provides utilities for testing trained models, including
checkpoint loading, multi-model testing, and results aggregation.
"""


import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .evaluator import Evaluator
from ..utils.checkpoint import load_checkpoint


logger = logging.getLogger(__name__)


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    num_classes: int,
    device: Optional[str] = None,
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Test a model on a test dataset.

    Args:
        model: Model to test.
        test_loader: Test data loader.
        num_classes: Number of classes.
        device: Device to test on. Default: auto-detect
        checkpoint_path: Path to checkpoint to load. If None, uses current weights.

    Returns:
        Dictionary of test results.

    Example:
        >>> results = test_model(
        ...     model=model,
        ...     test_loader=test_loader,
        ...     num_classes=10,
        ...     checkpoint_path="./checkpoints/best.pth"
        ... )
        >>> print(f"Test accuracy: {results['accuracy']:.4f}")
    """
    # Load checkpoint if provided
    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        load_checkpoint(checkpoint_path, model, map_location=device)

    # Create evaluator
    evaluator = Evaluator(model=model, num_classes=num_classes, device=device)

    # Run evaluation
    results = evaluator.evaluate(test_loader)

    return results


def test_multiple_checkpoints(
    model: nn.Module,
    test_loader: DataLoader,
    checkpoint_paths: List[str],
    num_classes: int,
    device: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Test multiple checkpoints on the same test dataset.

    Args:
        model: Model architecture (weights will be loaded from checkpoints).
        test_loader: Test data loader.
        checkpoint_paths: List of checkpoint paths to test.
        num_classes: Number of classes.
        device: Device to test on. Default: auto-detect

    Returns:
        Dictionary mapping checkpoint names to their results.

    Example:
        >>> checkpoints = [
        ...     "./checkpoints/epoch_10.pth",
        ...     "./checkpoints/epoch_20.pth",
        ...     "./checkpoints/best.pth"
        ... ]
        >>> results = test_multiple_checkpoints(
        ...     model, test_loader, checkpoints, num_classes=10
        ... )
        >>> for ckpt, res in results.items():
        ...     print(f"{ckpt}: {res['balanced_accuracy']:.4f}")
    """
    results_dict = {}

    for checkpoint_path in checkpoint_paths:
        checkpoint_name = Path(checkpoint_path).stem

        logger.info(f"Testing checkpoint: {checkpoint_name}")

        # Test this checkpoint
        results = test_model(
            model=model,
            test_loader=test_loader,
            num_classes=num_classes,
            device=device,
            checkpoint_path=checkpoint_path
        )

        results_dict[checkpoint_name] = results

    return results_dict


def test_experiment_checkpoints(
    model: nn.Module,
    test_loader: DataLoader,
    checkpoint_dir: str,
    experiment_name: str,
    num_classes: int,
    device: Optional[str] = None,
    test_best_only: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Test all checkpoints from an experiment.

    Args:
        model: Model architecture.
        test_loader: Test data loader.
        checkpoint_dir: Directory containing checkpoints.
        experiment_name: Name of the experiment.
        num_classes: Number of classes.
        device: Device to test on. Default: auto-detect
        test_best_only: If True, only test the best checkpoint. Default: False

    Returns:
        Dictionary mapping checkpoint names to their results.

    Example:
        >>> results = test_experiment_checkpoints(
        ...     model=model,
        ...     test_loader=test_loader,
        ...     checkpoint_dir="./checkpoints",
        ...     experiment_name="exp1",
        ...     num_classes=10,
        ...     test_best_only=True
        ... )
    """
    checkpoint_path = Path(checkpoint_dir)

    if test_best_only:
        # Test only the best checkpoint
        best_checkpoint = checkpoint_path / f"{experiment_name}_best.pth"

        if not best_checkpoint.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint}")

        checkpoint_paths = [str(best_checkpoint)]
    else:
        # Find all checkpoints for this experiment
        checkpoint_paths = list(checkpoint_path.glob(f"{experiment_name}_*.pth"))
        checkpoint_paths = [str(p) for p in checkpoint_paths]

        if not checkpoint_paths:
            raise FileNotFoundError(
                f"No checkpoints found for experiment '{experiment_name}' "
                f"in directory '{checkpoint_dir}'"
            )

    logger.info(f"Found {len(checkpoint_paths)} checkpoint(s) to test")

    # Test all checkpoints
    results = test_multiple_checkpoints(
        model=model,
        test_loader=test_loader,
        checkpoint_paths=checkpoint_paths,
        num_classes=num_classes,
        device=device
    )

    return results


def save_test_results(
    results: Dict[str, Dict[str, Any]],
    save_path: str,
    experiment_name: str = "test_results"
):
    """
    Save test results to disk.

    Args:
        results: Results dictionary from test_multiple_checkpoints().
        save_path: Path to save results.
        experiment_name: Name for the results file.

    Example:
        >>> save_test_results(
        ...     results,
        ...     save_path="./results",
        ...     experiment_name="exp1_test"
        ... )
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare results for JSON (exclude non-serializable items)
    results_to_save = {}
    for checkpoint_name, checkpoint_results in results.items():
        results_to_save[checkpoint_name] = {
            k: v for k, v in checkpoint_results.items()
            if k not in ['confusion_matrix', 'predictions', 'labels']
        }

    # Save as JSON
    results_file = save_dir / f"{experiment_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    logger.info(f"Saved test results to: {results_file}")


def print_test_summary(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'balanced_accuracy'
):
    """
    Print a summary table of test results.

    Args:
        results: Results from test_multiple_checkpoints().
        metric: Metric to display in summary. Default: 'balanced_accuracy'

    Example:
        >>> print_test_summary(results, metric='balanced_accuracy')
    """
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Checkpoint':<40} {'Accuracy':<12} {metric.replace('_', ' ').title():<12}")
    print("-" * 80)

    for checkpoint_name, checkpoint_results in results.items():
        accuracy = checkpoint_results.get('accuracy', 0.0)
        metric_value = checkpoint_results.get(metric, 0.0)

        print(f"{checkpoint_name:<40} {accuracy:<12.4f} {metric_value:<12.4f}")

    print("=" * 80 + "\n")


def find_best_checkpoint(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'balanced_accuracy',
    mode: str = 'max'
) -> tuple[str, Dict[str, Any]]:
    """
    Find the best checkpoint based on a metric.

    Args:
        results: Results from test_multiple_checkpoints().
        metric: Metric to use for comparison. Default: 'balanced_accuracy'
        mode: 'max' to find maximum, 'min' to find minimum. Default: 'max'

    Returns:
        Tuple of (best_checkpoint_name, best_results)

    Example:
        >>> best_name, best_results = find_best_checkpoint(
        ...     results,
        ...     metric='balanced_accuracy',
        ...     mode='max'
        ... )
        >>> print(f"Best checkpoint: {best_name}")
        >>> print(f"Best {metric}: {best_results[metric]:.4f}")
    """
    if not results:
        raise ValueError("No results provided")

    best_name = None
    best_results = None
    best_value = float('-inf') if mode == 'max' else float('inf')

    for checkpoint_name, checkpoint_results in results.items():
        metric_value = checkpoint_results.get(metric, None)

        if metric_value is None:
            logger.warning(
                f"Metric '{metric}' not found in results for '{checkpoint_name}'"
            )
            continue

        if mode == 'max' and metric_value > best_value:
            best_value = metric_value
            best_name = checkpoint_name
            best_results = checkpoint_results
        elif mode == 'min' and metric_value < best_value:
            best_value = metric_value
            best_name = checkpoint_name
            best_results = checkpoint_results

    if best_name is None:
        raise ValueError(f"Could not find best checkpoint for metric '{metric}'")

    logger.info(
        f"Best checkpoint: {best_name} ({metric}={best_value:.4f})"
    )

    return best_name, best_results


def compare_experiments(
    model: nn.Module,
    test_loader: DataLoader,
    experiments: Dict[str, str],
    num_classes: int,
    device: Optional[str] = None,
    test_best_only: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple experiments on the same test dataset.

    Args:
        model: Model architecture.
        test_loader: Test data loader.
        experiments: Dictionary mapping experiment names to checkpoint directories.
        num_classes: Number of classes.
        device: Device to test on. Default: auto-detect
        test_best_only: If True, only test best checkpoints. Default: True

    Returns:
        Dictionary mapping experiment names to their best results.

    Example:
        >>> experiments = {
        ...     'baseline': './checkpoints/baseline',
        ...     'cfw': './checkpoints/cfw'
        ... }
        >>> comparison = compare_experiments(
        ...     model, test_loader, experiments, num_classes=10
        ... )
        >>> for exp_name, results in comparison.items():
        ...     print(f"{exp_name}: {results['balanced_accuracy']:.4f}")
    """
    comparison_results = {}

    for exp_name, checkpoint_dir in experiments.items():
        logger.info(f"Testing experiment: {exp_name}")

        # Test this experiment
        exp_results = test_experiment_checkpoints(
            model=model,
            test_loader=test_loader,
            checkpoint_dir=checkpoint_dir,
            experiment_name=exp_name,
            num_classes=num_classes,
            device=device,
            test_best_only=test_best_only
        )

        # If testing best only, use those results
        if test_best_only:
            comparison_results[exp_name] = list(exp_results.values())[0]
        else:
            # Find best checkpoint
            _, best_results = find_best_checkpoint(exp_results)
            comparison_results[exp_name] = best_results

    return comparison_results


def print_experiment_comparison(
    comparison_results: Dict[str, Dict[str, Any]],
    metrics: Optional[List[str]] = None
):
    """
    Print a comparison table of experiment results.

    Args:
        comparison_results: Results from compare_experiments().
        metrics: List of metrics to display. If None, uses default set.

    Example:
        >>> print_experiment_comparison(
        ...     comparison_results,
        ...     metrics=['accuracy', 'balanced_accuracy']
        ... )
    """
    if metrics is None:
        metrics = ['accuracy', 'balanced_accuracy', 'loss']

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)

    # Header
    header = f"{'Experiment':<20}"
    for metric in metrics:
        header += f"{metric.replace('_', ' ').title():<20}"
    print(f"\n{header}")
    print("-" * 80)

    # Results
    for exp_name, results in comparison_results.items():
        row = f"{exp_name:<20}"
        for metric in metrics:
            value = results.get(metric, 0.0)
            row += f"{value:<20.4f}"
        print(row)

    print("=" * 80 + "\n")


__all__ = [
    'test_model',
    'test_multiple_checkpoints',
    'test_experiment_checkpoints',
    'save_test_results',
    'print_test_summary',
    'find_best_checkpoint',
    'compare_experiments',
    'print_experiment_comparison'
]
