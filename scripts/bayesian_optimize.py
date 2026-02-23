#!/usr/bin/env python3
"""
Bayesian Optimization entry point for CFW hyperparameter tuning.

This script runs BOHB (Bayesian Optimization HyperBand) to find optimal
hyperparameters for CFW training using Ray Tune.

Usage:
    # Basic usage with default settings
    python scripts/bayesian_optimize.py

    # With custom optimization config
    python scripts/bayesian_optimize.py +optimization=bayesian

    # Override number of trials
    python scripts/bayesian_optimize.py +optimization=bayesian optimization.tuner.num_samples=20

    # Override search space bounds
    python scripts/bayesian_optimize.py +optimization=bayesian \
        optimization.search_space.learning_rate.lower=1e-4 \
        optimization.search_space.learning_rate.upper=1e-2

    # Specify GPU resources
    python scripts/bayesian_optimize.py +optimization=bayesian \
        optimization.ray.num_gpus=2 \
        optimization.tuner.max_concurrent=2

    # Use with specific model/dataset config
    python scripts/bayesian_optimize.py model=dinov2_vitb14 dataset=driveact_binary \
        +optimization=bayesian

    # Quick test run
    python scripts/bayesian_optimize.py +optimization=bayesian \
        optimization.tuner.num_samples=2 \
        optimization.bohb.max_t=5

The script:
1. Loads Hydra configuration with optimization settings
2. Creates ConfigSpace search space from config
3. Sets up Ray Tune with BOHB scheduler and searcher
4. Runs trials with CFWTrainable wrapper
5. Reports best configuration and saves results

Results are saved to the path specified in optimization.storage.path
(default: ./ray_results/)
"""


import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from cfw.optimization.bayesian import (
    run_bayesian_optimization,
    get_best_config,
    get_all_trial_results,
)


def print_banner():
    """Print script banner."""
    print("=" * 70)
    print("CFW Bayesian Hyperparameter Optimization")
    print("=" * 70)


def print_config_summary(cfg: DictConfig):
    """Print optimization configuration summary."""
    print("\nOptimization Settings:")
    print("-" * 40)

    opt_cfg = cfg.optimization

    # Search space
    print("\nSearch Space:")
    if 'search_space' in opt_cfg:
        for hp_name, hp_cfg in opt_cfg.search_space.items():
            if isinstance(hp_cfg, DictConfig):
                if 'choices' in hp_cfg:
                    print(f"  {hp_name}: {list(hp_cfg.choices)}")
                elif 'lower' in hp_cfg and 'upper' in hp_cfg:
                    scale = "log" if hp_cfg.get('log', False) else "linear"
                    print(f"  {hp_name}: [{hp_cfg.lower}, {hp_cfg.upper}] ({scale})")

    # BOHB settings
    print("\nBOHB Settings:")
    print(f"  max_t (max epochs): {opt_cfg.bohb.max_t}")
    print(f"  reduction_factor: {opt_cfg.bohb.reduction_factor}")

    # Tuner settings
    print("\nTuner Settings:")
    print(f"  num_samples (total trials): {opt_cfg.tuner.num_samples}")
    print(f"  max_concurrent: {opt_cfg.tuner.max_concurrent}")
    print(f"  metric: {opt_cfg.tuner.metric}")
    print(f"  mode: {opt_cfg.tuner.mode}")
    print(f"  evaluate_test_on_best: {opt_cfg.get('evaluate_test_on_best', False)}")

    # Resource settings
    print("\nResources per Trial:")
    print(f"  CPUs: {opt_cfg.ray.resources_per_trial.cpu}")
    print(f"  GPUs: {opt_cfg.ray.resources_per_trial.gpu}")

    # Storage
    print("\nStorage:")
    print(f"  path: {opt_cfg.storage.path}")
    print(f"  experiment name: {opt_cfg.storage.name}")

    print("-" * 40)


def print_results(results, metric: str):
    """Print optimization results."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    # Get best result
    best_result = results.get_best_result()

    if best_result is None:
        print("\nNo successful trials completed.")
        return

    best_config = best_result.config
    best_metrics = best_result.metrics

    print("\nBest Configuration:")
    print("-" * 40)
    for k, v in best_config.items():
        if not k.startswith('_') and k != 'base_config_path':
            if isinstance(v, float):
                print(f"  {k}: {v:.6g}")
            else:
                print(f"  {k}: {v}")

    print("\nBest Metrics:")
    print("-" * 40)
    metric_value = best_metrics.get(metric, None)
    if metric_value is None:
        print(f"  {metric}: N/A")
    else:
        print(f"  {metric}: {metric_value:.4f}")
    if 'val_accuracy' in best_metrics:
        print(f"  val_accuracy: {best_metrics['val_accuracy']:.2f}%")
    if 'val_loss' in best_metrics:
        print(f"  val_loss: {best_metrics['val_loss']:.4f}")
    if 'test_balanced_accuracy' in best_metrics:
        print(f"  test_balanced_accuracy: {best_metrics['test_balanced_accuracy']:.4f}")
    if 'test_accuracy' in best_metrics:
        print(f"  test_accuracy: {best_metrics['test_accuracy']:.2f}%")
    if 'test_loss' in best_metrics:
        print(f"  test_loss: {best_metrics['test_loss']:.4f}")

    # Summary of all trials
    all_results = get_all_trial_results(results)
    num_trials = len(all_results)
    num_complete = sum(1 for r in all_results if metric in r['metrics'])

    print(f"\nTrials Summary:")
    print("-" * 40)
    print(f"  Total trials: {num_trials}")
    print(f"  Completed trials: {num_complete}")

    if num_complete > 0:
        metric_values = [r['metrics'][metric] for r in all_results
                        if metric in r['metrics']]
        print(f"  {metric} - min: {min(metric_values):.4f}, "
              f"max: {max(metric_values):.4f}, "
              f"mean: {sum(metric_values)/len(metric_values):.4f}")

    print("=" * 70)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Run Bayesian hyperparameter optimization.

    Args:
        cfg: Hydra configuration object
    """
    print_banner()

    # Check if optimization config is present
    if 'optimization' not in cfg:
        print("\nError: No optimization config found.")
        print("Please add optimization config:")
        print("  python scripts/bayesian_optimize.py +optimization=bayesian")
        print("\nOr create configs/optimization/bayesian.yaml")
        sys.exit(1)

    # Print configuration summary
    print_config_summary(cfg)

    non_interactive = cfg.optimization.get("non_interactive", False)
    if not non_interactive:
        print("\nPress Enter to start optimization (Ctrl+C to cancel)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nOptimization cancelled.")
            sys.exit(0)
    else:
        print("\nNon-interactive mode enabled. Starting optimization immediately...")

    # Run optimization
    print("\nStarting BOHB optimization...")
    print("(Ray Dashboard available at http://localhost:8265)\n")

    try:
        results = run_bayesian_optimization(cfg)

        # Print results
        metric = cfg.optimization.tuner.metric
        print_results(results, metric)

        # Get paths
        storage_path = cfg.optimization.storage.path
        experiment_name = cfg.optimization.storage.name

        print(f"\nResults saved to: {storage_path}/{experiment_name}")
        print(f"Best config saved to: {storage_path}/best_config.yaml")

    except Exception as e:
        print(f"\nOptimization failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
