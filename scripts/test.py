#!/usr/bin/env python3
"""
Testing script for evaluating trained CFW models.

This script loads a trained model checkpoint and evaluates it on the test set,
computing metrics like accuracy, balanced accuracy, and per-class recall.

Usage:
    # Test with specific checkpoint
    python scripts/test.py checkpoint_path=/path/to/checkpoint.pth

    # Test with experiment config
    python scripts/test.py experiment=cfw_dinov2_binary checkpoint_path=/path/to/checkpoint.pth

    # Test with config overrides
    python scripts/test.py model=vit_b_16 dataset=driveact_binary checkpoint_path=/path/to/checkpoint.pth
"""


import os
import sys
from pathlib import Path
from typing import Dict, Optional, Iterator, Tuple, Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from cfw.utils.logging import setup_logger
from cfw.utils.reproducibility import set_seeds
from cfw.utils.config_utils import validate_config
from cfw.data.dataloaders import create_dataloader
from cfw.models.builders import build_classifier_from_config
from cfw.evaluation.evaluator import Evaluator


def _iter_class_metric(metric_values: Any) -> Iterator[Tuple[int, float]]:
    """Iterate per-class metric values as (class_idx, value) for dicts or sequences."""
    if isinstance(metric_values, dict):
        def _sort_key(key: Any) -> Tuple[int, Any]:
            try:
                return (0, int(key))
            except (TypeError, ValueError):
                return (1, str(key))

        for key in sorted(metric_values.keys(), key=_sort_key):
            try:
                class_idx = int(key)
            except (TypeError, ValueError):
                continue
            yield class_idx, float(metric_values[key])
        return

    for class_idx, value in enumerate(metric_values):
        yield class_idx, float(value)


def _resolve_device(cfg: DictConfig) -> torch.device:
    """Resolve device from trainer.device with trainer.gpu_id fallback."""
    device_value = str(cfg.trainer.get("device", cfg.trainer.get("gpu_id", "0"))).lower()
    if device_value == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    if device_value.startswith("cuda:"):
        return torch.device(device_value)

    return torch.device(f"cuda:{device_value}")


def _resolve_azureml_output_root() -> Optional[Path]:
    """Resolve Azure ML output path if this script is running inside AML."""
    def _is_resolved_path(value: str) -> bool:
        stripped = value.strip()
        if not stripped:
            return False
        if "${{" in stripped or "${" in stripped:
            return False
        return True

    for key in ("AZUREML_OUTPUTS_DIR", "AZUREML_RUN_OUTPUT_PATH", "OUTPUTS_DIR"):
        value = os.environ.get(key)
        if value and _is_resolved_path(value):
            return Path(value)

    for key, value in os.environ.items():
        if key.startswith("AZUREML_OUTPUT_") and value and _is_resolved_path(value):
            return Path(value)

    return None


def load_model_from_checkpoint(
    checkpoint_path: str,
    cfg: DictConfig,
    device: torch.device
) -> torch.nn.Module:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        cfg: Hydra configuration
        device: Device to load model on

    Returns:
        Loaded model
    """
    # Build model architecture
    model = build_classifier_from_config(cfg)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def save_results(results: Dict, output_path: Path, logger) -> None:
    """
    Save test results to file.

    Args:
        results: Dictionary of test results
        output_path: Path to save results
        logger: Logger instance
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("Test Results\n")
        f.write("=" * 80 + "\n\n")

        # Overall metrics
        f.write("Overall Metrics:\n")
        f.write("-" * 40 + "\n")
        for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score"]:
            if key in results:
                f.write(f"{key.replace('_', ' ').title()}: {results[key]:.4f}\n")
        f.write("\n")

        # Per-class metrics
        if "per_class_recall" in results:
            f.write("Per-Class Recall:\n")
            f.write("-" * 40 + "\n")
            for class_idx, recall in _iter_class_metric(results["per_class_recall"]):
                f.write(f"Class {class_idx}: {recall:.4f}\n")
            f.write("\n")

        if "per_class_precision" in results:
            f.write("Per-Class Precision:\n")
            f.write("-" * 40 + "\n")
            for class_idx, precision in _iter_class_metric(results["per_class_precision"]):
                f.write(f"Class {class_idx}: {precision:.4f}\n")
            f.write("\n")

        if "per_class_f1" in results:
            f.write("Per-Class F1-Score:\n")
            f.write("-" * 40 + "\n")
            for class_idx, f1 in _iter_class_metric(results["per_class_f1"]):
                f.write(f"Class {class_idx}: {f1:.4f}\n")
            f.write("\n")

        # Confusion matrix
        if "confusion_matrix" in results:
            f.write("Confusion Matrix:\n")
            f.write("-" * 40 + "\n")
            cm = results["confusion_matrix"]
            for row in cm:
                formatted_row = []
                for val in row:
                    val_float = float(val)
                    if val_float.is_integer():
                        formatted_row.append(f"{int(val_float):6d}")
                    else:
                        formatted_row.append(f"{val_float:6.2f}")
                f.write(" ".join(formatted_row) + "\n")
            f.write("\n")

    logger.info(f"Results saved to {output_path}")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Run model evaluation on the configured test split.

    Args:
        cfg: Hydra configuration object
    """
    print("=" * 80)
    print("CFW Model Testing Script")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Check if checkpoint path is provided
    if not cfg.get("checkpoint_path"):
        raise ValueError(
            "checkpoint_path must be provided. "
            "Usage: python scripts/test.py checkpoint_path=/path/to/checkpoint.pth"
        )

    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Validate configuration
    validate_config(cfg)

    # Set random seeds for reproducibility
    set_seeds(cfg.experiment.seed)

    # Get output directory from Hydra (or Azure ML output mount when available)
    azure_output_root = _resolve_azureml_output_root()
    if azure_output_root is not None:
        output_dir = azure_output_root / cfg.experiment.name / "testing"
    else:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log_dir = output_dir / "logs"
    results_dir = output_dir / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(
        name="cfw_testing",
        log_file=str(log_dir / "testing.log"),
        level=cfg.get("log_level", "INFO")
    )

    logger.info("Starting CFW model testing")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output directory: {output_dir}")

    # Setup device
    device = _resolve_device(cfg)
    logger.info(f"Using device: {device}")

    # Create test dataloader
    logger.info("Creating test dataloader...")
    test_loader = create_dataloader(
        cfg=cfg,
        split="test",
        shuffle=False,
        drop_last=False
    )
    logger.info(f"Test batches: {len(test_loader)}")

    # Load model
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = load_model_from_checkpoint(str(checkpoint_path), cfg, device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {num_params:,}")
    logger.info(f"Trainable parameters: {num_trainable:,}")

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        device=device,
        num_classes=cfg.dataset.num_classes
    )

    # Evaluate
    logger.info("Running evaluation on test set...")
    test_results = evaluator.evaluate(test_loader)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("Test Results:")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")

    if "precision" in test_results:
        logger.info(f"Precision: {test_results['precision']:.4f}")
    if "recall" in test_results:
        logger.info(f"Recall: {test_results['recall']:.4f}")
    if "f1_score" in test_results:
        logger.info(f"F1-Score: {test_results['f1_score']:.4f}")

    if "per_class_recall" in test_results:
        logger.info("\nPer-Class Recall:")
        for class_idx, recall in _iter_class_metric(test_results["per_class_recall"]):
            logger.info(f"  Class {class_idx}: {recall:.4f}")

    logger.info("=" * 80)

    # Save results
    results_file = results_dir / "test_results.txt"
    save_results(test_results, results_file, logger)

    # Print summary
    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
    print(f"\nResults saved to: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
