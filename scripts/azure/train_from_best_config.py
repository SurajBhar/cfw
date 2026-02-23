#!/usr/bin/env python3
"""Train a final model from BOHB best_config.yaml and existing Hydra experiment config."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf


def _resolve_feature_split_dir(features_root: Path) -> Path:
    """Accept either extraction root (.../features/train under root) or split root."""
    nested = features_root / "features" / "train"
    if nested.exists():
        return nested
    return features_root


def _build_hparam_overrides(best_cfg: dict) -> list[str]:
    """Map BOHB keys to Hydra overrides for scripts/train.py."""
    overrides: list[str] = []
    mappings = [
        ("learning_rate", "optimizer.lr"),
        ("initial_lr", "optimizer.lr"),
        ("initial_lr", "scheduler.start_lr"),
        ("end_lr", "scheduler.end_lr"),
        ("weight_decay", "optimizer.weight_decay"),
        ("optimizer", "optimizer.name"),
        ("scheduler", "scheduler.name"),
        ("batch_size", "dataloader.batch_size"),
        ("dropout", "model.dropout"),
    ]

    for key, hydra_key in mappings:
        if key not in best_cfg:
            continue
        value = best_cfg[key]
        if value is None:
            continue
        overrides.append(f"{hydra_key}={value}")

    return overrides


def _format_cmd(cmd: Iterable[str]) -> str:
    return " ".join(str(part) for part in cmd)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for final training from BOHB best config."""
    parser = argparse.ArgumentParser(
        description="Run scripts/train.py using BOHB best_config.yaml overrides."
    )
    parser.add_argument("--best-config", required=True, help="Path to best_config.yaml")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root containing train/val/test subdirs",
    )
    parser.add_argument(
        "--features-root",
        required=True,
        help="Feature root (either extraction root or direct train split root)",
    )
    parser.add_argument("--output-dir", required=True, help="Training output directory")
    parser.add_argument(
        "--experiment",
        default="azure/cfw_dinov2_binary",
        help="Hydra experiment config name",
    )
    parser.add_argument(
        "--trainer-profile",
        default="single_gpu",
        help="Hydra trainer config (e.g., single_gpu, ddp_multi_node)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Final training epochs",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="trainer.device override (e.g., 0, cuda:0, cpu)",
    )
    return parser.parse_args()


def main() -> None:
    """Build Hydra overrides from BOHB output and launch `scripts/train.py`."""
    args = parse_args()

    best_cfg_path = Path(args.best_config)
    if not best_cfg_path.exists():
        raise FileNotFoundError(f"best_config not found: {best_cfg_path}")

    dataset_root = Path(args.dataset_root)
    features_root = Path(args.features_root)
    output_dir = Path(args.output_dir)
    feature_split_dir = _resolve_feature_split_dir(features_root)

    best_cfg_raw = OmegaConf.to_container(OmegaConf.load(best_cfg_path), resolve=True)
    if not isinstance(best_cfg_raw, dict):
        raise ValueError(f"Unexpected best_config format in {best_cfg_path}")

    cmd = [
        "python",
        "scripts/train.py",
        f"experiment={args.experiment}",
        f"trainer={args.trainer_profile}",
        f"dataset.train_dir={dataset_root}/train",
        f"dataset.val_dir={dataset_root}/val",
        f"dataset.test_dir={dataset_root}/test",
        "dataloader.cfw.cfw_train_only=true",
        f"dataloader.cfw.train_feature_file={feature_split_dir}/features.pkl",
        f"dataloader.cfw.train_label_file={feature_split_dir}/labels.pkl",
        f"dataloader.cfw.train_img_path_file={feature_split_dir}/image_paths.pkl",
        f"trainer.num_epochs={int(args.num_epochs)}",
        f"trainer.device={args.device}",
        f"output_dir={output_dir}",
    ]
    cmd.extend(_build_hparam_overrides(best_cfg_raw))

    print("Running final training with BOHB best config:")
    print(_format_cmd(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
