#!/usr/bin/env python3
"""
Create a small StateFarm class-balanced smoke pool.

Usage:
    Build Imbalanced Multi-class Statefarm Smoke Dataset:
        python scripts/data_ingestion/build_statefarm.py \
          --config-name ingestion/statefarm_splits_smoke \
          'variants.balanced_multiclass.enabled=false' \
          'variants.balanced_binary.enabled=false' \
          'variants.imbalanced_multiclass.enabled=true' \
          'variants.imbalanced_binary.enabled=false'
    
    Build Imbalanced binary Statefarm Smoke Dataset:
        python scripts/data_ingestion/build_statefarm.py \
          --config-name ingestion/statefarm_splits_smoke \
          'variants.balanced_multiclass.enabled=false' \
          'variants.balanced_binary.enabled=false' \
          'variants.imbalanced_multiclass.enabled=true' \
          'variants.imbalanced_binary.enabled=true'
"""


from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for StateFarm smoke-pool creation."""
    parser = argparse.ArgumentParser(description="Build StateFarm smoke pool")
    parser.add_argument(
        "--source-pool",
        default="data/raw/statefarm_pool",
        help="Source class-folder pool directory",
    )
    parser.add_argument(
        "--dest-pool",
        default="data/raw/statefarm_pool_smoke",
        help="Destination smoke pool directory",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=60,
        help="Number of images to sample per class",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clear-dest",
        action="store_true",
        help="Delete destination directory before writing",
    )
    return parser.parse_args()


def main() -> None:
    """Sample a small class-balanced StateFarm pool for smoke testing."""
    args = parse_args()
    random.seed(args.seed)

    source_pool = Path(args.source_pool)
    dest_pool = Path(args.dest_pool)

    if not source_pool.exists() or not source_pool.is_dir():
        raise FileNotFoundError(f"Source pool not found: {source_pool}")

    if args.clear_dest and dest_pool.exists():
        shutil.rmtree(dest_pool)

    dest_pool.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted(p for p in source_pool.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found under: {source_pool}")

    for class_dir in class_dirs:
        images = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
        if not images:
            print(f"Skipping class with no images: {class_dir.name}")
            continue

        random.shuffle(images)
        selected = images[: args.per_class]

        target_class_dir = dest_pool / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for src in selected:
            shutil.copy2(src, target_class_dir / src.name)

        print(
            f"Class {class_dir.name}: copied {len(selected)} / {len(images)} "
            f"images -> {target_class_dir}"
        )


if __name__ == "__main__":
    main()
