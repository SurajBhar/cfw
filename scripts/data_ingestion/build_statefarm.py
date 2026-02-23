#!/usr/bin/env python3
"""
Stage D: Build StateFarm dataset splits.

Creates balanced and imbalanced variants of the StateFarm dataset
with both binary and multi-class organizations.

Creates 4 variants:
- statefarm_balanced_multiclass
- statefarm_balanced_binary
- statefarm_imbalanced_multiclass
- statefarm_imbalanced_binary

Usage:
    # Build all variants (local storage)
    python scripts/data_ingestion/build_statefarm.py

    # Build with Azure storage
    python scripts/data_ingestion/build_statefarm.py storage=azure

    # Change split ratios
    python scripts/data_ingestion/build_statefarm.py \
        split_ratios.train=0.80 split_ratios.val=0.10 split_ratios.test=0.10

    # Build only balanced variants
    python scripts/data_ingestion/build_statefarm.py \
        variants.imbalanced_multiclass.enabled=false \
        variants.imbalanced_binary.enabled=false
"""


import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
src_path = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from cfw.utils.logging import setup_ingestion_logging
from cfw.data.ingestion import (
    StateFarmSplitBuilder,
    merge_statefarm_pool,
    create_storage_backend
)


def _normalize_cfg(cfg: DictConfig) -> DictConfig:
    """Support both flat cfg and nested cfg.ingestion layouts."""
    if "ingestion" in cfg:
        # Detach subtree so OmegaConf interpolations resolve from ingestion root.
        return OmegaConf.create(OmegaConf.to_container(cfg.ingestion, resolve=False))
    return cfg


def _resolve_statefarm_split_dirs(
    raw_dir: str,
    storage,
) -> tuple[str, str, str]:
    """
    Resolve StateFarm root and split directories across known layouts.

    Supported layouts:
    - <raw_dir>/train and <raw_dir>/test
    - <raw_dir>/imgs/train and <raw_dir>/imgs/test
    """
    join = storage.join_path if storage else lambda a, b: str(Path(a) / b)
    exists = storage.exists if storage else lambda p: Path(p).exists()

    candidates = [raw_dir, join(raw_dir, "imgs")]
    for root in candidates:
        train_dir = join(root, "train")
        test_dir = join(root, "test")
        if exists(train_dir) and exists(test_dir):
            return root, train_dir, test_dir

    # Deterministic fallback to canonical layout for clearer downstream logs.
    root = raw_dir
    return root, join(root, "train"), join(root, "test")


@hydra.main(
    config_path="../../configs",
    config_name="ingestion/statefarm_splits",
    version_base=None
)
def main(cfg: DictConfig) -> None:
    """Build StateFarm dataset splits."""
    cfg = _normalize_cfg(cfg)
    logger, log_file = setup_ingestion_logging(
        cfg=cfg,
        stage_name="stage_d_statefarm_splits",
        logger_name=__name__,
    )

    logger.info("=" * 60)
    logger.info("Stage D: Build StateFarm Dataset Splits")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Create storage backend
    storage = create_storage_backend(cfg) if 'storage' in cfg else None

    # Check if we need to merge train/test into pool first
    pool_dir = cfg.input.pool_dir
    raw_dir = cfg.input.statefarm_raw

    pool_exists = storage.exists(pool_dir) if storage else Path(pool_dir).exists()

    if not pool_exists:
        logger.info("\nMerging StateFarm train/test into pool...")
        resolved_root, train_dir, test_dir = _resolve_statefarm_split_dirs(raw_dir, storage)
        logger.info("Resolved StateFarm root: %s", resolved_root)
        logger.info("Resolved train dir: %s", train_dir)
        logger.info("Resolved test dir: %s", test_dir)

        merge_statefarm_pool(
            train_dir=train_dir,
            test_dir=test_dir,
            pool_dir=pool_dir,
            storage=storage,
            max_workers=cfg.processing.max_workers,
        )
    else:
        logger.info(f"Using existing pool: {pool_dir}")

    # Create split builder
    builder = StateFarmSplitBuilder(
        storage=storage,
        seed=cfg.processing.seed,
        max_workers=cfg.processing.max_workers
    )

    results = {}

    # Build balanced multiclass
    if cfg.variants.balanced_multiclass.enabled:
        logger.info("\nBuilding balanced multiclass...")
        results['balanced_multiclass'] = builder.build_balanced_multiclass(
            pool_dir=pool_dir,
            output_dir=cfg.output.balanced_multiclass,
            train_ratio=cfg.split_ratios.train,
            val_ratio=cfg.split_ratios.val,
            test_ratio=cfg.split_ratios.test
        )

    # Build balanced binary (from balanced multiclass)
    if cfg.variants.balanced_binary.enabled:
        if 'balanced_multiclass' in results:
            logger.info("\nBuilding balanced binary...")
            results['balanced_binary'] = builder.build_binary_from_multiclass(
                multiclass_dir=cfg.output.balanced_multiclass,
                output_dir=cfg.output.balanced_binary,
                rename_sequential=cfg.processing.rename_sequential
            )
        else:
            logger.warning("Balanced multiclass not built, skipping balanced binary")

    # Build imbalanced multiclass
    if cfg.variants.imbalanced_multiclass.enabled:
        logger.info("\nBuilding imbalanced multiclass...")
        results['imbalanced_multiclass'] = builder.build_imbalanced_multiclass(
            pool_dir=pool_dir,
            output_dir=cfg.output.imbalanced_multiclass,
            train_ratio=cfg.split_ratios.train,
            val_ratio=cfg.split_ratios.val,
            test_ratio=cfg.split_ratios.test
        )

    # Build imbalanced binary (from imbalanced multiclass)
    if cfg.variants.imbalanced_binary.enabled:
        if 'imbalanced_multiclass' in results:
            logger.info("\nBuilding imbalanced binary...")
            results['imbalanced_binary'] = builder.build_binary_from_multiclass(
                multiclass_dir=cfg.output.imbalanced_multiclass,
                output_dir=cfg.output.imbalanced_binary,
                rename_sequential=cfg.processing.rename_sequential
            )
        else:
            logger.warning("Imbalanced multiclass not built, skipping imbalanced binary")

    # Validate no overlap if configured
    if cfg.validation.check_overlap:
        logger.info("\nValidating splits for no overlap...")
        for variant_name in results.keys():
            variant_dir = getattr(cfg.output, variant_name)
            if builder.validate_no_overlap(variant_dir):
                logger.info(f"  ✓ {variant_name}: No overlap")
            else:
                logger.error(f"  ✗ {variant_name}: Overlap detected!")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Build Summary")
    logger.info("=" * 60)

    for variant_name, manifest in results.items():
        counts = manifest.get('counts', {})
        total = counts.get('total', {})
        logger.info(f"\n{variant_name}:")
        logger.info(f"  Train: {total.get('train', 0)}")
        logger.info(f"  Val: {total.get('val', 0)}")
        logger.info(f"  Test: {total.get('test', 0)}")

    logger.info("\nStage D complete!")


if __name__ == "__main__":
    main()
