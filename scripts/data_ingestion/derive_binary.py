#!/usr/bin/env python3
"""
Stage C: Derive binary datasets from multi-class datasets.

Converts Drive&Act multi-class datasets to binary classification
(distracted vs. non-distracted) based on configurable class mappings.

Usage:
    # Convert all enabled datasets (local storage)
    python scripts/data_ingestion/derive_binary.py

    # Convert with Azure storage
    python scripts/data_ingestion/derive_binary.py storage=azure

    # Convert only kinect_color
    python scripts/data_ingestion/derive_binary.py \
        daa_conversions.kinect_ir.enabled=false \
        daa_conversions.nir_right_top.enabled=false \
        daa_conversions.nir_front.enabled=false
"""


import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
src_path = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from cfw.utils.logging import setup_ingestion_logging
from cfw.data.ingestion import BinaryConverter, create_storage_backend


def _normalize_cfg(cfg: DictConfig) -> DictConfig:
    """Support both flat cfg and nested cfg.ingestion layouts."""
    if "ingestion" in cfg:
        # Detach subtree so OmegaConf interpolations resolve from ingestion root.
        return OmegaConf.create(OmegaConf.to_container(cfg.ingestion, resolve=False))
    return cfg


def _get_first_int_env(keys, default: int) -> int:
    """Return first available integer env var value from keys."""
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return default


def _get_rank_world_size() -> tuple[int, int]:
    """Detect distributed rank/world-size from common Azure/PyTorch env vars."""
    rank = _get_first_int_env(
        ["RANK", "OMPI_COMM_WORLD_RANK", "NODE_RANK", "AZUREML_NODE_RANK"],
        0,
    )
    world_size = _get_first_int_env(
        ["WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "AZUREML_NODE_COUNT"],
        1,
    )
    if world_size < 1:
        world_size = 1
    if rank < 0:
        rank = 0
    return rank, world_size


@hydra.main(
    config_path="../../configs",
    config_name="ingestion/binary_conversion",
    version_base=None
)
def main(cfg: DictConfig) -> None:
    """Convert multi-class datasets to binary."""
    cfg = _normalize_cfg(cfg)
    rank, world_size = _get_rank_world_size()
    stage_name = "stage_c_binary_conversion"
    if world_size > 1:
        stage_name = f"{stage_name}_rank{rank}"

    logger, log_file = setup_ingestion_logging(
        cfg=cfg,
        stage_name=stage_name,
        logger_name=__name__,
    )

    logger.info("=" * 60)
    logger.info("Stage C: Derive Binary Datasets from Multi-Class")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info("Distributed context: rank=%d world_size=%d", rank, world_size)
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Create storage backend
    storage = create_storage_backend(cfg) if 'storage' in cfg else None

    # Build class mapping from config
    class_mapping = {
        'non_distracted': list(cfg.daa_class_mapping.non_distracted),
        'distracted': list(cfg.daa_class_mapping.distracted)
    }

    # Create converter
    converter = BinaryConverter(
        storage=storage,
        max_workers=cfg.processing.max_workers,
        class_mapping=class_mapping
    )

    all_results = {}

    enabled_tasks = [
        (task_name, task_cfg)
        for task_name, task_cfg in cfg.daa_conversions.items()
        if task_cfg.enabled
    ]

    if world_size > 1:
        assigned_tasks = [
            (task_name, task_cfg)
            for idx, (task_name, task_cfg) in enumerate(enabled_tasks)
            if idx % world_size == rank
        ]
        logger.info(
            "Enabled tasks=%d, assigned to this rank=%d: %s",
            len(enabled_tasks),
            len(assigned_tasks),
            [name for name, _ in assigned_tasks],
        )
    else:
        assigned_tasks = enabled_tasks

    if not assigned_tasks:
        logger.info("No tasks assigned to this rank. Exiting.")
        return

    # Process each assigned conversion task
    for task_name, task_cfg in assigned_tasks:

        logger.info(f"\nConverting {task_name}...")
        logger.info(f"  Source: {task_cfg.source}")
        logger.info(f"  Destination: {task_cfg.destination}")
        logger.info(f"  Splits: {task_cfg.splits}")

        try:
            result = converter.convert_dataset(
                source_dir=task_cfg.source,
                dest_dir=task_cfg.destination,
                splits=list(task_cfg.splits),
                rename_sequential=cfg.processing.rename_sequential,
                verify_counts=cfg.verification.check_counts
            )
            all_results[task_name] = result

            # Log results
            for split_name, split_stats in result.get('splits', {}).items():
                counts = split_stats.get('counts', {})
                logger.info(
                    f"    {split_name}: "
                    f"non_distracted={counts.get('non_distracted', 0)}, "
                    f"distracted={counts.get('distracted', 0)}"
                )

        except Exception as e:
            logger.error(f"  Failed to convert {task_name}: {e}")
            all_results[task_name] = {'error': str(e)}

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Conversion Summary")
    logger.info("=" * 60)

    for task_name, result in all_results.items():
        if 'error' in result:
            logger.info(f"  ✗ {task_name}: FAILED - {result['error']}")
        else:
            total = sum(
                s.get('total', 0)
                for s in result.get('splits', {}).values()
            )
            logger.info(f"  ✓ {task_name}: {total} images")

    logger.info("\nStage C complete!")


if __name__ == "__main__":
    main()
