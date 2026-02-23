#!/usr/bin/env python3
"""
Stage E: Generate EDA statistics for all datasets.

Produces statistical summaries and integrity reports for each dataset variant:
- Class distribution per split
- Imbalance ratios
- Dataset sizes
- Integrity checks (corrupt images, split overlap)

Usage:
    # Generate EDA for all datasets (local storage)
    python scripts/data_ingestion/generate_eda.py

    # Generate with Azure storage
    python scripts/data_ingestion/generate_eda.py storage=azure

    # Skip integrity checking (faster)
    python scripts/data_ingestion/generate_eda.py integrity.enabled=false

    # Enable image file verification (slow but thorough)
    python scripts/data_ingestion/generate_eda.py integrity.check_image_files=true
"""


import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
src_path = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from cfw.utils.logging import setup_ingestion_logging
from cfw.data.ingestion import EDAGenerator, create_storage_backend


def _normalize_cfg(cfg: DictConfig) -> DictConfig:
    """Support both flat cfg and nested cfg.ingestion layouts."""
    if "ingestion" in cfg:
        # Detach subtree so OmegaConf interpolations resolve from ingestion root.
        return OmegaConf.create(OmegaConf.to_container(cfg.ingestion, resolve=False))
    return cfg


def _run_single_dataset_eda(
    storage,
    dataset_name: str,
    dataset_path: str,
    output_dir: str,
    per_dataset_workers: int,
    check_image_integrity: bool,
    check_integrity: bool,
    integrity_sample_ratio: float,
):
    """Generate an EDA report for one dataset."""
    eda = EDAGenerator(
        storage=storage,
        max_workers=per_dataset_workers,
        check_image_integrity=check_image_integrity,
    )
    return eda.generate_report(
        dataset_dir=dataset_path,
        output_dir=output_dir,
        dataset_name=dataset_name,
        check_integrity=check_integrity,
        integrity_sample_ratio=integrity_sample_ratio,
    )


@hydra.main(
    config_path="../../configs",
    config_name="ingestion/eda",
    version_base=None
)
def main(cfg: DictConfig) -> None:
    """Generate EDA statistics for all datasets."""
    cfg = _normalize_cfg(cfg)
    logger, log_file = setup_ingestion_logging(
        cfg=cfg,
        stage_name="stage_e_eda",
        logger_name=__name__,
    )

    logger.info("=" * 60)
    logger.info("Stage E: Generate EDA Statistics")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Create storage backend
    storage = create_storage_backend(cfg) if 'storage' in cfg else None

    all_reports = {}
    tasks = []

    # Build dataset task list
    for dataset_name, dataset_cfg in cfg.datasets.items():
        if not dataset_cfg.enabled:
            logger.info(f"Skipping disabled dataset: {dataset_name}")
            continue

        dataset_path = dataset_cfg.path
        logger.info(f"\nAnalyzing {dataset_name}...")
        logger.info(f"  Path: {dataset_path}")

        # Check if dataset exists
        if storage:
            exists = storage.exists(dataset_path)
        else:
            exists = Path(dataset_path).exists()

        if not exists:
            logger.warning(f"  Dataset not found: {dataset_path}")
            all_reports[dataset_name] = {'error': 'Dataset not found'}
            continue

        output_dir = str(Path(cfg.paths.artifacts_dir) / dataset_name)
        tasks.append((dataset_name, dataset_path, output_dir))

    # Process datasets in parallel
    if tasks:
        total_workers = max(1, int(cfg.processing.max_workers))
        if total_workers >= 4 and len(tasks) > 1:
            dataset_workers = min(len(tasks), max(1, total_workers // 2))
        else:
            dataset_workers = min(len(tasks), total_workers)
        per_dataset_workers = max(1, total_workers // dataset_workers)
        logger.info(
            "\nParallel EDA execution: %d datasets "
            "(dataset_workers=%d, per_dataset_workers=%d)",
            len(tasks),
            dataset_workers,
            per_dataset_workers,
        )

        with ThreadPoolExecutor(max_workers=dataset_workers) as executor:
            future_to_dataset = {
                executor.submit(
                    _run_single_dataset_eda,
                    storage,
                    dataset_name,
                    dataset_path,
                    output_dir,
                    per_dataset_workers,
                    bool(cfg.integrity.check_image_files),
                    bool(cfg.integrity.enabled),
                    float(cfg.integrity.sample_ratio),
                ): dataset_name
                for dataset_name, dataset_path, output_dir in tasks
            }

            for future in as_completed(future_to_dataset):
                dataset_name = future_to_dataset[future]
                try:
                    report = future.result()
                    all_reports[dataset_name] = report

                    # Print statistics
                    stats = report.get('statistics', {})
                    logger.info(f"\n{dataset_name}:")
                    logger.info(f"  Total images: {stats.get('total_images', 0)}")
                    logger.info(f"  Type: {stats.get('dataset_type', 'unknown')}")
                    logger.info(f"  Classes: {stats.get('num_classes', 0)}")

                    # Print split info
                    for split_name, split_stats in stats.get('splits', {}).items():
                        logger.info(
                            f"    {split_name}: {split_stats.get('total_images', 0)} images, "
                            f"imbalance={split_stats.get('imbalance_ratio', 0):.2f}"
                        )

                    # Print integrity status
                    if 'integrity' in report:
                        integrity = report['integrity']
                        if integrity.get('passed'):
                            logger.info("  ✓ Integrity check passed")
                        else:
                            logger.warning("  ✗ Integrity check failed")
                            if integrity.get('corrupt_images'):
                                logger.warning(
                                    f"    Corrupt images: {len(integrity['corrupt_images'])}"
                                )
                            if integrity.get('split_overlaps'):
                                logger.warning(
                                    f"    Split overlaps: {integrity['split_overlaps']}"
                                )

                except Exception as e:
                    logger.error(f"  Failed to analyze {dataset_name}: {e}")
                    all_reports[dataset_name] = {'error': str(e)}

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EDA Summary")
    logger.info("=" * 60)

    total_images = 0
    successful = 0
    failed = 0

    for dataset_name, report in all_reports.items():
        if 'error' in report:
            logger.info(f"  ✗ {dataset_name}: {report['error']}")
            failed += 1
        else:
            images = report.get('statistics', {}).get('total_images', 0)
            total_images += images
            logger.info(f"  ✓ {dataset_name}: {images} images")
            successful += 1

    logger.info(f"\nProcessed: {successful} datasets, Failed: {failed}")
    logger.info(f"Total images across all datasets: {total_images}")
    logger.info(f"Reports saved to: {cfg.paths.artifacts_dir}")
    logger.info("\nStage E complete!")


if __name__ == "__main__":
    main()
