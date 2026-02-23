#!/usr/bin/env python3
"""
Stage A: Download and extract datasets.

Downloads Drive&Act and StateFarm datasets from their respective sources
and extracts them to the raw data directory.

Usage:
    # Download all datasets (local storage)
    python scripts/data_ingestion/download_data.py

    # Download with Azure storage
    python scripts/data_ingestion/download_data.py storage=azure

    # Download only Drive&Act
    python scripts/data_ingestion/download_data.py statefarm.enabled=false

    # Download only specific Drive&Act components
    python scripts/data_ingestion/download_data.py \
        driveact.components.kinect_ir.enabled=false \
        driveact.components.nir_front.enabled=false
"""


import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
src_path = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from cfw.utils.logging import setup_ingestion_logging
from cfw.utils.reproducibility import set_seeds
from cfw.data.ingestion import DatasetDownloader, create_storage_backend


def _normalize_cfg(cfg: DictConfig) -> DictConfig:
    """Support both flat cfg and nested cfg.ingestion layouts."""
    if "ingestion" in cfg:
        # Detach subtree so OmegaConf interpolations resolve from ingestion root.
        return OmegaConf.create(OmegaConf.to_container(cfg.ingestion, resolve=False))
    return cfg


@hydra.main(
    config_path="../../configs",
    config_name="ingestion/download",
    version_base=None
)
def main(cfg: DictConfig) -> None:
    """Download and extract datasets."""
    cfg = _normalize_cfg(cfg)
    logger, log_file = setup_ingestion_logging(
        cfg=cfg,
        stage_name="stage_a_download",
        logger_name=__name__,
    )

    logger.info("=" * 60)
    logger.info("Stage A: Download & Extract Datasets")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Create storage backend
    storage = create_storage_backend(cfg) if 'storage' in cfg else None

    # Create downloader
    downloader = DatasetDownloader(
        storage=storage,
        cache_dir=cfg.cache.dir,
        max_retries=cfg.download.max_retries,
        retry_delay=cfg.download.retry_delay,
        chunk_size=cfg.download.chunk_size,
        timeout=cfg.download.timeout
    )

    results = {}

    # Download Drive&Act components
    if cfg.driveact:
        logger.info("\nDownloading Drive&Act dataset...")

        # Get enabled components
        components = [
            name for name, comp in cfg.driveact.components.items()
            if comp.get('enabled', True)
        ]

        if components:
            driveact_results = downloader.download_driveact(
                output_dir=cfg.driveact.output_dir,
                components=components,
                use_cache=cfg.download.use_cache
            )
            results['driveact'] = driveact_results

            for comp, success in driveact_results.items():
                status = "✓" if success else "✗"
                logger.info(f"  {status} {comp}")

    # Download StateFarm
    if cfg.statefarm.enabled:
        logger.info("\nDownloading StateFarm dataset...")

        statefarm_result = downloader.download_statefarm_kaggle(
            output_dir=cfg.statefarm.output_dir,
            kaggle_credentials=cfg.statefarm.kaggle_credentials,
            use_cache=cfg.download.use_cache
        )
        results['statefarm'] = statefarm_result

        status = "✓" if statefarm_result else "✗"
        logger.info(f"  {status} StateFarm")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)

    total_success = 0
    total_failed = 0

    for dataset, result in results.items():
        if isinstance(result, dict):
            for comp, success in result.items():
                if success:
                    total_success += 1
                else:
                    total_failed += 1
        else:
            if result:
                total_success += 1
            else:
                total_failed += 1

    logger.info(f"Success: {total_success}, Failed: {total_failed}")

    if total_failed > 0:
        logger.warning("Some downloads failed. Check logs for details.")
        sys.exit(1)

    logger.info("Stage A complete!")


if __name__ == "__main__":
    main()
