#!/usr/bin/env python3
"""
Run the full data ingestion pipeline (Stages A-E).

Orchestrates all stages of data ingestion:
- Stage A: Download & extract datasets
- Stage B: Build DAA multi-class image datasets from videos
- Stage C: Derive DAA binary datasets from multi-class
- Stage D: Build StateFarm balanced + imbalanced variants
- Stage E: Generate EDA statistics for all datasets

Usage:
    # Run full pipeline (local storage)
    python scripts/data_ingestion/run_full_pipeline.py

    # Run with Azure storage
    python scripts/data_ingestion/run_full_pipeline.py storage=azure

    # Skip download stage (data already available)
    python scripts/data_ingestion/run_full_pipeline.py \
        pipeline.stages.0.enabled=false

    # Stop on any error
    python scripts/data_ingestion/run_full_pipeline.py \
        pipeline.error_handling=stop

    # Continue even if stages fail
    python scripts/data_ingestion/run_full_pipeline.py \
        pipeline.error_handling=continue
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
from cfw.data.ingestion.storage import create_storage_backend
from cfw.data.ingestion.pipeline import (
    IngestionPipeline,
    PipelineConfig,
    ErrorHandling
)


def _normalize_cfg(cfg: DictConfig) -> DictConfig:
    """Support both flat cfg and nested cfg.ingestion layouts."""
    if "ingestion" in cfg:
        # Detach subtree so OmegaConf interpolations resolve from ingestion root.
        return OmegaConf.create(OmegaConf.to_container(cfg.ingestion, resolve=False))
    return cfg


@hydra.main(
    config_path="../../configs",
    config_name="ingestion/pipeline",
    version_base=None
)
def main(cfg: DictConfig) -> None:
    """Run the full data ingestion pipeline."""
    cfg = _normalize_cfg(cfg)
    logger, log_file = setup_ingestion_logging(
        cfg=cfg,
        stage_name="pipeline",
        logger_name=__name__,
    )

    logger.info("=" * 60)
    logger.info("CFW Data Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seeds for reproducibility
    seed = cfg.processing.get('seed', 42)
    set_seeds(seed)

    # Create storage backend
    storage = create_storage_backend(cfg)

    # Build pipeline config
    pipeline_config = PipelineConfig(
        raw_dir=cfg.storage.paths.raw,
        processed_dir=cfg.storage.paths.processed,
        artifacts_dir=cfg.storage.paths.artifacts,
        cache_dir=cfg.get('cache_dir', './data_cache'),
        error_handling=ErrorHandling(cfg.pipeline.error_handling),
        max_workers=cfg.processing.max_workers,
        seed=seed
    )

    # Configure enabled stages from config
    for stage_cfg in cfg.pipeline.stages:
        # Get the stage key (stage_a, stage_b, etc.)
        for key in stage_cfg.keys():
            if key.startswith('stage_'):
                stage_name = key
                pipeline_config.stages_enabled[stage_name] = stage_cfg.get('enabled', True)
                break

    logger.info("\nEnabled stages:")
    for stage, enabled in pipeline_config.stages_enabled.items():
        status = "✓" if enabled else "○"
        logger.info(f"  {status} {stage}")

    # Create and run pipeline
    pipeline = IngestionPipeline(config=pipeline_config, storage=storage)

    try:
        results = pipeline.run()

        # Check overall status
        all_passed = all(
            r.status.value in ['completed', 'skipped']
            for r in results.values()
        )

        if all_passed:
            logger.info("\n" + "=" * 60)
            logger.info("Pipeline completed successfully!")
            logger.info("=" * 60)

            # Print expected output structure
            logger.info("\nExpected output structure:")
            logger.info(f"""
data/
├── raw/
│   ├── driveandact/
│   │   ├── kinect_color/
│   │   └── iccv_activities_3s/
│   └── statefarm/
│       └── (train, test folders)
├── processed/
│   ├── daa_multiclass_kinect_color/
│   │   └── split_0/
│   ├── daa_binary_kinect_color/
│   │   └── split_0/
│   ├── statefarm_balanced_multiclass/
│   ├── statefarm_balanced_binary/
│   ├── statefarm_imbalanced_multiclass/
│   └── statefarm_imbalanced_binary/
└── artifacts/
    ├── daa_multiclass_kinect_color/
    ├── statefarm_balanced_multiclass/
    └── ...
            """)
        else:
            logger.warning("\n" + "=" * 60)
            logger.warning("Pipeline completed with failures")
            logger.warning("=" * 60)
            sys.exit(1)

    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
