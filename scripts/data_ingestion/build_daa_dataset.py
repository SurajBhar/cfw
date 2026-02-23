#!/usr/bin/env python3
"""
Stage B: Build Drive&Act multi-class image dataset from videos.

Extracts frames from Drive&Act videos based on annotation CSV files
to create the multi-class image dataset for driver distraction detection.

Usage:
    # Extract all enabled splits and views (local storage)
    python scripts/data_ingestion/build_daa_dataset.py
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
    DriveActFrameExtractor,
    DriveActMultiViewExtractor,
    create_storage_backend
)


def _normalize_cfg(cfg: DictConfig) -> DictConfig:
    """Support both flat cfg and nested cfg.ingestion layouts."""
    if "ingestion" in cfg:
        # Detach subtree so OmegaConf interpolations resolve from ingestion root.
        return OmegaConf.create(OmegaConf.to_container(cfg.ingestion, resolve=False))
    return cfg


@hydra.main(
    config_path="../../configs",
    config_name="ingestion/daa_extraction",
    version_base=None
)
def main(cfg: DictConfig) -> None:
    """Extract frames from Drive&Act videos."""
    cfg = _normalize_cfg(cfg)
    logger, log_file = setup_ingestion_logging(
        cfg=cfg,
        stage_name="stage_b_daa_extraction",
        logger_name=__name__,
    )

    logger.info("=" * 60)
    logger.info("Stage B: Build DAA Multi-Class Image Dataset")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Create storage backend
    storage = create_storage_backend(cfg) if 'storage' in cfg else None

    # Create extractor
    extractor = DriveActMultiViewExtractor(
        storage=storage,
        max_workers=cfg.extraction.max_workers,
        max_frames_per_chunk=cfg.extraction.max_frames_per_chunk,
        image_format=cfg.extraction.image_format
    )

    all_results = {}

    # Process enabled splits
    for split_cfg in cfg.splits:
        if not split_cfg.enabled:
            logger.info(f"Skipping disabled split: {split_cfg.name}")
            continue

        split_name = split_cfg.name
        logger.info(f"\nProcessing {split_name}")

        # Process enabled camera views
        for view_name, view_cfg in cfg.camera_views.items():
            if not view_cfg.enabled:
                logger.info(f"  Skipping disabled view: {view_name}")
                continue

            logger.info(f"\n  Extracting {view_name}...")

            # Build annotation file paths for enabled splits
            splits_to_extract = {}
            for split_type in view_cfg.extract_splits:
                ann_filename = split_cfg.annotation_files.get(split_type)
                if ann_filename:
                    splits_to_extract[split_type] = ann_filename

            if not splits_to_extract:
                logger.warning(f"  No annotation files for {view_name}")
                continue

            # Output directory layout expected by later stages:
            # <processed>/<dataset_name>/<split_name>/...
            output_dir = Path(cfg.output.base_dir) / cfg.dataset_names[view_name] / split_name

            try:
                stats = extractor.extract_view(
                    view_name=view_name,
                    raw_data_dir=cfg.input.raw_data_dir,
                    annotations_dir=cfg.input.annotations_dir,
                    output_dir=str(output_dir),
                    splits=splits_to_extract,
                    video_subdir=view_cfg.video_subdir,
                    annotation_subdir=view_cfg.annotation_subdir,
                    error_handling=cfg.processing.error_handling
                )
                all_results[f"{view_name}_{split_name}"] = stats

                # Log results
                for split_type, split_stats in stats.items():
                    logger.info(
                        f"    {split_type}: {split_stats.get('total_frames', 0)} frames "
                        f"from {split_stats.get('success', 0)} annotations"
                    )

            except Exception as e:
                logger.error(f"  Failed to extract {view_name}: {e}")
                if cfg.processing.error_handling == 'stop':
                    raise

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Extraction Summary")
    logger.info("=" * 60)

    total_frames = 0
    for key, stats in all_results.items():
        for split_type, split_stats in stats.items():
            frames = split_stats.get('total_frames', 0)
            total_frames += frames
            logger.info(f"  {key}/{split_type}: {frames} frames")

    logger.info(f"\nTotal frames extracted: {total_frames}")
    logger.info("Stage B complete!")


if __name__ == "__main__":
    main()
