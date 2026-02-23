"""
Data Ingestion Pipeline Orchestrator.

Orchestrates the full data ingestion pipeline (Stages A-E):
- Stage A: Download & extract datasets
- Stage B: Build DAA multi-class image datasets from videos
- Stage C: Derive DAA binary datasets from multi-class
- Stage D: Build StateFarm balanced + imbalanced variants
- Stage E: Generate EDA statistics for all datasets

Supports configurable stage execution, error handling, and recovery.
"""


import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .storage import StorageBackend, LocalStorage, create_storage_backend
from .downloader import DatasetDownloader, download_all_datasets
from .frame_extractor import DriveActFrameExtractor, DriveActMultiViewExtractor
from .binary_converter import BinaryConverter, convert_daa_to_binary
from .split_builder import StateFarmSplitBuilder, merge_statefarm_pool
from .eda_generator import EDAGenerator

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ErrorHandling(Enum):
    """Error handling strategy."""

    STOP = "stop"       # Stop pipeline on error
    SKIP = "skip"       # Skip failed stage and continue
    CONTINUE = "continue"  # Continue even after failure


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""

    stage_name: str
    status: StageStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    output: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class PipelineConfig:
    """Configuration for the ingestion pipeline."""

    # Storage paths
    raw_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    artifacts_dir: str = "./data/artifacts"
    cache_dir: str = "./data_cache"

    # Stage configuration
    stages_enabled: Dict[str, bool] = field(default_factory=lambda: {
        'stage_a': True,  # Download
        'stage_b': True,  # DAA extraction
        'stage_c': True,  # Binary conversion
        'stage_d': True,  # StateFarm splits
        'stage_e': True,  # EDA
    })

    # Error handling
    error_handling: ErrorHandling = ErrorHandling.STOP

    # Processing settings
    max_workers: int = 8
    seed: int = 42

    # DAA extraction settings
    daa_splits: List[str] = field(default_factory=lambda: ['split_0'])
    daa_max_frames_per_chunk: int = 48

    # StateFarm settings
    statefarm_train_ratio: float = 0.70
    statefarm_val_ratio: float = 0.15
    statefarm_test_ratio: float = 0.15


class IngestionPipeline:
    """
    Orchestrates the data ingestion pipeline.

    Executes stages A-E in order with configurable error handling
    and progress tracking.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        storage: Optional[StorageBackend] = None
    ):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
            storage: Storage backend
        """
        self.config = config or PipelineConfig()
        self.storage = storage or LocalStorage()
        self.results: Dict[str, StageResult] = {}

        # Create output directories
        for dir_path in [
            self.config.raw_dir,
            self.config.processed_dir,
            self.config.artifacts_dir,
            self.config.cache_dir
        ]:
            self.storage.makedirs(dir_path)

    def _run_stage(
        self,
        stage_name: str,
        stage_func: Callable,
        depends_on: Optional[List[str]] = None
    ) -> StageResult:
        """
        Run a single pipeline stage.

        Args:
            stage_name: Name of the stage
            stage_func: Function to execute
            depends_on: List of stages this depends on

        Returns:
            StageResult object
        """
        # Check if enabled
        if not self.config.stages_enabled.get(stage_name, True):
            logger.info(f"Stage {stage_name} is disabled, skipping")
            return StageResult(
                stage_name=stage_name,
                status=StageStatus.SKIPPED,
                start_time=datetime.now()
            )

        # Check dependencies
        if depends_on:
            for dep in depends_on:
                if dep in self.results:
                    dep_result = self.results[dep]
                    if dep_result.status == StageStatus.FAILED:
                        if self.config.error_handling == ErrorHandling.STOP:
                            logger.error(
                                f"Dependency {dep} failed, skipping {stage_name}"
                            )
                            return StageResult(
                                stage_name=stage_name,
                                status=StageStatus.SKIPPED,
                                start_time=datetime.now(),
                                error=f"Dependency {dep} failed"
                            )

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Stage: {stage_name}")
        logger.info(f"{'='*60}")

        start_time = datetime.now()
        result = StageResult(
            stage_name=stage_name,
            status=StageStatus.RUNNING,
            start_time=start_time
        )

        try:
            output = stage_func()
            result.status = StageStatus.COMPLETED
            result.output = output
            logger.info(f"Stage {stage_name} completed successfully")

        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}", exc_info=True)
            result.status = StageStatus.FAILED
            result.error = str(e)

            if self.config.error_handling == ErrorHandling.STOP:
                raise

        finally:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - start_time).total_seconds()
            self.results[stage_name] = result

        return result

    def _stage_a_download(self) -> Dict:
        """Stage A: Download and extract datasets."""
        downloader = DatasetDownloader(
            storage=self.storage,
            cache_dir=self.config.cache_dir
        )

        # Download Drive&Act
        driveact_dir = self.storage.join_path(self.config.raw_dir, 'driveandact')
        driveact_results = downloader.download_driveact(
            output_dir=driveact_dir,
            components=['kinect_color', 'annotations']  # Minimum for training
        )

        # Download StateFarm
        statefarm_dir = self.storage.join_path(self.config.raw_dir, 'statefarm')
        statefarm_result = downloader.download_statefarm_kaggle(
            output_dir=statefarm_dir
        )

        return {
            'driveact': driveact_results,
            'statefarm': statefarm_result
        }

    def _stage_b_daa_extraction(self) -> Dict:
        """Stage B: Extract frames from Drive&Act videos."""
        extractor = DriveActMultiViewExtractor(
            storage=self.storage,
            max_workers=self.config.max_workers,
            max_frames_per_chunk=self.config.daa_max_frames_per_chunk
        )

        raw_dir = self.storage.join_path(self.config.raw_dir, 'driveandact')
        annotations_dir = self.storage.join_path(raw_dir, 'iccv_activities_3s')

        all_stats = {}

        for split in self.config.daa_splits:
            splits = {
                'train': f'midlevel.chunks_90.{split}.train.csv',
                'val': f'midlevel.chunks_90.{split}.val.csv',
                'test': f'midlevel.chunks_90.{split}.test.csv',
            }
            output_dir = self.storage.join_path(
                self.config.processed_dir,
                'daa_multiclass_kinect_color',
                split,
            )

            # Extract training data (Kinect Color), aligned with stage-B script layout:
            # processed/daa_multiclass_kinect_color/split_X/{train,val,test}/...
            stats = extractor.extract_view(
                view_name='kinect_color',
                raw_data_dir=raw_dir,
                annotations_dir=annotations_dir,
                output_dir=output_dir,
                splits=splits,
            )
            all_stats[f'daa_multiclass_kinect_color/{split}'] = stats

        return all_stats

    def _stage_c_binary_conversion(self) -> Dict:
        """Stage C: Convert multi-class datasets to binary."""
        converter = BinaryConverter(
            storage=self.storage,
            max_workers=self.config.max_workers
        )

        all_stats = {}

        for split in self.config.daa_splits:
            multiclass_dir = self.storage.join_path(
                self.config.processed_dir,
                'daa_multiclass_kinect_color',
                split,
            )
            binary_dir = self.storage.join_path(
                self.config.processed_dir,
                'daa_binary_kinect_color',
                split,
            )

            if self.storage.exists(multiclass_dir):
                stats = converter.convert_dataset(
                    source_dir=multiclass_dir,
                    dest_dir=binary_dir
                )
                all_stats[f'daa_binary_kinect_color/{split}'] = stats

        return all_stats

    def _stage_d_statefarm_splits(self) -> Dict:
        """Stage D: Build StateFarm dataset splits."""
        # First merge train/test into pool
        raw_statefarm = self.storage.join_path(self.config.raw_dir, 'statefarm')
        train_dir, test_dir, resolved_root = self._resolve_statefarm_split_dirs(raw_statefarm)
        logger.info("Resolved StateFarm root: %s", resolved_root)
        logger.info("Resolved StateFarm train dir: %s", train_dir)
        logger.info("Resolved StateFarm test dir: %s", test_dir)
        pool_dir = self.storage.join_path(self.config.raw_dir, 'statefarm_pool')

        # Check if pool already exists
        if not self.storage.exists(pool_dir):
            merge_statefarm_pool(
                train_dir=train_dir,
                test_dir=test_dir,
                pool_dir=pool_dir,
                storage=self.storage,
                max_workers=self.config.max_workers,
            )

        # Build all variants
        builder = StateFarmSplitBuilder(
            storage=self.storage,
            seed=self.config.seed,
            max_workers=self.config.max_workers
        )

        results = builder.build_all_variants(
            pool_dir=pool_dir,
            output_base_dir=self.config.processed_dir,
            train_ratio=self.config.statefarm_train_ratio,
            val_ratio=self.config.statefarm_val_ratio,
            test_ratio=self.config.statefarm_test_ratio
        )

        return results

    def _resolve_statefarm_split_dirs(self, raw_statefarm: str) -> tuple[str, str, str]:
        """
        Resolve StateFarm train/test directories across known layouts.

        Supported layouts:
        - <raw_statefarm>/train and <raw_statefarm>/test
        - <raw_statefarm>/imgs/train and <raw_statefarm>/imgs/test
        """
        candidates = [raw_statefarm, self.storage.join_path(raw_statefarm, 'imgs')]
        for root in candidates:
            train_dir = self.storage.join_path(root, 'train')
            test_dir = self.storage.join_path(root, 'test')
            if self.storage.exists(train_dir) and self.storage.exists(test_dir):
                return train_dir, test_dir, root

        # Deterministic fallback to canonical layout for clearer downstream logs.
        return (
            self.storage.join_path(raw_statefarm, 'train'),
            self.storage.join_path(raw_statefarm, 'test'),
            raw_statefarm,
        )

    def _stage_e_eda(self) -> Dict:
        """Stage E: Generate EDA statistics."""
        eda = EDAGenerator(
            storage=self.storage,
            max_workers=self.config.max_workers
        )

        reports = eda.generate_all_reports(
            processed_dir=self.config.processed_dir,
            artifacts_dir=self.config.artifacts_dir,
            check_integrity=True
        )

        return reports

    def run(self) -> Dict[str, StageResult]:
        """
        Run the full pipeline.

        Returns:
            Dictionary of stage results
        """
        logger.info("=" * 60)
        logger.info("Starting Data Ingestion Pipeline")
        logger.info("=" * 60)

        pipeline_start = datetime.now()

        # Stage A: Download
        self._run_stage('stage_a', self._stage_a_download)

        # Stage B: DAA Extraction
        self._run_stage('stage_b', self._stage_b_daa_extraction, depends_on=['stage_a'])

        # Stage C: Binary Conversion
        self._run_stage('stage_c', self._stage_c_binary_conversion, depends_on=['stage_b'])

        # Stage D: StateFarm Splits (independent of B/C)
        self._run_stage('stage_d', self._stage_d_statefarm_splits, depends_on=['stage_a'])

        # Stage E: EDA (depends on B, C, D)
        self._run_stage('stage_e', self._stage_e_eda, depends_on=['stage_b', 'stage_c', 'stage_d'])

        pipeline_end = datetime.now()
        total_duration = (pipeline_end - pipeline_start).total_seconds()

        # Print summary
        self._print_summary(total_duration)

        # Save results
        self._save_results()

        return self.results

    def _print_summary(self, total_duration: float) -> None:
        """Print pipeline summary."""
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Summary")
        logger.info("=" * 60)

        for stage_name, result in self.results.items():
            status_icon = {
                StageStatus.COMPLETED: "✓",
                StageStatus.FAILED: "✗",
                StageStatus.SKIPPED: "○",
                StageStatus.PENDING: "·",
                StageStatus.RUNNING: "…"
            }.get(result.status, "?")

            logger.info(
                f"{status_icon} {stage_name}: {result.status.value} "
                f"({result.duration_seconds:.1f}s)"
            )
            if result.error:
                logger.info(f"    Error: {result.error}")

        logger.info(f"\nTotal Duration: {total_duration:.1f}s")
        logger.info("=" * 60)

    def _save_results(self) -> None:
        """Save pipeline results to file."""
        results_dict = {}
        for stage_name, result in self.results.items():
            results_dict[stage_name] = {
                'status': result.status.value,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'duration_seconds': result.duration_seconds,
                'error': result.error
            }

        results_path = self.storage.join_path(
            self.config.artifacts_dir,
            'pipeline_results.json'
        )
        self.storage.write_text(results_path, json.dumps(results_dict, indent=2))
        logger.info(f"Results saved to: {results_path}")


def run_pipeline_from_config(cfg) -> Dict[str, StageResult]:
    """
    Run the ingestion pipeline from a Hydra configuration.

    Args:
        cfg: Hydra DictConfig object

    Returns:
        Dictionary of stage results
    """
    # Create storage backend
    storage = create_storage_backend(cfg)

    # Create pipeline config from Hydra config
    config = PipelineConfig(
        raw_dir=cfg.storage.paths.raw,
        processed_dir=cfg.storage.paths.processed,
        artifacts_dir=cfg.storage.paths.artifacts,
        cache_dir=cfg.get('cache_dir', './data_cache'),
        error_handling=ErrorHandling(cfg.pipeline.get('error_handling', 'stop')),
        max_workers=cfg.get('processing', {}).get('max_workers', 8),
        seed=cfg.get('seed', 42)
    )

    # Configure enabled stages
    if 'stages' in cfg.pipeline:
        for stage in cfg.pipeline.stages:
            stage_key = stage.get('stage_a') or stage.get('stage_b') or \
                       stage.get('stage_c') or stage.get('stage_d') or stage.get('stage_e')
            if stage_key:
                config.stages_enabled[f'stage_{list(stage.keys())[0][-1]}'] = \
                    stage.get('enabled', True)

    # Run pipeline
    pipeline = IngestionPipeline(config=config, storage=storage)
    return pipeline.run()
