"""
Data Ingestion Pipeline for CFW.

This module provides tools for:
- Downloading and extracting datasets (Stage A)
- Frame extraction from videos (Stage B)
- Binary conversion from multi-class (Stage C)
- Dataset organization and splitting (Stage D)
- EDA statistics generation (Stage E)

Supports both local filesystem and Azure Blob Storage backends.
"""


from .storage import (
    StorageBackend,
    LocalStorage,
    AzureBlobStorage,
    create_storage_backend,
    get_storage_paths,
)

from .downloader import (
    DatasetDownloader,
    DownloadTask,
    download_all_datasets,
    DRIVEACT_URLS,
    STATEFARM_KAGGLE,
)

from .frame_extractor import (
    DriveActFrameExtractor,
    DriveActMultiViewExtractor,
    AnnotationRow,
    ExtractionResult,
)

from .binary_converter import (
    BinaryConverter,
    convert_daa_to_binary,
    get_daa_class_mapping,
    create_custom_mapping,
    DAA_DEFAULT_MAPPING,
)

from .split_builder import (
    StateFarmSplitBuilder,
    merge_statefarm_pool,
    STATEFARM_CLASSES,
    BINARY_MAPPING,
)

from .eda_generator import (
    EDAGenerator,
    DatasetStatistics,
    SplitStatistics,
    ClassDistribution,
    IntegrityReport,
    print_statistics,
)

from .pipeline import (
    IngestionPipeline,
    PipelineConfig,
    StageResult,
    StageStatus,
    ErrorHandling,
    run_pipeline_from_config,
)

__all__ = [
    # Storage
    'StorageBackend',
    'LocalStorage',
    'AzureBlobStorage',
    'create_storage_backend',
    'get_storage_paths',
    # Downloader
    'DatasetDownloader',
    'DownloadTask',
    'download_all_datasets',
    'DRIVEACT_URLS',
    'STATEFARM_KAGGLE',
    # Frame Extractor
    'DriveActFrameExtractor',
    'DriveActMultiViewExtractor',
    'AnnotationRow',
    'ExtractionResult',
    # Binary Converter
    'BinaryConverter',
    'convert_daa_to_binary',
    'get_daa_class_mapping',
    'create_custom_mapping',
    'DAA_DEFAULT_MAPPING',
    # Split Builder
    'StateFarmSplitBuilder',
    'merge_statefarm_pool',
    'STATEFARM_CLASSES',
    'BINARY_MAPPING',
    # EDA Generator
    'EDAGenerator',
    'DatasetStatistics',
    'SplitStatistics',
    'ClassDistribution',
    'IntegrityReport',
    'print_statistics',
    # Pipeline
    'IngestionPipeline',
    'PipelineConfig',
    'StageResult',
    'StageStatus',
    'ErrorHandling',
    'run_pipeline_from_config',
]
