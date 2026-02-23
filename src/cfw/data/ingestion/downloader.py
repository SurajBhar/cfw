"""
Dataset Downloader.

Downloads and extracts datasets from URLs with support for:
- Drive&Act dataset from official URLs
- StateFarm dataset from Kaggle
- Retry with exponential backoff
- Hash verification (optional)
- Archive extraction (.zip, .tar.gz, .7z)
- Caching to avoid repeated downloads
"""


import hashlib
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from .storage import StorageBackend, LocalStorage

logger = logging.getLogger(__name__)


@dataclass
class DownloadTask:
    """Represents a dataset download task."""

    name: str
    url: str
    output_dir: str
    extract: bool = True
    expected_hash: Optional[str] = None
    hash_algorithm: str = 'sha256'
    archive_format: Optional[str] = None  # auto-detect if None


# Drive&Act dataset URLs
DRIVEACT_URLS = {
    'kinect_color': {
        'name': 'Drive&Act Kinect Color (RGB)',
        'url': 'https://driveandact.com/dataset/kinect_color.zip',
        'description': 'Right Top View - Used for Model Training + Evaluation'
    },
    'kinect_ir': {
        'name': 'Drive&Act Kinect IR',
        'url': 'https://driveandact.com/dataset/kinect_ir.zip',
        'description': 'Right Top View - Cross-Modality Generalization'
    },
    'nir_right_top': {
        'name': 'Drive&Act NIR A-Column',
        'url': 'https://driveandact.com/dataset/a_column_co_driver.zip',
        'description': 'Right Top View - Cross-Modality Generalization'
    },
    'nir_front': {
        'name': 'Drive&Act NIR Inner Mirror',
        'url': 'https://driveandact.com/dataset/inner_mirror.zip',
        'description': 'Front View - Cross-Modality Generalization'
    },
    'annotations': {
        'name': 'Drive&Act Annotations',
        'url': 'https://driveandact.com/dataset/iccv_activities_3s.zip',
        'description': 'Activity annotations for dataset derivation'
    }
}

# Output folders expected by downstream extraction configs/scripts.
DRIVEACT_OUTPUT_DIR_MAP = {
    'kinect_color': 'kinect_color',
    'kinect_ir': 'kinect_ir',
    'nir_right_top': 'a_column_co_driver',
    'nir_front': 'inner_mirror',
    'annotations': 'iccv_activities_3s',
}

# Kaggle StateFarm dataset
STATEFARM_KAGGLE = {
    'name': 'StateFarm Distracted Driver Detection',
    'dataset': 'sinhasau/statefarm-dataset-training-split',
    'url': 'https://www.kaggle.com/api/v1/datasets/download/sinhasau/statefarm-dataset-training-split'
}


class DatasetDownloader:
    """Download and extract datasets with retry and caching support."""

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        cache_dir: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        chunk_size: int = 8192,
        timeout: int = 60
    ):
        """
        Initialize the downloader.

        Args:
            storage: Storage backend for file operations
            cache_dir: Directory to cache downloaded files
            max_retries: Maximum number of download retries
            retry_delay: Initial delay between retries (exponential backoff)
            chunk_size: Download chunk size in bytes
            timeout: Request timeout in seconds
        """
        self.storage = storage or LocalStorage()
        self.cache_dir = Path(cache_dir) if cache_dir else Path('./data_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.chunk_size = chunk_size
        self.timeout = timeout

    def _get_cache_path(self, url: str) -> Path:
        """Get cache path for a URL."""
        # Use URL hash as cache filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
        filename = urlparse(url).path.split('/')[-1] or f"download_{url_hash}"
        return self.cache_dir / filename

    def _verify_hash(
        self,
        file_path: Path,
        expected_hash: str,
        algorithm: str = 'sha256'
    ) -> bool:
        """
        Verify file hash.

        Args:
            file_path: Path to the file
            expected_hash: Expected hash value
            algorithm: Hash algorithm (sha256, md5, etc.)

        Returns:
            True if hash matches
        """
        hasher = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest()

        if actual_hash != expected_hash:
            logger.error(
                f"Hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
            return False
        return True

    def _download_with_retry(
        self,
        url: str,
        output_path: Path,
        progress_callback: Optional[Callable] = None
    ) -> bool:
        """
        Download a file with exponential backoff retry.

        Args:
            url: URL to download
            output_path: Path to save the file
            progress_callback: Optional callback for progress updates

        Returns:
            True if download successful
        """
        delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading {url} (attempt {attempt + 1}/{self.max_retries})")

                response = requests.get(
                    url,
                    stream=True,
                    timeout=self.timeout,
                    allow_redirects=True
                )
                response.raise_for_status()

                # Get total size for progress bar
                total_size = int(response.headers.get('content-length', 0))

                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'wb') as f:
                    with tqdm(
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        desc=output_path.name
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                                if progress_callback:
                                    progress_callback(len(chunk))

                logger.info(f"Download complete: {output_path}")
                return True

            except requests.RequestException as e:
                logger.warning(f"Download failed: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to download after {self.max_retries} attempts")
                    return False

        return False

    def _detect_archive_format(self, file_path: Path) -> Optional[str]:
        """Detect archive format from file extension or magic bytes."""
        suffix = file_path.suffix.lower()

        if suffix == '.zip':
            return 'zip'
        elif suffix in ['.tar', '.gz', '.tgz']:
            return 'tar'
        elif suffix == '.7z':
            return '7z'
        elif suffix == '.tar.gz':
            return 'tar.gz'

        # Check file content
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header[:2] == b'PK':  # ZIP magic bytes
                return 'zip'
            elif header[:2] == b'\x1f\x8b':  # GZIP magic bytes
                return 'tar.gz'

        return None

    def _extract_archive(
        self,
        archive_path: Path,
        output_dir: Path,
        archive_format: Optional[str] = None
    ) -> bool:
        """
        Extract an archive file.

        Args:
            archive_path: Path to the archive
            output_dir: Directory to extract to
            archive_format: Archive format (auto-detect if None)

        Returns:
            True if extraction successful
        """
        if archive_format is None:
            archive_format = self._detect_archive_format(archive_path)

        if archive_format is None:
            logger.error(f"Could not detect archive format: {archive_path}")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if archive_format == 'zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    # Get total size for progress
                    total_size = sum(f.file_size for f in zf.filelist)
                    extracted = 0

                    with tqdm(total=total_size, unit='B', unit_scale=True,
                              desc="Extracting") as pbar:
                        for file_info in zf.filelist:
                            zf.extract(file_info, output_dir)
                            extracted += file_info.file_size
                            pbar.update(file_info.file_size)

            elif archive_format in ['tar', 'tar.gz']:
                mode = 'r:gz' if archive_format == 'tar.gz' else 'r'
                with tarfile.open(archive_path, mode) as tf:
                    tf.extractall(output_dir)

            elif archive_format == '7z':
                # Requires 7z command-line tool
                try:
                    subprocess.run(
                        ['7z', 'x', str(archive_path), f'-o{output_dir}', '-y'],
                        check=True,
                        capture_output=True
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"7z extraction failed: {e}")
                    return False
                except FileNotFoundError:
                    logger.error("7z command not found. Install p7zip.")
                    return False

            logger.info(f"Extracted to {output_dir}")
            return True

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False

    def download(
        self,
        task: DownloadTask,
        use_cache: bool = True,
        delete_archive: bool = True
    ) -> bool:
        """
        Download and optionally extract a dataset.

        Args:
            task: Download task specification
            use_cache: Whether to use cached downloads
            delete_archive: Whether to delete archive after extraction

        Returns:
            True if download and extraction successful
        """
        logger.info(f"Processing: {task.name}")
        logger.info(f"URL: {task.url}")
        logger.info(f"Output: {task.output_dir}")

        cache_path = self._get_cache_path(task.url)

        # Check cache
        if use_cache and cache_path.exists():
            logger.info(f"Using cached file: {cache_path}")
        else:
            # Download
            if not self._download_with_retry(task.url, cache_path):
                return False

        # Verify hash if provided
        if task.expected_hash:
            if not self._verify_hash(cache_path, task.expected_hash, task.hash_algorithm):
                logger.error("Hash verification failed")
                cache_path.unlink()  # Remove corrupted file
                return False
            logger.info("Hash verification passed")

        # Extract if needed
        if task.extract:
            output_dir = Path(task.output_dir)
            if not self._extract_archive(cache_path, output_dir, task.archive_format):
                return False

            if delete_archive:
                logger.info(f"Deleting archive: {cache_path}")
                cache_path.unlink()
        else:
            # Just copy to output
            output_path = Path(task.output_dir) / cache_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_path, output_path)

        return True

    def download_driveact(
        self,
        output_dir: str,
        components: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Dict[str, bool]:
        """
        Download Drive&Act dataset components.

        Args:
            output_dir: Output directory for downloaded data
            components: List of components to download (default: all)
                        Options: kinect_color, kinect_ir, nir_right_top, nir_front, annotations
            use_cache: Whether to use cached downloads

        Returns:
            Dictionary mapping component names to success status
        """
        if components is None:
            components = list(DRIVEACT_URLS.keys())

        results = {}

        for component in components:
            if component not in DRIVEACT_URLS:
                logger.warning(f"Unknown component: {component}")
                continue

            info = DRIVEACT_URLS[component]
            component_output_dir = DRIVEACT_OUTPUT_DIR_MAP.get(component, component)
            task = DownloadTask(
                name=info['name'],
                url=info['url'],
                output_dir=os.path.join(output_dir, component_output_dir)
            )

            results[component] = self.download(task, use_cache=use_cache)

        return results

    def download_statefarm_kaggle(
        self,
        output_dir: str,
        kaggle_credentials: Optional[str] = None,
        use_cache: bool = True
    ) -> bool:
        """
        Download StateFarm dataset from Kaggle.

        Requires Kaggle API credentials (kaggle.json) or curl.

        Args:
            output_dir: Output directory
            kaggle_credentials: Path to kaggle.json (optional)
            use_cache: Whether to use cached downloads

        Returns:
            True if successful
        """
        logger.info(f"Downloading StateFarm dataset from Kaggle")

        cache_path = self.cache_dir / "statefarm-dataset-training-split.zip"

        # Check cache
        if use_cache and cache_path.exists():
            logger.info(f"Using cached file: {cache_path}")
        else:
            # Try using Kaggle CLI first
            try:
                kaggle_dir = Path.home() / '.kaggle'
                if kaggle_credentials:
                    kaggle_dir.mkdir(exist_ok=True)
                    shutil.copy(kaggle_credentials, kaggle_dir / 'kaggle.json')
                    os.chmod(kaggle_dir / 'kaggle.json', 0o600)

                # Download using kaggle CLI
                subprocess.run([
                    'kaggle', 'datasets', 'download',
                    '-d', STATEFARM_KAGGLE['dataset'],
                    '-p', str(self.cache_dir)
                ], check=True)

            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.info("Kaggle CLI not available, trying direct download")

                # Fallback to curl/direct download
                if not self._download_with_retry(STATEFARM_KAGGLE['url'], cache_path):
                    return False

        # Extract
        output_path = Path(output_dir)
        if not self._extract_archive(cache_path, output_path):
            return False

        logger.info(f"StateFarm dataset downloaded to {output_dir}")
        return True


def download_all_datasets(
    raw_data_dir: str,
    driveact_components: Optional[List[str]] = None,
    include_statefarm: bool = True,
    cache_dir: Optional[str] = None
) -> Dict[str, bool]:
    """
    Download all required datasets.

    Args:
        raw_data_dir: Base directory for raw data
        driveact_components: Drive&Act components to download (default: all)
        include_statefarm: Whether to download StateFarm dataset
        cache_dir: Cache directory for downloads

    Returns:
        Dictionary with download status for each dataset
    """
    downloader = DatasetDownloader(cache_dir=cache_dir)
    results = {}

    # Download Drive&Act
    driveact_dir = os.path.join(raw_data_dir, 'driveandact')
    driveact_results = downloader.download_driveact(
        output_dir=driveact_dir,
        components=driveact_components
    )
    results.update({f'driveact_{k}': v for k, v in driveact_results.items()})

    # Download StateFarm
    if include_statefarm:
        statefarm_dir = os.path.join(raw_data_dir, 'statefarm')
        results['statefarm'] = downloader.download_statefarm_kaggle(
            output_dir=statefarm_dir
        )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)
    for name, status in results.items():
        status_str = "✓ Success" if status else "✗ Failed"
        logger.info(f"{name}: {status_str}")

    return results
