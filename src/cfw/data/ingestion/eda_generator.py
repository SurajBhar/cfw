"""
EDA (Exploratory Data Analysis) Generator.

Generates statistics and integrity reports for all dataset variants:
- Class distribution per split
- Imbalance ratios
- Dataset sizes (image counts, disk usage)
- Integrity checks (corrupt images, split overlap, missing files)
"""


import json
import logging
import hashlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from tqdm import tqdm

from .storage import StorageBackend, LocalStorage

logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
OVERLAP_REPORT_LIMIT = 200


@dataclass
class ClassDistribution:
    """Class distribution statistics."""

    class_name: str
    count: int
    percentage: float


@dataclass
class SplitStatistics:
    """Statistics for a single split."""

    split_name: str
    total_images: int
    class_distribution: List[ClassDistribution]
    imbalance_ratio: float  # max_count / min_count
    disk_size_bytes: int


@dataclass
class DatasetStatistics:
    """Complete statistics for a dataset."""

    dataset_name: str
    dataset_type: str  # multiclass or binary
    splits: Dict[str, SplitStatistics]
    total_images: int
    total_disk_size_bytes: int
    num_classes: int
    class_names: List[str]


@dataclass
class IntegrityReport:
    """Dataset integrity report."""

    dataset_name: str
    checked_images: int
    corrupt_images: List[str]
    missing_files: List[str]
    split_overlaps: Dict[str, List[str]]
    passed: bool


def _check_image_integrity(image_path: str) -> Tuple[str, bool, Optional[str]]:
    """
    Check if an image file is valid.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (path, is_valid, error_message)
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img.verify()
        return image_path, True, None
    except Exception as e:
        return image_path, False, str(e)


def _normalize_path_str(path: str) -> str:
    """Normalize path separators for stable string operations."""
    return str(path).replace("\\", "/")


class EDAGenerator:
    """Generate EDA statistics and integrity reports for datasets."""

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        max_workers: int = 8,
        check_image_integrity: bool = False
    ):
        """
        Initialize the EDA generator.

        Args:
            storage: Storage backend
            max_workers: Number of parallel workers
            check_image_integrity: Whether to verify image files (slow)
        """
        self.storage = storage or LocalStorage()
        self.max_workers = max_workers
        self.check_image_integrity = check_image_integrity

    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        try:
            return self.storage.get_size(file_path)
        except Exception:
            return 0

    def _has_standard_splits(self, dataset_dir: str) -> bool:
        """Return True if dataset_dir contains train/val/test folders directly."""
        return any(
            self.storage.exists(self.storage.join_path(dataset_dir, split))
            for split in ["train", "val", "test"]
        )

    def _resolve_dataset_root(
        self,
        dataset_dir: str,
        dataset_name: Optional[str] = None,
    ) -> str:
        """
        Resolve dataset root that actually contains train/val/test.

        Supports both:
        - <dataset_dir>/{train,val,test}
        - <dataset_dir>/split_*/{train,val,test}
        """
        if self._has_standard_splits(dataset_dir):
            return dataset_dir

        split_dirs = [
            d for d in self.storage.list_dirs(dataset_dir)
            if self.storage.get_filename(d).startswith("split_")
            and self._has_standard_splits(d)
        ]

        if not split_dirs:
            return dataset_dir

        split_dirs = sorted(split_dirs, key=self.storage.get_filename)
        split_names = [self.storage.get_filename(d) for d in split_dirs]

        # Prefer split_0 when available (common training setup in this project).
        preferred = None
        for d in split_dirs:
            if self.storage.get_filename(d) == "split_0":
                preferred = d
                break

        if preferred is None:
            preferred = split_dirs[0]

        logger.info(
            "Resolved nested split dataset '%s': using %s from candidates %s",
            dataset_name or self.storage.get_filename(dataset_dir),
            preferred,
            split_names,
        )
        return preferred

    def _expected_extensions(self, dataset_name: Optional[str]) -> Set[str]:
        """
        Return expected image extensions for a dataset.

        Defaults:
        - Drive&Act datasets (`daa_*`): png
        - StateFarm datasets (`statefarm_*`): jpg/jpeg
        - Fallback: png/jpg/jpeg
        """
        if not dataset_name:
            return IMAGE_EXTENSIONS

        name = dataset_name.lower()
        if name.startswith("daa_"):
            return {".png"}
        if name.startswith("statefarm_"):
            return {".jpg", ".jpeg"}
        return IMAGE_EXTENSIONS

    def _list_images(
        self,
        path: str,
        recursive: bool = True,
        extensions: Optional[Set[str]] = None,
    ) -> List[str]:
        """List image files in one traversal and filter by extension."""
        ext_set = extensions or IMAGE_EXTENSIONS
        files = self.storage.list_files(path, recursive=recursive)
        return [
            f for f in files
            if Path(f).suffix.lower() in ext_set
        ]

    def _relative_to_root(self, path: str, root: str) -> str:
        """Return normalized path relative to root when possible."""
        norm_path = _normalize_path_str(path)
        norm_root = _normalize_path_str(root).rstrip("/")
        prefix = norm_root + "/"
        if norm_path.startswith(prefix):
            return norm_path[len(prefix):]
        return self.storage.get_filename(path)

    def _content_hash(self, file_path: str) -> str:
        """Compute a stable content hash for overlap detection."""
        data = self.storage.read_bytes(file_path)
        return hashlib.sha1(data).hexdigest()

    def _collect_class_stats(
        self,
        class_dir: str,
        extensions: Set[str],
    ) -> Tuple[str, int, int]:
        """Collect image count and disk usage for a single class directory."""
        class_name = self.storage.get_filename(class_dir)
        images = self._list_images(class_dir, recursive=True, extensions=extensions)
        total_size = sum(self._get_file_size(img) for img in images)
        return class_name, len(images), total_size

    def _collect_split_statistics(
        self,
        split_dir: str,
        split_name: str,
        dataset_name: Optional[str] = None,
    ) -> SplitStatistics:
        """
        Collect statistics for a single split.

        Args:
            split_dir: Path to split directory
            split_name: Name of the split

        Returns:
            SplitStatistics object
        """
        class_counts = defaultdict(int)
        total_size = 0
        extensions = self._expected_extensions(dataset_name)

        # Get class directories
        class_dirs = self.storage.list_dirs(split_dir)
        if class_dirs:
            workers = max(1, min(self.max_workers, len(class_dirs)))
            logger.info(
                "Collecting split '%s' stats: %d classes (%d workers, extensions=%s)",
                split_name,
                len(class_dirs),
                workers,
                sorted(extensions),
            )

            if workers == 1:
                for class_dir in class_dirs:
                    class_name, count, class_size = self._collect_class_stats(
                        class_dir, extensions
                    )
                    class_counts[class_name] = count
                    total_size += class_size
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    class_worker = partial(self._collect_class_stats, extensions=extensions)
                    for class_name, count, class_size in executor.map(
                        class_worker, class_dirs
                    ):
                        class_counts[class_name] = count
                        total_size += class_size

        # Calculate distribution
        total_images = sum(class_counts.values())
        class_distribution = []

        for class_name, count in sorted(class_counts.items()):
            percentage = (count / total_images * 100) if total_images > 0 else 0
            class_distribution.append(ClassDistribution(
                class_name=class_name,
                count=count,
                percentage=round(percentage, 2)
            ))

        # Calculate imbalance ratio
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        else:
            imbalance_ratio = 1.0

        return SplitStatistics(
            split_name=split_name,
            total_images=total_images,
            class_distribution=class_distribution,
            imbalance_ratio=round(imbalance_ratio, 2),
            disk_size_bytes=total_size
        )

    def generate_statistics(
        self,
        dataset_dir: str,
        dataset_name: Optional[str] = None
    ) -> DatasetStatistics:
        """
        Generate comprehensive statistics for a dataset.

        Args:
            dataset_dir: Path to dataset directory
            dataset_name: Name of the dataset (auto-detected if None)

        Returns:
            DatasetStatistics object
        """
        if dataset_name is None:
            dataset_name = self.storage.get_filename(dataset_dir)

        dataset_dir = self._resolve_dataset_root(dataset_dir, dataset_name)
        logger.info(f"Generating statistics for: {dataset_name}")

        splits = {}
        all_classes = set()

        for split_name in ['train', 'val', 'test']:
            split_dir = self.storage.join_path(dataset_dir, split_name)
            if not self.storage.exists(split_dir):
                continue

            stats = self._collect_split_statistics(
                split_dir=split_dir,
                split_name=split_name,
                dataset_name=dataset_name,
            )
            splits[split_name] = stats

            # Collect class names
            for dist in stats.class_distribution:
                all_classes.add(dist.class_name)

        # Detect dataset type
        class_names = sorted(all_classes)
        if len(class_names) == 2 and 'distracted' in class_names:
            dataset_type = 'binary'
        else:
            dataset_type = 'multiclass'

        # Calculate totals
        total_images = sum(s.total_images for s in splits.values())
        total_size = sum(s.disk_size_bytes for s in splits.values())

        return DatasetStatistics(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            splits=splits,
            total_images=total_images,
            total_disk_size_bytes=total_size,
            num_classes=len(class_names),
            class_names=class_names
        )

    def check_integrity(
        self,
        dataset_dir: str,
        dataset_name: Optional[str] = None,
        sample_ratio: float = 1.0
    ) -> IntegrityReport:
        """
        Check dataset integrity.

        Args:
            dataset_dir: Path to dataset directory
            dataset_name: Name of the dataset
            sample_ratio: Ratio of images to check (1.0 = all)

        Returns:
            IntegrityReport object
        """
        if dataset_name is None:
            dataset_name = self.storage.get_filename(dataset_dir)

        dataset_dir = self._resolve_dataset_root(dataset_dir, dataset_name)
        logger.info(f"Checking integrity for: {dataset_name}")

        corrupt_images = []
        missing_files = []
        split_files = {}
        split_hash_to_paths: Dict[str, Dict[str, List[str]]] = {}
        extensions = self._expected_extensions(dataset_name)

        # Collect all images per split
        for split_name in ['train', 'val', 'test']:
            split_dir = self.storage.join_path(dataset_dir, split_name)
            if not self.storage.exists(split_dir):
                continue

            images = self._list_images(split_dir, recursive=True, extensions=extensions)
            split_files[split_name] = set()
            split_hash_to_paths[split_name] = defaultdict(list)

            # Build hash index for robust overlap checking.
            workers = max(1, min(self.max_workers, len(images))) if images else 1

            def _hash_one(img: str) -> Tuple[str, Optional[str], Optional[str]]:
                try:
                    return img, self._content_hash(img), None
                except Exception as exc:
                    return img, None, str(exc)

            if workers == 1:
                hash_results = (_hash_one(img) for img in images)
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    hash_results = executor.map(_hash_one, images)

            for img_path, img_hash, err in hash_results:
                if err:
                    logger.warning("Failed to hash image for overlap check: %s - %s", img_path, err)
                    missing_files.append(img_path)
                    continue
                if img_hash is None:
                    continue
                split_files[split_name].add(img_hash)
                split_hash_to_paths[split_name][img_hash].append(
                    self._relative_to_root(img_path, split_dir)
                )

            # Check image integrity if enabled
            if self.check_image_integrity:
                import random
                sample_size = int(len(images) * sample_ratio)
                sampled = random.sample(images, min(sample_size, len(images)))

                with tqdm(total=len(sampled), desc=f"Checking {split_name}") as pbar:
                    with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = {
                            executor.submit(_check_image_integrity, img): img
                            for img in sampled
                        }
                        for future in as_completed(futures):
                            path, valid, error = future.result()
                            if not valid:
                                corrupt_images.append(path)
                                logger.warning(f"Corrupt image: {path} - {error}")
                            pbar.update(1)

        # Check for split overlaps
        overlaps = {}
        split_names = list(split_files.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                overlap = split_files[s1] & split_files[s2]
                if overlap:
                    key = f"{s1}_vs_{s2}"
                    details: List[str] = []
                    for idx, h in enumerate(sorted(overlap)):
                        if idx >= OVERLAP_REPORT_LIMIT:
                            break
                        p1 = split_hash_to_paths[s1].get(h, ["<unknown>"])[0]
                        p2 = split_hash_to_paths[s2].get(h, ["<unknown>"])[0]
                        details.append(f"sha1:{h} | {s1}:{p1} | {s2}:{p2}")
                    overlaps[key] = details
                    logger.warning(f"Overlap between {s1} and {s2}: {len(overlap)} files")

        # Calculate total checked
        total_checked = sum(len(files) for files in split_files.values())
        passed = len(corrupt_images) == 0 and len(overlaps) == 0

        return IntegrityReport(
            dataset_name=dataset_name,
            checked_images=total_checked,
            corrupt_images=corrupt_images,
            missing_files=missing_files,
            split_overlaps=overlaps,
            passed=passed
        )

    def generate_report(
        self,
        dataset_dir: str,
        output_dir: str,
        dataset_name: Optional[str] = None,
        check_integrity: bool = True,
        integrity_sample_ratio: float = 1.0,
    ) -> Dict:
        """
        Generate complete EDA report for a dataset.

        Args:
            dataset_dir: Path to dataset directory
            output_dir: Directory to save reports
            dataset_name: Name of the dataset
            check_integrity: Whether to check integrity
            integrity_sample_ratio: Fraction of images sampled for integrity checks

        Returns:
            Dictionary with all reports
        """
        self.storage.makedirs(output_dir)

        # Generate statistics
        stats = self.generate_statistics(dataset_dir, dataset_name)
        if not stats.splits:
            raise ValueError(
                f"No train/val/test splits found for dataset: {dataset_dir}"
            )
        if stats.total_images == 0:
            raise ValueError(
                f"Dataset contains zero images (check path/extensions): {dataset_dir}"
            )

        # Convert to dict for JSON serialization
        stats_dict = {
            'dataset_name': stats.dataset_name,
            'dataset_type': stats.dataset_type,
            'total_images': stats.total_images,
            'total_disk_size_mb': round(stats.total_disk_size_bytes / (1024 * 1024), 2),
            'num_classes': stats.num_classes,
            'class_names': stats.class_names,
            'splits': {}
        }

        for split_name, split_stats in stats.splits.items():
            stats_dict['splits'][split_name] = {
                'total_images': split_stats.total_images,
                'disk_size_mb': round(split_stats.disk_size_bytes / (1024 * 1024), 2),
                'imbalance_ratio': split_stats.imbalance_ratio,
                'class_distribution': {
                    d.class_name: {'count': d.count, 'percentage': d.percentage}
                    for d in split_stats.class_distribution
                }
            }

        # Save statistics
        stats_path = self.storage.join_path(output_dir, 'eda_statistics.json')
        self.storage.write_text(stats_path, json.dumps(stats_dict, indent=2))
        logger.info(f"Statistics saved to: {stats_path}")

        # Extract class distribution separately
        class_dist = {}
        for split_name, split_stats in stats.splits.items():
            class_dist[split_name] = {
                d.class_name: d.count for d in split_stats.class_distribution
            }

        class_dist_path = self.storage.join_path(output_dir, 'eda_class_distribution.json')
        self.storage.write_text(class_dist_path, json.dumps(class_dist, indent=2))

        # Extract imbalance ratios
        imbalance = {
            split_name: split_stats.imbalance_ratio
            for split_name, split_stats in stats.splits.items()
        }
        imbalance_path = self.storage.join_path(output_dir, 'eda_imbalance_ratios.json')
        self.storage.write_text(imbalance_path, json.dumps(imbalance, indent=2))

        # Extract dataset sizes
        sizes = {
            'total_images': stats.total_images,
            'total_disk_size_mb': round(stats.total_disk_size_bytes / (1024 * 1024), 2),
            'splits': {
                split_name: {
                    'images': split_stats.total_images,
                    'disk_size_mb': round(split_stats.disk_size_bytes / (1024 * 1024), 2)
                }
                for split_name, split_stats in stats.splits.items()
            }
        }
        sizes_path = self.storage.join_path(output_dir, 'eda_dataset_sizes.json')
        self.storage.write_text(sizes_path, json.dumps(sizes, indent=2))

        result = {
            'statistics': stats_dict,
            'class_distribution': class_dist,
            'imbalance_ratios': imbalance,
            'dataset_sizes': sizes
        }

        # Check integrity if requested
        if check_integrity:
            integrity = self.check_integrity(
                dataset_dir,
                dataset_name,
                sample_ratio=integrity_sample_ratio,
            )
            integrity_dict = {
                'dataset_name': integrity.dataset_name,
                'checked_images': integrity.checked_images,
                'corrupt_images': integrity.corrupt_images,
                'missing_files': integrity.missing_files,
                'split_overlaps': integrity.split_overlaps,
                'passed': integrity.passed
            }

            integrity_path = self.storage.join_path(output_dir, 'eda_integrity_report.json')
            self.storage.write_text(integrity_path, json.dumps(integrity_dict, indent=2))

            result['integrity'] = integrity_dict

        logger.info(f"EDA report generated in: {output_dir}")
        return result

    def generate_all_reports(
        self,
        processed_dir: str,
        artifacts_dir: str,
        check_integrity: bool = True,
        integrity_sample_ratio: float = 1.0,
    ) -> Dict[str, Dict]:
        """
        Generate EDA reports for all datasets in processed directory.

        Args:
            processed_dir: Directory containing processed datasets
            artifacts_dir: Base directory for artifacts

        Returns:
            Dictionary mapping dataset names to their reports
        """
        logger.info("Generating EDA reports for all datasets")

        all_reports = {}

        # Find all dataset directories
        dataset_dirs = self.storage.list_dirs(processed_dir)
        tasks = []

        for dataset_dir in dataset_dirs:
            dataset_name = self.storage.get_filename(dataset_dir)

            # Skip non-dataset directories
            if not any(
                self.storage.exists(self.storage.join_path(dataset_dir, split))
                for split in ['train', 'val', 'test']
            ):
                continue

            output_dir = self.storage.join_path(artifacts_dir, dataset_name)
            tasks.append((dataset_name, dataset_dir, output_dir))

        if not tasks:
            logger.info("No dataset directories found for EDA.")
        else:
            if self.max_workers >= 4 and len(tasks) > 1:
                dataset_workers = min(len(tasks), max(1, self.max_workers // 2))
            else:
                dataset_workers = min(len(tasks), max(1, self.max_workers))
            per_dataset_workers = max(1, self.max_workers // dataset_workers)
            logger.info(
                "Running EDA in parallel across datasets: %d tasks "
                "(dataset_workers=%d, per_dataset_workers=%d)",
                len(tasks),
                dataset_workers,
                per_dataset_workers,
            )

            def _run_task(task: Tuple[str, str, str]) -> Tuple[str, Dict]:
                dataset_name, dataset_dir, output_dir = task
                local_eda = EDAGenerator(
                    storage=self.storage,
                    max_workers=per_dataset_workers,
                    check_image_integrity=self.check_image_integrity,
                )
                report = local_eda.generate_report(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    dataset_name=dataset_name,
                    check_integrity=check_integrity,
                    integrity_sample_ratio=integrity_sample_ratio,
                )
                return dataset_name, report

            if dataset_workers == 1:
                for task in tasks:
                    dataset_name, dataset_dir, _ = task
                    try:
                        name, report = _run_task(task)
                        all_reports[name] = report
                    except Exception as e:
                        logger.error(f"Failed to generate report for {dataset_name}: {e}")
                        all_reports[dataset_name] = {'error': str(e)}
            else:
                with ThreadPoolExecutor(max_workers=dataset_workers) as executor:
                    future_to_dataset = {
                        executor.submit(_run_task, task): task[0] for task in tasks
                    }
                    for future in as_completed(future_to_dataset):
                        dataset_name = future_to_dataset[future]
                        try:
                            name, report = future.result()
                            all_reports[name] = report
                        except Exception as e:
                            logger.error(f"Failed to generate report for {dataset_name}: {e}")
                            all_reports[dataset_name] = {'error': str(e)}

        # Generate summary
        summary = {
            'total_datasets': len(all_reports),
            'datasets': list(all_reports.keys()),
            'total_images': sum(
                r.get('statistics', {}).get('total_images', 0)
                for r in all_reports.values()
                if 'error' not in r
            )
        }

        summary_path = self.storage.join_path(artifacts_dir, 'eda_summary.json')
        self.storage.write_text(summary_path, json.dumps(summary, indent=2))

        logger.info(f"Generated reports for {len(all_reports)} datasets")
        return all_reports


def print_statistics(stats: DatasetStatistics) -> None:
    """
    Print statistics in a human-readable format.

    Args:
        stats: DatasetStatistics object
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {stats.dataset_name}")
    print(f"Type: {stats.dataset_type}")
    print(f"Classes: {stats.num_classes}")
    print(f"Total Images: {stats.total_images:,}")
    print(f"Disk Size: {stats.total_disk_size_bytes / (1024*1024):.2f} MB")
    print(f"{'='*60}")

    for split_name, split_stats in stats.splits.items():
        print(f"\n{split_name.upper()}")
        print(f"  Total: {split_stats.total_images:,} images")
        print(f"  Imbalance Ratio: {split_stats.imbalance_ratio:.2f}")
        print("  Class Distribution:")
        for dist in split_stats.class_distribution:
            bar = '█' * int(dist.percentage / 5)
            print(f"    {dist.class_name:30} {dist.count:6,} ({dist.percentage:5.1f}%) {bar}")
