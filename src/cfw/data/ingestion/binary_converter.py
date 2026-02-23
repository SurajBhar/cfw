"""
Binary Dataset Converter.

Converts multi-class datasets to binary classification format
(distracted vs. non-distracted) with configurable class mappings.
"""


import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .storage import StorageBackend, LocalStorage

logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# Default Drive&Act class mapping for binary conversion
DAA_DEFAULT_MAPPING = {
    'non_distracted': [
        'sitting_still',
        'entering_car',
        'exiting_car'
    ],
    'distracted': [
        'closing_bottle',
        'closing_door_inside',
        'closing_door_outside',
        'closing_laptop',
        'drinking',
        'eating',
        'fastening_seat_belt',
        'fetching_an_object',
        'interacting_with_phone',
        'looking_or_moving_around_(e.g._searching)',
        'opening_backpack',
        'opening_bottle',
        'opening_door_inside',
        'opening_door_outside',
        'opening_laptop',
        'placing_an_object',
        'preparing_food',
        'pressing_automation_button',
        'putting_laptop_into_backpack',
        'putting_on_jacket',
        'putting_on_sunglasses',
        'reading_magazine',
        'reading_newspaper',
        'taking_laptop_from_backpack',
        'taking_off_jacket',
        'taking_off_sunglasses',
        'talking_on_phone',
        'unfastening_seat_belt',
        'using_multimedia_display',
        'working_on_laptop',
        'writing'
    ]
}


class BinaryConverter:
    """
    Convert multi-class datasets to binary format.

    Supports converting both Drive&Act and StateFarm multi-class
    datasets to binary (distracted vs non-distracted) classification.
    """

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        max_workers: int = 8,
        class_mapping: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize the binary converter.

        Args:
            storage: Storage backend (defaults to LocalStorage)
            max_workers: Number of parallel workers for file operations
            class_mapping: Dictionary mapping binary classes to source classes
                          Expected keys: 'non_distracted', 'distracted'
        """
        self.storage = storage or LocalStorage()
        self.max_workers = max_workers
        self.class_mapping = class_mapping or DAA_DEFAULT_MAPPING

        # Validate mapping
        required_keys = {'non_distracted', 'distracted'}
        if not required_keys.issubset(set(self.class_mapping.keys())):
            raise ValueError(
                f"class_mapping must have keys: {required_keys}"
            )

    def _get_binary_label(self, multiclass_label: str) -> Optional[str]:
        """
        Get binary label for a multiclass label.

        Args:
            multiclass_label: Original multi-class label

        Returns:
            Binary label ('non_distracted' or 'distracted') or None if not mapped
        """
        # Clean up label (handle variations in naming)
        label = multiclass_label.lower().strip()

        for binary_class, source_classes in self.class_mapping.items():
            # Check exact match first
            if label in [s.lower() for s in source_classes]:
                return binary_class

            # Check if label contains any source class (partial match)
            for source in source_classes:
                if source.lower() in label or label in source.lower():
                    return binary_class

        # Default: assume it's distracted if not in non_distracted
        non_distracted_labels = [s.lower() for s in self.class_mapping['non_distracted']]
        if label not in non_distracted_labels:
            return 'distracted'

        return None

    def _list_images(
        self,
        path: str,
        recursive: bool = True,
    ) -> List[str]:
        """
        List image files under a directory.

        Uses a single filesystem traversal and filters by extension to avoid
        multiple expensive recursive scans.
        """
        files = self.storage.list_files(path, recursive=recursive)
        return [
            f for f in files
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        ]

    def _plan_conversion(
        self,
        source_dir: str,
        dest_dir: str,
        split_name: str
    ) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
        """
        Plan the file mapping for conversion.

        Args:
            source_dir: Source multi-class directory
            dest_dir: Destination binary directory
            split_name: Name of the split (train/val/test)

        Returns:
            Tuple of (file_mapping, class_counts)
        """
        file_mapping = []
        class_counts = {'non_distracted': 0, 'distracted': 0}

        source_split_dir = self.storage.join_path(source_dir, split_name)
        if not self.storage.exists(source_split_dir):
            logger.warning(f"Source split not found: {source_split_dir}")
            return file_mapping, class_counts

        # Get all class directories
        class_dirs = sorted(self.storage.list_dirs(source_split_dir))
        logger.info(
            "Planning conversion for split '%s': %d source classes",
            split_name,
            len(class_dirs),
        )

        for idx, class_dir in enumerate(class_dirs, 1):
            class_name = self.storage.get_filename(class_dir)
            binary_label = self._get_binary_label(class_name)

            if binary_label is None:
                logger.warning(f"No mapping for class: {class_name}, defaulting to distracted")
                binary_label = 'distracted'

            logger.info(
                "  [%d/%d] Scanning class '%s' -> '%s'",
                idx,
                len(class_dirs),
                class_name,
                binary_label,
            )

            # Get all images (handle nested chunk directories)
            images = self._list_images(class_dir, recursive=True)
            logger.info(
                "  [%d/%d] Found %d images in class '%s'",
                idx,
                len(class_dirs),
                len(images),
                class_name,
            )

            for img_path in images:
                # Flatten chunk structure in binary dataset
                filename = self.storage.get_filename(img_path)
                dest_path = self.storage.join_path(
                    dest_dir, split_name, binary_label, filename
                )
                file_mapping.append((img_path, dest_path))
                class_counts[binary_label] += 1

        logger.info(
            "Planned %d files for split '%s' (non_distracted=%d, distracted=%d)",
            len(file_mapping),
            split_name,
            class_counts["non_distracted"],
            class_counts["distracted"],
        )
        return file_mapping, class_counts

    def _execute_conversion(
        self,
        file_mapping: List[Tuple[str, str]],
        desc: str = "Converting"
    ) -> int:
        """
        Execute the planned file mapping.

        Args:
            file_mapping: List of (source, destination) tuples
            desc: Description for progress bar

        Returns:
            Number of files copied successfully
        """
        # Create destination directories
        dest_dirs = set(self.storage.get_parent(dst) for _, dst in file_mapping)
        for d in dest_dirs:
            self.storage.makedirs(d)

        workers = max(1, int(self.max_workers))
        logger.info("Copy mode: parallel (%d workers)", workers)

        def _copy_one(item: Tuple[str, str]) -> Tuple[bool, str, Optional[str]]:
            src, dst = item
            try:
                self.storage.copy_file(src, dst)
                return True, src, None
            except Exception as e:
                return False, src, str(e)

        success_count = 0
        with tqdm(total=len(file_mapping), desc=desc) as pbar:
            if workers == 1:
                for item in file_mapping:
                    ok, src, error = _copy_one(item)
                    if ok:
                        success_count += 1
                    else:
                        logger.warning("Failed to copy %s: %s", src, error)
                    pbar.update(1)
                return success_count

            with ThreadPoolExecutor(max_workers=workers) as executor:
                for ok, src, error in executor.map(_copy_one, file_mapping):
                    if ok:
                        success_count += 1
                    else:
                        logger.warning("Failed to copy %s: %s", src, error)
                    pbar.update(1)

        return success_count

    def _rename_sequential(
        self,
        binary_dir: str,
        split_name: str
    ) -> None:
        """
        Rename files sequentially within each binary class.

        Args:
            binary_dir: Binary dataset directory
            split_name: Split name (train/val/test)
        """
        for binary_class in ['non_distracted', 'distracted']:
            class_dir = self.storage.join_path(binary_dir, split_name, binary_class)
            if not self.storage.exists(class_dir):
                continue

            images = self._list_images(class_dir, recursive=False)
            images = sorted(images)
            logger.info(
                "Renaming %d files in %s/%s sequentially",
                len(images),
                split_name,
                binary_class,
            )

            for idx, old_path in enumerate(tqdm(
                images, desc=f"Renaming {split_name}/{binary_class}"
            )):
                ext = Path(old_path).suffix
                new_filename = f"img_{idx:06d}{ext}"
                new_path = self.storage.join_path(class_dir, new_filename)

                if old_path != new_path:
                    # Create temp name to avoid conflicts
                    temp_path = self.storage.join_path(class_dir, f"_temp_{idx}{ext}")
                    self.storage.move_file(old_path, temp_path)

            # Second pass: rename from temp to final
            temp_files = self.storage.list_files(class_dir, pattern='_temp_*')
            for temp_path in temp_files:
                filename = self.storage.get_filename(temp_path)
                idx = int(filename.split('_')[1])
                ext = Path(temp_path).suffix
                new_filename = f"img_{idx:06d}{ext}"
                new_path = self.storage.join_path(class_dir, new_filename)
                self.storage.move_file(temp_path, new_path)

    def convert_split(
        self,
        source_dir: str,
        dest_dir: str,
        split_name: str,
        rename_sequential: bool = True
    ) -> Dict:
        """
        Convert a single split from multiclass to binary.

        Args:
            source_dir: Source multi-class directory
            dest_dir: Destination binary directory
            split_name: Split name (train/val/test)
            rename_sequential: Whether to rename files sequentially

        Returns:
            Conversion statistics
        """
        logger.info(f"Converting {split_name} split to binary")

        # Plan conversion
        file_mapping, class_counts = self._plan_conversion(
            source_dir, dest_dir, split_name
        )

        if not file_mapping:
            logger.warning(f"No files found for {split_name}")
            return {'split': split_name, 'counts': class_counts, 'success': 0}

        logger.info(
            "Starting copy for split '%s' with %d files",
            split_name,
            len(file_mapping),
        )

        # Execute conversion
        success_count = self._execute_conversion(
            file_mapping, desc=f"Converting {split_name}"
        )

        # Optionally rename sequentially
        if rename_sequential:
            self._rename_sequential(dest_dir, split_name)

        stats = {
            'split': split_name,
            'counts': class_counts,
            'total': sum(class_counts.values()),
            'success': success_count
        }

        logger.info(
            f"{split_name}: non_distracted={class_counts['non_distracted']}, "
            f"distracted={class_counts['distracted']}"
        )

        return stats

    def convert_dataset(
        self,
        source_dir: str,
        dest_dir: str,
        splits: Optional[List[str]] = None,
        rename_sequential: bool = True,
        verify_counts: bool = True,
    ) -> Dict:
        """
        Convert entire multi-class dataset to binary.

        Args:
            source_dir: Source multi-class dataset directory
            dest_dir: Destination binary dataset directory
            splits: List of splits to convert (default: train, val, test)
            rename_sequential: Whether to rename files sequentially
            verify_counts: Whether to verify source/destination counts

        Returns:
            Dictionary with statistics for all splits
        """
        logger.info("Converting multi-class dataset to binary")
        logger.info(f"Source: {source_dir}")
        logger.info(f"Destination: {dest_dir}")

        if splits is None:
            splits = ['train', 'val', 'test']

        all_stats = {}

        for split_name in splits:
            stats = self.convert_split(
                source_dir=source_dir,
                dest_dir=dest_dir,
                split_name=split_name,
                rename_sequential=rename_sequential
            )
            all_stats[split_name] = stats

        # Verify counts
        if verify_counts:
            self._verify_conversion(source_dir, dest_dir, splits)
        else:
            logger.info("Skipping count verification (verify_counts=false)")

        # Save manifest
        manifest = {
            'dataset_type': 'binary',
            'source_dir': source_dir,
            'class_mapping': self.class_mapping,
            'splits': all_stats,
            'total': {
                split: stats['total']
                for split, stats in all_stats.items()
            }
        }

        manifest_path = self.storage.join_path(dest_dir, 'manifest.json')
        self.storage.write_text(manifest_path, json.dumps(manifest, indent=2))

        logger.info("Binary conversion complete")
        for split, stats in all_stats.items():
            logger.info(f"{split}: {stats['total']} images")

        return manifest

    def _verify_conversion(
        self,
        source_dir: str,
        dest_dir: str,
        splits: List[str]
    ) -> bool:
        """
        Verify that conversion preserved all images.

        Args:
            source_dir: Source multi-class directory
            dest_dir: Destination binary directory
            splits: List of splits to verify

        Returns:
            True if verification passed
        """
        logger.info("Verifying conversion")
        all_passed = True

        for split_name in splits:
            # Count source images
            source_split = self.storage.join_path(source_dir, split_name)
            if not self.storage.exists(source_split):
                continue

            source_images = self._list_images(source_split, recursive=True)
            source_count = len(source_images)

            # Count dest images
            dest_split = self.storage.join_path(dest_dir, split_name)
            dest_images = self._list_images(dest_split, recursive=True)
            dest_count = len(dest_images)

            if source_count != dest_count:
                logger.error(
                    f"Count mismatch in {split_name}: "
                    f"source={source_count}, dest={dest_count}"
                )
                all_passed = False
            else:
                logger.info(f"✓ {split_name}: {source_count} images verified")

        return all_passed


def convert_daa_to_binary(
    multiclass_dir: str,
    binary_dir: str,
    storage: Optional[StorageBackend] = None,
    class_mapping: Optional[Dict[str, List[str]]] = None,
    splits: Optional[List[str]] = None
) -> Dict:
    """
    Convert Drive&Act multiclass data to binary labels.

    Args:
        multiclass_dir: Path to DAA multi-class dataset
        binary_dir: Output path for binary dataset
        storage: Optional storage backend
        class_mapping: Optional custom class mapping
        splits: Optional list of splits (default: train, val, test)

    Returns:
        Conversion statistics
    """
    converter = BinaryConverter(
        storage=storage,
        class_mapping=class_mapping or DAA_DEFAULT_MAPPING
    )

    return converter.convert_dataset(
        source_dir=multiclass_dir,
        dest_dir=binary_dir,
        splits=splits
    )


def get_daa_class_mapping() -> Dict[str, List[str]]:
    """Get the default Drive&Act binary class mapping."""
    return DAA_DEFAULT_MAPPING.copy()


def create_custom_mapping(
    non_distracted_classes: List[str],
    all_classes: List[str]
) -> Dict[str, List[str]]:
    """
    Create a custom binary class mapping.

    Args:
        non_distracted_classes: List of classes to map to 'non_distracted'
        all_classes: List of all classes in the dataset

    Returns:
        Binary class mapping dictionary
    """
    distracted_classes = [
        c for c in all_classes
        if c not in non_distracted_classes
    ]

    return {
        'non_distracted': non_distracted_classes,
        'distracted': distracted_classes
    }
