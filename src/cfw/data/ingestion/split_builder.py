"""
StateFarm Dataset Split Builder.

Creates balanced and imbalanced variants of the StateFarm dataset
with proper train/val/test splits and binary/multiclass organizations.
"""


import json
import logging
import random
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from tqdm import tqdm

from .storage import StorageBackend, LocalStorage

logger = logging.getLogger(__name__)
OVERLAP_REPORT_LIMIT = 50


# StateFarm class definitions
STATEFARM_CLASSES = {
    'c0': 'safe_driving',
    'c1': 'texting_right',
    'c2': 'talking_phone_right',
    'c3': 'texting_left',
    'c4': 'talking_phone_left',
    'c5': 'operating_radio',
    'c6': 'drinking',
    'c7': 'reaching_behind',
    'c8': 'hair_makeup',
    'c9': 'talking_passenger'
}

# Binary mapping: c0 = non_distracted, c1-c9 = distracted
BINARY_MAPPING = {
    'non_distracted': ['c0'],
    'distracted': ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
}


def _parallel_copy_files(
    storage: StorageBackend,
    file_mapping: List[Tuple[str, str]],
    max_workers: int,
    desc: str,
) -> int:
    """
    Copy files using a thread pool.

    Args:
        storage: Storage backend used to copy files
        file_mapping: List of (src, dst) pairs
        max_workers: Number of worker threads
        desc: Progress bar description

    Returns:
        Number of files copied successfully
    """
    if not file_mapping:
        logger.info("%s: no files to copy", desc)
        return 0

    workers = max(1, int(max_workers))
    logger.info("%s: %d files (%d workers)", desc, len(file_mapping), workers)

    def _copy_one(item: Tuple[str, str]) -> Tuple[bool, str, Optional[str]]:
        src, dst = item
        try:
            storage.copy_file(src, dst)
            return True, src, None
        except Exception as exc:
            return False, src, str(exc)

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


class StateFarmSplitBuilder:
    """
    Build StateFarm dataset splits with balanced and imbalanced variants.

    Supports creating:
    - Balanced multiclass (70/15/15 stratified split)
    - Balanced binary (derived from balanced multiclass)
    - Imbalanced multiclass (exponential decay on train only)
    - Imbalanced binary (derived from imbalanced multiclass)
    """

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        seed: int = 42,
        max_workers: int = 8
    ):
        """
        Initialize the split builder.

        Args:
            storage: Storage backend to use (defaults to LocalStorage)
            seed: Random seed for reproducibility
            max_workers: Number of parallel workers for file operations
        """
        self.storage = storage or LocalStorage()
        self.seed = seed
        self.max_workers = max_workers
        random.seed(seed)

    def _collect_images_by_class(self, pool_dir: str) -> Dict[str, List[str]]:
        """
        Collect all images organized by class from the pool directory.

        Args:
            pool_dir: Path to the StateFarm pool directory

        Returns:
            Dictionary mapping class names to list of image paths
        """
        images_by_class = defaultdict(list)

        for class_name in STATEFARM_CLASSES.keys():
            class_dir = self.storage.join_path(pool_dir, class_name)
            if self.storage.exists(class_dir):
                images = self.storage.list_files(class_dir, pattern='*.jpg')
                # Also check for png files
                images.extend(self.storage.list_files(class_dir, pattern='*.png'))
                images_by_class[class_name] = sorted(images)
                logger.info(f"Found {len(images)} images for class {class_name}")

        return dict(images_by_class)

    def _stratified_split(
        self,
        images_by_class: Dict[str, List[str]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Perform stratified split maintaining class proportions.

        Args:
            images_by_class: Dictionary of class -> image paths
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set

        Returns:
            Tuple of (train_dict, val_dict, test_dict)
        """
        train_split = {}
        val_split = {}
        test_split = {}

        for class_name, images in images_by_class.items():
            # Shuffle images
            shuffled = images.copy()
            random.shuffle(shuffled)

            n_total = len(shuffled)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            train_split[class_name] = shuffled[:n_train]
            val_split[class_name] = shuffled[n_train:n_train + n_val]
            test_split[class_name] = shuffled[n_train + n_val:]

            logger.debug(
                f"Class {class_name}: train={len(train_split[class_name])}, "
                f"val={len(val_split[class_name])}, test={len(test_split[class_name])}"
            )

        return train_split, val_split, test_split

    def _compute_retention(self, class_idx: int) -> float:
        """
        Compute retention ratio for imbalanced dataset.

        Uses exponential decay: c0 retains 70%, each subsequent class
        retains half of the previous.

        Args:
            class_idx: Class index (0-9)

        Returns:
            Retention ratio (0.0 to 1.0)
        """
        base = 0.70
        decay = 0.5
        return base * (decay ** class_idx)

    def _apply_imbalance(
        self,
        train_split: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Apply exponential decay imbalance to training split.

        Args:
            train_split: Balanced training split

        Returns:
            Imbalanced training split
        """
        imbalanced_train = {}

        for class_name, images in train_split.items():
            class_idx = int(class_name[1])  # Extract number from 'cX'
            retention = self._compute_retention(class_idx)
            n_keep = max(1, int(len(images) * retention))  # Keep at least 1

            # Randomly sample
            imbalanced_train[class_name] = random.sample(images, n_keep)

            logger.info(
                f"Class {class_name}: {len(images)} -> {n_keep} "
                f"(retention={retention:.4f})"
            )

        return imbalanced_train

    def _copy_files_parallel(
        self,
        file_mapping: List[Tuple[str, str]],
        desc: str = "Copying files"
    ) -> int:
        """
        Copy files in parallel.

        Args:
            file_mapping: List of (src, dst) tuples
            desc: Description for progress bar

        Returns:
            Number of files copied successfully
        """
        return _parallel_copy_files(
            storage=self.storage,
            file_mapping=file_mapping,
            max_workers=self.max_workers,
            desc=desc,
        )

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize path separators for stable relative-path rendering."""
        return str(path).replace("\\", "/")

    def _relative_to_root(self, path: str, root: str) -> str:
        """Return path relative to split root when possible."""
        norm_path = self._normalize_path(path)
        norm_root = self._normalize_path(root).rstrip("/")
        prefix = norm_root + "/"
        if norm_path.startswith(prefix):
            return norm_path[len(prefix):]
        return self.storage.get_filename(path)

    def _content_hash(self, file_path: str) -> str:
        """Compute stable content hash for overlap detection."""
        data = self.storage.read_bytes(file_path)
        return hashlib.sha1(data).hexdigest()

    def _list_split_images(self, split_dir: str) -> List[str]:
        """List supported images from split in a single traversal."""
        files = self.storage.list_files(split_dir, recursive=True)
        return [
            f for f in files
            if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

    def _write_split_to_disk(
        self,
        split_data: Dict[str, List[str]],
        output_dir: str,
        split_name: str,
        rename_sequential: bool = False
    ) -> Dict[str, int]:
        """
        Write a split to disk with proper directory structure.

        Args:
            split_data: Dictionary of class -> image paths
            output_dir: Base output directory
            split_name: Name of the split (train/val/test)
            rename_sequential: If True, rename images sequentially

        Returns:
            Dictionary with counts per class
        """
        file_mapping = []
        counts = {}

        for class_name, images in split_data.items():
            class_output_dir = self.storage.join_path(output_dir, split_name, class_name)
            self.storage.makedirs(class_output_dir)

            for idx, src_path in enumerate(images):
                if rename_sequential:
                    ext = Path(src_path).suffix
                    dst_filename = f"img_{idx:06d}{ext}"
                else:
                    dst_filename = self.storage.get_filename(src_path)

                dst_path = self.storage.join_path(class_output_dir, dst_filename)
                file_mapping.append((src_path, dst_path))

            counts[class_name] = len(images)

        # Copy all files
        self._copy_files_parallel(file_mapping, desc=f"Writing {split_name}")

        return counts

    def _convert_to_binary(
        self,
        multiclass_split: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Convert multiclass split to binary.

        Args:
            multiclass_split: Dictionary of multiclass -> images

        Returns:
            Dictionary with 'non_distracted' and 'distracted' keys
        """
        binary_split = {'non_distracted': [], 'distracted': []}

        for class_name, images in multiclass_split.items():
            if class_name in BINARY_MAPPING['non_distracted']:
                binary_split['non_distracted'].extend(images)
            else:
                binary_split['distracted'].extend(images)

        return binary_split

    def build_balanced_multiclass(
        self,
        pool_dir: str,
        output_dir: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict:
        """
        Build balanced multiclass StateFarm dataset.

        Args:
            pool_dir: Path to StateFarm pool directory
            output_dir: Output directory for the dataset
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio

        Returns:
            Dictionary with split statistics
        """
        logger.info("Building balanced multiclass StateFarm dataset")
        logger.info(f"Pool directory: {pool_dir}")
        logger.info(f"Output directory: {output_dir}")

        # Collect images
        images_by_class = self._collect_images_by_class(pool_dir)
        total_images = sum(len(v) for v in images_by_class.values())
        logger.info(f"Total images found: {total_images}")

        # Create stratified split
        train_split, val_split, test_split = self._stratified_split(
            images_by_class, train_ratio, val_ratio, test_ratio
        )

        # Write splits to disk
        train_counts = self._write_split_to_disk(train_split, output_dir, 'train')
        val_counts = self._write_split_to_disk(val_split, output_dir, 'val')
        test_counts = self._write_split_to_disk(test_split, output_dir, 'test')

        # Save manifest
        manifest = {
            'dataset_type': 'statefarm_balanced_multiclass',
            'seed': self.seed,
            'split_ratios': {
                'train': train_ratio,
                'val': val_ratio,
                'test': test_ratio
            },
            'counts': {
                'train': train_counts,
                'val': val_counts,
                'test': test_counts,
                'total': {
                    'train': sum(train_counts.values()),
                    'val': sum(val_counts.values()),
                    'test': sum(test_counts.values())
                }
            }
        }

        manifest_path = self.storage.join_path(output_dir, 'manifest.json')
        self.storage.write_text(manifest_path, json.dumps(manifest, indent=2))

        logger.info("Balanced multiclass dataset created successfully")
        logger.info(f"Train: {manifest['counts']['total']['train']}")
        logger.info(f"Val: {manifest['counts']['total']['val']}")
        logger.info(f"Test: {manifest['counts']['total']['test']}")

        return manifest

    def build_imbalanced_multiclass(
        self,
        pool_dir: str,
        output_dir: str,
        balanced_output_dir: Optional[str] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict:
        """
        Build imbalanced multiclass StateFarm dataset.

        Applies exponential decay imbalance to training set only.
        Val and test sets remain balanced.

        Args:
            pool_dir: Path to StateFarm pool directory
            output_dir: Output directory for the dataset
            balanced_output_dir: Optional path to reuse balanced val/test
            train_ratio: Training set ratio (before imbalance)
            val_ratio: Validation set ratio
            test_ratio: Test set ratio

        Returns:
            Dictionary with split statistics
        """
        logger.info("Building imbalanced multiclass StateFarm dataset")

        # Collect images
        images_by_class = self._collect_images_by_class(pool_dir)

        # Create stratified split
        train_split, val_split, test_split = self._stratified_split(
            images_by_class, train_ratio, val_ratio, test_ratio
        )

        # Apply imbalance to training set only
        imbalanced_train = self._apply_imbalance(train_split)

        # Write splits to disk
        train_counts = self._write_split_to_disk(imbalanced_train, output_dir, 'train')
        val_counts = self._write_split_to_disk(val_split, output_dir, 'val')
        test_counts = self._write_split_to_disk(test_split, output_dir, 'test')

        # Calculate imbalance ratios
        max_train = max(train_counts.values())
        imbalance_ratios = {
            k: max_train / v for k, v in train_counts.items()
        }

        # Save manifest
        manifest = {
            'dataset_type': 'statefarm_imbalanced_multiclass',
            'seed': self.seed,
            'split_ratios': {
                'train': train_ratio,
                'val': val_ratio,
                'test': test_ratio
            },
            'imbalance_strategy': 'exponential_decay',
            'imbalance_base': 0.70,
            'imbalance_decay': 0.5,
            'counts': {
                'train': train_counts,
                'val': val_counts,
                'test': test_counts,
                'total': {
                    'train': sum(train_counts.values()),
                    'val': sum(val_counts.values()),
                    'test': sum(test_counts.values())
                }
            },
            'imbalance_ratios': imbalance_ratios
        }

        manifest_path = self.storage.join_path(output_dir, 'manifest.json')
        self.storage.write_text(manifest_path, json.dumps(manifest, indent=2))

        logger.info("Imbalanced multiclass dataset created successfully")
        logger.info(f"Train: {manifest['counts']['total']['train']} (imbalanced)")
        logger.info(f"Val: {manifest['counts']['total']['val']}")
        logger.info(f"Test: {manifest['counts']['total']['test']}")

        return manifest

    def build_binary_from_multiclass(
        self,
        multiclass_dir: str,
        output_dir: str,
        rename_sequential: bool = True
    ) -> Dict:
        """
        Build binary dataset from existing multiclass dataset.

        Args:
            multiclass_dir: Path to multiclass dataset
            output_dir: Output directory for binary dataset
            rename_sequential: If True, rename images sequentially

        Returns:
            Dictionary with split statistics
        """
        logger.info("Building binary dataset from multiclass")
        logger.info(f"Source: {multiclass_dir}")
        logger.info(f"Output: {output_dir}")

        manifest = {'counts': {}}

        for split_name in ['train', 'val', 'test']:
            split_dir = self.storage.join_path(multiclass_dir, split_name)
            if not self.storage.exists(split_dir):
                logger.warning(f"Split {split_name} not found, skipping")
                continue

            # Collect images from multiclass
            multiclass_split = {}
            for class_name in STATEFARM_CLASSES.keys():
                class_dir = self.storage.join_path(split_dir, class_name)
                if self.storage.exists(class_dir):
                    images = self.storage.list_files(class_dir, pattern='*.jpg')
                    images.extend(self.storage.list_files(class_dir, pattern='*.png'))
                    multiclass_split[class_name] = images

            # Convert to binary
            binary_split = self._convert_to_binary(multiclass_split)

            # Write to disk
            for binary_class, images in binary_split.items():
                output_class_dir = self.storage.join_path(
                    output_dir, split_name, binary_class
                )
                self.storage.makedirs(output_class_dir)

                file_mapping = []
                for idx, src_path in enumerate(images):
                    if rename_sequential:
                        ext = Path(src_path).suffix
                        dst_filename = f"img_{idx:06d}{ext}"
                    else:
                        dst_filename = self.storage.get_filename(src_path)

                    dst_path = self.storage.join_path(output_class_dir, dst_filename)
                    file_mapping.append((src_path, dst_path))

                self._copy_files_parallel(
                    file_mapping,
                    desc=f"Writing {split_name}/{binary_class}"
                )

            manifest['counts'][split_name] = {
                'non_distracted': len(binary_split['non_distracted']),
                'distracted': len(binary_split['distracted'])
            }

        # Read source manifest to determine type
        source_manifest_path = self.storage.join_path(multiclass_dir, 'manifest.json')
        source_type = 'balanced'
        if self.storage.exists(source_manifest_path):
            source_manifest = json.loads(
                self.storage.read_text(source_manifest_path)
            )
            if 'imbalanced' in source_manifest.get('dataset_type', ''):
                source_type = 'imbalanced'

        manifest['dataset_type'] = f'statefarm_{source_type}_binary'
        manifest['source_multiclass_dir'] = multiclass_dir
        manifest['seed'] = self.seed
        manifest['total'] = {
            split: sum(counts.values())
            for split, counts in manifest['counts'].items()
        }

        manifest_path = self.storage.join_path(output_dir, 'manifest.json')
        self.storage.write_text(manifest_path, json.dumps(manifest, indent=2))

        logger.info("Binary dataset created successfully")
        for split, counts in manifest['counts'].items():
            logger.info(
                f"{split}: non_distracted={counts['non_distracted']}, "
                f"distracted={counts['distracted']}"
            )

        return manifest

    def validate_no_overlap(self, dataset_dir: str) -> bool:
        """
        Validate that there is no overlap between train/val/test splits.

        Args:
            dataset_dir: Path to dataset directory

        Returns:
            True if no overlap found, False otherwise
        """
        logger.info(f"Validating no overlap in {dataset_dir}")

        splits: Dict[str, Set[str]] = {}
        split_hash_to_paths: Dict[str, Dict[str, List[str]]] = {}
        for split_name in ['train', 'val', 'test']:
            split_dir = self.storage.join_path(dataset_dir, split_name)
            if self.storage.exists(split_dir):
                files = self._list_split_images(split_dir)
                workers = max(1, min(self.max_workers, len(files))) if files else 1

                splits[split_name] = set()
                split_hash_to_paths[split_name] = defaultdict(list)

                def _hash_one(path: str) -> Tuple[str, Optional[str], Optional[str]]:
                    try:
                        return path, self._content_hash(path), None
                    except Exception as exc:
                        return path, None, str(exc)

                if workers == 1:
                    hash_results = (_hash_one(path) for path in files)
                else:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        hash_results = executor.map(_hash_one, files)

                for file_path, file_hash, error in hash_results:
                    if error:
                        logger.warning(
                            "Failed to hash file during overlap check: %s - %s",
                            file_path,
                            error,
                        )
                        continue
                    if file_hash is None:
                        continue
                    splits[split_name].add(file_hash)
                    split_hash_to_paths[split_name][file_hash].append(
                        self._relative_to_root(file_path, split_dir)
                    )

        # Check overlaps
        has_overlap = False
        split_names = list(splits.keys())

        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                overlap = splits[s1] & splits[s2]
                if overlap:
                    logger.error(
                        "Overlap found between %s and %s: %d content matches",
                        s1,
                        s2,
                        len(overlap),
                    )
                    for idx, file_hash in enumerate(sorted(overlap)):
                        if idx >= OVERLAP_REPORT_LIMIT:
                            break
                        p1 = split_hash_to_paths[s1].get(file_hash, ["<unknown>"])[0]
                        p2 = split_hash_to_paths[s2].get(file_hash, ["<unknown>"])[0]
                        logger.error(
                            "  sha1:%s | %s:%s | %s:%s",
                            file_hash,
                            s1,
                            p1,
                            s2,
                            p2,
                        )
                    has_overlap = True
                else:
                    logger.info(f"No overlap between {s1} and {s2}")

        return not has_overlap

    def build_all_variants(
        self,
        pool_dir: str,
        output_base_dir: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, Dict]:
        """
        Build all four StateFarm dataset variants.

        Creates:
        - statefarm_balanced_multiclass
        - statefarm_balanced_binary
        - statefarm_imbalanced_multiclass
        - statefarm_imbalanced_binary

        Args:
            pool_dir: Path to StateFarm pool directory
            output_base_dir: Base output directory
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio

        Returns:
            Dictionary with manifests for all variants
        """
        logger.info("Building all StateFarm dataset variants")

        results = {}

        # 1. Balanced multiclass
        balanced_mc_dir = self.storage.join_path(
            output_base_dir, 'statefarm_balanced_multiclass'
        )
        results['balanced_multiclass'] = self.build_balanced_multiclass(
            pool_dir, balanced_mc_dir, train_ratio, val_ratio, test_ratio
        )

        # 2. Balanced binary (from balanced multiclass)
        balanced_bin_dir = self.storage.join_path(
            output_base_dir, 'statefarm_balanced_binary'
        )
        results['balanced_binary'] = self.build_binary_from_multiclass(
            balanced_mc_dir, balanced_bin_dir
        )

        # 3. Imbalanced multiclass
        imbalanced_mc_dir = self.storage.join_path(
            output_base_dir, 'statefarm_imbalanced_multiclass'
        )
        results['imbalanced_multiclass'] = self.build_imbalanced_multiclass(
            pool_dir, imbalanced_mc_dir, train_ratio=train_ratio,
            val_ratio=val_ratio, test_ratio=test_ratio
        )

        # 4. Imbalanced binary (from imbalanced multiclass)
        imbalanced_bin_dir = self.storage.join_path(
            output_base_dir, 'statefarm_imbalanced_binary'
        )
        results['imbalanced_binary'] = self.build_binary_from_multiclass(
            imbalanced_mc_dir, imbalanced_bin_dir
        )

        # Validate all variants
        logger.info("Validating all variants for no overlap")
        for variant_name in ['balanced_multiclass', 'balanced_binary',
                            'imbalanced_multiclass', 'imbalanced_binary']:
            variant_dir = self.storage.join_path(
                output_base_dir, f'statefarm_{variant_name}'
            )
            if self.validate_no_overlap(variant_dir):
                logger.info(f"✓ {variant_name}: No overlap")
            else:
                logger.error(f"✗ {variant_name}: Overlap detected!")

        logger.info("All StateFarm dataset variants created successfully")
        return results


def merge_statefarm_pool(
    train_dir: str,
    test_dir: str,
    pool_dir: str,
    storage: Optional[StorageBackend] = None,
    max_workers: int = 8,
) -> Dict[str, int]:
    """
    Merge StateFarm train and test directories into a single pool.

    The downloaded StateFarm dataset has separate train and test folders.
    This function merges them into a single pool for re-splitting.

    Args:
        train_dir: Path to StateFarm train directory
        test_dir: Path to StateFarm test directory
        pool_dir: Output pool directory
        max_workers: Number of parallel workers for copy operations

    Returns:
        Dictionary with counts per class
    """
    storage = storage or LocalStorage()

    logger.info("Merging StateFarm train and test into pool")
    logger.info(f"Train: {train_dir}")
    logger.info(f"Test: {test_dir}")
    logger.info(f"Pool: {pool_dir}")

    counts = {}

    for class_name in STATEFARM_CLASSES.keys():
        class_pool_dir = storage.join_path(pool_dir, class_name)
        storage.makedirs(class_pool_dir)

        images = []

        # Collect from train
        train_class_dir = storage.join_path(train_dir, class_name)
        if storage.exists(train_class_dir):
            train_images = storage.list_files(train_class_dir, pattern='*.jpg')
            train_images.extend(storage.list_files(train_class_dir, pattern='*.png'))
            images.extend(train_images)

        # Collect from test
        test_class_dir = storage.join_path(test_dir, class_name)
        if storage.exists(test_class_dir):
            test_images = storage.list_files(test_class_dir, pattern='*.jpg')
            test_images.extend(storage.list_files(test_class_dir, pattern='*.png'))
            images.extend(test_images)

        # Copy to pool in parallel
        file_mapping = []
        for img_path in images:
            filename = storage.get_filename(img_path)
            dst_path = storage.join_path(class_pool_dir, filename)
            file_mapping.append((img_path, dst_path))

        _parallel_copy_files(
            storage=storage,
            file_mapping=file_mapping,
            max_workers=max_workers,
            desc=f"Merging {class_name}",
        )

        counts[class_name] = len(images)
        logger.info(f"Class {class_name}: {len(images)} images")

    total = sum(counts.values())
    logger.info(f"Total images in pool: {total}")

    return counts
