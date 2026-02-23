"""
Drive&Act Frame Extractor.

Extracts frames from Drive&Act video dataset based on annotation CSV files
to create multi-class image datasets for driver distraction detection.
"""


import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
from tqdm import tqdm

from .storage import StorageBackend, LocalStorage

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv")


@dataclass
class AnnotationRow:
    """Represents a single annotation row from the CSV file."""

    participant_id: str
    file_id: str
    annotation_id: str
    frame_start: int
    frame_end: int
    activity: str
    chunk_id: str

    @classmethod
    def from_csv_row(cls, row: List[str]) -> 'AnnotationRow':
        """Create from CSV row."""
        return cls(
            participant_id=row[0],
            file_id=row[1],
            annotation_id=row[2],
            frame_start=int(row[3]),
            frame_end=int(row[4]),
            activity=row[5],
            chunk_id=row[6] if len(row) > 6 else '0'
        )


@dataclass
class ExtractionResult:
    """Result of a frame extraction operation."""

    annotation: AnnotationRow
    frames_extracted: int
    success: bool
    error_message: Optional[str] = None


def _extract_frames_worker(
    video_path: str,
    output_dir: str,
    frame_start: int,
    frame_end: int,
    max_frames: int,
    image_format: str = 'png'
) -> Tuple[int, Optional[str]]:
    """
    Worker function to extract frames from a video file.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_start: Starting frame number
        frame_end: Ending frame number
        max_frames: Maximum frames to extract per chunk
        image_format: Output image format (png or jpg)

    Returns:
        Tuple of (frames_extracted, error_message)
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, f"Failed to open video: {video_path}"

    frame_count = 0
    error_message = None

    try:
        # Set starting frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        for frame_num in range(frame_start, frame_end + 1):
            ret, frame = cap.read()
            if not ret:
                logger.debug(f"Frame {frame_num} missing in {video_path}")
                break

            output_filename = f'img_{frame_num:06d}.{image_format}'
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, frame)
            frame_count += 1

            if frame_count >= max_frames:
                break

    except Exception as e:
        error_message = str(e)
    finally:
        cap.release()

    return frame_count, error_message


class DriveActFrameExtractor:
    """
    Extract frames from Drive&Act video dataset.

    Processes annotation CSV files to extract labeled frames from videos
    and organizes them into a multi-class dataset structure.
    """

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        max_workers: int = 8,
        max_frames_per_chunk: int = 48,
        image_format: str = 'png'
    ):
        """
        Initialize the frame extractor.

        Args:
            storage: Storage backend (defaults to LocalStorage)
            max_workers: Number of parallel workers for extraction
            max_frames_per_chunk: Maximum frames to extract per annotation chunk
            image_format: Output image format (png or jpg)
        """
        self.storage = storage or LocalStorage()
        self.max_workers = max_workers
        self.max_frames_per_chunk = max_frames_per_chunk
        self.image_format = image_format
        self._video_path_cache: Dict[Tuple[str, str], str] = {}

    def _file_id_candidates(self, file_id: str) -> List[str]:
        """
        Build likely file_id candidates from annotation value.

        Handles both slash-separated and backslash-separated file IDs.
        """
        normalized = file_id.replace("\\", "/").strip().lstrip("./")
        if not normalized:
            return []

        candidates = [normalized]
        suffix = Path(normalized).suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            return candidates

        for ext in VIDEO_EXTENSIONS:
            candidates.append(normalized + ext)
        return candidates

    def _parse_annotations(self, annotation_file: str) -> List[AnnotationRow]:
        """
        Parse annotation CSV file.

        Args:
            annotation_file: Path to the annotation CSV file

        Returns:
            List of AnnotationRow objects
        """
        annotations = []

        with open(annotation_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header

            for row in reader:
                if len(row) >= 6:
                    annotations.append(AnnotationRow.from_csv_row(row))

        logger.info(f"Parsed {len(annotations)} annotations from {annotation_file}")
        return annotations

    def _get_video_path(self, data_dir: str, file_id: str) -> str:
        """
        Construct video file path from file_id.

        Args:
            data_dir: Base video directory
            file_id: File identifier from annotation

        Returns:
            Full path to video file
        """
        cache_key = (data_dir, file_id)
        if cache_key in self._video_path_cache:
            return self._video_path_cache[cache_key]

        root = Path(data_dir)

        # Fast path: exact relative candidate (with/without extension).
        for rel_candidate in self._file_id_candidates(file_id):
            candidate = root / rel_candidate
            if candidate.exists() and candidate.is_file():
                resolved = str(candidate)
                self._video_path_cache[cache_key] = resolved
                return resolved

        # Fallback: recursive search by stem + known extensions.
        normalized = file_id.replace("\\", "/").strip().lstrip("./")
        file_stem = Path(normalized).stem if normalized else ""
        if file_stem:
            matches: List[Path] = []
            for ext in VIDEO_EXTENSIONS:
                matches.extend(root.rglob(file_stem + ext))

            if not matches and Path(normalized).suffix:
                matches.extend(root.rglob(Path(normalized).name))
                for ext in VIDEO_EXTENSIONS:
                    matches.extend(root.rglob(Path(normalized).name + ext))

            if matches:
                matches = sorted(matches, key=lambda p: (len(p.parts), str(p)))
                resolved = str(matches[0])
                logger.warning(
                    "Auto-resolved video path for file_id '%s' to: %s",
                    file_id,
                    resolved,
                )
                self._video_path_cache[cache_key] = resolved
                return resolved

        # Keep a deterministic fallback for clear downstream errors.
        fallback = str(root / (normalized + ".mp4" if normalized else file_id + ".mp4"))
        self._video_path_cache[cache_key] = fallback
        return fallback

    def _get_output_dir(
        self,
        base_output_dir: str,
        annotation: AnnotationRow
    ) -> str:
        """
        Construct output directory path for extracted frames.

        Creates a directory structure: activity/participant_fileid_frames_chunk/

        Args:
            base_output_dir: Base output directory
            annotation: Annotation row with metadata

        Returns:
            Output directory path
        """
        # Sanitize file_id (replace / with _)
        safe_file_id = annotation.file_id.replace("/", "_")

        chunk_dir_name = (
            f"{annotation.participant_id}_{safe_file_id}_"
            f"frames_{annotation.frame_start}_{annotation.frame_end}_"
            f"ann_{annotation.annotation_id}_chunk_{annotation.chunk_id}"
        )

        return os.path.join(
            base_output_dir,
            annotation.activity,
            chunk_dir_name
        )

    def extract_split(
        self,
        annotation_file: str,
        video_dir: str,
        output_dir: str,
        error_handling: str = 'skip'
    ) -> Dict:
        """
        Extract frames for a single split (train/val/test).

        Args:
            annotation_file: Path to annotation CSV file
            video_dir: Path to video directory
            output_dir: Output directory for extracted frames
            error_handling: How to handle errors ('skip', 'stop', 'retry')

        Returns:
            Dictionary with extraction statistics
        """
        logger.info(f"Extracting frames from: {annotation_file}")
        logger.info(f"Video directory: {video_dir}")
        logger.info(f"Output directory: {output_dir}")

        # Parse annotations
        annotations = self._parse_annotations(annotation_file)

        if not annotations:
            logger.warning("No annotations found")
            return {'total': 0, 'success': 0, 'failed': 0, 'frames': 0}

        # Prepare extraction tasks
        tasks = []
        missing_video_annotations = 0
        missing_examples: List[str] = []
        for ann in annotations:
            video_path = self._get_video_path(video_dir, ann.file_id)
            if not os.path.exists(video_path):
                missing_video_annotations += 1
                if len(missing_examples) < 5 and ann.file_id not in missing_examples:
                    missing_examples.append(ann.file_id)
                continue
            frame_output_dir = self._get_output_dir(output_dir, ann)
            tasks.append((ann, video_path, frame_output_dir))

        if missing_video_annotations:
            logger.warning(
                "Skipped %d annotations due to missing video files. Example file_ids: %s",
                missing_video_annotations,
                ", ".join(missing_examples),
            )
        if not tasks:
            logger.error("No valid video files found for any annotations in split")
            return {
                'total_annotations': len(annotations),
                'success': 0,
                'failed': missing_video_annotations,
                'total_frames': 0,
                'class_counts': {}
            }

        # Process with multiprocessing
        results = []
        success_count = 0
        failed_count = missing_video_annotations
        total_frames = 0

        with tqdm(total=len(tasks), desc="Extracting frames") as pbar:
            # Use ProcessPoolExecutor for CPU-bound video processing
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}

                for ann, video_path, frame_output_dir in tasks:
                    future = executor.submit(
                        _extract_frames_worker,
                        video_path,
                        frame_output_dir,
                        ann.frame_start,
                        ann.frame_end,
                        self.max_frames_per_chunk,
                        self.image_format
                    )
                    futures[future] = ann

                for future in as_completed(futures):
                    ann = futures[future]
                    try:
                        frames_extracted, error_msg = future.result()

                        if error_msg:
                            if error_handling == 'stop':
                                raise RuntimeError(
                                    f"Extraction failed for {ann.file_id}: {error_msg}"
                                )
                            logger.warning(
                                f"Extraction error for {ann.file_id}: {error_msg}"
                            )
                            failed_count += 1
                        else:
                            success_count += 1
                            total_frames += frames_extracted

                        results.append(ExtractionResult(
                            annotation=ann,
                            frames_extracted=frames_extracted,
                            success=error_msg is None,
                            error_message=error_msg
                        ))

                    except Exception as e:
                        logger.error(f"Unexpected error for {ann.file_id}: {e}")
                        failed_count += 1
                        results.append(ExtractionResult(
                            annotation=ann,
                            frames_extracted=0,
                            success=False,
                            error_message=str(e)
                        ))

                    pbar.update(1)

        # Compute class distribution
        class_counts = {}
        for result in results:
            if result.success:
                activity = result.annotation.activity
                class_counts[activity] = class_counts.get(activity, 0) + result.frames_extracted

        stats = {
            'total_annotations': len(annotations),
            'success': success_count,
            'failed': failed_count,
            'total_frames': total_frames,
            'class_counts': class_counts
        }

        logger.info(f"Extraction complete: {success_count}/{len(annotations)} successful")
        logger.info(f"Total frames extracted: {total_frames}")

        return stats

    def extract_dataset(
        self,
        annotation_files: Dict[str, str],
        video_dir: str,
        output_dir: str,
        error_handling: str = 'skip'
    ) -> Dict:
        """
        Extract frames for all splits of a dataset.

        Args:
            annotation_files: Dictionary mapping split names to annotation file paths
                              e.g., {'train': 'path/to/train.csv', 'val': 'path/to/val.csv'}
            video_dir: Path to video directory
            output_dir: Base output directory
            error_handling: How to handle errors ('skip', 'stop', 'retry')

        Returns:
            Dictionary with statistics for all splits
        """
        logger.info("Extracting Drive&Act dataset")
        logger.info(f"Output base directory: {output_dir}")

        all_stats = {}

        for split_name, annotation_file in annotation_files.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing split: {split_name}")
            logger.info(f"{'='*60}")

            split_output_dir = os.path.join(output_dir, split_name)
            stats = self.extract_split(
                annotation_file=annotation_file,
                video_dir=video_dir,
                output_dir=split_output_dir,
                error_handling=error_handling
            )
            all_stats[split_name] = stats

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Extraction Summary")
        logger.info(f"{'='*60}")

        total_frames_all = 0
        for split_name, stats in all_stats.items():
            total_frames_all += stats['total_frames']
            logger.info(
                f"{split_name}: {stats['success']}/{stats['total_annotations']} successful, "
                f"{stats['total_frames']} frames"
            )

        logger.info(f"Total frames across all splits: {total_frames_all}")

        return all_stats


class DriveActMultiViewExtractor:
    """
    Extract frames from multiple Drive&Act camera views.

    Supports extracting from:
    - Kinect Color (RGB) - Right Top View
    - Kinect IR - Right Top View
    - NIR - Right Top View (a_column_co_driver)
    - NIR - Front View (inner_mirror)
    """

    # Camera view configurations
    CAMERA_VIEWS = {
        'kinect_color': {
            'name': 'Kinect Color RGB',
            'position': 'Right Top View',
            'video_subdir': 'kinect_color',
            'annotation_subdir': 'kinect_color',
            'usage': 'training_evaluation'
        },
        'kinect_ir': {
            'name': 'Kinect IR',
            'position': 'Right Top View',
            'video_subdir': 'kinect_ir',
            'annotation_subdir': 'kinect_ir',
            'usage': 'generalization'
        },
        'nir_right_top': {
            'name': 'NIR A-Column',
            'position': 'Right Top View',
            'video_subdir': 'a_column_co_driver',
            'annotation_subdir': 'a_column_co_driver',
            'usage': 'generalization'
        },
        'nir_front': {
            'name': 'NIR Inner Mirror',
            'position': 'Front View',
            'video_subdir': 'inner_mirror',
            'annotation_subdir': 'inner_mirror',
            'usage': 'generalization'
        }
    }

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        max_workers: int = 8,
        max_frames_per_chunk: int = 48,
        image_format: str = 'png'
    ):
        """
        Initialize the multi-view extractor.

        Args:
            storage: Storage backend
            max_workers: Number of parallel workers
            max_frames_per_chunk: Maximum frames per annotation chunk
            image_format: Output image format
        """
        self.storage = storage or LocalStorage()
        self.extractor = DriveActFrameExtractor(
            storage=self.storage,
            max_workers=max_workers,
            max_frames_per_chunk=max_frames_per_chunk,
            image_format=image_format
        )

    def _resolve_video_dir(
        self,
        raw_data_dir: str,
        view_name: str,
        video_subdir: str,
    ) -> str:
        """Resolve video directory across current and legacy extraction layouts."""
        raw_root = Path(raw_data_dir)
        candidates = [
            raw_root / video_subdir / video_subdir,
            raw_root / video_subdir,
            raw_root / view_name,
            raw_root / view_name / video_subdir,
        ]

        def _has_videos(directory: Path) -> bool:
            if not directory.exists() or not directory.is_dir():
                return False
            for ext in VIDEO_EXTENSIONS:
                if any(directory.rglob(f"*{ext}")):
                    return True
            return False

        existing_without_videos: List[Path] = []
        for candidate in candidates:
            if not candidate.exists() or not candidate.is_dir():
                continue
            if _has_videos(candidate):
                return str(candidate)
            existing_without_videos.append(candidate)

        # Fallback: recursive search for a directory named video_subdir that contains mp4 files.
        if raw_root.exists():
            matches: List[Path] = []
            for candidate in raw_root.rglob(video_subdir):
                if not candidate.is_dir():
                    continue
                if _has_videos(candidate):
                    matches.append(candidate)
            if matches:
                matches = sorted(matches, key=lambda p: (len(p.parts), str(p)))
                resolved = matches[0]
                logger.warning(
                    "Auto-resolved video directory for view '%s' to: %s",
                    view_name,
                    resolved,
                )
                return str(resolved)

        if existing_without_videos:
            logger.warning(
                "Found candidate video directories for view '%s' but none contain known video files: %s",
                view_name,
                ", ".join(str(p) for p in existing_without_videos[:3]),
            )

        # Return canonical expected location to keep downstream error messages clear.
        return str(raw_root / video_subdir)

    def _annotation_base_candidates(
        self,
        annotations_dir: str,
        annotation_subdir: str,
    ) -> List[Path]:
        """Build likely base paths for annotation CSV files."""
        ann_root = Path(annotations_dir)
        candidates = [
            ann_root / annotation_subdir,
            ann_root / ann_root.name / annotation_subdir,
            ann_root / "activities_3s" / annotation_subdir,
            ann_root / "iccv_activities_3s" / annotation_subdir,
            ann_root / "annotations" / annotation_subdir,
            ann_root / "annotations" / "activities_3s" / annotation_subdir,
            ann_root / "annotations" / "iccv_activities_3s" / annotation_subdir,
            ann_root.parent / "annotations" / annotation_subdir,
            ann_root.parent / "annotations" / "activities_3s" / annotation_subdir,
            ann_root.parent / "annotations" / "iccv_activities_3s" / annotation_subdir,
            ann_root.parent / "activities_3s" / annotation_subdir,
            ann_root.parent / "iccv_activities_3s" / annotation_subdir,
        ]

        # Keep order, remove duplicates.
        deduped: List[Path] = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _resolve_annotation_file(
        self,
        annotations_dir: str,
        annotation_subdir: str,
        ann_filename: str,
    ) -> Optional[str]:
        """Resolve annotation CSV path, supporting legacy/nested directory layouts."""
        for base in self._annotation_base_candidates(annotations_dir, annotation_subdir):
            candidate = base / ann_filename
            if candidate.exists():
                return str(candidate)

        # Fallback: recursive search by filename, constrained to matching subdir name.
        ann_root = Path(annotations_dir)
        search_roots = [ann_root, ann_root.parent]
        matches: List[Path] = []

        for root in search_roots:
            if not root.exists():
                continue
            for candidate in root.rglob(ann_filename):
                if annotation_subdir in candidate.parts:
                    matches.append(candidate)

        if matches:
            matches = sorted(matches, key=lambda p: (len(p.parts), str(p)))
            resolved = matches[0]
            logger.warning(
                "Auto-resolved annotation file '%s' to: %s",
                ann_filename,
                resolved,
            )
            return str(resolved)

        return None

    def extract_view(
        self,
        view_name: str,
        raw_data_dir: str,
        annotations_dir: str,
        output_dir: str,
        splits: Dict[str, str],
        video_subdir: Optional[str] = None,
        annotation_subdir: Optional[str] = None,
        error_handling: str = 'skip'
    ) -> Dict:
        """
        Extract frames for a specific camera view.

        Args:
            view_name: Camera view name (e.g., 'kinect_color', 'kinect_ir')
            raw_data_dir: Base directory containing raw video data
            annotations_dir: Base directory containing annotation files
            output_dir: Output directory for extracted frames
            splits: Dictionary mapping split names to annotation file names
                    e.g., {'train': 'midlevel.chunks_90.split_0.train.csv'}
            error_handling: Error handling strategy

        Returns:
            Extraction statistics
        """
        if view_name not in self.CAMERA_VIEWS:
            raise ValueError(f"Unknown camera view: {view_name}")

        view_config = self.CAMERA_VIEWS[view_name]
        effective_video_subdir = video_subdir or view_config['video_subdir']
        effective_annotation_subdir = annotation_subdir or view_config['annotation_subdir']
        logger.info(f"Extracting {view_config['name']} ({view_config['position']})")

        # Construct paths
        video_dir = self._resolve_video_dir(
            raw_data_dir=raw_data_dir,
            view_name=view_name,
            video_subdir=effective_video_subdir,
        )

        # Build annotation file paths
        annotation_files = {}
        for split_name, ann_filename in splits.items():
            ann_path = self._resolve_annotation_file(
                annotations_dir=annotations_dir,
                annotation_subdir=effective_annotation_subdir,
                ann_filename=ann_filename,
            )
            if ann_path and os.path.exists(ann_path):
                annotation_files[split_name] = ann_path
            else:
                expected = os.path.join(
                    annotations_dir, effective_annotation_subdir, ann_filename
                )
                logger.warning(f"Annotation file not found: {expected}")

        if not annotation_files:
            logger.error("No annotation files found")
            return {}

        return self.extractor.extract_dataset(
            annotation_files=annotation_files,
            video_dir=video_dir,
            output_dir=output_dir,
            error_handling=error_handling
        )

    def extract_all_training_data(
        self,
        raw_data_dir: str,
        annotations_dir: str,
        output_base_dir: str,
        split: str = 'split_0'
    ) -> Dict:
        """
        Extract training data from Kinect Color (primary training view).

        Args:
            raw_data_dir: Base raw data directory
            annotations_dir: Base annotations directory
            output_base_dir: Output base directory
            split: Split name (default: split_0)

        Returns:
            Extraction statistics
        """
        splits = {
            'train': f'midlevel.chunks_90.{split}.train.csv',
            'val': f'midlevel.chunks_90.{split}.val.csv',
            'test': f'midlevel.chunks_90.{split}.test.csv'
        }

        output_dir = os.path.join(output_base_dir, f'daa_multiclass_{split}')

        return self.extract_view(
            view_name='kinect_color',
            raw_data_dir=raw_data_dir,
            annotations_dir=annotations_dir,
            output_dir=output_dir,
            splits=splits
        )

    def extract_generalization_test_data(
        self,
        raw_data_dir: str,
        annotations_dir: str,
        output_base_dir: str,
        split: str = 'split_0',
        views: Optional[List[str]] = None
    ) -> Dict:
        """
        Extract test-only data for generalization evaluation.

        Args:
            raw_data_dir: Base raw data directory
            annotations_dir: Base annotations directory
            output_base_dir: Output base directory
            split: Split name (default: split_0)
            views: List of views to extract (default: all generalization views)

        Returns:
            Dictionary with stats for each view
        """
        if views is None:
            views = ['kinect_ir', 'nir_right_top', 'nir_front']

        all_stats = {}

        for view_name in views:
            logger.info(f"\nExtracting generalization view: {view_name}")

            splits = {
                'test': f'midlevel.chunks_90.{split}.test.csv'
            }

            output_dir = os.path.join(
                output_base_dir,
                f'daa_multiclass_{view_name}_{split}'
            )

            stats = self.extract_view(
                view_name=view_name,
                raw_data_dir=raw_data_dir,
                annotations_dir=annotations_dir,
                output_dir=output_dir,
                splits=splits
            )
            all_stats[view_name] = stats

        return all_stats
