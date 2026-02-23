#!/usr/bin/env python3
"""
Create small smoke annotation CSVs from full Drive&Act split CSVs.

Build smoke dataset:
python scripts/data_ingestion/build_daa_dataset.py \
  --config-name ingestion/daa_extraction_smoke \
  'extraction.max_frames_per_chunk=6'
"""


from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Drive&Act smoke CSV generation."""
    parser = argparse.ArgumentParser(description="Build Drive&Act smoke CSV files")
    parser.add_argument(
        "--source-dir",
        default="data/raw/driveandact/iccv_activities_3s/kinect_color",
        help="Directory containing full split CSV files",
    )
    parser.add_argument(
        "--dest-dir",
        default="data/raw/driveandact/iccv_activities_3s_smoke/kinect_color",
        help="Destination directory for reduced CSV files",
    )
    parser.add_argument("--train-cap", type=int, default=120)
    parser.add_argument("--val-cap", type=int, default=40)
    parser.add_argument("--test-cap", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-id",
        type=int,
        default=0,
        help="Drive&Act split id (0/1/2)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle rows before applying caps",
    )
    parser.add_argument(
        "--ensure-class-coverage",
        action="store_true",
        help=(
            "Ensure every class present in source CSV is represented in the reduced CSV "
            "(subject to cap)."
        ),
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=1,
        help=(
            "Minimum rows per class when --ensure-class-coverage is enabled. "
            "Default: 1."
        ),
    )
    return parser.parse_args()


def cap_for_file(filename: str, args: argparse.Namespace) -> int:
    """Return the row cap for a split inferred from the CSV filename."""
    if ".train." in filename:
        return args.train_cap
    if ".val." in filename:
        return args.val_cap
    if ".test." in filename:
        return args.test_cap
    raise ValueError(f"Could not infer split type from filename: {filename}")


def get_activity_column_index(header: List[str]) -> int:
    """Find activity/class column index in Drive&Act CSV header."""
    normalized = [h.strip().lower() for h in header]
    for key in ("activity", "action", "label", "class"):
        if key in normalized:
            return normalized.index(key)

    # Fallback to known Drive&Act layout:
    # participant_id,file_id,annotation_id,frame_start,frame_end,activity,chunk_id
    if len(header) > 5:
        return 5
    raise ValueError(
        "Could not infer activity column from CSV header. "
        "Expected one of: activity/action/label/class."
    )


def classes_in_rows(rows: List[List[str]], activity_idx: int) -> List[str]:
    """Return sorted unique class names from rows."""
    return sorted({row[activity_idx] for row in rows if len(row) > activity_idx})


def sample_rows_with_class_coverage(
    rows: List[List[str]],
    cap: int,
    activity_idx: int,
    min_per_class: int,
    shuffle: bool,
) -> List[List[str]]:
    """
    Sample rows while preserving class coverage.

    Strategy:
    1. Take up to `min_per_class` rows from each class.
    2. Fill remaining capacity from leftover rows.
    """
    grouped: Dict[str, List[List[str]]] = defaultdict(list)
    for row in rows:
        if len(row) <= activity_idx:
            continue
        grouped[row[activity_idx]].append(row)

    if not grouped:
        return []

    class_names = sorted(grouped.keys())
    min_per_class = max(1, int(min_per_class))

    required = sum(min(len(grouped[name]), min_per_class) for name in class_names)
    if cap < required:
        raise ValueError(
            "Cap is too small for requested class coverage: "
            f"cap={cap}, required={required} "
            f"(classes={len(class_names)}, min_per_class={min_per_class})"
        )

    selected: List[List[str]] = []
    leftovers: List[List[str]] = []

    for name in class_names:
        class_rows = grouped[name].copy()
        if shuffle:
            random.shuffle(class_rows)

        take = min(len(class_rows), min_per_class)
        selected.extend(class_rows[:take])
        leftovers.extend(class_rows[take:])

    if shuffle:
        random.shuffle(leftovers)

    remaining = cap - len(selected)
    if remaining > 0:
        selected.extend(leftovers[:remaining])

    return selected[:cap]


def main() -> None:
    """Create reduced Drive&Act split CSVs for smoke-scale experiments."""
    args = parse_args()
    random.seed(args.seed)

    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    expected_files = [
        f"midlevel.chunks_90.split_{args.split_id}.train.csv",
        f"midlevel.chunks_90.split_{args.split_id}.val.csv",
        f"midlevel.chunks_90.split_{args.split_id}.test.csv",
    ]

    for filename in expected_files:
        source_file = source_dir / filename
        if not source_file.exists():
            raise FileNotFoundError(f"Missing source CSV: {source_file}")

        with open(source_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"CSV is empty: {source_file}")

        header, body = rows[0], rows[1:]
        activity_idx = get_activity_column_index(header)
        source_classes = classes_in_rows(body, activity_idx)

        cap = cap_for_file(filename, args)
        if args.ensure_class_coverage:
            reduced = sample_rows_with_class_coverage(
                rows=body,
                cap=cap,
                activity_idx=activity_idx,
                min_per_class=args.min_per_class,
                shuffle=args.shuffle,
            )
        else:
            body_sample = body.copy()
            if args.shuffle:
                random.shuffle(body_sample)
            reduced = body_sample[:cap]

        reduced_classes = classes_in_rows(reduced, activity_idx)
        missing_classes = sorted(set(source_classes) - set(reduced_classes))

        dest_file = dest_dir / filename
        with open(dest_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(reduced)

        print(
            f"Wrote {dest_file} with {len(reduced)} rows "
            f"(from {len(body)} original rows)"
        )
        print(
            f"  Classes retained: {len(reduced_classes)}/{len(source_classes)} "
            f"(missing: {len(missing_classes)})"
        )
        if missing_classes:
            print(
                "  Missing classes: "
                + ", ".join(missing_classes[:10])
                + (" ..." if len(missing_classes) > 10 else "")
            )


if __name__ == "__main__":
    main()
