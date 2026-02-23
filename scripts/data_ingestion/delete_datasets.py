#!/usr/bin/env python3
"""
Delete processed dataset directories recursively.

This utility removes full dataset folders (including nested subdirectories
and images) under processed data paths.

Examples:
    # Dry-run (default): show what would be deleted
    python scripts/data_ingestion/delete_datasets.py \
        data/processed/daa_multiclass_kinect_color

    # Execute deletion
    python scripts/data_ingestion/delete_datasets.py \
        data/processed/daa_multiclass_kinect_color \
        data/processed/daa_binary_kinect_color \
        --execute --yes

    # Allow paths outside processed roots (advanced)
    python scripts/data_ingestion/delete_datasets.py \
        /custom/path/to/dataset --execute --yes --allow-outside-processed
"""


from __future__ import annotations

import argparse
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for safe dataset deletion."""
    parser = argparse.ArgumentParser(
        description="Delete dataset directories recursively."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Dataset directory path(s) to delete.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform deletion. Without this flag, runs as dry-run.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompt (use with --execute).",
    )
    parser.add_argument(
        "--allow-outside-processed",
        action="store_true",
        help=(
            "Allow deleting paths outside data/processed and data/processed_smoke. "
            "Use with care."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Maximum parallel worker processes for deletion.",
    )
    return parser.parse_args()


def is_subpath(candidate: Path, root: Path) -> bool:
    """Return True if candidate is inside root directory."""
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_targets(
    raw_paths: List[str],
    project_root: Path,
    safe_roots: List[Path],
    allow_outside_processed: bool,
) -> List[Path]:
    """Resolve and validate dataset target paths."""
    targets: List[Path] = []
    errors: List[str] = []

    safe_roots_resolved = [p.resolve() for p in safe_roots]

    for raw in raw_paths:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        else:
            p = p.resolve()

        if not p.exists():
            errors.append(f"Path does not exist: {p}")
            continue

        if p.is_symlink():
            errors.append(f"Refusing to delete symlink path: {p}")
            continue

        if not p.is_dir():
            errors.append(f"Path is not a directory: {p}")
            continue

        if not allow_outside_processed:
            if not any(is_subpath(p, root) for root in safe_roots_resolved):
                errors.append(
                    "Path is outside allowed processed roots "
                    f"(use --allow-outside-processed to override): {p}"
                )
                continue

        if any(p == root for root in safe_roots_resolved):
            errors.append(f"Refusing to delete protected root directory: {p}")
            continue

        targets.append(p)

    if errors:
        joined = "\n".join(f"- {msg}" for msg in errors)
        raise ValueError(f"Validation failed:\n{joined}")

    if not targets:
        raise ValueError("No valid dataset paths to delete.")

    # Deduplicate while preserving order.
    deduped: List[Path] = []
    seen = set()
    for t in targets:
        key = str(t)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    return deduped


def confirm_or_abort(targets: List[Path]) -> None:
    """Prompt user for explicit confirmation."""
    print("About to permanently delete these directories:")
    for t in targets:
        print(f"  - {t}")
    response = input("Type DELETE to confirm: ").strip()
    if response != "DELETE":
        raise RuntimeError("Deletion cancelled by user.")


def _delete_tree(path_str: str) -> Tuple[str, bool, str, float]:
    """
    Delete one directory tree.

    Returns:
        (path, success, error_message, duration_seconds)
    """
    start = time.perf_counter()
    try:
        path = Path(path_str)
        if not path.exists():
            return path_str, False, "Path does not exist at deletion time", 0.0
        if not path.is_dir():
            return path_str, False, "Path is not a directory at deletion time", 0.0
        shutil.rmtree(path)
        return path_str, True, "", time.perf_counter() - start
    except Exception as exc:
        return path_str, False, str(exc), time.perf_counter() - start


def _delete_path(path_str: str) -> Tuple[str, bool, str, float]:
    """
    Delete a path that may be a file, symlink, or directory.

    Returns:
        (path, success, error_message, duration_seconds)
    """
    start = time.perf_counter()
    try:
        path = Path(path_str)
        if not path.exists() and not path.is_symlink():
            return path_str, False, "Path does not exist at deletion time", 0.0
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            return path_str, False, "Unsupported path type", 0.0
        return path_str, True, "", time.perf_counter() - start
    except Exception as exc:
        return path_str, False, str(exc), time.perf_counter() - start


def main() -> None:
    """Validate targets and delete selected dataset directories."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    safe_roots = [
        project_root / "data" / "processed",
        project_root / "data" / "processed_smoke",
    ]

    targets = resolve_targets(
        raw_paths=args.paths,
        project_root=project_root,
        safe_roots=safe_roots,
        allow_outside_processed=args.allow_outside_processed,
    )

    print("Targets:")
    for t in targets:
        print(f"  - {t}")

    if not args.execute:
        print("\nDry-run only. Re-run with --execute to delete.")
        return

    if not args.yes:
        confirm_or_abort(targets)

    target_strs = [str(t) for t in targets]
    workers = max(1, min(int(args.max_workers), len(target_strs)))
    results: List[Tuple[str, bool, str, float]] = []

    # If user passed exactly one dataset path and requested parallel workers,
    # parallelize deletion across that dataset's top-level entries.
    if len(targets) == 1 and int(args.max_workers) > 1:
        target = targets[0]
        child_entries = [str(p) for p in target.iterdir()]
        child_workers = max(1, min(int(args.max_workers), len(child_entries) or 1))
        print(
            f"\nDeleting single dataset using {child_workers} process(es) "
            "over top-level entries..."
        )

        if child_entries:
            if child_workers == 1:
                with tqdm(total=len(child_entries), desc="Deleting", unit="entry") as pbar:
                    for entry in child_entries:
                        result = _delete_path(entry)
                        results.append(result)
                        pbar.update(1)
            else:
                with ProcessPoolExecutor(max_workers=child_workers) as executor:
                    with tqdm(total=len(child_entries), desc="Deleting", unit="entry") as pbar:
                        for result in executor.map(_delete_path, child_entries):
                            results.append(result)
                            pbar.update(1)

            # Remove now-empty dataset root if all child deletions succeeded.
            if all(ok for _, ok, _, _ in results):
                root_result = _delete_tree(str(target))
                results.append(root_result)
        else:
            print("\nDataset directory is empty; deleting root directory...")
            results.append(_delete_tree(str(target)))
    else:
        print(f"\nDeleting with {workers} process(es)...")
        if workers == 1:
            with tqdm(total=len(target_strs), desc="Deleting", unit="dir") as pbar:
                for target in target_strs:
                    result = _delete_tree(target)
                    results.append(result)
                    pbar.update(1)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                with tqdm(total=len(target_strs), desc="Deleting", unit="dir") as pbar:
                    for result in executor.map(_delete_tree, target_strs):
                        results.append(result)
                        pbar.update(1)

    failed = []
    for path_str, ok, error, elapsed in results:
        if ok:
            print(f"Deleted: {path_str} ({elapsed:.2f}s)")
        else:
            failed.append((path_str, error))
            print(f"FAILED: {path_str} -> {error}")

    if failed:
        print("\nCompleted with failures:")
        for path_str, error in failed:
            print(f"  - {path_str}: {error}")
        raise SystemExit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
