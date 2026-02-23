#!/usr/bin/env python3
"""
Generate a persisted CFW weights table from pre-extracted features.

This script runs clustered feature weighting (HDBSCAN + inverse cluster-size
weighting) and writes a train table with:
    - image_path
    - label
    - cluster_label
    - weight

Outputs:
    - cfw_weights_<split>.csv
    - cfw_weights_<split>.pkl
    - cfw_weights_<split>_summary.json

Usage:
    python scripts/generate_cfw_weights_table.py \
        +split=train \
        dataloader.cfw.train_feature_file=./data/features/.../train/features.pkl \
        dataloader.cfw.train_label_file=./data/features/.../train/labels.pkl \
        dataloader.cfw.train_img_path_file=./data/features/.../train/image_paths.pkl \
        output_dir=./data/artifacts/statefarm_imbalanced_multiclass_smoke/vit_h_14
"""


import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from cfw.core.cfw_dataloader import CFWDataLoaderBuilder
from cfw.core.clustering import ClusteringConfig
from cfw.core.weighting import WeightingConfig
from cfw.utils.logging import setup_logger
from cfw.utils.reproducibility import set_seeds


def _resolve_output_dir(cfg: DictConfig) -> Path:
    """Resolve output directory from config or Hydra runtime directory."""
    if cfg.get("output_dir"):
        return Path(str(cfg.output_dir))
    runtime_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    return Path(str(runtime_dir))


def _build_clustering_config(cfw_cfg: DictConfig) -> ClusteringConfig:
    """Build typed clustering config from Hydra dict config."""
    clustering_cfg = cfw_cfg.get("clustering", {})
    return ClusteringConfig(
        min_cluster_size=int(clustering_cfg.get("min_cluster_size", 25)),
        min_samples=int(clustering_cfg.get("min_samples", 1)),
        cluster_selection_epsilon=float(clustering_cfg.get("cluster_selection_epsilon", 0.0)),
        cluster_selection_method=str(clustering_cfg.get("cluster_selection_method", "eom")),
        allow_single_cluster=bool(clustering_cfg.get("allow_single_cluster", False)),
        metric=str(clustering_cfg.get("metric", "cosine")),
    )


def _build_weighting_config(cfw_cfg: DictConfig) -> WeightingConfig:
    """Build typed weighting config from Hydra dict config."""
    weighting_cfg = cfw_cfg.get("weighting", {})
    strategy = str(
        weighting_cfg.get(
            "strategy",
            weighting_cfg.get("weighting_strategy", "inverse_cluster_size"),
        )
    )
    return WeightingConfig(
        outlier_weight=float(weighting_cfg.get("outlier_weight", 0.001)),
        max_outlier_cluster_size=int(weighting_cfg.get("max_outlier_cluster_size", 50)),
        weighting_strategy=strategy,
    )


def _resolve_feature_paths(cfw_cfg: DictConfig, split: str) -> Tuple[Path, Path, Path]:
    """Resolve and validate required feature paths for the selected split."""
    feature_file = cfw_cfg.get(f"{split}_feature_file")
    label_file = cfw_cfg.get(f"{split}_label_file")
    img_path_file = cfw_cfg.get(f"{split}_img_path_file")

    missing_keys = []
    if not feature_file:
        missing_keys.append(f"dataloader.cfw.{split}_feature_file")
    if not label_file:
        missing_keys.append(f"dataloader.cfw.{split}_label_file")
    if not img_path_file:
        missing_keys.append(f"dataloader.cfw.{split}_img_path_file")

    if missing_keys:
        missing_str = ", ".join(missing_keys)
        raise ValueError(
            f"Missing required CFW feature inputs for split={split}: {missing_str}"
        )

    feature_path = Path(str(feature_file))
    label_path = Path(str(label_file))
    img_path_path = Path(str(img_path_file))

    for path in (feature_path, label_path, img_path_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    return feature_path, label_path, img_path_path


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run clustered feature weighting and persist weights table."""
    print("=" * 80)
    print("Generate CFW Weights Table")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    set_seeds(cfg.experiment.seed)

    split = str(cfg.get("split", "train"))
    if split != "train":
        print(f"Warning: split={split}. Most CFW workflows generate weights for train split.")

    cfw_cfg = cfg.dataloader.get("cfw", {})
    feature_path, label_path, img_path_path = _resolve_feature_paths(cfw_cfg, split)

    output_dir = _resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="cfw_weight_table",
        log_file=str(log_dir / "cfw_weight_table.log"),
        level=cfg.get("log_level", "INFO"),
    )

    logger.info("Starting clustered feature weighting table generation")
    logger.info(f"split={split}")
    logger.info(f"feature_file={feature_path}")
    logger.info(f"label_file={label_path}")
    logger.info(f"img_path_file={img_path_path}")

    clustering_cfg = _build_clustering_config(cfw_cfg)
    weighting_cfg = _build_weighting_config(cfw_cfg)
    clustering_batch_size = int(cfw_cfg.get("clustering_batch_size", 1024))

    builder = CFWDataLoaderBuilder(
        feature_file_path=str(feature_path),
        label_file_path=str(label_path),
        img_path_file_path=str(img_path_path),
        clustering_config=clustering_cfg,
        weighting_config=weighting_cfg,
    )

    builder.load_data()
    builder.batch_process_features(clustering_batch_size=clustering_batch_size)

    image_paths = [str(p) for p in builder.img_paths]
    labels = np.asarray(builder.labels).astype(np.int64, copy=False)
    cluster_labels = np.asarray(builder.all_cluster_labels).astype(np.int64, copy=False)
    weights = np.asarray(builder.all_weights).astype(np.float64, copy=False)

    n_samples = len(image_paths)
    if not (len(labels) == n_samples == len(cluster_labels) == len(weights)):
        raise ValueError(
            "Length mismatch in generated table columns: "
            f"paths={len(image_paths)}, labels={len(labels)}, "
            f"cluster_labels={len(cluster_labels)}, weights={len(weights)}"
        )

    csv_path = output_dir / f"cfw_weights_{split}.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["index", "image_path", "label", "cluster_label", "weight"])
        for idx, (path, label, cluster_label, weight) in enumerate(
            zip(image_paths, labels, cluster_labels, weights)
        ):
            writer.writerow(
                [
                    int(idx),
                    path,
                    int(label),
                    int(cluster_label),
                    float(weight),
                ]
            )

    pkl_path = output_dir / f"cfw_weights_{split}.pkl"
    with open(pkl_path, "wb") as pkl_file:
        pickle.dump(
            {
                "image_paths": image_paths,
                "labels": labels,
                "cluster_labels": cluster_labels,
                "weights": weights.astype(np.float32, copy=False),
            },
            pkl_file,
        )

    summary = {
        "split": split,
        "num_samples": int(n_samples),
        "num_unique_clusters": int(np.unique(cluster_labels).size),
        "mean_weight": float(np.mean(weights)),
        "std_weight": float(np.std(weights)),
        "min_weight": float(np.min(weights)),
        "max_weight": float(np.max(weights)),
        "clustering_batch_size": int(clustering_batch_size),
        "inputs": {
            "feature_file": str(feature_path),
            "label_file": str(label_path),
            "img_path_file": str(img_path_path),
        },
        "outputs": {
            "weights_csv": str(csv_path),
            "weights_pkl": str(pkl_path),
        },
    }

    summary_path = output_dir / f"cfw_weights_{split}_summary.json"
    with open(summary_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=2)

    logger.info(f"Saved weights CSV: {csv_path}")
    logger.info(f"Saved weights PKL: {pkl_path}")
    logger.info(f"Saved summary JSON: {summary_path}")

    print("\n" + "=" * 80)
    print("CFW weight table generation complete")
    print("=" * 80)
    print(f"split:            {split}")
    print(f"samples:          {n_samples}")
    print(f"unique clusters:  {summary['num_unique_clusters']}")
    print(f"mean weight:      {summary['mean_weight']:.6f}")
    print(f"std weight:       {summary['std_weight']:.6f}")
    print(f"csv table:        {csv_path}")
    print(f"pkl table:        {pkl_path}")
    print(f"summary:          {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
