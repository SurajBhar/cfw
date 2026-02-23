# Grid Search Configurations

This directory contains Hydra sweep configurations for hyperparameter grid search experiments.

## Overview

The grid search is conducted on the **Drive & Act Kinect Color Split 0** dataset on both:
- **Supervised ViT (vit_b_16)** - Standard Vision Transformer
- **DINOv2 (dinov2_vitb14)** - Self-supervised DINOv2 backbone

## Available Configurations

### DINOv2 Experiments

| Config File | Description |
|------------|-------------|
| `baseline_dinov2_multiclass_lr_sweep.yaml` | DINOv2 baseline on 34-class multiclass |
| `baseline_dinov2_binary_lr_sweep.yaml` | DINOv2 baseline on binary classification |
| `cfw_dinov2_multiclass_lr_sweep.yaml` | DINOv2 with CFW dataloader on multiclass |

### ViT Experiments

| Config File | Description |
|------------|-------------|
| `baseline_vit_multiclass_lr_sweep.yaml` | ViT baseline on 34-class multiclass |
| `baseline_vit_binary_lr_sweep.yaml` | ViT baseline on binary classification |

## Hyperparameter Search Space

Based on the original experiments, the grid search covers:

### Learning Rates
- 0.003, 0.01, 0.03, 0.06

### Optimizers
- Adam
- SGD (with momentum=0.9)

### Schedulers
- CosineAnnealingLR
- LinearInterpolationLR
- ExponentialDecay
- LambdaLR (step decay)

## Running Grid Search

Use Hydra's multi-run mode (`-m` flag):

```bash
# DINOv2 multiclass grid search
python scripts/train.py -m experiment=grid_search/baseline_dinov2_multiclass_lr_sweep

# ViT binary grid search
python scripts/train.py -m experiment=grid_search/baseline_vit_binary_lr_sweep

# Custom subset of parameters
python scripts/train.py -m experiment=grid_search/baseline_dinov2_multiclass_lr_sweep \
    optimizer.lr=0.01,0.03 \
    optimizer=sgd \
    scheduler=cosine_annealing
```

## Output Structure

Results are organized by Hydra in:
```
outputs/multirun/<date>/<time>/
├── 0/   # First combination
├── 1/   # Second combination
├── 2/   # Third combination
└── ...
```

Each subdirectory contains:
- `train.log` - Training logs
- `checkpoints/` - Model checkpoints
- `.hydra/` - Hydra configuration files

## Dataset Paths

The configs use original server paths from the Polaris cluster:
- **Multiclass**: `/net/polaris/storage/deeplearning/sur_data/rgb_daa/split_0/`
- **Binary**: `/net/polaris/storage/deeplearning/sur_data/binary_rgb_daa/split_0/`

**Update these paths** to your local or Azure storage paths before running.

## Notes

1. Grid search configs use **shorter epochs (80)** for faster iteration
2. The multiclass dataset has **34 classes**
3. The binary dataset has **2 classes** (distracted vs not distracted)
4. CFW configs require **pre-extracted features** - run feature extraction first
