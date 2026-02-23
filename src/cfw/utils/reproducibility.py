"""Reproducibility utilities for setting random seeds.

Ensures reproducible results across runs by setting seeds for all
random number generators used in training.
"""


import random
import os
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Sets seeds for:
    - Python random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CuDNN (deterministic mode)

    Args:
        seed: Random seed value

    Example:
        >>> from cfw.utils.reproducibility import set_seed
        >>> set_seed(42)
        >>> # All random operations will now be reproducible
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # CuDNN
    # Note: Setting these may reduce performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set PYTHONHASHSEED for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_seed_for_training(seed: int, allow_cudnn_benchmark: bool = False) -> None:
    """
    Set seed for training with optional CuDNN benchmark mode.

    This is useful when you want reproducibility but also want to allow
    CuDNN to benchmark and select the fastest convolution algorithms.

    Args:
        seed: Random seed value
        allow_cudnn_benchmark: If True, allows CuDNN benchmark mode for
                              better performance at the cost of some
                              non-determinism

    Example:
        >>> set_seed_for_training(42, allow_cudnn_benchmark=True)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN settings
    if allow_cudnn_benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def get_random_state() -> dict:
    """
    Get the current random state for all RNGs.

    Returns:
        Dictionary containing random states for Python, NumPy, and PyTorch

    Example:
        >>> state = get_random_state()
        >>> # ... do some random operations ...
        >>> restore_random_state(state)  # Restore to previous state
    """
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }


def restore_random_state(state: dict) -> None:
    """
    Restore random state for all RNGs.

    Args:
        state: Dictionary containing random states (from get_random_state())

    Example:
        >>> state = get_random_state()
        >>> # ... do some random operations ...
        >>> restore_random_state(state)  # Restore to previous state
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['torch_cuda'])


def worker_init_fn(worker_id: int, seed: Optional[int] = None) -> None:
    """
    Worker initialization function for DataLoader.

    Ensures each worker has a different random seed to avoid duplicate
    data augmentations across workers.

    Args:
        worker_id: Worker ID (provided by DataLoader)
        seed: Base random seed (if None, uses current random state)

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataloader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed=42)
        ... )
    """
    if seed is None:
        seed = torch.initial_seed() % (2**32)

    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Alias for backwards compatibility — referenced by __init__.py and tests
set_seeds = set_seed
