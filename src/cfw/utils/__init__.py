"""
Utility modules for CFW.

This module provides:
- Logging utilities
- Time formatting utilities
- Reproducibility utilities (seed setting)
- Configuration utilities for Hydra
- Checkpoint management utilities
- Distributed training utilities
"""


from .logging import get_logger, configure_logger
from .time_utils import format_time, format_time_compact, format_eta
from .reproducibility import set_seeds
from .config_utils import (
    validate_config,
    validate_paths,
    print_config,
    get_output_dir,
    create_experiment_name,
    merge_configs,
    resolve_interpolations,
    get_config_value,
    save_config,
    load_config,
)
from .checkpoint import CheckpointManager, save_checkpoint, load_checkpoint
from .distributed import (
    setup_ddp,
    cleanup_ddp,
    is_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    reduce_tensor,
    gather_tensors,
    broadcast_object,
    get_device,
    wrap_model_ddp,
    save_on_master
)

__all__ = [
    # Logging
    'get_logger',
    'configure_logger',
    # Time utils
    'format_time',
    'format_time_compact',
    'format_eta',
    # Reproducibility
    'set_seeds',
    # Config utils
    'validate_config',
    'validate_paths',
    'print_config',
    'get_output_dir',
    'create_experiment_name',
    'merge_configs',
    'resolve_interpolations',
    'get_config_value',
    'save_config',
    'load_config',
    # Checkpoint
    'CheckpointManager',
    'save_checkpoint',
    'load_checkpoint',
    # Distributed
    'setup_ddp',
    'cleanup_ddp',
    'is_distributed',
    'get_rank',
    'get_world_size',
    'is_main_process',
    'barrier',
    'reduce_tensor',
    'gather_tensors',
    'broadcast_object',
    'get_device',
    'wrap_model_ddp',
    'save_on_master',
]
