"""Distributed training utilities for DDP (DistributedDataParallel).

This module provides helper functions for setting up and managing distributed training
across multiple GPUs and nodes.
"""


import os
import logging
from typing import Optional
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


logger = logging.getLogger(__name__)


def setup_ddp(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    local_rank: Optional[int] = None,
    backend: str = "nccl",
    master_addr: Optional[str] = None,
    master_port: Optional[str] = None,
) -> None:
    """
    Initialize the distributed process group for DDP training.

    This function sets up communication between processes running on different GPUs.
    Each GPU runs one process with a unique rank.

    Args:
        rank: Unique identifier for this process (0 to world_size-1).
            Rank 0 is typically the master/coordinator process.
        world_size: Total number of processes in the process group.
            Usually equals the number of GPUs being used.
        backend: Communication backend. Options:
            - "nccl": NVIDIA Collective Communications Library (recommended for CUDA)
            - "gloo": CPU-based backend
            - "mpi": MPI backend
            Default: "nccl"
        master_addr: IP address of the machine running rank 0 process.
            Use "localhost" for single-node multi-GPU training.
            Use actual IP for multi-node training.
            Default: "localhost"
        master_port: Port for process communication.
            Must be free on the master node.
            Default: "12355"

    Example:
        Single-node multi-GPU:
        >>> setup_ddp(rank=0, world_size=4)  # On GPU 0
        >>> setup_ddp(rank=1, world_size=4)  # On GPU 1
        >>> # ... etc for each GPU

        Multi-node:
        >>> # On node 0, GPU 0:
        >>> setup_ddp(rank=0, world_size=8, master_addr="192.168.1.100")
        >>> # On node 1, GPU 0:
        >>> setup_ddp(rank=4, world_size=8, master_addr="192.168.1.100")

    Note:
        - This function must be called at the start of each training process
        - The master node (rank 0) coordinates communication
        - NCCL backend only works with CUDA GPUs
        - Remember to call cleanup_ddp() at the end of training
    """
    # Resolve process topology from explicit args first, then environment.
    # This matches torchrun/AzureML launch behavior.
    rank = int(os.environ.get("RANK", "0")) if rank is None else int(rank)
    world_size = (
        int(os.environ.get("WORLD_SIZE", "1"))
        if world_size is None
        else int(world_size)
    )
    local_rank = (
        int(os.environ.get("LOCAL_RANK", str(rank)))
        if local_rank is None
        else int(local_rank)
    )

    # Populate rendezvous env if explicit values are provided.
    if master_addr is not None:
        os.environ["MASTER_ADDR"] = master_addr
    if master_port is not None:
        os.environ["MASTER_PORT"] = str(master_port)

    # Ensure env has defaults even when launcher does not provide them.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    if dist.is_initialized():
        logger.info("DDP process group already initialized; skipping setup")
        return

    # NCCL requires CUDA; fall back to gloo in CPU-only environments.
    if backend == "nccl" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; overriding DDP backend from nccl to gloo")
        backend = "gloo"

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )

    # Set current GPU device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    logger.info(
        "DDP initialized: "
        f"rank={rank}, local_rank={local_rank}, world_size={world_size}, "
        f"backend={backend}, master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    )


def cleanup_ddp() -> None:
    """
    Cleanup the distributed process group.

    This function should be called at the end of distributed training to
    properly shut down communication between processes.

    Example:
        >>> setup_ddp(rank=0, world_size=4)
        >>> # ... training code ...
        >>> cleanup_ddp()
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("DDP process group destroyed")


def is_distributed() -> bool:
    """
    Check if distributed training is currently active.

    Returns:
        True if running in distributed mode, False otherwise.

    Example:
        >>> if is_distributed():
        ...     print(f"Running on rank {get_rank()} of {get_world_size()}")
        ... else:
        ...     print("Running in single-GPU mode")
    """
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """
    Get the rank of the current process.

    Returns:
        Process rank (0 to world_size-1) if distributed, 0 otherwise.

    Example:
        >>> rank = get_rank()
        >>> if rank == 0:
        ...     print("I am the master process!")
    """
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """
    Get the total number of processes in the distributed group.

    Returns:
        World size (total number of processes) if distributed, 1 otherwise.

    Example:
        >>> world_size = get_world_size()
        >>> print(f"Training with {world_size} GPUs")
    """
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0).

    Useful for operations that should only happen once (e.g., logging, saving).

    Returns:
        True if this is the main process, False otherwise.

    Example:
        >>> if is_main_process():
        ...     # Only rank 0 saves checkpoints
        ...     save_checkpoint(model, "checkpoint.pth")
    """
    return get_rank() == 0


def barrier():
    """
    Synchronize all processes.

    All processes will wait at this barrier until all processes reach it.
    Useful for ensuring all processes complete a task before continuing.

    Example:
        >>> # Ensure all processes finish validation before continuing
        >>> run_validation()
        >>> barrier()
        >>> print("All processes finished validation")
    """
    if is_distributed():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce (sum or average) a tensor across all processes.

    Args:
        tensor: Tensor to reduce.
        average: If True, compute average. If False, compute sum.
            Default: True

    Returns:
        Reduced tensor. On non-main processes, the return value is undefined.

    Example:
        >>> loss = torch.tensor(0.5).cuda()
        >>> avg_loss = reduce_tensor(loss, average=True)
        >>> # avg_loss now contains the average loss across all GPUs
    """
    if not is_distributed():
        return tensor

    # Clone to avoid modifying the original
    rt = tensor.clone()

    # Sum across all processes
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)

    # Average if requested
    if average:
        rt /= get_world_size()

    return rt


def gather_tensors(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Gather tensors from all processes.

    Args:
        tensor: Tensor to gather from this process.

    Returns:
        List of tensors from all processes (only valid on rank 0).
        On non-main processes, returns list with only the local tensor.

    Example:
        >>> predictions = torch.tensor([1, 2, 3]).cuda()
        >>> all_predictions = gather_tensors(predictions)
        >>> if is_main_process():
        ...     # all_predictions contains predictions from all GPUs
        ...     print(f"Total predictions: {torch.cat(all_predictions)}")
    """
    if not is_distributed():
        return [tensor]

    # Prepare list to gather into
    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]

    # Gather
    dist.all_gather(gathered, tensor)

    return gathered


def broadcast_object(obj: any, src: int = 0):
    """
    Broadcast an arbitrary Python object from source process to all processes.

    Args:
        obj: Object to broadcast (only used on src rank).
        src: Source rank to broadcast from. Default: 0

    Returns:
        The broadcasted object on all ranks.

    Example:
        >>> if is_main_process():
        ...     config = load_config("config.yaml")
        ... else:
        ...     config = None
        >>> config = broadcast_object(config, src=0)
        >>> # Now all processes have the config
    """
    if not is_distributed():
        return obj

    # Create list to store object
    obj_list = [obj]

    # Broadcast
    dist.broadcast_object_list(obj_list, src=src)

    return obj_list[0]


def setup_for_distributed(is_master: bool):
    """
    Disable printing on non-master processes.

    This is useful to avoid cluttered output when running on multiple GPUs.

    Args:
        is_master: True if this is the master process, False otherwise.

    Example:
        >>> setup_for_distributed(is_main_process())
        >>> print("This will only print on rank 0")
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        """Print only on master process."""
        if is_master:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_device() -> torch.device:
    """
    Get the appropriate device for the current process.

    Returns:
        torch.device for this process (cuda:rank or cpu).

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    if torch.cuda.is_available() and is_distributed():
        return torch.device(f"cuda:{get_rank()}")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def wrap_model_ddp(
    model: torch.nn.Module,
    device_id: Optional[int] = None,
    find_unused_parameters: bool = False
) -> torch.nn.Module:
    """
    Wrap a model with DistributedDataParallel.

    Args:
        model: Model to wrap.
        device_id: GPU device ID for this process. If None, uses current rank.
            Default: None
        find_unused_parameters: Whether to find unused parameters (slower).
            Set to True if your model has conditional branches.
            Default: False

    Returns:
        DDP-wrapped model if distributed, otherwise returns the original model.

    Example:
        >>> model = MyModel()
        >>> model = wrap_model_ddp(model)
        >>> # Now model is ready for distributed training
    """
    if not is_distributed():
        return model

    if device_id is None:
        device_id = get_rank()

    model = model.to(device_id)
    model = DDP(
        model,
        device_ids=[device_id],
        find_unused_parameters=find_unused_parameters
    )

    logger.info(f"Model wrapped with DDP on device {device_id}")

    return model


def save_on_master(*args, **kwargs):
    """
    Save checkpoint only on the master process.

    Wrapper around torch.save that only executes on rank 0.

    Args:
        *args: Arguments to pass to torch.save
        **kwargs: Keyword arguments to pass to torch.save

    Example:
        >>> save_on_master(model.state_dict(), "checkpoint.pth")
        >>> # File is only saved once, on rank 0
    """
    if is_main_process():
        torch.save(*args, **kwargs)
