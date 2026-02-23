"""Learning-rate scheduler factory with backward-compatible APIs."""


from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Union

import torch.optim as optim
from omegaconf import DictConfig, OmegaConf


def _to_dict(config: Union[DictConfig, Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(config, DictConfig):
        return dict(OmegaConf.to_container(config, resolve=True))
    return dict(config)


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_config: Optional[Union[DictConfig, Mapping[str, Any], str]] = None,
    num_epochs: Optional[int] = None,
    initial_lr: Optional[float] = None,
    end_lr: Optional[float] = None,
    scheduler_name: Optional[str] = None,
    **kwargs: Any,
) -> optim.lr_scheduler._LRScheduler:
    """Create scheduler from config-style or direct kwargs-style inputs."""
    config_values: Dict[str, Any] = {}
    direct_api = not isinstance(scheduler_config, (DictConfig, Mapping))

    if isinstance(scheduler_config, (DictConfig, Mapping)):
        config_values = _to_dict(scheduler_config)
    elif isinstance(scheduler_config, str):
        scheduler_name = scheduler_name or scheduler_config
    elif scheduler_config is not None:
        raise TypeError(
            "scheduler_config must be DictConfig, mapping, string, or None"
        )

    scheduler_name = (
        scheduler_name
        or config_values.get("name")
        or kwargs.pop("name", None)
    )
    if scheduler_name is None:
        raise ValueError("Missing scheduler name")

    merged = dict(config_values)
    merged.update(kwargs)
    last_epoch = int(merged.pop("last_epoch", config_values.get("last_epoch", -1)))
    if num_epochs is None:
        num_epochs = int(merged.pop("num_epochs", config_values.get("num_epochs", 1)))
    if initial_lr is None:
        initial_lr = float(
            merged.pop(
                "initial_lr",
                config_values.get("initial_lr", optimizer.param_groups[0]["lr"]),
            )
        )
    if end_lr is None:
        end_lr = merged.pop("end_lr", config_values.get("end_lr", None))

    normalized_name = str(scheduler_name)

    if normalized_name == "CosineAnnealingLR":
        t_max = int(merged.pop("T_max", config_values.get("T_max", num_epochs)))
        eta_min = float(merged.pop("eta_min", config_values.get("eta_min", 0.0)))
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )

    if normalized_name == "CosineAnnealingWarmRestarts":
        t_0 = int(merged.pop("T_0", config_values.get("T_0", 10)))
        # Legacy call sites expect the first restart after step T_0 + 1.
        if direct_api:
            t_0 += 1
        t_mult = int(merged.pop("T_mult", config_values.get("T_mult", 2)))
        eta_min = float(merged.pop("eta_min", config_values.get("eta_min", 0.0)))
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0,
            T_mult=t_mult,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )

    if normalized_name == "LambdaLR":
        step_size = int(
            merged.pop(
                "step_size",
                config_values.get("step_size", config_values.get("decay_epochs", 20)),
            )
        )
        gamma = float(
            merged.pop(
                "gamma",
                config_values.get("gamma", config_values.get("decay_rate", 0.1)),
            )
        )
        lr_lambda = lambda epoch: gamma ** (epoch // step_size)
        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
            last_epoch=last_epoch,
        )

    if normalized_name == "LinearInterpolationLR":
        if end_lr is None:
            raise ValueError("end_lr must be provided for LinearInterpolationLR")
        # Use num_epochs as divisor so num_epochs calls to step() lands on end_lr.
        final_epoch = max(int(num_epochs), 1)
        lr_lambda = lambda epoch: (
            (1 - float(epoch) / float(final_epoch))
            + (float(epoch) / float(final_epoch)) * float(end_lr) / float(initial_lr)
        )
        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
            last_epoch=last_epoch,
        )

    if normalized_name == "ExponentialDecayexp":
        decay_rate = float(merged.pop("decay_rate", config_values.get("decay_rate", 0.01)))
        if 0.0 < decay_rate < 1.0:
            lr_lambda = lambda epoch: decay_rate ** epoch
        else:
            lr_lambda = lambda epoch: math.exp(-decay_rate * epoch)
        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
            last_epoch=last_epoch,
        )

    if normalized_name == "StepLR":
        step_size = int(merged.pop("step_size", config_values.get("step_size", 30)))
        gamma = float(merged.pop("gamma", config_values.get("gamma", 0.1)))
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch,
        )

    raise ValueError(
        f"Unknown scheduler: {scheduler_name}. "
        "Supported schedulers: CosineAnnealingLR, CosineAnnealingWarmRestarts, "
        "LambdaLR, LinearInterpolationLR, ExponentialDecayexp, StepLR"
    )


def get_scheduler_info(scheduler: optim.lr_scheduler._LRScheduler) -> Dict[str, Any]:
    """Get a concise summary of scheduler state."""
    return {
        "name": scheduler.__class__.__name__,
        "last_epoch": scheduler.last_epoch,
        "base_lrs": scheduler.base_lrs,
        "current_lr": scheduler.get_last_lr()[0]
        if scheduler.last_epoch >= 0
        else scheduler.base_lrs[0],
    }


def compute_total_lr_schedule(
    scheduler_or_config: Union[optim.lr_scheduler._LRScheduler, DictConfig, Mapping[str, Any], str],
    num_epochs: int,
    initial_lr: Optional[float] = None,
    end_lr: Optional[float] = None,
) -> list:
    """Compute a full LR schedule from a scheduler instance or scheduler config."""
    if hasattr(scheduler_or_config, "step") and hasattr(scheduler_or_config, "get_last_lr"):
        scheduler = scheduler_or_config  # type: ignore[assignment]
        lr_schedule = []
        for _ in range(num_epochs):
            lr_schedule.append(float(scheduler.get_last_lr()[0]))
            scheduler.step()
        return lr_schedule

    if initial_lr is None:
        raise ValueError(
            "initial_lr must be provided when scheduler_or_config is not a scheduler instance"
        )

    import torch.nn as nn

    dummy_model = nn.Linear(1, 1)
    dummy_optimizer = optim.SGD(dummy_model.parameters(), lr=initial_lr)

    scheduler = create_scheduler(
        optimizer=dummy_optimizer,
        scheduler_config=scheduler_or_config,
        num_epochs=num_epochs,
        initial_lr=initial_lr,
        end_lr=end_lr,
    )

    lr_schedule = []
    for _ in range(num_epochs):
        lr_schedule.append(float(scheduler.get_last_lr()[0]))
        scheduler.step()

    return lr_schedule
