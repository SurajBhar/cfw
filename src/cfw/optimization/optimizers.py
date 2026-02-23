"""Optimizer factory with backward-compatible APIs."""


from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import torch.optim as optim
from omegaconf import DictConfig, OmegaConf


def _to_dict(config: Union[DictConfig, Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(config, DictConfig):
        return dict(OmegaConf.to_container(config, resolve=True))
    return dict(config)


def create_optimizer(
    model_parameters,
    optimizer_config: Optional[Union[DictConfig, Mapping[str, Any], str]] = None,
    optimizer_name: Optional[str] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    momentum: Optional[float] = None,
    **kwargs: Any,
) -> optim.Optimizer:
    """Create optimizer from config-style or direct kwargs-style inputs."""
    config_values: Dict[str, Any] = {}

    if isinstance(optimizer_config, (DictConfig, Mapping)):
        config_values = _to_dict(optimizer_config)
    elif isinstance(optimizer_config, str):
        optimizer_name = optimizer_name or optimizer_config
    elif optimizer_config is not None:
        raise TypeError(
            "optimizer_config must be DictConfig, mapping, string, or None"
        )

    optimizer_name = (
        optimizer_name
        or config_values.get("name")
        or kwargs.pop("name", None)
    )
    if optimizer_name is None:
        raise ValueError("Missing optimizer name")

    lr = (
        lr
        if lr is not None
        else config_values.get("lr", kwargs.pop("lr", None))
    )
    if lr is None:
        raise ValueError("Missing learning rate (lr)")

    weight_decay = (
        weight_decay
        if weight_decay is not None
        else config_values.get("weight_decay", kwargs.pop("weight_decay", 0.0))
    )
    momentum = (
        momentum
        if momentum is not None
        else config_values.get("momentum", kwargs.pop("momentum", 0.9))
    )

    extra_args = dict(config_values)
    for key in ("name", "lr", "weight_decay", "momentum", "optimizer_type"):
        extra_args.pop(key, None)
    extra_args.update(kwargs)
    for key in ("optimizer_type", "optimizer_name"):
        extra_args.pop(key, None)

    normalized_name = str(optimizer_name).upper()

    def _with_legacy_zero_grad(opt: optim.Optimizer) -> optim.Optimizer:
        # Keep legacy behavior where optimizer.zero_grad() sets gradients to zero
        # rather than None when called without arguments.
        original_zero_grad = opt.zero_grad

        def _zero_grad(*, set_to_none: bool = False):
            return original_zero_grad(set_to_none=set_to_none)

        opt.zero_grad = _zero_grad  # type: ignore[assignment]
        return opt

    if normalized_name == "ADAM":
        return _with_legacy_zero_grad(optim.Adam(
            model_parameters,
            lr=float(lr),
            weight_decay=float(weight_decay),
            **extra_args,
        ))
    if normalized_name == "SGD":
        return _with_legacy_zero_grad(optim.SGD(
            model_parameters,
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
            **extra_args,
        ))
    if normalized_name == "ADAMW":
        return _with_legacy_zero_grad(optim.AdamW(
            model_parameters,
            lr=float(lr),
            weight_decay=float(weight_decay),
            **extra_args,
        ))

    raise ValueError(
        f"Unknown optimizer: {optimizer_name}. "
        "Supported optimizers: ADAM, SGD, ADAMW"
    )


def get_optimizer_info(optimizer: optim.Optimizer) -> Dict[str, Any]:
    """Get a concise summary of optimizer configuration."""
    param_groups = optimizer.param_groups
    first_group = param_groups[0]
    total_params = sum(
        sum(p.numel() for p in group["params"])
        for group in param_groups
    )

    return {
        "name": optimizer.__class__.__name__,
        "param_groups": len(param_groups),
        "lr": first_group.get("lr"),
        "weight_decay": first_group.get("weight_decay", 0.0),
        "total_params": total_params,
    }
