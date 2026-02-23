"""
Model registry system for CFW.

This module provides a registration system for models, allowing them to be
registered with a decorator and instantiated via a factory function.
"""


from typing import Callable, Dict, List, Optional, Any
import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# Global registry to store model builders
_MODEL_REGISTRY: Dict[str, Callable] = {}
_DEFAULT_MODELS_IMPORTED = False

_MODEL_INFO_MAP: Dict[str, Dict[str, Any]] = {
    "dinov2_vits14": {"name": "dinov2_vits14", "feature_dim": 384},
    "dinov2_vitb14": {"name": "dinov2_vitb14", "feature_dim": 768},
    "dinov2_vitl14": {"name": "dinov2_vitl14", "feature_dim": 1024},
    "dinov2_vitg14": {"name": "dinov2_vitg14", "feature_dim": 1536},
    "vit_s_16": {"name": "vit_s_16", "feature_dim": 384},
    "vit_b_16": {"name": "vit_b_16", "feature_dim": 768},
    "vit_l_16": {"name": "vit_l_16", "feature_dim": 1024},
    "vit_h_14": {"name": "vit_h_14", "feature_dim": 1280},
    "resnet50": {"name": "resnet50", "feature_dim": 2048},
}


def _ensure_default_models_registered() -> None:
    """Lazily import built-in model modules so decorators run."""
    global _DEFAULT_MODELS_IMPORTED
    if _DEFAULT_MODELS_IMPORTED:
        return

    from .backbones import dinov2  # noqa: F401
    from .backbones import resnet  # noqa: F401
    from .backbones import vit  # noqa: F401

    _DEFAULT_MODELS_IMPORTED = True


def register_model(name: str) -> Callable:
    """Register a model builder function.

    Example:
        ```python
        @register_model('dinov2_vitb14')
        def build_dinov2_vitb14(cfg: DictConfig):
            # Build and return model
            return model
        ```

    Args:
        name: Name to register the model under

    Returns:
        Decorator function

    Raises:
        ValueError: If model name is already registered
    """
    def decorator(builder_fn: Callable) -> Callable:
        if name in _MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered. "
                f"Please use a different name or unregister the existing model."
            )

        logger.debug(f"Registering model: {name}")
        _MODEL_REGISTRY[name] = builder_fn
        return builder_fn

    return decorator


def unregister_model(name: str) -> None:
    """Unregister a model from the registry.

    Args:
        name: Name of the model to unregister

    Raises:
        KeyError: If model name is not registered
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered")

    logger.debug(f"Unregistering model: {name}")
    del _MODEL_REGISTRY[name]


def create_model(name: str, cfg: Optional[DictConfig] = None, **kwargs: Any):
    """Create a model by name.

    This function looks up the model in the registry and calls its builder
    function with the provided configuration.

    Example:
        ```python
        # Using config
        model = create_model('dinov2_vitb14', cfg=model_cfg)

        # Using kwargs
        model = create_model('dinov2_vitb14', num_classes=10, freeze_backbone=True)
        ```

    Args:
        name: Name of the model to create
        cfg: Optional Hydra configuration object
        **kwargs: Additional keyword arguments to pass to the builder function

    Returns:
        Instantiated model

    Raises:
        KeyError: If model name is not registered
        TypeError: If builder function raises an exception
    """
    _ensure_default_models_registered()

    if name not in _MODEL_REGISTRY:
        available = list_models()
        raise KeyError(
            f"Model '{name}' is not registered. "
            f"Available models: {available}"
        )

    builder_fn = _MODEL_REGISTRY[name]

    try:
        # Call builder with config and/or kwargs
        if cfg is not None:
            logger.info(f"Creating model '{name}' with config")
            return builder_fn(cfg, **kwargs)
        else:
            logger.info(f"Creating model '{name}' with kwargs")
            return builder_fn(**kwargs)
    except Exception as e:
        logger.error(f"Failed to create model '{name}': {str(e)}")
        raise TypeError(
            f"Error creating model '{name}': {str(e)}"
        ) from e


def list_models() -> List[str]:
    """List all registered model names.

    Returns:
        List of registered model names, sorted alphabetically
    """
    _ensure_default_models_registered()
    return sorted(_MODEL_REGISTRY.keys())


def is_model_registered(name: str) -> bool:
    """Check if a model is registered.

    Args:
        name: Name of the model to check

    Returns:
        True if model is registered, False otherwise
    """
    _ensure_default_models_registered()
    return name in _MODEL_REGISTRY


def get_model_builder(name: str) -> Callable:
    """Get the builder function for a registered model.

    Args:
        name: Name of the model

    Returns:
        Builder function

    Raises:
        KeyError: If model name is not registered
    """
    _ensure_default_models_registered()
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered")

    return _MODEL_REGISTRY[name]


def clear_registry() -> None:
    """Clear all registered models.

    This is primarily useful for testing purposes.
    """
    logger.debug("Clearing model registry")
    _MODEL_REGISTRY.clear()


# ----------------------------------------------------------------------
# Backward-compatible aliases expected by tests and legacy callers
# ----------------------------------------------------------------------

def get_model(name: str, cfg: Optional[DictConfig] = None, **kwargs: Any):
    """Backward-compatible alias for create_model()."""
    try:
        return create_model(name=name, cfg=cfg, **kwargs)
    except KeyError as exc:
        raise ValueError(f"Model '{name}' not found") from exc


def model_exists(name: str) -> bool:
    """Backward-compatible alias for is_model_registered()."""
    return is_model_registered(name)


def get_model_info(name: str) -> Dict[str, Any]:
    """Return static info for a registered model name."""
    _ensure_default_models_registered()

    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found")

    if name in _MODEL_INFO_MAP:
        return dict(_MODEL_INFO_MAP[name])

    return {
        "name": name,
        "feature_dim": None,
    }
