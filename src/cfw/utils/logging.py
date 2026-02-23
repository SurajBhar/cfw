"""Centralized logging setup for CFW.

Provides consistent logging configuration across all modules with both
file and console handlers.
"""


import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple


def get_logger(
    name: str,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    console: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Create and configure logger with consistent formatting.

    Args:
        name: Logger name (typically __name__ from calling module)
        log_dir: Directory to save log files (created if doesn't exist)
        log_file: Log file name (defaults to '{name}.log' if not provided)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to add console (stdout) handler
        file_output: Whether to add file handler

    Returns:
        Configured logger instance

    Example:
        >>> from cfw.utils.logging import get_logger
        >>> logger = get_logger(__name__, log_dir='./logs')
        >>> logger.info("Training started")
        >>> logger.error("An error occurred")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatter
    log_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Add file handler
    if file_output and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            log_file = f"{name.split('.')[-1]}.log"

        log_path = log_dir / log_file

        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def _safe_get(cfg: Any, path: str, default: Optional[Any] = None) -> Optional[Any]:
    """
    Safely get nested values from DictConfig/dict-like config.

    Args:
        cfg: Config object
        path: Dot-separated key path
        default: Default if key path does not exist

    Returns:
        Retrieved value or default
    """
    if cfg is None:
        return default

    current = cfg
    for key in path.split("."):
        try:
            if hasattr(current, key):
                current = getattr(current, key)
            elif isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        except Exception:
            return default
    return current


def _attach_root_file_handler(log_path: Path, log_level: int = logging.INFO) -> None:
    """
    Attach a file handler to the root logger to capture module logs.

    This ensures logs emitted via `logging.getLogger(__name__)` in other modules
    (e.g., data ingestion internals) are persisted in the same run log file.
    """
    root = logging.getLogger()
    target = str(log_path.resolve())

    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            existing = getattr(handler, "baseFilename", None)
            if existing and str(Path(existing).resolve()) == target:
                return

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def setup_ingestion_logging(
    cfg: Any,
    stage_name: str,
    logger_name: str,
    log_level: int = logging.INFO,
) -> Tuple[logging.Logger, Path]:
    """
    Set up timestamped file + console logging for ingestion scripts.

    Logs are stored under:
    - local storage backend: <storage.paths.artifacts>/logs/ingestion/<stage>/<timestamp>/
    - otherwise fallback: ./outputs/logs/ingestion/<stage>/<timestamp>/

    Args:
        cfg: Ingestion config object (DictConfig/dict-like)
        stage_name: Stage identifier (e.g., stage_a_download)
        logger_name: Logger name (typically __name__)
        log_level: Log level

    Returns:
        Tuple of (configured logger, absolute log file path)
    """
    backend = str(_safe_get(cfg, "storage.backend", "local")).lower()
    artifacts_dir = _safe_get(cfg, "storage.paths.artifacts")

    if backend == "local" and artifacts_dir:
        base_dir = Path(str(artifacts_dir))
    else:
        base_dir = Path("./outputs")

    run_dir = (
        base_dir
        / "logs"
        / "ingestion"
        / stage_name
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file_name = f"{stage_name}.log"
    log_file_path = run_dir / log_file_name

    logger = get_logger(
        name=logger_name,
        log_dir=str(run_dir),
        log_file=log_file_name,
        log_level=log_level,
        console=True,
        file_output=True,
    )
    _attach_root_file_handler(log_file_path, log_level=log_level)

    return logger, log_file_path


def setup_logging(
    log_dir: str,
    experiment_name: str,
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logging for an experiment.

    Creates a logger with both file and console output specifically
    configured for training experiments.

    Args:
        log_dir: Directory for log files
        experiment_name: Name of the experiment (used for log filename)
        log_level: Logging level

    Returns:
        Configured logger for the experiment

    Example:
        >>> logger = setup_logging('./outputs/my_exp/logs', 'my_exp')
        >>> logger.info("Starting experiment")
    """
    return get_logger(
        name=f"cfw.{experiment_name}",
        log_dir=log_dir,
        log_file=f"{experiment_name}.log",
        log_level=log_level,
        console=True,
        file_output=True
    )


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with optional file output.

    This is an alias function that provides a simpler interface for scripts.
    It maps string log levels to logging constants.

    Args:
        name: Logger name (typically __name__ from calling module)
        log_file: Path to log file (if None, only console logging)
        level: Log level as string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        console: Whether to output to console

    Returns:
        Configured logger instance

    Example:
        >>> from cfw.utils.logging import setup_logger
        >>> logger = setup_logger("my_script", log_file="./logs/script.log")
        >>> logger.info("Starting script")
    """
    # Map string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = level_map.get(level.upper(), logging.INFO)

    # Extract log_dir and log_file name from log_file path
    log_dir = None
    log_file_name = None
    if log_file:
        log_path = Path(log_file)
        log_dir = str(log_path.parent)
        log_file_name = log_path.name

    return get_logger(
        name=name,
        log_dir=log_dir,
        log_file=log_file_name,
        log_level=log_level,
        console=console,
        file_output=log_file is not None,
    )


# Alias for backwards compatibility — referenced by __init__.py and tests
configure_logger = setup_logger


def log_config(logger: logging.Logger, config: dict) -> None:
    """
    Log configuration parameters in a readable format.

    Args:
        logger: Logger instance
        config: Configuration dictionary (or OmegaConf DictConfig)

    Example:
        >>> logger = get_logger(__name__)
        >>> config = {'batch_size': 128, 'lr': 0.001}
        >>> log_config(logger, config)
    """
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info("=" * 80)

    # Handle OmegaConf DictConfig
    if hasattr(config, 'items'):
        for key, value in config.items():
            if isinstance(value, dict) or (hasattr(value, 'items') and not isinstance(value, str)):
                logger.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"  {sub_key}: {sub_value}")
            else:
                logger.info(f"{key}: {value}")
    else:
        logger.info(str(config))

    logger.info("=" * 80)


class TqdmLoggingHandler(logging.Handler):
    """
    Logging handler that works with tqdm progress bars.

    Prevents log messages from interfering with tqdm progress bars
    by using tqdm.write() instead of print.
    """

    def __init__(self, level=logging.NOTSET):
        """Initialize a tqdm-compatible stream handler."""
        super().__init__(level)

    def emit(self, record):
        """Emit a formatted record using `tqdm.write`."""
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger_with_tqdm(
    name: str,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Create logger that works with tqdm progress bars.

    Args:
        name: Logger name
        log_dir: Directory for log files
        log_file: Log file name
        log_level: Logging level

    Returns:
        Logger configured for tqdm compatibility
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers = []

    log_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler (normal)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            log_file = f"{name.split('.')[-1]}.log"

        log_path = log_dir / log_file
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    # Console handler (tqdm-compatible)
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setLevel(log_level)
    tqdm_handler.setFormatter(log_format)
    logger.addHandler(tqdm_handler)

    logger.propagate = False

    return logger
