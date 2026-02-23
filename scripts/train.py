#!/usr/bin/env python3
"""Main training script for CFW (Clustered Feature Weighting) experiments.

Responsibilities:
- Run end-to-end training and evaluation from Hydra configurations.
- Support baseline and CFW dataloader modes with experiment logging.
- Serve as the primary entrypoint for paper reproduction runs.
"""


import os
import sys
import json
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

import hydra
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

try:
    import mlflow
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None

# Add src to path for imports
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from cfw.utils.logging import setup_logger
from cfw.utils.reproducibility import set_seeds
from cfw.utils.config_utils import validate_config
from cfw.data.dataloaders import create_dataloader
from cfw.models.builders import build_classifier_from_config
from cfw.optimization.optimizers import create_optimizer
from cfw.optimization.schedulers import create_scheduler
from cfw.training.trainer import create_trainer_from_config
from cfw.utils.checkpoint import CheckpointManager
from cfw.utils.distributed import barrier, cleanup_ddp, get_rank, is_distributed, setup_ddp


def _is_distributed_training(cfg: DictConfig) -> bool:
    """Infer whether the current run should initialize DDP."""
    trainer_mode = str(cfg.trainer.get("mode", "")).lower()
    return bool(
        cfg.trainer.get("distributed", False)
        or trainer_mode in {"ddp", "distributed"}
    )


def _resolve_azureml_output_root() -> Optional[Path]:
    """Resolve Azure ML output root from known environment variables."""
    def _is_resolved_path(value: str) -> bool:
        """Return True when an env value is a concrete path, not a template token."""
        stripped = value.strip()
        if not stripped:
            return False
        # Ignore unresolved AML/Hydra-style placeholders (e.g. ${{outputs.foo}}).
        if "${{" in stripped or "${" in stripped:
            return False
        return True

    explicit_keys = [
        "AZUREML_OUTPUTS_DIR",
        "AZUREML_RUN_OUTPUT_PATH",
        "OUTPUTS_DIR",
    ]

    for key in explicit_keys:
        value = os.environ.get(key)
        if value and _is_resolved_path(value):
            return Path(value)

    # AzureML often exposes named outputs as AZUREML_OUTPUT_<name>
    for key, value in os.environ.items():
        if key.startswith("AZUREML_OUTPUT_") and value and _is_resolved_path(value):
            return Path(value)

    return None


def setup_directories(cfg: DictConfig) -> dict[str, Path]:
    """Create output/checkpoint/log directories."""
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    azure_output_root = _resolve_azureml_output_root()

    if azure_output_root is not None:
        output_dir = azure_output_root / cfg.experiment.name

    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    results_dir = output_dir / "results"

    if not is_distributed() or get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
        "results_dir": results_dir,
    }


def _get_device_config_value(cfg: DictConfig) -> str:
    """Return normalized device selector from trainer config."""
    device_value = cfg.trainer.get("device", cfg.trainer.get("gpu_id", "0"))
    return str(device_value)


def setup_device(cfg: DictConfig) -> torch.device:
    """Resolve runtime device from trainer config and distributed context."""
    device_value = _get_device_config_value(cfg).lower()

    if device_value == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        return torch.device("cpu")

    if is_distributed():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")

    if device_value.startswith("cuda:"):
        return torch.device(device_value)

    return torch.device(f"cuda:{device_value}")


def _is_distributed_eval_loader(dataloader) -> bool:
    """Return True when evaluation loader is sharded across ranks."""
    if not is_distributed():
        return False
    sampler = getattr(dataloader, "sampler", None)
    return sampler is not None and hasattr(sampler, "set_epoch")


def _resolve_lr_scale_factor(cfg: DictConfig) -> float:
    """Resolve LR scale factor for distributed runs (default: 1.0)."""
    if not bool(cfg.trainer.get("scale_lr_with_world_size", False)):
        return 1.0

    configured_factor = cfg.trainer.get("lr_scale_factor", None)
    if configured_factor is not None:
        factor = float(configured_factor)
    elif is_distributed():
        factor = float(int(os.environ.get("WORLD_SIZE", "1")))
    else:
        factor = 1.0

    if factor <= 0:
        raise ValueError(
            f"Invalid lr scale factor: {factor}. Expected a positive number."
        )
    return factor


def _broadcast_model_state(model: torch.nn.Module, src_rank: int = 0) -> None:
    """Broadcast model parameters and buffers from src rank to all ranks."""
    if not is_distributed():
        return

    module = model.module if hasattr(model, "module") else model
    for tensor in module.state_dict().values():
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src=src_rank)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run model training and evaluation with the configured experiment settings."""
    distributed_requested = _is_distributed_training(cfg)

    if distributed_requested:
        setup_ddp(
            backend=cfg.trainer.get("backend", "nccl"),
            master_addr=os.environ.get("MASTER_ADDR"),
            master_port=os.environ.get("MASTER_PORT"),
        )

    rank = get_rank() if is_distributed() else 0
    is_main_process = rank == 0

    dirs = setup_directories(cfg)
    log_level = cfg.get("log_level", cfg.get("logging", {}).get("log_level", "INFO"))

    logger = setup_logger(
        name=f"cfw_training_rank_{rank}",
        log_file=str(dirs["log_dir"] / "training.log") if is_main_process else None,
        level=log_level,
    )

    try:
        if is_main_process:
            print("=" * 80)
            print("CFW Training Script")
            print("=" * 80)
            print("\nConfiguration:")
            print(OmegaConf.to_yaml(cfg))
            print("=" * 80)

        validate_config(cfg)
        set_seeds(cfg.experiment.seed)

        if is_main_process:
            logger.info("Starting CFW training experiment")
            logger.info(f"Experiment name: {cfg.experiment.name}")
            logger.info(f"Output directory: {dirs['output_dir']}")
            logger.info(f"Seed: {cfg.experiment.seed}")

        device = setup_device(cfg)
        logger.info(f"Using device: {device}")

        if is_main_process:
            logger.info("Creating dataloaders...")

        train_drop_last = bool(cfg.dataloader.get("drop_last", False))

        train_loader = create_dataloader(
            cfg=cfg,
            split="train",
            shuffle=True,
            drop_last=train_drop_last,
        )
        val_loader = create_dataloader(
            cfg=cfg,
            split="val",
            shuffle=False,
            drop_last=False,
        )
        test_loader = create_dataloader(
            cfg=cfg,
            split="test",
            shuffle=False,
            drop_last=False,
        )

        if is_main_process:
            logger.info(f"Train batches: {len(train_loader)}")
            logger.info(f"Val batches: {len(val_loader)}")
            logger.info(f"Test batches: {len(test_loader)}")

        if is_main_process:
            logger.info(f"Building model: {cfg.model.name}")

        model = build_classifier_from_config(cfg).to(device)

        if is_main_process:
            num_params = sum(p.numel() for p in model.parameters())
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {num_params:,}")
            logger.info(f"Trainable parameters: {num_trainable:,}")

        if is_main_process:
            logger.info(f"Creating optimizer: {cfg.optimizer.name}")

        lr_scale_factor = _resolve_lr_scale_factor(cfg)
        base_optimizer_lr = float(cfg.optimizer.lr)
        effective_optimizer_lr = base_optimizer_lr * lr_scale_factor
        if is_main_process:
            if bool(cfg.trainer.get("scale_lr_with_world_size", False)):
                logger.info(
                    "LR scaling enabled: base_lr=%.8f, scale_factor=%.4f, effective_lr=%.8f",
                    base_optimizer_lr,
                    lr_scale_factor,
                    effective_optimizer_lr,
                )
            else:
                logger.info("LR scaling disabled. Using fixed lr=%.8f", effective_optimizer_lr)

        optimizer = create_optimizer(
            model.parameters(),
            cfg.optimizer,
            lr=effective_optimizer_lr,
        )

        scheduler = None
        if "scheduler" in cfg and cfg.scheduler is not None:
            base_initial_lr = float(cfg.scheduler.get("start_lr", cfg.optimizer.lr))
            initial_lr = base_initial_lr * lr_scale_factor
            if optimizer.param_groups and optimizer.param_groups[0]["lr"] != initial_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = initial_lr

            end_lr = cfg.scheduler.get("end_lr", None)
            if end_lr is not None:
                end_lr = float(end_lr) * lr_scale_factor
            if is_main_process:
                logger.info(f"Creating scheduler: {cfg.scheduler.name}")

            scheduler = create_scheduler(
                optimizer=optimizer,
                scheduler_config=cfg.scheduler,
                num_epochs=cfg.trainer.num_epochs,
                initial_lr=initial_lr,
                end_lr=end_lr,
            )

        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(dirs["checkpoint_dir"]),
            experiment_name=cfg.experiment.name,
            save_every=cfg.get("checkpoint", {}).get("save_frequency", 1),
            keep_last_n=cfg.trainer.get("max_checkpoints", None),
        )

        start_epoch = 0
        resume_path = cfg.trainer.get("resume_from_checkpoint", None)
        if resume_path:
            logger.info(f"Resuming from checkpoint: {resume_path}")
            checkpoint = checkpoint_manager.load_checkpoint(
                checkpoint_path=resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=str(device),
            )
            start_epoch = checkpoint.get("epoch", 0) + 1
            logger.info(f"Resumed from epoch {start_epoch}")

        # Trainer callback settings are consumed by create_trainer_from_config.
        checkpoint_cfg = cfg.get("checkpoint", {})
        monitor_metric = checkpoint_cfg.get("monitor_metric", "val_balanced_accuracy")
        monitor_mode = checkpoint_cfg.get("mode", "max")

        with open_dict(cfg):
            cfg.trainer.device = _get_device_config_value(cfg)
            cfg.trainer.log_dir = str(dirs["log_dir"])
            cfg.trainer.checkpoint_dir = str(dirs["checkpoint_dir"])
            cfg.trainer.save_every = checkpoint_cfg.get("save_frequency", 1)
            cfg.trainer.monitor = monitor_metric
            cfg.trainer.monitor_mode = monitor_mode
            cfg.trainer.save_checkpoints = True

        trainer = create_trainer_from_config(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        if is_main_process:
            logger.info("Starting training...")
            logger.info(f"Training for {cfg.trainer.num_epochs} epochs")

        trainer.train(num_epochs=cfg.trainer.num_epochs, start_epoch=start_epoch)

        test_metrics = {}
        distributed_test_eval = _is_distributed_eval_loader(test_loader)
        best_checkpoint_path = checkpoint_manager.get_best_checkpoint() if is_main_process else None
        has_best_checkpoint = bool(best_checkpoint_path)

        if is_distributed():
            has_best_checkpoint_tensor = torch.tensor(
                1 if has_best_checkpoint else 0,
                device=device,
                dtype=torch.int32,
            )
            dist.broadcast(has_best_checkpoint_tensor, src=0)
            has_best_checkpoint = bool(has_best_checkpoint_tensor.item())

            if has_best_checkpoint and is_main_process:
                checkpoint_manager.load_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    map_location=str(device),
                )
                logger.info(f"Loaded best checkpoint: {best_checkpoint_path}")
            elif is_main_process:
                logger.warning("No best checkpoint found. Evaluating final model weights.")

            if has_best_checkpoint:
                barrier()
                _broadcast_model_state(trainer.model, src_rank=0)
                barrier()

            if is_main_process and distributed_test_eval:
                logger.info("Evaluating model on test split with distributed sharding...")
            elif is_main_process:
                logger.warning(
                    "Test dataloader is not sharded across ranks. "
                    "Running distributed evaluation on all ranks to avoid DDP deadlock."
                )
            # In distributed mode, all ranks must enter DDPTrainer.evaluate()
            # because it performs collective ops (all_reduce/all_gather).
            test_metrics = trainer.evaluate(test_loader)

        else:
            if has_best_checkpoint:
                checkpoint_manager.load_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    map_location=str(device),
                )
                logger.info(f"Loaded best checkpoint: {best_checkpoint_path}")
            else:
                logger.warning("No best checkpoint found. Evaluating final model weights.")

            logger.info("Evaluating model on test split...")
            test_metrics = trainer.evaluate(test_loader)

        if is_main_process and test_metrics:
            logger.info("Test Results:")
            for metric_name, metric_value in test_metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")

            results_file = dirs["results_dir"] / "test_results.txt"
            with open(results_file, "w", encoding="utf-8") as f:
                f.write("Test Results\n")
                f.write("=" * 50 + "\n")
                for metric_name, metric_value in test_metrics.items():
                    f.write(f"{metric_name}: {metric_value:.4f}\n")

            if mlflow is not None and mlflow.active_run() is not None:
                test_metrics_for_mlflow = {}
                if "loss" in test_metrics:
                    test_metrics_for_mlflow["test_loss"] = float(test_metrics["loss"])
                if "accuracy" in test_metrics:
                    test_metrics_for_mlflow["test_accuracy"] = float(test_metrics["accuracy"])
                if "balanced_accuracy" in test_metrics:
                    test_metrics_for_mlflow["test_balanced_accuracy"] = float(
                        test_metrics["balanced_accuracy"]
                    )

                if test_metrics_for_mlflow:
                    mlflow.log_metrics(test_metrics_for_mlflow, step=cfg.trainer.num_epochs)
                mlflow.log_artifact(str(results_file))

                best_checkpoint_for_log = checkpoint_manager.get_best_checkpoint()
                if best_checkpoint_for_log:
                    mlflow.log_artifact(best_checkpoint_for_log)

            logger.info(f"Test results saved to {results_file}")
            print("\n" + "=" * 80)
            print("Training Complete!")
            print("=" * 80)
            print(f"\nResults saved to: {dirs['output_dir']}")
            print(f"Checkpoints saved to: {dirs['checkpoint_dir']}")
            print(f"Logs saved to: {dirs['log_dir']}")
            print("=" * 80)

        if is_main_process:
            metrics_summary_file = dirs["results_dir"] / "metrics_summary.json"
            metrics_summary = {
                "experiment_name": str(cfg.experiment.name),
                "dataset": str(cfg.dataset.name),
                "model": str(cfg.model.name),
                "dataloader": str(cfg.dataloader.type),
                "num_epochs": int(cfg.trainer.num_epochs),
                "output_dir": str(dirs["output_dir"]),
                "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
                "test_metrics": {
                    metric_name: float(metric_value)
                    for metric_name, metric_value in test_metrics.items()
                },
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }

            with open(metrics_summary_file, "w", encoding="utf-8") as f:
                json.dump(metrics_summary, f, indent=2)

            if mlflow is not None and mlflow.active_run() is not None:
                mlflow.log_artifact(str(metrics_summary_file))

            logger.info(f"Metrics summary saved to {metrics_summary_file}")

    except Exception as exc:
        logger.error(f"Training failed with error: {exc}", exc_info=True)
        raise

    finally:
        if is_main_process and mlflow is not None and mlflow.active_run() is not None:
            mlflow.end_run()
        if distributed_requested:
            cleanup_ddp()


if __name__ == "__main__":
    main()
