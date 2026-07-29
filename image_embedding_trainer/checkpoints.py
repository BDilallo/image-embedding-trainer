from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .config import TrainConfig


def _convert_paths_to_strings(value: Any) -> Any:

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            key: _convert_paths_to_strings(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _convert_paths_to_strings(item)
            for item in value
        ]

    return value


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    loss_fn: nn.Module,
    model_optimizer: Optimizer,
    loss_optimizer: Optional[Optimizer],
    model_scheduler: LRScheduler,
    loss_scheduler: Optional[LRScheduler],
    epoch: int,
    class_to_idx: Dict[str, int],
    config: TrainConfig,
    metrics: Dict[str, float],
    best_top1: float,
) -> None:

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = _convert_paths_to_strings(asdict(config))

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss_state_dict": loss_fn.state_dict(),
        "model_optimizer_state_dict": model_optimizer.state_dict(),
        "loss_optimizer_state_dict": (
            loss_optimizer.state_dict() if loss_optimizer is not None else None
        ),
        "model_scheduler_state_dict": model_scheduler.state_dict(),
        "loss_scheduler_state_dict": (
            loss_scheduler.state_dict() if loss_scheduler is not None else None
        ),
        "class_to_idx": class_to_idx,
        "idx_to_class": {
            class_index: class_name
            for class_name, class_index in class_to_idx.items()
        },
        "config": config_dict,
        "metrics": metrics,
        "best_top1": best_top1,
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    model_optimizer: Optional[Optimizer] = None,
    loss_optimizer: Optional[Optimizer] = None,
    model_scheduler: Optional[LRScheduler] = None,
    loss_scheduler: Optional[LRScheduler] = None,
) -> Dict[str, Any]:

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file was not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if loss_fn is not None:
        loss_fn.load_state_dict(
            checkpoint["loss_state_dict"]
        )

    if model_optimizer is not None:
        model_optimizer.load_state_dict(
            checkpoint["model_optimizer_state_dict"]
        )

    if loss_optimizer is not None and checkpoint.get("loss_optimizer_state_dict") is not None:
        loss_optimizer.load_state_dict(
            checkpoint["loss_optimizer_state_dict"]
        )

    if model_scheduler is not None:
        model_scheduler.load_state_dict(
            checkpoint["model_scheduler_state_dict"]
        )

    if loss_scheduler is not None and checkpoint.get("loss_scheduler_state_dict") is not None:
        loss_scheduler.load_state_dict(
            checkpoint["loss_scheduler_state_dict"]
        )

    return checkpoint
