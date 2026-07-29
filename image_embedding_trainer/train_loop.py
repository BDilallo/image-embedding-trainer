from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .checkpoints import save_checkpoint
from .config import TrainConfig
from .evaluate import validate_retrieval
from .utils import save_json


def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    model_optimizer: Optimizer,
    loss_optimizer: Optional[Optimizer],
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> float:

    model.train()
    loss_fn.train()

    total_loss = 0.0
    amp_enabled = use_amp and device.type == "cuda"

    progress_bar = tqdm(
        train_loader,
        desc="Train",
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        model_optimizer.zero_grad(set_to_none=True)

        if loss_optimizer is not None:
            loss_optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            embeddings = model(images)
            loss = loss_fn(embeddings, labels)

        scaler.scale(loss).backward()

        scaler.step(model_optimizer)

        if loss_optimizer is not None:
            scaler.step(loss_optimizer)

        scaler.update()

        batch_size = images.size(0)
        batch_loss = loss.item()

        total_loss += batch_loss * batch_size

        progress_bar.set_postfix(
            loss=f"{batch_loss:.4f}"
        )

    return total_loss / len(train_loader.dataset)


def run_training(
    model: nn.Module,
    loss_fn: nn.Module,
    train_dataset: ImageFolder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_optimizer: Optimizer,
    loss_optimizer: Optional[Optimizer],
    model_scheduler: LRScheduler,
    loss_scheduler: Optional[LRScheduler],
    config: TrainConfig,
    device: torch.device,
    start_epoch: int = 1,
    best_top1: float = -1.0,
) -> List[Dict]:

    output_directory = Path(config.out_dir)

    best_checkpoint_path = output_directory / "best_model.pt"
    last_checkpoint_path = output_directory / "last_model.pt"
    metrics_history_path = output_directory / "metrics_history.json"

    amp_enabled = config.use_amp and device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    metrics_history: List[Dict] = []

    for epoch in range(start_epoch, config.epochs + 1):
        print(f"\n=== Epoch {epoch}/{config.epochs} ===")

        train_loss = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            model_optimizer=model_optimizer,
            loss_optimizer=loss_optimizer,
            device=device,
            scaler=scaler,
            use_amp=config.use_amp,
        )

        validation_metrics = validate_retrieval(
            model=model,
            val_loader=val_loader,
            device=device,
        )

        model_scheduler.step()

        if loss_scheduler is not None:
            loss_scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            **validation_metrics,
        }

        metrics_history.append(epoch_metrics)

        print(
            f"train_loss={train_loss:.5f} | "
            f"val_top1={validation_metrics['top1']:.4f} | "
            f"val_top5={validation_metrics['top5']:.4f} | "
            f"best_threshold="
            f"{validation_metrics['best_threshold']:.3f} | "
            f"best_f1={validation_metrics['best_f1']:.4f}"
        )

        print(
            f"pos_mean={validation_metrics['pos_mean']:.4f} | "
            f"neg_mean={validation_metrics['neg_mean']:.4f}"
        )

        if validation_metrics["top1"] > best_top1:
            best_top1 = validation_metrics["top1"]

        save_checkpoint(
            checkpoint_path=last_checkpoint_path,
            model=model,
            loss_fn=loss_fn,
            model_optimizer=model_optimizer,
            loss_optimizer=loss_optimizer,
            model_scheduler=model_scheduler,
            loss_scheduler=loss_scheduler,
            epoch=epoch,
            class_to_idx=train_dataset.class_to_idx,
            config=config,
            metrics=epoch_metrics,
            best_top1=best_top1,
        )

        if validation_metrics["top1"] == best_top1:
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                loss_fn=loss_fn,
                model_optimizer=model_optimizer,
                loss_optimizer=loss_optimizer,
                model_scheduler=model_scheduler,
                loss_scheduler=loss_scheduler,
                epoch=epoch,
                class_to_idx=train_dataset.class_to_idx,
                config=config,
                metrics=epoch_metrics,
                best_top1=best_top1,
            )

            print(
                f"[SAVED BEST] {best_checkpoint_path}"
            )

        save_json(
            metrics_history_path,
            {"history": metrics_history},
        )

    return metrics_history
