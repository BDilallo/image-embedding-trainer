import torch
from torch import nn
from pytorch_metric_learning import losses

from .config import TrainConfig


def build_arcface_loss(
    num_classes: int,
    config: TrainConfig,
    device: torch.device,
) -> nn.Module:

    if num_classes < 2:
        raise ValueError(
            "ArcFace training requires at least two classes. "
            f"Found {num_classes}."
        )

    if config.embedding_dim <= 0:
        raise ValueError(
            "Embedding dimension must be greater than zero."
        )

    arcface_loss = losses.ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=config.embedding_dim,
        margin=config.arcface_margin,
        scale=config.arcface_scale,
    )

    return arcface_loss.to(device)