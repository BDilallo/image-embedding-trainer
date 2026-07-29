import torch
from torch import nn
from pytorch_metric_learning import losses

from .config import TrainConfig


def build_loss(
    num_classes: int,
    config: TrainConfig,
    device: torch.device,
) -> nn.Module:

    if num_classes < 2:
        raise ValueError(
            "Metric-learning training requires at least two classes. "
            f"Found {num_classes}."
        )

    if config.embedding_dim <= 0:
        raise ValueError(
            "Embedding dimension must be greater than zero."
        )

    loss_name = config.loss_name.lower()

    if loss_name == "arcface":
        loss_fn = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=config.embedding_dim,
            margin=config.arcface_margin,
            scale=config.arcface_scale,
        )
    elif loss_name == "subcenter_arcface":
        loss_fn = losses.SubCenterArcFaceLoss(
            num_classes=num_classes,
            embedding_size=config.embedding_dim,
            margin=config.arcface_margin,
            scale=config.arcface_scale,
            sub_centers=config.arcface_sub_centers,
        )
    elif loss_name == "triplet":
        loss_fn = losses.TripletMarginLoss(
            margin=config.triplet_margin,
        )
    elif loss_name == "contrastive":
        loss_fn = losses.ContrastiveLoss(
            pos_margin=config.contrastive_pos_margin,
            neg_margin=config.contrastive_neg_margin,
        )
    else:
        raise ValueError(
            f"Unknown loss_name: {config.loss_name!r}. Expected one of: "
            f"arcface, subcenter_arcface, triplet, contrastive."
        )

    return loss_fn.to(device)
