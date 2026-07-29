import argparse
import multiprocessing
from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .checkpoints import load_checkpoint
from .config import TrainConfig
from .data import build_dataloaders
from .embedder import ImageEmbedder
from .metric_losses import build_loss
from .train_loop import run_training
from .utils import ensure_dir, get_device, save_json, set_seed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an image embedding model with a configurable metric-learning loss."
    )

    parser.add_argument(
        "--train_dir",
        type=Path,
        required=True,
        help="Folder containing the training class folders.",
    )

    parser.add_argument(
        "--val_dir",
        type=Path,
        required=True,
        help="Folder containing the validation class folders.",
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("image_embedder_output"),
        help="Folder where checkpoints and metrics will be saved.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="convnext_tiny.fb_in22k_ft_in1k",
        help="Any timm model name.",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--num_channels",
        type=int,
        default=3,
        choices=(1, 3),
        help="1 for grayscale images, 3 for RGB.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--model_lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--loss_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the loss function's own parameters, if any (ignored by parameter-free losses like triplet/contrastive).",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--loss_name",
        type=str,
        default="arcface",
        choices=("arcface", "subcenter_arcface", "triplet", "contrastive"),
        help="Metric-learning loss to train the embedding with.",
    )

    parser.add_argument(
        "--arcface_margin",
        type=float,
        default=28.6,
        help="Used by loss_name=arcface/subcenter_arcface.",
    )

    parser.add_argument(
        "--arcface_scale",
        type=float,
        default=64.0,
        help="Used by loss_name=arcface/subcenter_arcface.",
    )

    parser.add_argument(
        "--arcface_sub_centers",
        type=int,
        default=3,
        help="Used by loss_name=subcenter_arcface.",
    )

    parser.add_argument(
        "--triplet_margin",
        type=float,
        default=0.05,
        help="Used by loss_name=triplet.",
    )

    parser.add_argument(
        "--contrastive_pos_margin",
        type=float,
        default=0.0,
        help="Used by loss_name=contrastive.",
    )

    parser.add_argument(
        "--contrastive_neg_margin",
        type=float,
        default=1.0,
        help="Used by loss_name=contrastive.",
    )

    parser.add_argument(
        "--no_horizontal_flip",
        action="store_true",
        help="Disable random horizontal flip augmentation (use for domains where mirroring changes class identity, e.g. text/asymmetric marks).",
    )

    parser.add_argument(
        "--no_color_jitter",
        action="store_true",
        help="Disable color jitter augmentation (use for domains where color is discriminative).",
    )

    parser.add_argument(
        "--color_jitter_brightness",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--color_jitter_contrast",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--color_jitter_saturation",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--random_crop_scale_min",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--random_crop_scale_max",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--normalize_mean",
        type=float,
        nargs="+",
        default=None,
        help="Per-channel normalization mean. Must match --num_channels. Defaults to ImageNet stats for 3 channels.",
    )

    parser.add_argument(
        "--normalize_std",
        type=float,
        nargs="+",
        default=None,
        help="Per-channel normalization std. Must match --num_channels. Defaults to ImageNet stats for 3 channels.",
    )

    parser.add_argument(
        "--weights_path",
        type=Path,
        default=None,
        help=(
            "Optional local pretrained backbone weights file. "
            "When omitted, the model is trained without pretrained weights."
        ),
    )

    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed-precision training.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint file (e.g. last_model.pt) to resume training from.",
    )

    return parser.parse_args()


def build_config(arguments: argparse.Namespace) -> TrainConfig:
    color_jitter = None

    if not arguments.no_color_jitter:
        color_jitter = (
            arguments.color_jitter_brightness,
            arguments.color_jitter_contrast,
            arguments.color_jitter_saturation,
        )

    config_kwargs = dict(
        train_dir=arguments.train_dir,
        val_dir=arguments.val_dir,
        out_dir=arguments.out_dir,
        model_name=arguments.model_name,
        image_size=arguments.image_size,
        embedding_dim=arguments.embedding_dim,
        num_channels=arguments.num_channels,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        epochs=arguments.epochs,
        model_lr=arguments.model_lr,
        loss_lr=arguments.loss_lr,
        weight_decay=arguments.weight_decay,
        loss_name=arguments.loss_name,
        arcface_margin=arguments.arcface_margin,
        arcface_scale=arguments.arcface_scale,
        arcface_sub_centers=arguments.arcface_sub_centers,
        triplet_margin=arguments.triplet_margin,
        contrastive_pos_margin=arguments.contrastive_pos_margin,
        contrastive_neg_margin=arguments.contrastive_neg_margin,
        use_horizontal_flip=not arguments.no_horizontal_flip,
        color_jitter=color_jitter,
        random_crop_scale=(arguments.random_crop_scale_min, arguments.random_crop_scale_max),
        pretrained=arguments.weights_path is not None,
        use_amp=not arguments.no_amp,
        seed=arguments.seed,
        weights_path=arguments.weights_path,
    )

    if arguments.normalize_mean is not None:
        config_kwargs["normalize_mean"] = tuple(arguments.normalize_mean)

    if arguments.normalize_std is not None:
        config_kwargs["normalize_std"] = tuple(arguments.normalize_std)

    return TrainConfig(**config_kwargs)


def main() -> None:
    arguments = parse_arguments()
    config = build_config(arguments)

    ensure_dir(config.out_dir)
    set_seed(config.seed)

    device = get_device()

    print(f"Using device: {device}")
    print(f"Training directory:   {config.train_dir}")
    print(f"Validation directory: {config.val_dir}")
    print(f"Output directory:     {config.out_dir}")

    print("\nBuilding datasets...")

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    ) = build_dataloaders(config)

    num_classes = len(train_dataset.classes)

    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Training classes:  {num_classes}")

    print("\nBuilding embedding model...")

    model = ImageEmbedder(
        model_name=config.model_name,
        embedding_dim=config.embedding_dim,
        in_channels=config.num_channels,
        pretrained=config.pretrained,
        weights_path=config.weights_path,
    ).to(device)

    loss_fn = build_loss(
        num_classes=num_classes,
        config=config,
        device=device,
    )

    model_optimizer = AdamW(
        model.parameters(),
        lr=config.model_lr,
        weight_decay=config.weight_decay,
    )

    model_scheduler = CosineAnnealingLR(
        model_optimizer,
        T_max=config.epochs,
    )

    loss_params = list(loss_fn.parameters())

    if loss_params:
        loss_optimizer = AdamW(
            loss_params,
            lr=config.loss_lr,
            weight_decay=config.weight_decay,
        )

        loss_scheduler = CosineAnnealingLR(
            loss_optimizer,
            T_max=config.epochs,
        )
    else:
        # Losses like triplet/contrastive have no learnable parameters of
        # their own, so there is nothing for a second optimizer to update.
        loss_optimizer = None
        loss_scheduler = None

    save_json(
        config.out_dir / "config.json",
        config.to_dict(),
    )

    start_epoch = 1
    best_top1 = -1.0

    if arguments.resume is not None:
        print(f"\nResuming from checkpoint: {arguments.resume}")

        checkpoint = load_checkpoint(
            checkpoint_path=arguments.resume,
            model=model,
            device=device,
            loss_fn=loss_fn,
            model_optimizer=model_optimizer,
            loss_optimizer=loss_optimizer,
            model_scheduler=model_scheduler,
            loss_scheduler=loss_scheduler,
        )

        start_epoch = checkpoint["epoch"] + 1
        best_top1 = checkpoint.get("best_top1", -1.0)

        print(f"Resuming at epoch {start_epoch} (best_top1={best_top1:.4f})")

    run_training(
        model=model,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        model_optimizer=model_optimizer,
        loss_optimizer=loss_optimizer,
        model_scheduler=model_scheduler,
        loss_scheduler=loss_scheduler,
        config=config,
        device=device,
        start_epoch=start_epoch,
        best_top1=best_top1,
    )

    print("\nTraining complete.")
    print(
        f"Best checkpoint: {config.out_dir / 'best_model.pt'}"
    )
    print(
        f"Last checkpoint: {config.out_dir / 'last_model.pt'}"
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
