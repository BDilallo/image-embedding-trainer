import argparse
import multiprocessing
from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import TrainConfig
from .data import build_dataloaders
from .embedder import ImageEmbedder
from .metric_losses import build_arcface_loss
from .train_loop import run_training
from .utils import ensure_dir, get_device, save_json, set_seed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an image embedding model using ArcFace loss."
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
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--arcface_margin",
        type=float,
        default=28.6,
    )

    parser.add_argument(
        "--arcface_scale",
        type=float,
        default=64.0,
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

    return parser.parse_args()


def build_config(arguments: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        train_dir=arguments.train_dir,
        val_dir=arguments.val_dir,
        out_dir=arguments.out_dir,
        model_name=arguments.model_name,
        image_size=arguments.image_size,
        embedding_dim=arguments.embedding_dim,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        epochs=arguments.epochs,
        model_lr=arguments.model_lr,
        loss_lr=arguments.loss_lr,
        weight_decay=arguments.weight_decay,
        arcface_margin=arguments.arcface_margin,
        arcface_scale=arguments.arcface_scale,
        pretrained=arguments.weights_path is not None,
        use_amp=not arguments.no_amp,
        seed=arguments.seed,
        weights_path=arguments.weights_path,
    )


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
        pretrained=config.pretrained,
        weights_path=config.weights_path,
    ).to(device)

    arcface_loss = build_arcface_loss(
        num_classes=num_classes,
        config=config,
        device=device,
    )

    model_optimizer = AdamW(
        model.parameters(),
        lr=config.model_lr,
        weight_decay=config.weight_decay,
    )

    loss_optimizer = AdamW(
        arcface_loss.parameters(),
        lr=config.loss_lr,
        weight_decay=config.weight_decay,
    )

    model_scheduler = CosineAnnealingLR(
        model_optimizer,
        T_max=config.epochs,
    )

    loss_scheduler = CosineAnnealingLR(
        loss_optimizer,
        T_max=config.epochs,
    )

    save_json(
        config.out_dir / "config.json",
        config.to_dict(),
    )

    run_training(
        model=model,
        arcface_loss=arcface_loss,
        train_dataset=train_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        model_optimizer=model_optimizer,
        loss_optimizer=loss_optimizer,
        model_scheduler=model_scheduler,
        loss_scheduler=loss_scheduler,
        config=config,
        device=device,
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