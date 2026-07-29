from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class TrainConfig:
    train_dir: Path
    val_dir: Path
    out_dir: Path = Path("image_embedder_output")


    model_name: str = "convnext_tiny.fb_in22k_ft_in1k"
    image_size: int = 224
    embedding_dim: int = 512
    num_channels: int = 3

    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 30

    model_lr: float = 1e-4
    loss_lr: float = 1e-3
    weight_decay: float = 1e-4

    # Selects the metric-learning loss. One of:
    # "arcface", "subcenter_arcface", "triplet", "contrastive".
    loss_name: str = "arcface"

    arcface_margin: float = 28.6
    arcface_scale: float = 64.0
    arcface_sub_centers: int = 3

    triplet_margin: float = 0.05

    contrastive_pos_margin: float = 0.0
    contrastive_neg_margin: float = 1.0

    # Augmentation. Defaults suit natural RGB photos (e.g. faces/characters);
    # override for domains where flipping/color jitter would break class identity.
    use_horizontal_flip: bool = True
    color_jitter: Optional[Tuple[float, float, float]] = (0.10, 0.10, 0.08)
    random_crop_scale: Tuple[float, float] = (0.85, 1.0)

    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    pretrained: bool = False
    use_amp: bool = True
    seed: int = 10

    weights_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.train_dir = Path(self.train_dir)
        self.val_dir = Path(self.val_dir)
        self.out_dir = Path(self.out_dir)

        if self.weights_path is not None:
            self.weights_path = Path(self.weights_path)

        if self.num_channels not in (1, 3):
            raise ValueError(
                f"num_channels must be 1 (grayscale) or 3 (RGB). Got {self.num_channels}."
            )

        if len(self.normalize_mean) != self.num_channels:
            raise ValueError(
                f"normalize_mean has {len(self.normalize_mean)} values but "
                f"num_channels={self.num_channels}. Pass a matching normalize_mean "
                "when changing num_channels."
            )

        if len(self.normalize_std) != self.num_channels:
            raise ValueError(
                f"normalize_std has {len(self.normalize_std)} values but "
                f"num_channels={self.num_channels}. Pass a matching normalize_std "
                "when changing num_channels."
            )

    def to_dict(self) -> dict:
        return {
            "train_dir": str(self.train_dir),
            "val_dir": str(self.val_dir),
            "out_dir": str(self.out_dir),
            "model_name": self.model_name,
            "image_size": self.image_size,
            "embedding_dim": self.embedding_dim,
            "num_channels": self.num_channels,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "epochs": self.epochs,
            "model_lr": self.model_lr,
            "loss_lr": self.loss_lr,
            "weight_decay": self.weight_decay,
            "loss_name": self.loss_name,
            "arcface_margin": self.arcface_margin,
            "arcface_scale": self.arcface_scale,
            "arcface_sub_centers": self.arcface_sub_centers,
            "triplet_margin": self.triplet_margin,
            "contrastive_pos_margin": self.contrastive_pos_margin,
            "contrastive_neg_margin": self.contrastive_neg_margin,
            "use_horizontal_flip": self.use_horizontal_flip,
            "color_jitter": self.color_jitter,
            "random_crop_scale": self.random_crop_scale,
            "normalize_mean": self.normalize_mean,
            "normalize_std": self.normalize_std,
            "pretrained": self.pretrained,
            "use_amp": self.use_amp,
            "seed": self.seed,
            "weights_path": str(self.weights_path) if self.weights_path else None,
        }
