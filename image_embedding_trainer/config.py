from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    train_dir: Path
    val_dir: Path
    out_dir: Path = Path("image_embedder_output")

    
    model_name: str = "convnext_tiny.fb_in22k_ft_in1k"
    image_size: int = 224
    embedding_dim: int = 512

    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 30

    model_lr: float = 1e-4
    loss_lr: float = 1e-3
    weight_decay: float = 1e-4

    arcface_margin: float = 28.6
    arcface_scale: float = 64.0

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

    def to_dict(self) -> dict:
        return {
            "train_dir": str(self.train_dir),
            "val_dir": str(self.val_dir),
            "out_dir": str(self.out_dir),
            "model_name": self.model_name,
            "image_size": self.image_size,
            "embedding_dim": self.embedding_dim,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "epochs": self.epochs,
            "model_lr": self.model_lr,
            "loss_lr": self.loss_lr,
            "weight_decay": self.weight_decay,
            "arcface_margin": self.arcface_margin,
            "arcface_scale": self.arcface_scale,
            "pretrained": self.pretrained,
            "use_amp": self.use_amp,
            "seed": self.seed,
            "weights_path": str(self.weights_path) if self.weights_path else None,
        }