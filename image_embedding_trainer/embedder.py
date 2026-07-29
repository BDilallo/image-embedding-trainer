from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ImageEmbedder(nn.Module):
    def __init__(
        self,
        model_name: str,
        embedding_dim: int,
        in_channels: int = 3,
        pretrained: bool = False,
        weights_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
            in_chans=in_channels,
        )

        if pretrained:
            if weights_path is None:
                raise RuntimeError(
                    "Pretrained weights were requested, but no weights path was provided."
                )

            weights_path = Path(weights_path)

            if not weights_path.is_file():
                raise FileNotFoundError(f"Pretrained weights file not found: {weights_path}")

            self._load_backbone_weights(weights_path)

        in_features = self.backbone.num_features
        self.embedding = nn.Linear(in_features, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        embeddings = self.embedding(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def _load_backbone_weights(self, weights_path: Path) -> None:
        print(f"Loading pretrained backbone weights from: {weights_path}")

        state_dict = torch.load(weights_path, map_location="cpu")

        if isinstance(state_dict, dict):
            if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]

        cleaned_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]

            if new_key.startswith("backbone."):
                new_key = new_key[len("backbone.") :]

            cleaned_state_dict[new_key] = value

        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )

        if missing_keys:
            print(f"[INFO] Missing backbone keys: {len(missing_keys)}")

        if unexpected_keys:
            print(f"[INFO] Unexpected backbone keys: {len(unexpected_keys)}")