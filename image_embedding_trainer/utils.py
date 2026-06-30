import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def resource_path(relative_path: str | Path) -> Path:
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return Path(base_path) / relative_path