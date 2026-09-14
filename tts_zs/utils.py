import random
from pathlib import Path

import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def encoder_short_name(encoder: str) -> str:
    return encoder.split("/")[-1]

def encoder_uses_e5_prefix(encoder: str) -> bool:
    return "e5" in encoder.lower()

def labels_short_name(labels_path) -> str:
    stem = Path(labels_path).stem
    return stem.removeprefix("labels_").lower().replace("-", "_")
