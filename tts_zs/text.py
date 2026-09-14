import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from tts_zs.config import CACHE_DIR, LABELS_LANG, N_VARIANTS, TEXT_EMB_MODEL_DEFAULT
from tts_zs.utils import encoder_short_name, encoder_uses_e5_prefix, labels_short_name

def text_cache_name(encoder: str, labels_path) -> str:
    short = encoder_short_name(encoder)
    lab = labels_short_name(labels_path)
    if encoder_uses_e5_prefix(encoder):
        return f"{lab}_passage_{short}.npy"
    return f"{lab}_{short}.npy"

def encode_text_variants(
    labels: dict,
    keys: list,
    encoder: str = TEXT_EMB_MODEL_DEFAULT,
    labels_path=LABELS_LANG,
) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / text_cache_name(encoder, labels_path)
    if cache_path.exists():
        print(f"[text-cache] {cache_path.name}")
        arr = np.load(cache_path)
        if arr.ndim != 3 or arr.shape[:2] != (len(keys), N_VARIANTS):
            raise AssertionError(
                f"cached shape {arr.shape} incompatible with "
                f"({len(keys)}, {N_VARIANTS}, *); delete the cache"
            )
        return arr

    print(f"[text-encode] {encoder}")
    use_prefix = encoder_uses_e5_prefix(encoder)
    flat = []
    for k in keys:
        variants = labels[k]
        if len(variants) != N_VARIANTS:
            raise ValueError(
                f"'{k}' has {len(variants)} variants, expected {N_VARIANTS}"
            )
        if use_prefix:
            flat.extend(f"passage: {v}" for v in variants)
        else:
            flat.extend(variants)

    st = SentenceTransformer(encoder, trust_remote_code=True)
    embs = st.encode(
        flat,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    text_dim = embs.shape[-1]
    embs = embs.reshape(len(keys), N_VARIANTS, text_dim).astype(np.float32)
    np.save(cache_path, embs)
    return embs

def pick_held_out(n_keys: int, seed: int) -> np.ndarray:
    fixed = os.environ.get("HELD_OUT_IDX")
    if fixed is not None and fixed != "":
        return np.full(n_keys, int(fixed), dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.integers(0, N_VARIANTS, size=n_keys).astype(np.int64)

def encode_text_variants_augmented(
    aug_labels: dict,
    keys: list,
    n_aug: int,
    encoder: str = TEXT_EMB_MODEL_DEFAULT,
    cache_path=None,
) -> np.ndarray:
    if cache_path is None or not Path(cache_path).exists():
        raise FileNotFoundError(
            f"Augmented text cache not found: {cache_path}. "
            "Run scripts/generate_augmented_text.py first."
        )
    cache_path = Path(cache_path)
    print(f"[aug-text-cache] {cache_path.name}")
    arr = np.load(cache_path)
    if arr.ndim != 3 or arr.shape[:2] != (len(keys), n_aug):
        raise AssertionError(
            f"Augmented cache shape {arr.shape} incompatible with "
            f"({len(keys)}, {n_aug}, *); delete and regenerate."
        )
    return arr

def anchors_from(
    text_variants: np.ndarray, held_out_idx: np.ndarray
) -> torch.Tensor:
    picked = text_variants[np.arange(len(held_out_idx)), held_out_idx]
    picked = picked / (np.linalg.norm(picked, axis=1, keepdims=True) + 1e-12)
    return torch.from_numpy(picked.astype(np.float32))

def anchors_all_variants(text_variants: np.ndarray) -> torch.Tensor:
    arr = text_variants / (
        np.linalg.norm(text_variants, axis=-1, keepdims=True) + 1e-12
    )
    return torch.from_numpy(arr.astype(np.float32))
