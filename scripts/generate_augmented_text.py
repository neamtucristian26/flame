import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_zs.config import CACHE_DIR, DATA_DIR, LABELS_LANG, TEXT_EMB_MODEL_DEFAULT
from tts_zs.utils import encoder_short_name, encoder_uses_e5_prefix, labels_short_name

_HEADER_RE = re.compile(r"^\[Language:.*?\]", re.IGNORECASE)
_SENT_SEP = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str):
    text = text.strip()
    m = _HEADER_RE.match(text)
    if m:
        header = m.group(0)
        rest = text[m.end():].strip()
    else:
        header = ""
        rest = text
    sentences = [s.strip() for s in _SENT_SEP.split(rest) if s.strip()]
    return header, sentences

def join_sentences(header: str, sentences: list[str]) -> str:
    parts = ([header] if header else []) + sentences
    return " ".join(parts)

def augment_variant(header: str, sentences: list[str], strategy: str, rng) -> str:
    n = len(sentences)
    if n == 0:
        return join_sentences(header, sentences)

    if strategy == "dropout":
        p = rng.uniform(0.2, 0.5)
        kept = [s for s in sentences if rng.random() > p]
        if not kept:
            kept = [sentences[rng.integers(0, n)]]
    else:
        if n == 1:
            kept = sentences
        else:
            k = rng.integers(math.ceil(n / 2), n)
            kept = sentences[:k]

    return join_sentences(header, kept)

def generate_augmented_labels(labels: dict, keys: list, n_aug: int, seed: int,
                              held_out_idx: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    aug_labels = {}
    for key in keys:
        variants = labels[key]
        if held_out_idx is not None:
            src_variants = [v for j, v in enumerate(variants) if j != held_out_idx]
        else:
            src_variants = variants
        augmented = []
        for i in range(n_aug):
            src = src_variants[i % len(src_variants)]
            header, sentences = split_sentences(src)
            strategy = "dropout" if i % 2 == 0 else "truncation"
            aug_text = augment_variant(header, sentences, strategy, rng)
            augmented.append(aug_text)
        aug_labels[key] = augmented
    return aug_labels

def encode_and_cache(aug_labels: dict, keys: list, n_aug: int,
                     encoder: str, out_path: Path) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    use_prefix = encoder_uses_e5_prefix(encoder)
    flat = []
    for k in keys:
        variants = aug_labels[k]
        assert len(variants) == n_aug, f"{k}: expected {n_aug} variants, got {len(variants)}"
        if use_prefix:
            flat.extend(f"passage: {v}" for v in variants)
        else:
            flat.extend(variants)

    print(f"[encode] {len(flat)} strings with {encoder} ...")
    st = SentenceTransformer(encoder, trust_remote_code=True)
    embs = st.encode(
        flat,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    text_dim = embs.shape[-1]
    embs = embs.reshape(len(keys), n_aug, text_dim).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embs)
    print(f"[save] {out_path}  shape={embs.shape}")
    return embs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels-path", type=Path, default=LABELS_LANG)
    p.add_argument("--n-aug", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoder", type=str, default=TEXT_EMB_MODEL_DEFAULT)
    p.add_argument("--held-out-idx", type=int, default=None,
                   help="If set, exclude this original variant index from the "
                        "augmentation source set (keeps the held-out eval variant "
                        "strictly unseen). Output filename gets a 'ho<idx>' tag.")
    p.add_argument("--skip-encode", action="store_true",
                   help="Only write the JSON; skip E5 encoding.")
    args = p.parse_args()

    with open(args.labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    keys = sorted(labels.keys())
    print(f"Loaded {len(keys)} pairs from {args.labels_path}")
    ho_tag = "" if args.held_out_idx is None else f"ho{args.held_out_idx}"

    out_json = DATA_DIR / f"labels_lang_aug{args.n_aug}{ho_tag}.json"
    aug_labels = generate_augmented_labels(labels, keys, args.n_aug, args.seed,
                                           held_out_idx=args.held_out_idx)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(aug_labels, f, ensure_ascii=False, indent=2)
    print(f"[save] {out_json}")

    if args.skip_encode:
        return

    enc_short = encoder_short_name(args.encoder)
    lab_short = labels_short_name(args.labels_path)
    cache_name = f"{lab_short}_passage_aug{args.n_aug}{ho_tag}_{enc_short}.npy"
    out_npy = CACHE_DIR / cache_name

    if out_npy.exists():
        print(f"[cache] {out_npy} already exists — skipping encoding.")
        return

    encode_and_cache(aug_labels, keys, args.n_aug, args.encoder, out_npy)

if __name__ == "__main__":
    main()
