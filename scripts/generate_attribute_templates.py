import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_zs.attributes import build_pair_attributes
from tts_zs.config import CACHE_DIR, LABELS_LANG, TEXT_EMB_MODEL_DEFAULT
from tts_zs.templates import AXES, display_name, template_variants
from tts_zs.utils import encoder_short_name, encoder_uses_e5_prefix

_AXES = AXES
N_TMPL_PER_AXIS = 5
N_TMPL_TOTAL = N_TMPL_PER_AXIS * len(_AXES)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels-path", type=Path, default=LABELS_LANG)
    p.add_argument("--encoder", type=str, default=TEXT_EMB_MODEL_DEFAULT)
    p.add_argument("--out-path", type=Path, default=None)
    args = p.parse_args()

    with open(args.labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    keys = list(labels.keys())
    n_keys = len(keys)

    attrs = build_pair_attributes(keys)

    use_prefix = encoder_uses_e5_prefix(args.encoder)
    flat = []
    for key, a in zip(keys, attrs):
        tmpl_strings = []
        for axis in _AXES:
            tmpl_strings.extend(template_variants(axis, display_name(axis, a[axis])))
        assert len(tmpl_strings) == N_TMPL_TOTAL, f"{key}: {len(tmpl_strings)} templates"
        if use_prefix:
            flat.extend(f"passage: {s}" for s in tmpl_strings)
        else:
            flat.extend(tmpl_strings)

    print(f"[attr-tmpl] Encoding {len(flat)} strings ({n_keys} keys × {N_TMPL_TOTAL} templates) ...")

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(args.encoder, trust_remote_code=True)
    embs = st.encode(
        flat,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    text_dim = embs.shape[-1]
    embs = embs.reshape(n_keys, N_TMPL_TOTAL, text_dim).astype(np.float32)

    norms = np.linalg.norm(embs, axis=-1)
    print(f"[attr-tmpl] L2-norm  mean={norms.mean():.4f}  min={norms.min():.4f}  max={norms.max():.4f}")

    if args.out_path is None:
        short = encoder_short_name(args.encoder)
        args.out_path = CACHE_DIR / f"lang_passage_attrtmpl{N_TMPL_TOTAL}_{short}.npy"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(args.out_path, embs)
    print(f"[attr-tmpl] Saved → {args.out_path}  shape={embs.shape}")

if __name__ == "__main__":
    main()
