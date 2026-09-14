import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tts_zs.config import AUDIO_DIM, AUDIO_ROOT, N_AUDIO_LAYERS, N_VARIANTS, SEP

class TTSAudioDataset(Dataset):

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        cls_idx, path = self.samples[i]
        emb = torch.load(path, map_location="cpu", weights_only=True)
        if emb.shape != (N_AUDIO_LAYERS, AUDIO_DIM):
            raise ValueError(
                f"{path}: expected ({N_AUDIO_LAYERS}, {AUDIO_DIM}), "
                f"got {tuple(emb.shape)}"
            )
        return emb.float(), cls_idx

def build_clip_index(model_names: list, audio_root: Path = AUDIO_ROOT) -> dict:
    by_model = {name: [] for name in model_names}
    lang_dirs = [p for p in audio_root.iterdir() if p.is_dir()]
    for lang in lang_dirs:
        for name in model_names:
            model_dir = lang / name
            if model_dir.is_dir():
                by_model[name].extend(sorted(model_dir.glob("*.pt")))
    missing = [n for n, paths in by_model.items() if not paths]
    if missing:
        raise RuntimeError(
            f"{len(missing)} models have no audio clips: {missing[:5]}..."
        )
    for name in model_names:
        by_model[name].sort()
    return by_model

def split_train_val(
    clips_by_model: dict, val_frac: float, seed: int
) -> tuple[list, list]:
    rng = random.Random(seed)
    train, val = [], []
    for model_idx, name in enumerate(sorted(clips_by_model.keys())):
        paths = list(clips_by_model[name])
        rng.shuffle(paths)
        n_val = max(1, int(round(len(paths) * val_frac)))
        train.extend((model_idx, p) for p in paths[n_val:])
        val.extend((model_idx, p) for p in paths[:n_val])
    return train, val

def build_clip_index_lang(
    pair_keys: list, audio_root: Path = AUDIO_ROOT
) -> dict:
    key_set = set(pair_keys)
    by_pair: dict = {k: [] for k in pair_keys}
    for lang_dir in sorted(audio_root.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        for model_dir in sorted(lang_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            key = f"{model_dir.name}{SEP}{lang}"
            if key in key_set:
                by_pair[key].extend(sorted(model_dir.glob("*.pt")))
    missing = [k for k, v in by_pair.items() if not v]
    if missing:
        raise RuntimeError(f"{len(missing)} pairs have no clips: {missing[:5]}")
    for k in by_pair:
        by_pair[k].sort()
    return by_pair

def split_train_val_lc(
    clips_by_pair: dict, val_frac: float, seed: int
) -> tuple[list, list]:
    rng = random.Random(seed)
    train, val = [], []
    for pair_idx, key in enumerate(sorted(clips_by_pair)):
        paths = list(clips_by_pair[key])
        rng.shuffle(paths)
        n_val = max(1, int(round(len(paths) * val_frac)))
        val.extend((pair_idx, p) for p in paths[:n_val])
        train.extend((pair_idx, p) for p in paths[n_val:])
    return train, val

def select_holdout_models(
    all_pair_keys: list, n_hold: int, seed: int,
    fold_id: int = -1, arch_labels: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list, list]:
    import random as _random
    model_names = sorted({k.split(SEP)[0] for k in all_pair_keys})
    if fold_id >= 0:
        n_folds = max(1, len(model_names) // n_hold)
        if arch_labels is not None:
            family_to_models: dict = {}
            for m in model_names:
                fam = arch_labels.get(m, "neural_tts")
                family_to_models.setdefault(fam, []).append(m)
            r = _random.Random(seed)
            fold_assignments: dict = {}
            fold_sizes = [0] * n_folds
            fam_order = sorted(
                family_to_models,
                key=lambda f: (-len(family_to_models[f]), f),
            )
            for fam in fam_order:
                fam_models = list(family_to_models[fam])
                r.shuffle(fam_models)
                for m in fam_models:
                    f = min(range(n_folds), key=lambda i: (fold_sizes[i], i))
                    fold_assignments[m] = f
                    fold_sizes[f] += 1
            unseen_m = {m for m, f in fold_assignments.items() if f == fold_id}
        else:
            unseen_m = set(model_names[fold_id::n_folds])
    else:
        rng = np.random.default_rng(seed)
        unseen_m = set(rng.choice(model_names, size=n_hold, replace=False).tolist())
    seen_pair_idx = np.array(
        [i for i, k in enumerate(all_pair_keys) if k.split(SEP)[0] not in unseen_m],
        dtype=np.int64,
    )
    unseen_pair_idx = np.array(
        [i for i, k in enumerate(all_pair_keys) if k.split(SEP)[0] in unseen_m],
        dtype=np.int64,
    )
    seen_m = [m for m in model_names if m not in unseen_m]
    return seen_pair_idx, unseen_pair_idx, sorted(seen_m), sorted(unseen_m)

def reindex_lc(samples: list, old_to_new: dict) -> list:
    return [(old_to_new[idx], p) for idx, p in samples if idx in old_to_new]

def make_collate_fn(text_variants: np.ndarray, held_out_idx: np.ndarray,
                    aug_variants: np.ndarray | None = None,
                    n_attr_tmpl: int = 0, attr_tmpl_prob: float | None = None):
    if aug_variants is not None:
        combined = np.concatenate([text_variants, aug_variants], axis=1)
    else:
        combined = text_variants
    n_pool = combined.shape[1]
    n_desc = n_pool - n_attr_tmpl

    text_tensor = torch.from_numpy(combined)

    desc_allowed = torch.stack(
        [
            torch.tensor(
                [v for v in range(n_desc) if v != int(h)], dtype=torch.long
            )
            for h in held_out_idx
        ],
        dim=0,
    )
    n_desc_allowed = n_desc - 1
    tmpl_start = n_desc

    weighted = (n_attr_tmpl > 0 and attr_tmpl_prob is not None)
    p_tmpl = float(attr_tmpl_prob) if weighted else 0.0

    uniform_allowed = torch.stack(
        [
            torch.tensor(
                [v for v in range(n_pool) if v != int(h)], dtype=torch.long
            )
            for h in held_out_idx
        ],
        dim=0,
    )
    n_uniform_allowed = n_pool - 1

    def collate(batch):
        audio = torch.stack([b[0] for b in batch], dim=0)
        cls_idx = torch.tensor([b[1] for b in batch], dtype=torch.long)
        B = len(batch)
        if not weighted:
            choice = torch.randint(0, n_uniform_allowed, (B,))
            v_idx = uniform_allowed[cls_idx, choice]
            is_tmpl = v_idx >= tmpl_start
        else:
            use_tmpl = torch.bernoulli(torch.full((B,), p_tmpl)).bool()
            v_idx = torch.empty(B, dtype=torch.long)
            if use_tmpl.any():
                tc = torch.randint(0, n_attr_tmpl, (int(use_tmpl.sum()),))
                v_idx[use_tmpl] = tmpl_start + tc
            if (~use_tmpl).any():
                dc = torch.randint(0, n_desc_allowed, (int((~use_tmpl).sum()),))
                v_idx[~use_tmpl] = desc_allowed[cls_idx[~use_tmpl], dc]
            is_tmpl = use_tmpl
        text = text_tensor[cls_idx, v_idx]
        return audio, text, cls_idx, is_tmpl

    return collate

def eval_collate(batch):
    audio = torch.stack([b[0] for b in batch])
    cls_idx = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return audio, torch.zeros(len(batch)), cls_idx
