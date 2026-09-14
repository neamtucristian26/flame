import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from tts_zs.attributes import build_pair_attributes
from tts_zs.config import (
    AUDIO_ROOT,
    CKPT_DIR,
    LABELS_LANG,
    RESULTS_DIR,
    SEED,
    TEXT_EMB_MODEL_DEFAULT,
    VAL_FRAC,
)
from tts_zs.data import (
    TTSAudioDataset,
    build_clip_index_lang,
    eval_collate,
    split_train_val_lc,
)
from tts_zs.eval import project_centroids
from tts_zs.model import ContrastiveModel
from tts_zs.templates import (
    ACOUSTIC_NAME,
    AXES,
    NAME_MAP,
    VOCODER_NAME,
    display_name,
    template_text,
    template_text_using,
    template_variants,
)
from tts_zs.text import (
    anchors_all_variants,
    anchors_from,
    encode_text_variants,
    pick_held_out,
)
from tts_zs.utils import encoder_uses_e5_prefix

CKPT_TMPL = "best_lc_zs_e5-large-v2__ho14_fold{f}_uniform_f{f}_s42.pt"
CONDITIONS = (
    "bare",
    "template",
    "template_using",
    "template_ensemble",
    "desc_centroid",
    "desc_centroid_ens",
)

def build_model(device) -> ContrastiveModel:
    return ContrastiveModel(
        proj_dim=256,
        hidden_dim=256,
        dropout=0.0,
        text_dim=1024,
        proj_style="linear1",
        text_proj_style="linear1",
        scale_max=30.0,
        normalize=True,
    ).to(device)

_ST_CACHE: dict = {}

def _get_st(encoder: str):
    if encoder not in _ST_CACHE:
        from sentence_transformers import SentenceTransformer

        _ST_CACHE[encoder] = SentenceTransformer(encoder, trust_remote_code=True)
    return _ST_CACHE[encoder]

def encode_strings(
    strings: list[str], encoder: str, device, prefix: str = "passage"
) -> torch.Tensor:
    if encoder_uses_e5_prefix(encoder):
        strings = [f"{prefix}: {s}" for s in strings]
    st = _get_st(encoder)
    embs = st.encode(
        strings,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return torch.from_numpy(embs.astype(np.float32)).to(device)

def acc_counts(pred_top: torch.Tensor, true_col: torch.Tensor, mask: torch.Tensor):
    if mask.sum() == 0:
        return 0, 0, 0
    p = pred_top[mask]
    t = true_col[mask]
    n = int(mask.sum())
    top1 = int((p[:, 0] == t).sum())
    top2 = int((p[:, :2] == t.unsqueeze(1)).any(dim=1).sum())
    return n, top1, top2

@torch.no_grad()
def run_fold(
    fold,
    *,
    all_pair_keys,
    all_text_variants,
    all_held_out_idx,
    all_attrs,
    clips_by_pair,
    encoder,
    device,
    num_workers,
    text_encoder_cache,
    anchor_prefix="passage",
):
    ckpt_path = CKPT_DIR / CKPT_TMPL.format(f=fold)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    model = build_model(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.layer_sum.load_state_dict(ckpt["layer_sum"])
    model.audio_proj.load_state_dict(ckpt["audio_proj"])
    model.text_proj.load_state_dict(ckpt["text_proj"])
    model.eval()

    cfg = ckpt.get("config", {})
    seen_keys = list(cfg["seen_pair_keys"])
    unseen_keys = list(cfg["unseen_pair_keys"])
    key_to_global = {k: i for i, k in enumerate(all_pair_keys)}
    seen_idx = np.array([key_to_global[k] for k in seen_keys], dtype=np.int64)
    unseen_idx = np.array([key_to_global[k] for k in unseen_keys], dtype=np.int64)

    all_anchors = anchors_from(all_text_variants, all_held_out_idx).to(device)
    all_centroids = project_centroids(model, all_anchors, device)

    all_var_anchors = anchors_all_variants(all_text_variants).to(device)
    n_var = all_var_anchors.shape[1]
    all_centroids_var = [
        project_centroids(model, all_var_anchors[:, v, :], device) for v in range(n_var)
    ]

    galleries = {}
    proj = {}
    seen_vals_by_axis = {}
    for axis in AXES:
        _SKIP = {"unknown", "proprietary"}
        values = sorted(
            {
                all_attrs[i][axis]
                for i in range(len(all_pair_keys))
                if all_attrs[i][axis] not in _SKIP
            }
        )
        val2col = {v: c for c, v in enumerate(values)}
        galleries[axis] = val2col
        seen_vals_by_axis[axis] = {all_attrs[i][axis] for i in seen_idx}

        for cond, mk in (
            ("bare", lambda v: display_name(axis, v)),
            ("template", lambda v: template_text(axis, display_name(axis, v))),
            (
                "template_using",
                lambda v: template_text_using(axis, display_name(axis, v)),
            ),
        ):
            ck = (axis, cond, anchor_prefix)
            if ck not in text_encoder_cache:
                raw = encode_strings(
                    [mk(v) for v in values], encoder, device, prefix=anchor_prefix
                )
                text_encoder_cache[ck] = (values, raw)
            cvals, raw = text_encoder_cache[ck]
            assert cvals == values
            proj[(axis, cond)] = project_centroids(model, raw, device)

        ck = (axis, "template_ensemble", anchor_prefix)
        if ck not in text_encoder_cache:
            all_strings = []
            for v in values:
                all_strings.extend(template_variants(axis, display_name(axis, v)))
            n_tmpl = len(all_strings) // len(values)
            raw_all = encode_strings(all_strings, encoder, device, prefix=anchor_prefix)
            raw_variants = raw_all.view(len(values), n_tmpl, -1)
            raw_mean = F.normalize(raw_variants.mean(dim=1), dim=-1)
            text_encoder_cache[ck] = (values, raw_mean)
        cvals, raw = text_encoder_cache[ck]
        assert cvals == values
        proj[(axis, "template_ensemble")] = project_centroids(model, raw, device)

        cols = []
        for v in values:
            members = [i for i in seen_idx if all_attrs[i][axis] == v]
            if members:
                c = F.normalize(all_centroids[members].mean(0, keepdim=True), dim=-1)
            else:
                c = torch.full((1, all_centroids.shape[1]), float("nan"), device=device)
            cols.append(c)
        proj[(axis, "desc_centroid")] = torch.cat(cols, 0)

        ens = []
        for v in range(n_var):
            cols_v = []
            for val in values:
                members = [i for i in seen_idx if all_attrs[i][axis] == val]
                if members:
                    c = F.normalize(
                        all_centroids_var[v][members].mean(0, keepdim=True), dim=-1
                    )
                else:
                    c = torch.full(
                        (1, all_centroids_var[v].shape[1]), float("nan"), device=device
                    )
                cols_v.append(c)
            ens.append(torch.cat(cols_v, 0))
        proj[(axis, "desc_centroid_ens")] = torch.stack(ens, 0)

    unseen_set = set(int(g) for g in unseen_idx)
    samples = [
        (g, p)
        for g, key in enumerate(all_pair_keys)
        if g in unseen_set
        for p in clips_by_pair[key]
    ]
    ds = TTSAudioDataset(samples)
    ld = DataLoader(
        ds,
        batch_size=256,
        shuffle=False,
        collate_fn=eval_collate,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    acc = {
        (axis, cond, sub): [0, 0, 0]
        for axis in AXES
        for cond in CONDITIONS
        for sub in ("overall", "seen", "novel")
    }
    model_votes = {
        (axis, cond): defaultdict(Counter) for axis in AXES for cond in CONDITIONS
    }
    model_true: dict = {axis: {} for axis in AXES}
    model_novel: dict = {axis: {} for axis in AXES}

    for audio, _t, idx in ld:
        audio = audio.to(device, non_blocking=True)
        idx_cpu = idx.tolist()
        models = [all_attrs[g]["model"] for g in idx_cpu]
        a = F.normalize(model.audio_proj(model.layer_sum(audio)), dim=-1)
        for axis in AXES:
            val2col = galleries[axis]
            _SKIP = {"unknown", "proprietary"}
            true_col_list = [val2col.get(all_attrs[g][axis], -1) for g in idx_cpu]
            novel_list = [
                all_attrs[g][axis] not in seen_vals_by_axis[axis] for g in idx_cpu
            ]
            true_col = torch.tensor(true_col_list, device=device)
            is_novel = torch.tensor(novel_list, device=device)
            evaluable = torch.tensor(
                [all_attrs[g][axis] not in _SKIP for g in idx_cpu], device=device
            )
            for bi, m in enumerate(models):
                if all_attrs[idx_cpu[bi]][axis] in _SKIP:
                    continue
                model_true[axis][m] = true_col_list[bi]
                model_novel[axis][m] = bool(novel_list[bi])
            for cond in CONDITIONS:
                G = proj[(axis, cond)]
                anchor_sets = G if cond == "desc_centroid_ens" else G.unsqueeze(0)
                for Gv in anchor_sets:
                    sim = a @ Gv.T
                    sim = torch.nan_to_num(sim, nan=float("-inf"))
                    top = sim.topk(min(2, sim.shape[1]), dim=1).indices
                    for sub, mask in (
                        ("overall", evaluable),
                        ("seen", evaluable & ~is_novel),
                        ("novel", evaluable & is_novel),
                    ):
                        n, t1, t2 = acc_counts(top, true_col, mask)
                        a3 = acc[(axis, cond, sub)]
                        a3[0] += n
                        a3[1] += t1
                        a3[2] += t2
                    top1 = top[:, 0].tolist()
                    for bi, m in enumerate(models):
                        if all_attrs[idx_cpu[bi]][axis] in _SKIP:
                            continue
                        model_votes[(axis, cond)][m][top1[bi]] += 1

    def pack(axis, cond, sub):
        n, t1, t2 = acc[(axis, cond, sub)]
        return {
            "n": n,
            "acc1": (t1 / n if n else None),
            "acc2": (t2 / n if n else None),
        }

    def pack_model(axis, cond, sub):
        n = correct = 0
        for m, cnt in model_votes[(axis, cond)].items():
            nov = model_novel[axis][m]
            if sub == "seen" and nov:
                continue
            if sub == "novel" and not nov:
                continue
            n += 1
            pred = cnt.most_common(1)[0][0]
            if pred == model_true[axis][m]:
                correct += 1
        return {
            "n_models": n,
            "n_correct": correct,
            "acc1": (correct / n if n else None),
        }

    novel_counts = {axis: acc[(axis, "bare", "novel")][0] for axis in AXES}
    novel_models = {
        axis: sum(1 for m in model_novel[axis] if model_novel[axis][m]) for axis in AXES
    }
    return {
        "fold": fold,
        "n_unseen_clips": len(samples),
        "n_unseen_models": len(model_true[AXES[0]]),
        "novel_clip_counts": novel_counts,
        "novel_model_counts": novel_models,
        "results": {
            axis: {
                cond: {
                    sub: pack(axis, cond, sub) for sub in ("overall", "seen", "novel")
                }
                for cond in CONDITIONS
            }
            for axis in AXES
        },
        "model_results": {
            axis: {
                cond: {
                    sub: pack_model(axis, cond, sub)
                    for sub in ("overall", "seen", "novel")
                }
                for cond in CONDITIONS
            }
            for axis in AXES
        },
    }

def pool(per_fold):
    out = {axis: {cond: {} for cond in CONDITIONS} for axis in AXES}
    for axis in AXES:
        for cond in CONDITIONS:
            for sub in ("overall", "seen", "novel"):
                N = T1 = T2 = 0
                for f in per_fold:
                    c = f["results"][axis][cond][sub]
                    if c["n"]:
                        N += c["n"]
                        T1 += round(c["acc1"] * c["n"])
                        T2 += round(c["acc2"] * c["n"])
                out[axis][cond][sub] = {
                    "n": N,
                    "acc1": (T1 / N if N else None),
                    "acc2": (T2 / N if N else None),
                }
    return out

def pool_model(per_fold):
    out = {axis: {cond: {} for cond in CONDITIONS} for axis in AXES}
    for axis in AXES:
        for cond in CONDITIONS:
            for sub in ("overall", "seen", "novel"):
                N = C = 0
                for f in per_fold:
                    c = f["model_results"][axis][cond][sub]
                    N += c["n_models"]
                    C += c["n_correct"]
                out[axis][cond][sub] = {
                    "n_models": N,
                    "n_correct": C,
                    "acc1": (C / N if N else None),
                }
    return out

def main():
    global CKPT_TMPL
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-tmpl", type=str, default=CKPT_TMPL)
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Split/held-out seed; match the checkpoint training seed.",
    )
    p.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--text-encoder", type=str, default=TEXT_EMB_MODEL_DEFAULT)
    p.add_argument("--audio-root", type=Path, default=AUDIO_ROOT)
    p.add_argument("--labels-path", type=Path, default=LABELS_LANG)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--anchor-prefix",
        choices=("query", "passage"),
        default="passage",
        help="E5 role prefix for the short bare/template anchors "
        "(desc_centroid always uses the passage descriptions).",
    )
    p.add_argument("--out-suffix", type=str, default="")
    args = p.parse_args()

    CKPT_TMPL = args.ckpt_tmpl
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  seed: {args.seed}")

    with open(args.labels_path) as f:
        lc_labels = json.load(f)
    all_pair_keys = sorted(lc_labels.keys())
    n_pairs = len(all_pair_keys)
    all_text_variants = encode_text_variants(
        lc_labels,
        all_pair_keys,
        encoder=args.text_encoder,
        labels_path=args.labels_path,
    )
    all_held_out_idx = pick_held_out(n_pairs, seed=args.seed)
    all_attrs = build_pair_attributes(all_pair_keys)
    clips_by_pair = build_clip_index_lang(all_pair_keys, args.audio_root)

    text_cache: dict = {}
    per_fold = []
    for fold in args.folds:
        print(f"\n=== fold {fold} ===")
        r = run_fold(
            fold,
            all_pair_keys=all_pair_keys,
            all_text_variants=all_text_variants,
            all_held_out_idx=all_held_out_idx,
            all_attrs=all_attrs,
            clips_by_pair=clips_by_pair,
            encoder=args.text_encoder,
            device=device,
            num_workers=args.num_workers,
            text_encoder_cache=text_cache,
            anchor_prefix=args.anchor_prefix,
        )
        per_fold.append(r)
        vc = r["results"]["vocoder"]
        ac = r["results"]["acoustic_model"]
        print(
            f"  unseen clips {r['n_unseen_clips']}  novel(voc/ac)="
            f"{r['novel_clip_counts']['vocoder']}/{r['novel_clip_counts']['acoustic_model']}"
        )
        for axis, d in (("voc", vc), ("ac", ac)):
            print(
                f"  {axis}: bare={d['bare']['overall']['acc1']:.3f} "
                f"tmpl={d['template']['overall']['acc1']:.3f} "
                f"ens={d['template_ensemble']['overall']['acc1']:.3f} "
                f"desc={d['desc_centroid']['overall']['acc1']:.3f}"
            )

    pooled = pool(per_fold)
    pooled_model = pool_model(per_fold)
    out = {
        "seed": args.seed,
        "anchor_prefix": args.anchor_prefix,
        "axes": list(AXES),
        "conditions": list(CONDITIONS),
        "per_fold": per_fold,
        "pooled": pooled,
        "pooled_model": pooled_model,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"attribute_retrieval{args.out_suffix}.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_json}")
    for axis in AXES:
        print(f"\n[{axis}] CLIP acc1 (overall / seen / novel):")
        for cond in CONDITIONS:
            d = pooled[axis][cond]

            def s(x):
                return (
                    f"{x['acc1']:.3f}(n={x['n']})"
                    if x["acc1"] is not None
                    else f"--(n={x['n']})"
                )

            print(f"  {cond:13s}  {s(d['overall'])}  {s(d['seen'])}  {s(d['novel'])}")
        print(f"[{axis}] PER-MODEL acc1 (overall / seen / novel):")
        for cond in CONDITIONS:
            d = pooled_model[axis][cond]

            def sm(x):
                return (
                    f"{x['acc1']:.3f}(m={x['n_models']})"
                    if x["acc1"] is not None
                    else f"--(m={x['n_models']})"
                )

            print(
                f"  {cond:13s}  {sm(d['overall'])}  {sm(d['seen'])}  {sm(d['novel'])}"
            )

if __name__ == "__main__":
    main()
