import argparse
import json
import sys
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
    AUDIO_ROOT, CKPT_DIR, LABELS_LANG, RESULTS_DIR, SEED, TEXT_EMB_MODEL_DEFAULT,
    VAL_FRAC,
)
from tts_zs.data import (
    TTSAudioDataset, build_clip_index_lang, eval_collate, split_train_val_lc,
)
from tts_zs.eval import project_centroids
from tts_zs.model import ContrastiveModel
from tts_zs.text import (
    anchors_all_variants, anchors_from, encode_text_variants, pick_held_out,
)

CKPT_TMPL = "best_lc_zs_e5-large-v2__ho14_fold{f}_uniform_f{f}_s42.pt"
LEVELS = ("pair", "model")
TREATMENTS = ("heldout", "ens_meanmetrics", "ens_meanemb")
METS = ("Hit@1", "P@10", "mAP", "random")

def build_model(device) -> ContrastiveModel:
    return ContrastiveModel(
        proj_dim=256, hidden_dim=256, dropout=0.0, text_dim=1024,
        proj_style="linear1", text_proj_style="linear1",
        scale_max=30.0, normalize=True,
    ).to(device)

@torch.no_grad()
def query_metrics(sim_row: torch.Tensor, relevant: torch.Tensor):
    total = int(relevant.sum())
    if total == 0:
        return None
    order = torch.argsort(sim_row, descending=True)
    rel = relevant[order].float()
    cum = torch.cumsum(rel, 0)
    ranks = torch.arange(1, rel.numel() + 1, device=rel.device, dtype=torch.float)
    prec_at_k = cum / ranks
    ap = (prec_at_k * rel).sum() / total
    r1 = rel[0]
    k = min(10, rel.numel())
    p10 = cum[k - 1] / k
    return float(r1), float(p10), float(ap)

@torch.no_grad()
def run_fold(fold, *, all_pair_keys, all_text_variants, all_held_out_idx, all_attrs,
             clips_by_pair, device, num_workers, gallery="unseen", max_clips_per_pair=0,
             val_by_g=None):
    ckpt_path = CKPT_DIR / CKPT_TMPL.format(f=fold)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    model = build_model(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.layer_sum.load_state_dict(ckpt["layer_sum"])
    model.audio_proj.load_state_dict(ckpt["audio_proj"])
    model.text_proj.load_state_dict(ckpt["text_proj"])
    model.eval()

    cfg = ckpt["config"]
    key_to_global = {k: i for i, k in enumerate(all_pair_keys)}
    unseen_idx = [key_to_global[k] for k in cfg["unseen_pair_keys"]]
    seen_idx = [key_to_global[k] for k in cfg["seen_pair_keys"]]

    if gallery == "seen":
        query_idx = list(seen_idx)
        gallery_idx = list(seen_idx)
    elif gallery == "full":
        query_idx = list(unseen_idx)
        gallery_idx = list(range(len(all_pair_keys)))
    else:
        query_idx = list(unseen_idx)
        gallery_idx = list(unseen_idx)

    qidx = torch.tensor(query_idx, device=device)
    model_id = {m: i for i, m in
                enumerate(sorted({all_attrs[g]["model"] for g in gallery_idx} |
                                 {all_attrs[g]["model"] for g in query_idx}))}
    q_pair = qidx
    q_model = torch.tensor([model_id[all_attrs[g]["model"]] for g in query_idx], device=device)

    Q_ho = project_centroids(model, anchors_from(all_text_variants, all_held_out_idx).to(device),
                             device)[qidx]
    Q_var = Q_me = None
    if gallery != "seen":
        av5 = anchors_all_variants(all_text_variants).to(device)
        Q_var = [project_centroids(model, av5[:, k, :], device)[qidx] for k in range(av5.shape[1])]
        Q_me = project_centroids(model, av5[query_idx], device)

    samples = []
    for g in gallery_idx:
        if gallery == "seen":
            clips = val_by_g.get(g, [])
        else:
            clips = clips_by_pair[all_pair_keys[g]]
            if max_clips_per_pair:
                clips = clips[:max_clips_per_pair]
        for p in clips:
            samples.append((g, p))
    ds = TTSAudioDataset(samples)
    ld = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=eval_collate,
                    num_workers=num_workers, pin_memory=(device.type == "cuda"))
    embs, clip_pair = [], []
    for audio, _t, idx in ld:
        audio = audio.to(device, non_blocking=True)
        a = F.normalize(model.audio_proj(model.layer_sum(audio)), dim=-1)
        embs.append(a)
        clip_pair.append(idx.to(device))
    A = torch.cat(embs, 0)
    clip_pair = torch.cat(clip_pair, 0)
    clip_model = torch.tensor([model_id[all_attrs[int(g)]["model"]] for g in clip_pair.cpu()],
                              device=device)
    N = A.shape[0]

    def eval_queries(Q):
        sims = Q @ A.T
        agg = {lvl: {"hit": [], "p10": [], "ap": [], "rand": []} for lvl in LEVELS}
        for i in range(Q.shape[0]):
            row = sims[i]
            for lvl in LEVELS:
                rel = (clip_pair == q_pair[i]) if lvl == "pair" else (clip_model == q_model[i])
                m = query_metrics(row, rel)
                if m is None:
                    continue
                agg[lvl]["hit"].append(m[0]); agg[lvl]["p10"].append(m[1]); agg[lvl]["ap"].append(m[2])
                agg[lvl]["rand"].append(float(rel.float().mean()))
        res = {}
        for lvl in LEVELS:
            d = agg[lvl]; n = len(d["ap"])
            res[lvl] = {"n_queries": n,
                        "Hit@1": float(np.mean(d["hit"])) if n else None,
                        "P@10": float(np.mean(d["p10"])) if n else None,
                        "mAP": float(np.mean(d["ap"])) if n else None,
                        "random": float(np.mean(d["rand"])) if n else None}
        return res

    def avg_res(reslist):
        out = {}
        for lvl in LEVELS:
            out[lvl] = {"n_queries": reslist[0][lvl]["n_queries"]}
            for met in METS:
                vals = [r[lvl][met] for r in reslist if r[lvl][met] is not None]
                out[lvl][met] = float(np.mean(vals)) if vals else None
        return out

    if gallery == "seen":
        treatments = {"heldout": eval_queries(Q_ho)}
    else:
        treatments = {
            "heldout": eval_queries(Q_ho),
            "ens_meanmetrics": avg_res([eval_queries(Qk) for Qk in Q_var]),
            "ens_meanemb": eval_queries(Q_me),
        }
    return {"fold": fold, "n_gallery": int(N), "treatments": treatments}

def pool(per_fold):
    present = list(per_fold[0]["treatments"].keys())
    out = {tr: {} for tr in present}
    for tr in present:
        for lvl in LEVELS:
            agg = {}
            for met in METS:
                num = den = 0.0
                for f in per_fold:
                    c = f["treatments"][tr][lvl]
                    if c["n_queries"] and c[met] is not None:
                        num += c[met] * c["n_queries"]; den += c["n_queries"]
                agg[met] = (num / den if den else None)
            agg["n_queries"] = sum(f["treatments"][tr][lvl]["n_queries"] for f in per_fold)
            out[tr][lvl] = agg
    return out

def main():
    global CKPT_TMPL
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-tmpl", type=str, default=CKPT_TMPL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--text-encoder", type=str, default=TEXT_EMB_MODEL_DEFAULT)
    p.add_argument("--audio-root", type=Path, default=AUDIO_ROOT)
    p.add_argument("--labels-path", type=Path, default=LABELS_LANG)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-suffix", type=str, default="")
    p.add_argument("--gallery", choices=["unseen", "full", "seen"], default="unseen",
                   help="'unseen': held-out clips only; 'full': all 298 systems' clips; "
                        "'seen': seen systems' 10%% val clips (sanity mirror)")
    p.add_argument("--max-clips-per-pair", type=int, default=0,
                   help="cap gallery clips per pair (0 = all); e.g. 100 bounds full-gallery cost")
    args = p.parse_args()

    CKPT_TMPL = args.ckpt_tmpl
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  seed: {args.seed}  gallery: {args.gallery}  "
          f"max_clips_per_pair: {args.max_clips_per_pair or 'all'}")

    with open(args.labels_path) as f:
        lc_labels = json.load(f)
    all_pair_keys = sorted(lc_labels.keys())
    n_pairs = len(all_pair_keys)
    all_text_variants = encode_text_variants(
        lc_labels, all_pair_keys, encoder=args.text_encoder, labels_path=args.labels_path)
    all_held_out_idx = pick_held_out(n_pairs, seed=args.seed)
    all_attrs = build_pair_attributes(all_pair_keys)
    clips_by_pair = build_clip_index_lang(all_pair_keys, args.audio_root)

    val_by_g = None
    if args.gallery == "seen":
        _, all_val_g = split_train_val_lc(clips_by_pair, VAL_FRAC, args.seed)
        val_by_g = {}
        for g, p in all_val_g:
            val_by_g.setdefault(g, []).append(p)

    per_fold = []
    for fold in args.folds:
        print(f"\n=== fold {fold} ===")
        r = run_fold(fold, all_pair_keys=all_pair_keys, all_text_variants=all_text_variants,
                     all_held_out_idx=all_held_out_idx, all_attrs=all_attrs,
                     clips_by_pair=clips_by_pair, device=device, num_workers=args.num_workers,
                     gallery=args.gallery, max_clips_per_pair=args.max_clips_per_pair,
                     val_by_g=val_by_g)
        per_fold.append(r)
        for tr in r["treatments"]:
            d = r["treatments"][tr]["model"]
            print(f"  {tr:16s} model: Hit@1={d['Hit@1']:.3f}  P@10={d['P@10']:.3f}  "
                  f"mAP={d['mAP']:.3f}  (random~{d['random']:.4f})  q={d['n_queries']}")

    pooled = pool(per_fold)
    out = {"seed": args.seed, "gallery": args.gallery,
           "max_clips_per_pair": args.max_clips_per_pair,
           "levels": list(LEVELS), "treatments": list(pooled.keys()),
           "per_fold": per_fold, "pooled": pooled}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gtag = {"full": "_full", "seen": "_seen"}.get(args.gallery, "")
    out_json = RESULTS_DIR / f"text_to_audio{gtag}{args.out_suffix}.json"
    json.dump(out, open(out_json, "w"), indent=2)
    print(f"\nWrote {out_json}")
    for tr in pooled:
        for lvl in LEVELS:
            d = pooled[tr][lvl]
            print(f"[{tr:16s} {lvl:5s}] Hit@1={d['Hit@1']:.3f}  P@10={d['P@10']:.3f}  "
                  f"mAP={d['mAP']:.3f}  random~{d['random']:.4f}  (q={d['n_queries']})")

if __name__ == "__main__":
    main()
