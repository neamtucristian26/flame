import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tts_zs.attributes import build_pair_attributes
from tts_zs.config import AUDIO_ROOT, CKPT_DIR, RESULTS_DIR, VAL_FRAC
from tts_zs.data import (
    TTSAudioDataset, build_clip_index_lang, eval_collate, split_train_val_lc,
)
from tts_zs.eval import SIBLING_AXES, _metrics
from tts_zs.model import ContrastiveModel
from tts_zs.text import anchors_from, encode_text_variants, pick_held_out

LABELS_PATH = ROOT / "data" / "labels_lang.json"
TEXT_ENCODER = "intfloat/e5-large-v2"
SEEDS = [42, 43, 44]
CKPT_TMPL = "best_lc_zs_e5-large-v2__ho14_fold{f}_ablsc_f5_p10_f{f}_s{s}.pt"
SCENARIOS = ("seen_sanity", "unseen_restricted", "unseen_full")
METRIC_KEYS = ("R@1", "R@5", "R@10", "MRR")
REPORT_AXES = ("pair", "model", "arch_family", "acoustic_model", "vocoder",
               "speaker_type", "lang_family")

def build_model(device):
    return ContrastiveModel(
        proj_dim=256, hidden_dim=256, dropout=0.0, text_dim=1024,
        proj_style="linear1", text_proj_style="linear1",
        scale_max=30.0, normalize=True,
    ).to(device)

@torch.no_grad()
def audio_embeds(model, samples, device):
    ld = DataLoader(TTSAudioDataset(samples), batch_size=256, shuffle=False,
                    collate_fn=eval_collate, num_workers=4,
                    pin_memory=(device.type == "cuda"))
    A, P = [], []
    for audio, _t, pair_idx in ld:
        a = F.normalize(model.audio_proj(model.layer_sum(audio.to(device))), dim=-1)
        A.append(a); P.append(pair_idx.to(device))
    return torch.cat(A), torch.cat(P)

@torch.no_grad()
def sibling_from_embeds(A, pair_idx, centroids, gallery_attrs, query_attrs, device):
    n_gallery = len(gallery_attrs)
    axis_vocab, gallery_axis_ids = {}, {}
    for axis in SIBLING_AXES:
        vocab = {}
        for g in gallery_attrs:
            if g[axis] not in vocab:
                vocab[g[axis]] = len(vocab)
        axis_vocab[axis] = vocab
        gallery_axis_ids[axis] = torch.tensor(
            [vocab[g[axis]] for g in gallery_attrs], dtype=torch.long, device=device)

    pair_ranks, axis_ranks, recalls = [], {ax: [] for ax in SIBLING_AXES}, []
    recall_axes = ("arch_family", "vocoder", "speaker_type", "lang_family")
    CH = 4096
    for s in range(0, A.shape[0], CH):
        a = A[s:s + CH]; pj = pair_idx[s:s + CH]
        order = (a @ centroids.T).argsort(dim=1, descending=True)
        pair_ranks.append((order == pj.unsqueeze(1)).nonzero()[:, 1].cpu())
        pj_cpu = pj.cpu().tolist()
        for axis in SIBLING_AXES:
            vocab = axis_vocab[axis]
            q_ids = torch.tensor([vocab.get(query_attrs[i][axis], -1) for i in pj_cpu],
                                 dtype=torch.long, device=device)
            match = gallery_axis_ids[axis][order] == q_ids.unsqueeze(1)
            has = match.any(dim=1)
            rank = torch.where(has, match.int().argmax(dim=1),
                               torch.full((a.shape[0],), n_gallery, device=device))
            axis_ranks[axis].append(rank.cpu())
        for bi, gi in zip(pj_cpu, order[:, 0].cpu().tolist()):
            q, g = query_attrs[bi], gallery_attrs[gi]
            recalls.append(sum(1 for ax in recall_axes if q[ax] == g[ax]) / 4.0)

    out = {"pair": _metrics(torch.cat(pair_ranks))}
    for axis in SIBLING_AXES:
        out[axis] = _metrics(torch.cat(axis_ranks[axis]))
    out["attribute_recall@1"] = {"mean": float(sum(recalls) / max(len(recalls), 1)),
                                 "n_val": len(recalls)}
    return out

def avg_metrics(results):
    out = {}
    for axis in list(results[0].keys()):
        if axis == "attribute_recall@1":
            out[axis] = {"mean": float(np.mean([r[axis]["mean"] for r in results])),
                         "n_val": results[0][axis]["n_val"]}
        else:
            out[axis] = {k: (results[0][axis]["n_val"] if k == "n_val"
                             else float(np.mean([r[axis][k] for r in results])))
                         for k in (*METRIC_KEYS, "n_val")}
    return out

def run_seed(seed, all_pair_keys, all_text_variants, attrs, clips_by_pair, device):
    n_pairs = len(all_pair_keys)
    held_out_idx = pick_held_out(n_pairs, seed=seed)
    _, all_val_g = split_train_val_lc(clips_by_pair, VAL_FRAC, seed)
    n_var = all_text_variants.shape[1]
    per_fold = []
    for fold in range(10):
        ckpt_path = CKPT_DIR / CKPT_TMPL.format(f=fold, s=seed)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        model = build_model(device)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.layer_sum.load_state_dict(ck["layer_sum"])
        model.audio_proj.load_state_dict(ck["audio_proj"])
        model.text_proj.load_state_dict(ck["text_proj"])
        model.eval()
        cfg = ck["config"]
        k2g = {k: i for i, k in enumerate(all_pair_keys)}
        seen_idx = [k2g[k] for k in cfg["seen_pair_keys"]]
        unseen_idx = [k2g[k] for k in cfg["unseen_pair_keys"]]
        g2seen = {g: i for i, g in enumerate(seen_idx)}
        g2uns = {g: i for i, g in enumerate(unseen_idx)}
        seen_attrs = [attrs[i] for i in seen_idx]
        uns_attrs = [attrs[i] for i in unseen_idx]

        heldout = F.normalize(
            model.text_proj(anchors_from(all_text_variants, held_out_idx).to(device)), dim=-1)
        var_cents = [F.normalize(
            model.text_proj(torch.from_numpy(all_text_variants[:, k, :]).float().to(device)), dim=-1)
            for k in range(n_var)]

        res = {}
        seen_set = set(seen_idx)
        seen_samples = [(g2seen[int(g)], p) for g, p in all_val_g if int(g) in seen_set]
        As, Ps = audio_embeds(model, seen_samples, device)
        res["seen_sanity"] = sibling_from_embeds(
            As, Ps, heldout[seen_idx], seen_attrs, seen_attrs, device)

        uns_samples = [(g, p) for g in unseen_idx for p in clips_by_pair[all_pair_keys[g]]]
        Au, Pu_g = audio_embeds(model, uns_samples, device)
        Pu_local = torch.tensor([g2uns[int(g)] for g in Pu_g.cpu().tolist()],
                                dtype=torch.long, device=device)
        restr = [sibling_from_embeds(Au, Pu_local, var_cents[k][unseen_idx],
                                     uns_attrs, uns_attrs, device) for k in range(n_var)]
        full = [sibling_from_embeds(Au, Pu_g, var_cents[k], attrs, attrs, device)
                for k in range(n_var)]
        res["unseen_restricted"] = avg_metrics(restr)
        res["unseen_full"] = avg_metrics(full)
        per_fold.append(res)
        print(f"  seed {seed} fold {fold}: "
              f"u-full pairR@1={res['unseen_full']['pair']['R@1']:.3f} "
              f"modelMRR={res['unseen_full']['model']['MRR']:.3f}")

    pooled = {}
    for sc in SCENARIOS:
        pooled[sc] = {}
        for axis in REPORT_AXES:
            nv = np.array([f[sc][axis]["n_val"] for f in per_fold], dtype=float)
            pooled[sc][axis] = {k: float(np.average([f[sc][axis][k] for f in per_fold], weights=nv))
                                for k in METRIC_KEYS}
        nv = np.array([f[sc]["attribute_recall@1"]["n_val"] for f in per_fold], dtype=float)
        pooled[sc]["attribute_recall@1"] = float(
            np.average([f[sc]["attribute_recall@1"]["mean"] for f in per_fold], weights=nv))
    return pooled

def main():
    global CKPT_TMPL
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-tmpl", type=str, default=CKPT_TMPL,
                    help="Per-(fold,seed) checkpoint template; use {f} and {s}.")
    ap.add_argument("--out-tag", type=str, default="f5_p10",
                    help="Output files: sibling_ensemble_<tag>.{json,md}.")
    ap.add_argument("--labels-path", type=Path, default=LABELS_PATH,
                    help="Description-text source; override for text-side ablations "
                         "(e.g. language-redacted labels), no retraining needed since "
                         "the E5 text encoder is frozen.")
    args = ap.parse_args()
    CKPT_TMPL = args.ckpt_tmpl
    out_tag = args.out_tag
    labels_path = args.labels_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  labels: {labels_path}")
    lc = json.load(open(labels_path))
    all_pair_keys = sorted(lc.keys())
    attrs = build_pair_attributes(all_pair_keys)
    atv = np.asarray(encode_text_variants(
        lc, all_pair_keys, encoder=TEXT_ENCODER, labels_path=labels_path))
    clips_by_pair = build_clip_index_lang(all_pair_keys, AUDIO_ROOT)

    per_seed = {}
    for seed in SEEDS:
        print(f"=== seed {seed} ===")
        per_seed[seed] = run_seed(seed, all_pair_keys, atv, attrs, clips_by_pair, device)

    agg = {}
    for sc in SCENARIOS:
        agg[sc] = {}
        for axis in REPORT_AXES:
            agg[sc][axis] = {}
            for k in METRIC_KEYS:
                vals = np.array([per_seed[s][sc][axis][k] for s in SEEDS])
                agg[sc][axis][k] = [float(vals.mean()), float(vals.std(ddof=1))]
        vals = np.array([per_seed[s][sc]["attribute_recall@1"] for s in SEEDS])
        agg[sc]["attribute_recall@1"] = [float(vals.mean()), float(vals.std(ddof=1))]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"per_seed": per_seed, "agg": agg, "seeds": SEEDS,
               "convention": "seen=held-out variant; unseen=5-variant mean-of-metrics"},
              open(RESULTS_DIR / f"sibling_ensemble_{out_tag}.json", "w"), indent=2)

    L = [f"# {out_tag} sibling-ensemble (seen=held-out variant, unseen=5-variant mean-of-metrics)",
         f"3 seeds {SEEDS}, clip-weighted over 10 folds, mean +/- sd (%).", ""]
    for sc in SCENARIOS:
        L += [f"## {sc}", "", "| axis | R@1 | R@5 | R@10 | MRR |", "|---|---|---|---|---|"]
        for axis in REPORT_AXES:
            r = agg[sc][axis]
            L.append(f"| {axis} | " + " | ".join(
                f"{100*r[k][0]:.1f} ± {100*r[k][1]:.1f}" for k in METRIC_KEYS) + " |")
        ar = agg[sc]["attribute_recall@1"]
        L.append(f"| attr_recall@1 | {100*ar[0]:.1f} ± {100*ar[1]:.1f} | — | — | — |")
        L.append("")
    (RESULTS_DIR / f"sibling_ensemble_{out_tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote sibling_ensemble_{out_tag}.{{json,md}}")

if __name__ == "__main__":
    main()
