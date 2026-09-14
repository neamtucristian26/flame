import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_zs.attributes import build_pair_attributes
from tts_zs.config import (
    AUDIO_DIM, AUDIO_ROOT, CKPT_DIR, DATA_DIR, LABELS_LANG, LAYER_WEIGHTS_CKPT,
    N_AUDIO_LAYERS, RESULTS_DIR, SEP, VAL_FRAC,
)
from tts_zs.data import (
    TTSAudioDataset, build_clip_index_lang, eval_collate,
    select_holdout_models, split_train_val_lc,
)
from tts_zs.model import ProjectionMLP, WeightedLayerSum
from tts_zs.utils import set_seed

HEADS = ("pair", "model", "arch_family", "acoustic_model", "vocoder",
         "speaker_type", "lang_family")

class SingleHeadBaseline(nn.Module):

    def __init__(self, n_classes: int, proj_dim: int = 256, hidden_dim: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.layer_sum = WeightedLayerSum(N_AUDIO_LAYERS)
        self.proj = ProjectionMLP(
            AUDIO_DIM, hidden_dim, proj_dim, dropout, style="linear1", normalize=True
        )
        self.head = nn.Linear(proj_dim, n_classes)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.head(self.proj(self.layer_sum(audio)))

@torch.no_grad()
def evaluate(models, loader, lut, n_classes, device) -> dict:
    for m in models.values():
        m.eval()
    preds = {h: [] for h in HEADS}
    gts = {h: [] for h in HEADS}
    for audio, _z, pair_idx in loader:
        audio = audio.to(device, non_blocking=True)
        pair_idx = pair_idx.to(device)
        for h in HEADS:
            preds[h].append(models[h](audio).argmax(dim=1).cpu())
            gts[h].append(lut[h][pair_idx].cpu())
    out = {}
    for h in HEADS:
        y = torch.cat(gts[h]).numpy()
        p = torch.cat(preds[h]).numpy()
        out[h] = {
            "acc": float((y == p).mean()),
            "macro_f1": float(f1_score(y, p, average="macro",
                                       labels=list(range(n_classes[h])), zero_division=0)),
            "macro_acc": float(balanced_accuracy_score(y, p)),
            "n": int(len(y)),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--held-out-models", type=int, default=14)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--layer-weights-ckpt", type=Path, default=LAYER_WEIGHTS_CKPT)
    ap.add_argument("--labels-path", type=Path, default=LABELS_LANG)
    ap.add_argument("--arch-labels-path", type=Path, default=DATA_DIR / "arch_labels.json")
    ap.add_argument("--run-suffix", type=str, default="_allheads",
                    help="Appended to run tag; default '_allheads' separates these results "
                         "from the legacy 3-head baseline.")
    ap.add_argument("--no-layer-warmstart", action="store_true",
                    help="Uniform layer-sum init (1/n_layers); matches the contrastive model.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  fold: {args.fold}  seed: {args.seed}")

    with open(args.labels_path, encoding="utf-8") as f:
        all_pair_keys = sorted(json.load(f).keys())
    n_pairs = len(all_pair_keys)
    with open(args.arch_labels_path, encoding="utf-8") as f:
        arch_labels = json.load(f)

    seen_pair_idx, _unseen_idx, seen_model_names, _unseen_models = select_holdout_models(
        all_pair_keys, args.held_out_models, args.seed,
        fold_id=args.fold, arch_labels=arch_labels,
    )
    seen_set = set(int(i) for i in seen_pair_idx)
    attrs = build_pair_attributes(all_pair_keys)

    pair_classes   = sorted([all_pair_keys[i] for i in seen_pair_idx])
    model_classes  = sorted(seen_model_names)
    enc = {
        "pair":          {c: j for j, c in enumerate(pair_classes)},
        "model":         {c: j for j, c in enumerate(model_classes)},
        "arch_family":   {c: j for j, c in enumerate(sorted(
                              {attrs[i]["arch_family"] for i in seen_pair_idx}))},
        "acoustic_model":{c: j for j, c in enumerate(sorted(
                              {attrs[i]["acoustic_model"] for i in seen_pair_idx}))},
        "vocoder":       {c: j for j, c in enumerate(sorted(
                              {attrs[i]["vocoder"] for i in seen_pair_idx}))},
        "speaker_type":  {c: j for j, c in enumerate(sorted(
                              {attrs[i]["speaker_type"] for i in seen_pair_idx}))},
        "lang_family":   {c: j for j, c in enumerate(sorted(
                              {attrs[i]["lang_family"] for i in seen_pair_idx}))},
    }
    n_classes = {h: len(enc[h]) for h in HEADS}
    print("[classes] " + "  ".join(f"{h}={n_classes[h]}" for h in HEADS))
    print(f"seen pairs={len(seen_pair_idx)}")

    lut = {h: torch.full((n_pairs,), -1, dtype=torch.long) for h in HEADS}
    for i in seen_pair_idx:
        i = int(i)
        key = all_pair_keys[i]
        lut["pair"][i]           = enc["pair"][key]
        lut["model"][i]          = enc["model"][key.split(SEP)[0]]
        lut["arch_family"][i]    = enc["arch_family"][attrs[i]["arch_family"]]
        lut["acoustic_model"][i] = enc["acoustic_model"][attrs[i]["acoustic_model"]]
        lut["vocoder"][i]        = enc["vocoder"][attrs[i]["vocoder"]]
        lut["speaker_type"][i]   = enc["speaker_type"][attrs[i]["speaker_type"]]
        lut["lang_family"][i]    = enc["lang_family"][attrs[i]["lang_family"]]
    lut = {h: t.to(device) for h, t in lut.items()}

    clips_by_pair = build_clip_index_lang(all_pair_keys, AUDIO_ROOT)
    train_g, val_g = split_train_val_lc(clips_by_pair, VAL_FRAC, args.seed)
    train_seen = [(pi, p) for pi, p in train_g if pi in seen_set]
    val_seen   = [(pi, p) for pi, p in val_g   if pi in seen_set]
    print(f"[clips] train={len(train_seen)} val={len(val_seen)}")

    def make_loader(samples, shuffle):
        return DataLoader(
            TTSAudioDataset(samples), batch_size=args.batch_size, shuffle=shuffle,
            collate_fn=eval_collate, num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"), drop_last=False,
        )

    train_loader = make_loader(train_seen, shuffle=True)
    val_loader   = make_loader(val_seen,   shuffle=False)

    layer_w = None
    if not args.no_layer_warmstart:
        ckpt = torch.load(args.layer_weights_ckpt, map_location="cpu", weights_only=False)
        layer_w = ckpt["model"]["combiner.weights"]

    models, optimisers, schedulers = {}, {}, {}
    steps_per_epoch = max(1, len(train_loader))
    total_steps  = args.epochs * steps_per_epoch
    warmup_steps = steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    for h in HEADS:
        m = SingleHeadBaseline(n_classes[h], proj_dim=args.proj_dim,
                               hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
        if layer_w is not None:
            m.layer_sum.raw_weights.data.copy_(layer_w)
        models[h]    = m
        optimisers[h]  = torch.optim.AdamW(m.parameters(), lr=args.lr,
                                            weight_decay=args.weight_decay)
        schedulers[h]  = torch.optim.lr_scheduler.LambdaLR(optimisers[h], lr_lambda)
    print("layer-sum: " + ("uniform init (1/n_layers)" if args.no_layer_warmstart
                           else f"warm-started from {args.layer_weights_ckpt}"))

    best = {h: {"macro_f1": -1.0, "epoch": -1, "state_dict": None, "metrics": None}
            for h in HEADS}
    for epoch in range(1, args.epochs + 1):
        for m in models.values():
            m.train()
        running = {h: 0.0 for h in HEADS}
        for audio, _z, pair_idx in train_loader:
            audio    = audio.to(device, non_blocking=True)
            pair_idx = pair_idx.to(device)
            for h in HEADS:
                logits = models[h](audio)
                loss   = F.cross_entropy(logits, lut[h][pair_idx])
                optimisers[h].zero_grad()
                loss.backward()
                optimisers[h].step()
                schedulers[h].step()
                running[h] += loss.item()
        val = evaluate(models, val_loader, lut, n_classes, device)
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              + "  ".join(f"{h}: loss {running[h]/steps_per_epoch:.3f} "
                          f"acc {val[h]['acc']:.3f} mF1 {val[h]['macro_f1']:.3f}"
                          for h in HEADS))
        for h in HEADS:
            if val[h]["macro_f1"] > best[h]["macro_f1"]:
                best[h] = {"macro_f1": val[h]["macro_f1"], "epoch": epoch,
                           "state_dict": {k: v.cpu() for k, v in models[h].state_dict().items()},
                           "metrics": val[h]}

    run_tag = f"baseline_sep_fold{args.fold}{args.run_suffix}"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    classes_by_head = {
        "pair": pair_classes, "model": model_classes,
        **{h: sorted(enc[h], key=enc[h].get)
           for h in ("arch_family", "acoustic_model", "vocoder", "speaker_type", "lang_family")},
    }
    torch.save({
        "run_tag": run_tag, "fold": args.fold, "seed": args.seed,
        "n_classes": n_classes, "classes_by_head": classes_by_head,
        "config": vars(args) | {"layer_weights_ckpt": str(args.layer_weights_ckpt),
                                "labels_path": str(args.labels_path),
                                "arch_labels_path": str(args.arch_labels_path)},
        "best_epoch": {h: best[h]["epoch"] for h in HEADS},
        "state_dict": {h: best[h]["state_dict"] for h in HEADS},
        "metrics": {h: best[h]["metrics"] for h in HEADS},
    }, CKPT_DIR / f"best_{run_tag}.pt")

    summary = {"run_tag": run_tag, "fold": args.fold, "seed": args.seed,
               "n_classes": n_classes,
               "best_epoch": {h: best[h]["epoch"] for h in HEADS},
               "metrics": {h: best[h]["metrics"] for h in HEADS}}
    out_json = RESULTS_DIR / f"baseline_{run_tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nBest epochs: " + ", ".join(f"{h}={best[h]['epoch']}" for h in HEADS))
    print("Macro-F1:    " + ", ".join(f"{h}={best[h]['macro_f1']:.3f}" for h in HEADS))
    print(f"Wrote {out_json}")

if __name__ == "__main__":
    main()
