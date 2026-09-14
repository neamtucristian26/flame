import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from baseline import SingleHeadBaseline

from tts_zs.attributes import build_pair_attributes
from tts_zs.config import AUDIO_ROOT, CKPT_DIR, DATA_DIR, LABELS_LANG, RESULTS_DIR
from tts_zs.data import (
    TTSAudioDataset,
    build_clip_index_lang,
    eval_collate,
    select_holdout_models,
)

AXES = ("arch_family", "acoustic_model", "vocoder", "speaker_type", "lang_family")
SKIP = {"unknown", "proprietary"}
SUBS = ("overall", "seen", "novel")
CKPT_TMPL = "best_baseline_sep_fold{f}_allheads{sfx}.pt"

def ckpt_name(fold: int, seed: int) -> str:
    sfx = "" if seed == 42 else f"_s{seed}"
    return CKPT_TMPL.format(f=fold, sfx=sfx)

@torch.no_grad()
def run_fold(
    fold,
    seed,
    *,
    all_pair_keys,
    all_attrs,
    clips_by_pair,
    arch_labels,
    device,
    num_workers,
    batch_size=256,
):
    ckpt_path = CKPT_DIR / ckpt_name(fold, seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    assert cfg["fold"] == fold, (cfg["fold"], fold)
    assert cfg["seed"] == seed, (cfg["seed"], seed)

    seen_idx, unseen_idx, _seen_models, _unseen_models = select_holdout_models(
        all_pair_keys,
        cfg["held_out_models"],
        cfg["seed"],
        fold_id=cfg["fold"],
        arch_labels=arch_labels,
    )

    for axis in AXES:
        recomputed = sorted({all_attrs[i][axis] for i in seen_idx})
        stored = ckpt["classes_by_head"][axis]
        if recomputed != stored:
            raise RuntimeError(
                f"class-vocab mismatch fold={fold} seed={seed} axis={axis}\n"
                f"  recomputed={recomputed}\n  stored={stored}"
            )

    seen_vals_by_axis = {axis: {all_attrs[i][axis] for i in seen_idx} for axis in AXES}

    models = {}
    for axis in AXES:
        m = SingleHeadBaseline(
            ckpt["n_classes"][axis], proj_dim=256, hidden_dim=256, dropout=0.0
        ).to(device)
        m.load_state_dict(ckpt["state_dict"][axis])
        m.eval()
        models[axis] = m

    unseen_set = set(int(i) for i in unseen_idx)
    samples = [
        (g, p)
        for g, key in enumerate(all_pair_keys)
        if g in unseen_set
        for p in clips_by_pair[key]
    ]
    ds = TTSAudioDataset(samples)
    ld = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eval_collate,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    acc = {(axis, sub): [0, 0] for axis in AXES for sub in SUBS}

    for audio, _t, idx in ld:
        audio = audio.to(device, non_blocking=True)
        idx_cpu = idx.tolist()
        preds = {
            axis: models[axis](audio).argmax(dim=1).cpu().tolist() for axis in AXES
        }
        for axis in AXES:
            classes = ckpt["classes_by_head"][axis]
            for bi, g in enumerate(idx_cpu):
                true_val = all_attrs[g][axis]
                if true_val in SKIP:
                    continue
                correct = int(classes[preds[axis][bi]] == true_val)
                is_novel = true_val not in seen_vals_by_axis[axis]
                for sub in ("overall", "novel" if is_novel else "seen"):
                    a2 = acc[(axis, sub)]
                    a2[0] += 1
                    a2[1] += correct

    return {
        "fold": fold,
        "n_unseen_clips": len(samples),
        "results": {
            axis: {
                sub: {
                    "n": acc[(axis, sub)][0],
                    "acc1": (
                        acc[(axis, sub)][1] / acc[(axis, sub)][0]
                        if acc[(axis, sub)][0]
                        else None
                    ),
                }
                for sub in SUBS
            }
            for axis in AXES
        },
    }

def pool(per_fold):
    out = {axis: {} for axis in AXES}
    for axis in AXES:
        for sub in SUBS:
            N = C = 0
            for f in per_fold:
                c = f["results"][axis][sub]
                if c["n"]:
                    N += c["n"]
                    C += round(c["acc1"] * c["n"])
            out[axis][sub] = {"n": N, "acc1": (C / N if N else None)}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--audio-root", type=Path, default=AUDIO_ROOT)
    ap.add_argument("--labels-path", type=Path, default=LABELS_LANG)
    ap.add_argument(
        "--arch-labels-path", type=Path, default=DATA_DIR / "arch_labels.json"
    )
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  seed: {args.seed}")

    with open(args.labels_path, encoding="utf-8") as f:
        all_pair_keys = sorted(json.load(f).keys())
    with open(args.arch_labels_path, encoding="utf-8") as f:
        arch_labels = json.load(f)
    all_attrs = build_pair_attributes(all_pair_keys)
    clips_by_pair = build_clip_index_lang(all_pair_keys, args.audio_root)

    per_fold = []
    for fold in args.folds:
        print(f"\n=== fold {fold} ===")
        r = run_fold(
            fold,
            args.seed,
            all_pair_keys=all_pair_keys,
            all_attrs=all_attrs,
            clips_by_pair=clips_by_pair,
            arch_labels=arch_labels,
            device=device,
            num_workers=args.num_workers,
        )
        per_fold.append(r)
        print(f"  unseen clips {r['n_unseen_clips']}")
        for axis in AXES:
            d = r["results"][axis]

            def s(x):
                return (
                    f"{x['acc1']:.3f}(n={x['n']})"
                    if x["acc1"] is not None
                    else f"--(n={x['n']})"
                )

            print(
                f"  {axis:16s} overall={s(d['overall'])}  seen={s(d['seen'])}  novel={s(d['novel'])}"
            )

    pooled = pool(per_fold)
    out = {
        "seed": args.seed,
        "axes": list(AXES),
        "per_fold": per_fold,
        "pooled": pooled,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"baseline_unseen_attrs{args.out_suffix}.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_json}")
    print(
        f"\n[pooled, seed {args.seed}] axis          overall(full)   novel(restricted)"
    )
    for axis in AXES:
        d = pooled[axis]

        def s(x):
            return (
                f"{x['acc1']:.3f}(n={x['n']})"
                if x["acc1"] is not None
                else f"--(n={x['n']})"
            )

        print(f"  {axis:16s} {s(d['overall']):>18} {s(d['novel']):>18}")

if __name__ == "__main__":
    main()
