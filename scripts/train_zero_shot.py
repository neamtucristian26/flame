import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_zs.attributes import build_pair_attributes
from tts_zs.config import (
    AUDIO_DIM,
    AUDIO_ROOT,
    CKPT_DIR,
    LABELS_LANG,
    LAYER_WEIGHTS_CKPT,
    LOGS_DIR,
    N_VARIANTS,
    RESULTS_DIR,
    SEED,
    SEP,
    TEXT_EMB_MODEL_DEFAULT,
    VAL_FRAC,
)
from tts_zs.data import (
    TTSAudioDataset,
    build_clip_index_lang,
    eval_collate,
    make_collate_fn,
    reindex_lc,
    select_holdout_models,
    split_train_val_lc,
)
from tts_zs.eval import (
    evaluate_lc,
    evaluate_lc_ensemble,
    evaluate_seen_within_epoch,
    project_centroids,
)
from tts_zs.losses import (
    cross_modal_supcon_loss,
    cross_modal_supcon_loss_bank,
    info_nce_loss,
    same_lang_margin_loss,
    supcon_loss,
    unified_supcon_loss,
    uniformity_loss,
)
from tts_zs.model import ContrastiveModel
from tts_zs.samplers import (
    DistinctModelBatchSampler,
    DistinctModelBatchSamplerLC,
    KPerClassBatchSampler,
    LangStratifiedKPerClassSampler,
    RotatingHoldoutKPerClassSampler,
)
from tts_zs.templates import AXES as ATTR_AXES
from tts_zs.templates import encode_value_templates
from tts_zs.text import (
    anchors_all_variants,
    anchors_from,
    encode_text_variants,
    encode_text_variants_augmented,
    pick_held_out,
)
from tts_zs.utils import encoder_short_name, set_seed

_TAU_MIN, _TAU_MAX = 0.005, 0.5
_LOG_TAU_MIN, _LOG_TAU_MAX = float(np.log(_TAU_MIN)), float(np.log(_TAU_MAX))

def local_pair_to_model(keys: list) -> dict:
    ms = sorted({k.split(SEP)[0] for k in keys})
    m2i = {m: i for i, m in enumerate(ms)}
    return {i: m2i[k.split(SEP)[0]] for i, k in enumerate(keys)}

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.labels_path, encoding="utf-8") as f:
        lc_labels = json.load(f)
    all_pair_keys = sorted(lc_labels.keys())
    n_pairs = len(all_pair_keys)
    print(f"(model, lang) pairs: {n_pairs}  |  labels: {args.labels_path}")

    all_text_variants = encode_text_variants(
        lc_labels,
        all_pair_keys,
        encoder=args.text_encoder,
        labels_path=args.labels_path,
    )
    text_dim = int(all_text_variants.shape[-1])
    print(f"Text encoder: {args.text_encoder}  |  dim={text_dim}")

    aug_variants_all = None
    if args.augmented_text_path is not None:
        n_aug = np.load(args.augmented_text_path).shape[1]
        aug_variants_all = encode_text_variants_augmented(
            {}, all_pair_keys, n_aug, cache_path=args.augmented_text_path
        )
        print(
            f"[aug-text] pool extended: {N_VARIANTS} orig + {n_aug} aug = {N_VARIANTS + n_aug} total"
        )

    attr_tmpl_all = None
    if args.attribute_template_path is not None:
        attr_tmpl_all = np.load(args.attribute_template_path)
        print(
            f"[attr-tmpl] loaded {attr_tmpl_all.shape[1]} attribute templates per pair"
        )

    if args.held_out_models <= 0:
        raise ValueError("--held-out-models must be > 0")
    arch_labels = None
    if args.arch_labels_path is not None:
        with open(args.arch_labels_path, encoding="utf-8") as f:
            arch_labels = json.load(f)
    seen_pair_idx, unseen_pair_idx, seen_model_names, unseen_model_names = (
        select_holdout_models(
            all_pair_keys,
            args.held_out_models,
            args.seed,
            fold_id=args.fold,
            arch_labels=arch_labels,
        )
    )
    n_seen_pairs = len(seen_pair_idx)
    n_unseen_pairs = len(unseen_pair_idx)
    n_seen_models = len(seen_model_names)
    n_unseen_models = len(unseen_model_names)
    print(
        f"[holdout] Seen {n_seen_models} models ({n_seen_pairs} pairs) | "
        f"Unseen {n_unseen_models} models ({n_unseen_pairs} pairs)"
    )
    print("Held-out model names:")
    for n in unseen_model_names:
        print(f"  - {n}")

    seen_pair_keys = [all_pair_keys[i] for i in seen_pair_idx]
    unseen_pair_keys = [all_pair_keys[i] for i in unseen_pair_idx]

    all_held_out_idx = pick_held_out(n_pairs, seed=args.seed)
    seen_held_out_idx = all_held_out_idx[seen_pair_idx]
    unseen_held_out_idx = all_held_out_idx[unseen_pair_idx]
    text_variants_seen = all_text_variants[seen_pair_idx]
    text_eval_anchors_seen = anchors_from(text_variants_seen, seen_held_out_idx).to(
        device
    )

    print(
        f"Held-out variant dist (seen):   "
        f"{np.bincount(seen_held_out_idx, minlength=N_VARIANTS).tolist()}"
    )
    print(
        f"Held-out variant dist (unseen): "
        f"{np.bincount(unseen_held_out_idx, minlength=N_VARIANTS).tolist()}"
    )

    clips_by_pair = build_clip_index_lang(all_pair_keys, args.audio_root)
    n_clips = sum(len(v) for v in clips_by_pair.values())
    print(f"Clips: {n_clips} across {n_pairs} pairs")
    all_train_g, all_val_g = split_train_val_lc(clips_by_pair, VAL_FRAC, args.seed)

    global_to_seen = {int(g): s for s, g in enumerate(seen_pair_idx)}
    global_to_unseen = {int(g): u for u, g in enumerate(unseen_pair_idx)}
    train_samples = reindex_lc(all_train_g, global_to_seen)
    val_seen_local = reindex_lc(all_val_g, global_to_seen)

    unseen_all_global = [
        (global_idx, p)
        for global_idx, key in enumerate(sorted(clips_by_pair))
        if global_idx in global_to_unseen
        for p in clips_by_pair[key]
    ]
    val_unseen_local = [(global_to_unseen[g], p) for g, p in unseen_all_global]
    val_unseen_global = unseen_all_global
    val_seen_global = [(idx, p) for idx, p in all_val_g if idx in global_to_seen]
    val_all_global = val_seen_global + unseen_all_global
    print(
        f"Train clips: {len(train_samples)}  |  "
        f"Val seen: {len(val_seen_local)}  |  Val unseen: {len(val_unseen_local)}"
    )

    seen_ptm = local_pair_to_model(seen_pair_keys)
    unseen_ptm = local_pair_to_model(unseen_pair_keys)
    all_ptm = local_pair_to_model(all_pair_keys)

    seen_pair_lang_map = {
        i: seen_pair_keys[i].split(SEP)[1] for i in range(n_seen_pairs)
    }
    lang_vocab = sorted(set(seen_pair_lang_map.values()))
    lang_to_idx = {l: i for i, l in enumerate(lang_vocab)}
    seen_pair_lang_idx = [
        lang_to_idx[seen_pair_lang_map[i]] for i in range(n_seen_pairs)
    ]
    print(f"[lang] {len(lang_vocab)} distinct languages in seen pairs")

    _KPERCLASS_LOSS_MODES = {
        "cross_modal_supcon",
        "unified_supcon",
        "text_supcon",
        "all_supcon",
    }
    _use_kperclass = args.k_per_class > 1 and (
        args.supcon_weight > 0 or args.loss_mode in _KPERCLASS_LOSS_MODES
    )
    if _use_kperclass:
        effective_batch_size = (args.batch_size // args.k_per_class) * args.k_per_class
        if args.rotation_holdout > 0:
            pair_to_model_name = {
                i: seen_pair_keys[i].split(SEP)[0] for i in range(n_seen_pairs)
            }
            train_sampler = RotatingHoldoutKPerClassSampler(
                train_samples,
                pair_to_model=pair_to_model_name,
                k=args.k_per_class,
                batch_size=effective_batch_size,
                n_holdout=args.rotation_holdout,
                seed=args.seed,
            )
            sampler_label = (
                f"rotating-holdout(k={args.k_per_class},R={args.rotation_holdout})"
            )
        elif args.lang_stratified:
            train_sampler = LangStratifiedKPerClassSampler(
                train_samples,
                pair_to_lang=seen_pair_lang_map,
                k=args.k_per_class,
                batch_size=effective_batch_size,
                seed=args.seed,
            )
            sampler_label = f"lang-stratified(k={args.k_per_class})"
        else:
            train_sampler = KPerClassBatchSampler(
                train_samples,
                k=args.k_per_class,
                batch_size=effective_batch_size,
                seed=args.seed,
            )
            sampler_label = f"k-per-class(k={args.k_per_class})"
    elif args.sampler == "model":
        effective_batch_size = min(args.batch_size, n_seen_models)
        train_sampler = DistinctModelBatchSamplerLC(
            train_samples,
            pair_to_model=seen_ptm,
            batch_size=effective_batch_size,
            seed=args.seed,
        )
        sampler_label = "model-distinct"
    else:
        effective_batch_size = min(args.batch_size, n_seen_pairs)
        train_sampler = DistinctModelBatchSampler(
            train_samples, batch_size=effective_batch_size, seed=args.seed
        )
        sampler_label = "pair-distinct"
    print(f"[sampler] {sampler_label}  batch_size={effective_batch_size}")

    train_ds = TTSAudioDataset(train_samples)
    val_seen_ds = TTSAudioDataset(val_seen_local)
    aug_variants_seen = (
        aug_variants_all[seen_pair_idx] if aug_variants_all is not None else None
    )
    n_attr_tmpl = 0
    if attr_tmpl_all is not None:
        attr_tmpl_seen = attr_tmpl_all[seen_pair_idx]
        n_attr_tmpl = attr_tmpl_seen.shape[1]
        aug_variants_seen = (
            np.concatenate([aug_variants_seen, attr_tmpl_seen], axis=1)
            if aug_variants_seen is not None
            else attr_tmpl_seen
        )
        print(
            f"[attr-tmpl] p_tmpl={args.attr_tmpl_prob} "
            f"mask_text_supcon={args.mask_tmpl_from_text_supcon}"
        )
    collate_fn = make_collate_fn(
        text_variants_seen,
        seen_held_out_idx,
        aug_variants=aug_variants_seen,
        n_attr_tmpl=n_attr_tmpl,
        attr_tmpl_prob=args.attr_tmpl_prob,
    )

    mp_pool_t = mp_desc_idx = mp_tmpl_idx = None
    if args.all_desc_positives:
        mp_pool_np = (
            np.concatenate([text_variants_seen, aug_variants_seen], axis=1)
            if aug_variants_seen is not None
            else text_variants_seen
        )
        mp_pool_t = torch.from_numpy(mp_pool_np).to(device)
        n_pool_mp = mp_pool_np.shape[1]
        n_desc_mp = n_pool_mp - n_attr_tmpl
        mp_desc_idx = [
            torch.tensor(
                [v for v in range(n_desc_mp) if v != int(h)],
                dtype=torch.long,
                device=device,
            )
            for h in seen_held_out_idx
        ]
        mp_tmpl_idx = torch.arange(
            n_desc_mp, n_pool_mp, dtype=torch.long, device=device
        )
        print(
            f"[multi-pos] all-desc positives ON: {n_desc_mp} desc/pair "
            f"+ {n_attr_tmpl} tmpl (p={args.attr_tmpl_prob})"
        )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader_seen = DataLoader(
        val_seen_ds,
        batch_size=256,
        shuffle=False,
        collate_fn=eval_collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = ContrastiveModel(
        proj_dim=args.proj_dim,
        hidden_dim=args.proj_dim,
        dropout=0.0,
        text_dim=text_dim,
        proj_style="linear1",
        text_proj_style="linear1",
        scale_max=args.scale_max,
        normalize=not args.no_l2_norm,
    ).to(device)
    print(
        f"[proj] audio=linear1({AUDIO_DIM}->{args.proj_dim})  "
        f"text=linear1({text_dim}->{args.proj_dim})"
    )

    if args.no_layer_warmstart:
        print("[layer-sum] uniform init (1/n_layers); warm-start skipped")
    else:
        ckpt = torch.load(
            args.layer_weights_ckpt, map_location="cpu", weights_only=False
        )
        model.layer_sum.raw_weights.data.copy_(ckpt["model"]["combiner.weights"])
        print(f"[layer-sum] Loaded from {args.layer_weights_ckpt}")

    log_tau = None
    if args.learnable_supcon_temps:
        init_cm = (
            args.cross_modal_temperature
            if args.cross_modal_temperature is not None
            else args.supcon_temperature
        )
        init_t = (
            args.text_supcon_temperature
            if args.text_supcon_temperature is not None
            else args.supcon_temperature
        )
        init_a = args.supcon_temperature
        log_tau = torch.nn.Parameter(
            torch.log(
                torch.tensor(
                    [init_cm, init_a, init_t], dtype=torch.float32, device=device
                )
            )
        )
        print(
            f"[learn-temps] learnable τ init: τ_cm={init_cm} τ_a={init_a} τ_t={init_t} "
            f"(clamp [{_TAU_MIN}, {_TAU_MAX}])"
        )

    if log_tau is not None:
        optimiser = torch.optim.AdamW(
            [
                {"params": list(model.parameters())},
                {"params": [log_tau], "weight_decay": 0.0},
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    total_steps = max(1, args.epochs * len(train_sampler))
    warmup_steps = max(1, len(train_sampler))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    for d in (CKPT_DIR, RESULTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    enc_short = encoder_short_name(args.text_encoder)
    fold_suffix = f"_fold{args.fold}" if args.fold >= 0 else ""
    run_tag = (
        f"lc_zs_{enc_short}__ho{args.held_out_models}{fold_suffix}{args.run_suffix}"
    )
    csv_path = RESULTS_DIR / f"metrics_{run_tag}.csv"
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                "epoch",
                "train_loss",
                "logit_scale",
                "R@1_pair",
                "R@5_pair",
                "MRR_pair",
                "n_val",
            ]
        )

    pair_to_model_t = torch.tensor(
        [seen_ptm[i] for i in range(n_seen_pairs)], dtype=torch.long, device=device
    )
    pair_to_lang_t = torch.tensor(seen_pair_lang_idx, dtype=torch.long, device=device)

    template_cfg = {
        "vocoder": (args.vocoder_template_weight, args.vocoder_template_prefix),
        "acoustic_model": (
            args.acoustic_template_weight,
            args.acoustic_template_prefix,
        ),
    }
    template_galleries = {}
    if any(w > 0 for w, _ in template_cfg.values()):
        all_attrs = build_pair_attributes(all_pair_keys)
        for axis in ATTR_AXES:
            w, prefix = template_cfg[axis]
            if w <= 0:
                continue
            _SKIP = {"unknown", "proprietary"}
            values = sorted(
                {
                    all_attrs[i][axis]
                    for i in range(n_pairs)
                    if all_attrs[i][axis] not in _SKIP
                }
            )
            val2col = {v: c for c, v in enumerate(values)}
            T_emb = encode_value_templates(
                values, axis, args.text_encoder, device, prefix=prefix
            )
            value_col_t = torch.tensor(
                [val2col.get(all_attrs[g][axis], -1) for g in seen_pair_idx],
                dtype=torch.long,
                device=device,
            )
            template_galleries[axis] = (T_emb, value_col_t)
            n_skip = sum(1 for g in seen_pair_idx if all_attrs[g][axis] in _SKIP)
            print(
                f"[template] {axis}: {len(values)} values, prefix={prefix}:, "
                f"weight={w} (skipping {n_skip} unknown/proprietary pairs)"
            )

    def _save_ckpt(path, epoch, metrics):
        cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        cfg.update(
            {
                "text_dim": text_dim,
                "n_pairs": n_pairs,
                "seen_pair_keys": seen_pair_keys,
                "unseen_pair_keys": unseen_pair_keys,
                "seen_model_names": seen_model_names,
                "unseen_model_names": unseen_model_names,
            }
        )
        torch.save(
            {
                "layer_sum": model.layer_sum.state_dict(),
                "audio_proj": model.audio_proj.state_dict(),
                "text_proj": model.text_proj.state_dict(),
                "logit_scale": model.logit_scale.detach().cpu(),
                "layer_weights": model.layer_sum.weights().detach().cpu().tolist(),
                "config": cfg,
                "epoch": epoch,
                "metrics": metrics,
            },
            path,
        )

    best_mrr = -1.0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        losses = []
        tmpl_losses = {axis: [] for axis in template_galleries}
        for audio, text, pair_idx, is_tmpl in train_loader:
            audio = audio.to(device, non_blocking=True)
            text = text.to(device, non_blocking=True)
            pair_idx = pair_idx.to(device)
            is_tmpl = is_tmpl.to(device)
            a, t, scale = model(audio, text)
            if args.no_l2_norm:
                a = F.normalize(a, dim=-1)
                t = F.normalize(t, dim=-1)
            mode = args.loss_mode
            K = args.k_per_class if args.k_per_class > 1 else 1
            tau_text = (
                args.text_supcon_temperature
                if args.text_supcon_temperature is not None
                else args.supcon_temperature
            )
            tau_cm = (
                args.cross_modal_temperature
                if args.cross_modal_temperature is not None
                else args.supcon_temperature
            )
            tau_a_eff = args.supcon_temperature
            tau_cm_eff, tau_text_eff = tau_cm, tau_text
            if log_tau is not None:
                lt = log_tau.clamp(_LOG_TAU_MIN, _LOG_TAU_MAX)
                tau_cm_eff, tau_a_eff, tau_text_eff = (
                    lt[0].exp(),
                    lt[1].exp(),
                    lt[2].exp(),
                )
            if mode == "default":
                if args.supcon_weight > 0 and K > 1:
                    nce = info_nce_loss(a[::K], t[::K], scale)
                    sc = supcon_loss(a, pair_idx, temperature=args.supcon_temperature)
                    loss = nce + args.supcon_weight * sc
                else:
                    loss = info_nce_loss(a, t, scale)
            elif mode == "cross_modal_supcon":
                cm = cross_modal_supcon_loss(a, t, pair_idx, temperature=tau_cm)
                loss = cm
                if args.supcon_weight > 0 and K > 1:
                    sc = supcon_loss(a, pair_idx, temperature=args.supcon_temperature)
                    loss = loss + args.supcon_weight * sc
            elif mode == "unified_supcon":
                loss = unified_supcon_loss(
                    a, t, pair_idx, temperature=args.supcon_temperature
                )
            elif mode == "text_supcon":
                if args.supcon_weight > 0 and K > 1:
                    nce = info_nce_loss(a[::K], t[::K], scale)
                    sc = supcon_loss(a, pair_idx, temperature=args.supcon_temperature)
                    sc_text = supcon_loss(t, pair_idx, temperature=tau_text)
                    loss = (
                        nce
                        + args.supcon_weight * sc
                        + args.text_supcon_weight * sc_text
                    )
                else:
                    loss = info_nce_loss(a, t, scale)
            elif mode == "all_supcon":
                if args.all_desc_positives:
                    bank_rows, bank_lbl, bank_istmpl = [], [], []
                    for p in pair_idx.unique():
                        pi = int(p)
                        di = mp_desc_idx[pi]
                        bank_rows.append(mp_pool_t[pi, di])
                        bank_lbl.append(p.repeat(di.numel()))
                        bank_istmpl.append(
                            torch.zeros(di.numel(), dtype=torch.bool, device=device)
                        )
                        if (
                            n_attr_tmpl > 0
                            and args.attr_tmpl_prob is not None
                            and torch.rand(1).item() < args.attr_tmpl_prob
                        ):
                            ti = mp_tmpl_idx[
                                torch.randint(0, n_attr_tmpl, (1,), device=device)
                            ]
                            bank_rows.append(mp_pool_t[pi, ti])
                            bank_lbl.append(p.repeat(1))
                            bank_istmpl.append(
                                torch.ones(1, dtype=torch.bool, device=device)
                            )
                    bank_raw = torch.cat(bank_rows, dim=0)
                    bank_lbl = torch.cat(bank_lbl, dim=0)
                    bank_istmpl = torch.cat(bank_istmpl, dim=0)
                    bank_t = model.text_proj(bank_raw)
                    if args.no_l2_norm:
                        bank_t = F.normalize(bank_t, dim=-1)
                    loss = cross_modal_supcon_loss_bank(
                        a, pair_idx, bank_t, bank_lbl, temperature=tau_cm
                    )
                    if args.supcon_weight > 0 and K > 1:
                        loss = loss + args.supcon_weight * supcon_loss(
                            a, pair_idx, temperature=args.supcon_temperature
                        )
                    if args.text_supcon_weight > 0:
                        if args.mask_tmpl_from_text_supcon:
                            keep = ~bank_istmpl
                            t_txt, pidx_txt = bank_t[keep], bank_lbl[keep]
                        else:
                            t_txt, pidx_txt = bank_t, bank_lbl
                        loss = loss + args.text_supcon_weight * supcon_loss(
                            t_txt, pidx_txt, temperature=tau_text
                        )
                else:
                    if args.cross_modal_loss == "infonce":
                        loss = info_nce_loss(a[::K], t[::K], scale)
                    else:
                        loss = cross_modal_supcon_loss(
                            a, t, pair_idx, temperature=tau_cm_eff
                        )
                    if args.supcon_weight > 0 and K > 1:
                        loss = loss + args.supcon_weight * supcon_loss(
                            a, pair_idx, temperature=tau_a_eff
                        )
                    if args.text_supcon_weight > 0 and K > 1:
                        if args.mask_tmpl_from_text_supcon:
                            keep = ~is_tmpl
                            t_txt, pidx_txt = t[keep], pair_idx[keep]
                        else:
                            t_txt, pidx_txt = t, pair_idx
                        loss = loss + args.text_supcon_weight * supcon_loss(
                            t_txt, pidx_txt, temperature=tau_text_eff
                        )
            else:
                raise ValueError(f"unknown --loss-mode {mode!r}")
            if (
                args.model_supcon_weight > 0
                and args.supcon_weight > 0
                and args.k_per_class > 1
            ):
                model_idx = pair_to_model_t[pair_idx]
                sc_model = supcon_loss(
                    a, model_idx, temperature=args.model_supcon_temperature
                )
                loss = loss + args.model_supcon_weight * sc_model
            for axis, (T_emb, value_col_t) in template_galleries.items():
                w = template_cfg[axis][0]
                T = model.text_proj(T_emb)
                labels = value_col_t[pair_idx]
                valid = labels >= 0
                if not valid.any():
                    continue
                logit_scale = (
                    1.0 / args.attr_template_temperature
                    if args.attr_template_temperature is not None
                    else scale
                )
                logits = logit_scale * (a @ T.T)
                term = F.cross_entropy(logits[valid], labels[valid])
                if args.attr_template_symmetric:
                    V = T.shape[0]
                    a_v = a[valid]
                    labels_v = labels[valid]
                    sim_va = logit_scale * (T @ a_v.T)
                    pos = (
                        labels_v.unsqueeze(0)
                        == torch.arange(V, device=device).unsqueeze(1)
                    ).float()
                    n_pos = pos.sum(dim=1)
                    has = n_pos > 0
                    logp = sim_va - torch.logsumexp(sim_va, dim=1, keepdim=True)
                    loss_va = -(pos * logp).sum(dim=1) / n_pos.clamp(min=1)
                    term = 0.5 * (term + loss_va[has].mean())
                loss = loss + w * term
                tmpl_losses[axis].append(term.item())
            if args.lang_margin > 0:
                lang_ids = pair_to_lang_t[pair_idx]
                loss = loss + args.lang_margin * same_lang_margin_loss(
                    a, lang_ids, pair_idx, margin=0.3
                )
            if args.uniform_weight > 0:
                K_u = (
                    args.k_per_class
                    if (args.supcon_weight > 0 and args.k_per_class > 1)
                    else 1
                )
                loss = loss + args.uniform_weight * uniformity_loss(t[::K_u])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            scheduler.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        centroids_seen_eval = F.normalize(
            model.text_proj(text_eval_anchors_seen), dim=-1
        )
        metrics = evaluate_seen_within_epoch(
            model, val_loader_seen, centroids_seen_eval, device
        )
        ls_val = float(model.logit_scale.detach().exp().item())
        layer_w = model.layer_sum.weights().detach().cpu().numpy()
        top_layer = int(layer_w.argmax())

        tmpl_str = "".join(
            f" {axis[:3]}CE {np.mean(v):.3f}" for axis, v in tmpl_losses.items() if v
        )
        temp_str = ""
        if log_tau is not None:
            lt = log_tau.detach().clamp(_LOG_TAU_MIN, _LOG_TAU_MAX).exp().cpu().numpy()
            temp_str = f" | τ_cm={lt[0]:.4f} τ_a={lt[1]:.4f} τ_t={lt[2]:.4f}"
        print(
            f"Epoch {epoch:3d}/{args.epochs} | loss {train_loss:.4f} | "
            f"scale {ls_val:5.1f} | [seen-pair] R@1 {metrics['R@1']:.3f} "
            f"R@5 {metrics['R@5']:.3f} MRR {metrics['MRR']:.3f} | "
            f"top layer {top_layer + 1} (w={layer_w[top_layer]:.3f}){tmpl_str}{temp_str}"
        )
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{ls_val:.4f}",
                    f"{metrics['R@1']:.6f}",
                    f"{metrics['R@5']:.6f}",
                    f"{metrics['MRR']:.6f}",
                    metrics["n_val"],
                ]
            )

        if metrics["MRR"] > best_mrr:
            best_mrr = metrics["MRR"]
            _save_ckpt(ckpt_path, epoch, metrics)

        if args.save_every > 0 and (
            epoch % args.save_every == 0 or epoch == args.epochs
        ):
            ep_path = CKPT_DIR / f"epoch_{run_tag}_ep{epoch}.pt"
            _save_ckpt(ep_path, epoch, metrics)

    print(f"\nBest seen-pair MRR: {best_mrr:.4f}  (ckpt: {ckpt_path})")

    if args.skip_final_eval:
        print("[skip-final-eval] checkpoint saved; built-in zero-shot eval skipped.")
        return

    print("\n=== Zero-shot evaluation ===")
    best = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.layer_sum.load_state_dict(best["layer_sum"])
    model.audio_proj.load_state_dict(best["audio_proj"])
    model.text_proj.load_state_dict(best["text_proj"])
    model.to(device).eval()

    def make_centroids(text_variants, held_out_idx):
        return project_centroids(
            model, anchors_from(text_variants, held_out_idx), device
        )

    tv_unseen = all_text_variants[unseen_pair_idx]
    c_seen = make_centroids(text_variants_seen, seen_held_out_idx)
    c_unseen = make_centroids(tv_unseen, unseen_held_out_idx)
    c_all = make_centroids(all_text_variants, all_held_out_idx)

    tv_seen_all = anchors_all_variants(text_variants_seen)
    tv_unseen_all = anchors_all_variants(tv_unseen)
    tv_all_all = anchors_all_variants(all_text_variants)

    def make_loader(samples):
        return DataLoader(
            TTSAudioDataset(samples),
            batch_size=256,
            shuffle=False,
            collate_fn=eval_collate,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    ld_seen = make_loader(val_seen_local)
    ld_unseen_local = make_loader(val_unseen_local)
    ld_unseen_global = make_loader(val_unseen_global)
    ld_all = make_loader(val_all_global)

    def run_eval(ensemble: bool) -> dict:
        if ensemble:
            return {
                "seen_sanity": evaluate_lc_ensemble(
                    model, ld_seen, tv_seen_all, device, seen_ptm, seen_ptm
                ),
                "unseen_restricted": evaluate_lc_ensemble(
                    model,
                    ld_unseen_local,
                    tv_unseen_all,
                    device,
                    unseen_ptm,
                    unseen_ptm,
                ),
                "unseen_full": evaluate_lc_ensemble(
                    model, ld_unseen_global, tv_all_all, device, all_ptm, all_ptm
                ),
                "full": evaluate_lc_ensemble(
                    model, ld_all, tv_all_all, device, all_ptm, all_ptm
                ),
            }
        return {
            "seen_sanity": evaluate_lc(
                model, ld_seen, c_seen, device, seen_ptm, seen_ptm
            ),
            "unseen_restricted": evaluate_lc(
                model, ld_unseen_local, c_unseen, device, unseen_ptm, unseen_ptm
            ),
            "unseen_full": evaluate_lc(
                model, ld_unseen_global, c_all, device, all_ptm, all_ptm
            ),
            "full": evaluate_lc(model, ld_all, c_all, device, all_ptm, all_ptm),
        }

    def fmt(d):
        if isinstance(d, float):
            return round(d, 4)
        if isinstance(d, dict):
            return {k: fmt(v) for k, v in d.items()}
        return d

    def print_block(label, res):
        print(f"\n--- {label} ---")
        for k in ("seen_sanity", "unseen_restricted", "unseen_full", "full"):
            p = res[k]["pair"]
            m = res[k]["model"]
            print(
                f"  {k:24s}  "
                f"pair  R@1 {p['R@1']:.4f}  R@5 {p['R@5']:.4f}  MRR {p['MRR']:.4f}  "
                f"|  model R@1 {m['R@1']:.4f}  R@5 {m['R@5']:.4f}  MRR {m['MRR']:.4f}"
                f"  (n={p['n_val']})"
            )

    print("\n[single held-out variant]")
    res_single = run_eval(ensemble=False)
    print("\n[ensemble: mean metrics over 5 variants]")
    res_ens = run_eval(ensemble=True)
    print_block("Single held-out", res_single)
    print_block("Ensemble (mean 5)", res_ens)

    summary = {
        "run_tag": run_tag,
        "n_seen_models": n_seen_models,
        "n_unseen_models": n_unseen_models,
        "n_seen_pairs": n_seen_pairs,
        "n_unseen_pairs": n_unseen_pairs,
        "unseen_model_names": unseen_model_names,
        "unseen_pair_keys": unseen_pair_keys,
        "best_epoch": int(best["epoch"]),
        "single_variant": fmt(res_single),
        "ensemble_5": fmt(res_ens),
    }
    out_path = RESULTS_DIR / f"zeroshot_{run_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")
    best["metrics_zs"] = summary
    torch.save(best, ckpt_path)
    print(f"Checkpoint updated: {ckpt_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--audio-root", type=Path, default=AUDIO_ROOT)
    p.add_argument("--labels-path", type=Path, default=LABELS_LANG)
    p.add_argument(
        "--augmented-text-path",
        type=Path,
        default=None,
        help="Path to pre-generated augmented text cache .npy "
        "(shape n_keys×M×D). If set, training pool = 5 orig + M aug.",
    )
    p.add_argument(
        "--attribute-template-path",
        type=Path,
        default=None,
        help="Path to attribute template cache .npy (shape n_keys×N_TMPL×D). "
        "Templates are appended to the training pool after aug variants.",
    )
    p.add_argument(
        "--attr-tmpl-prob",
        type=float,
        default=None,
        help="Per-clip probability of sampling an attribute template instead "
        "of a description. None → uniform over the whole pool (legacy).",
    )
    p.add_argument(
        "--mask-tmpl-from-text-supcon",
        action="store_true",
        help="Exclude template-sampled clips from the text-text SupCon term "
        "(removes the destructive shared-template collision there).",
    )
    p.add_argument(
        "--all-desc-positives",
        action="store_true",
        help="Multi-positive mode: each batch builds a text bank of ALL "
        "description variants for its pairs (templates added per pair "
        "with prob --attr-tmpl-prob) instead of one sampled text per clip.",
    )
    p.add_argument(
        "--cross-modal-loss",
        type=str,
        default="supcon",
        choices=["supcon", "infonce"],
        help="all_supcon term 1: 'supcon' (cross-modal SupCon) or 'infonce' "
        "(CLIP InfoNCE on one clip/pair with learnable logit-scale).",
    )
    p.add_argument(
        "--learnable-supcon-temps",
        action="store_true",
        help="Make τ_cm, τ_a, τ_t learnable (clamped log-params) instead of "
        "fixed. Ablation; weights λ, γ stay fixed.",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="If >0, also save a periodic checkpoint every N epochs (and at the "
        "final epoch) as epoch_<run_tag>_ep<E>.pt — for unseen-vs-epoch study.",
    )
    p.add_argument("--text-encoder", type=str, default=TEXT_EMB_MODEL_DEFAULT)
    p.add_argument("--layer-weights-ckpt", type=Path, default=LAYER_WEIGHTS_CKPT)
    p.add_argument(
        "--no-layer-warmstart",
        action="store_true",
        help="skip the LayerCombiner warm-start; use uniform 1/n_layers init",
    )
    p.add_argument(
        "--held-out-models",
        type=int,
        required=True,
        help="Number of model names to hold out entirely.",
    )
    p.add_argument(
        "--run-suffix",
        type=str,
        default="",
        help="Suffix appended to run_tag (e.g. '_v2').",
    )
    p.add_argument(
        "--sampler",
        type=str,
        default="pair",
        choices=["pair", "model"],
        help="pair: ≤1 clip per (model,lang) per batch. "
        "model: ≤1 clip per model per batch.",
    )
    p.add_argument(
        "--rotation-holdout",
        type=int,
        default=0,
        help="Per-epoch virtual-unseen rotation count (0=off).",
    )
    p.add_argument(
        "--supcon-weight",
        type=float,
        default=0.298,
        help="Weight λ for the audio-audio SupCon term.",
    )
    p.add_argument(
        "--k-per-class",
        type=int,
        default=32,
        help="Clips per pair per batch when SupCon is active.",
    )
    p.add_argument("--supcon-temperature", type=float, default=0.045)
    p.add_argument(
        "--loss-mode",
        type=str,
        default="all_supcon",
        choices=[
            "default",
            "cross_modal_supcon",
            "unified_supcon",
            "text_supcon",
            "all_supcon",
        ],
        help="Loss formulation. default: InfoNCE(class-reps) + λ·SupCon(audio). "
        "cross_modal_supcon: cross-modal SupCon on full batch + λ·SupCon(audio). "
        "unified_supcon: single SupCon over (audio ∪ text). "
        "text_supcon: default + γ·SupCon(text) (ablation). "
        "all_supcon: cross-modal SupCon + λ·SupCon(audio) + γ·SupCon(text).",
    )
    p.add_argument(
        "--text-supcon-weight",
        type=float,
        default=0.401,
        help="Weight γ for the text-text SupCon term.",
    )
    p.add_argument(
        "--text-supcon-temperature",
        type=float,
        default=0.070,
        help="Temperature τ_t for the text-text SupCon term. If unset, "
        "falls back to --supcon-temperature (shared τ).",
    )
    p.add_argument(
        "--cross-modal-temperature",
        type=float,
        default=0.020,
        help="Temperature τ_cm for the cross-modal SupCon term. If unset, "
        "falls back to --supcon-temperature (shared τ_a).",
    )
    p.add_argument(
        "--uniform-weight",
        type=float,
        default=0.0,
        help="Weight μ for uniformity regularisation.",
    )
    p.add_argument("--scale-max", type=float, default=30.0)
    p.add_argument(
        "--no-l2-norm", action="store_true", help="Disable L2 norm in projection heads."
    )
    p.add_argument("--model-supcon-weight", type=float, default=0.0)
    p.add_argument("--model-supcon-temperature", type=float, default=0.15)
    p.add_argument(
        "--lang-stratified",
        action="store_true",
        help="≥2 models per language per batch (hard same-lang negatives).",
    )
    p.add_argument(
        "--lang-margin",
        type=float,
        default=0.0,
        help="Weight for same-lang different-pair cosine margin loss.",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="Fold index for stratified k-fold CV (0-based). "
        "-1 = random seed-based holdout (default).",
    )
    p.add_argument(
        "--arch-labels-path",
        type=Path,
        default=None,
        help="JSON mapping model name -> architecture family. "
        "Required for stratified-by-family k-fold (--fold >= 0).",
    )
    p.add_argument(
        "--vocoder-template-weight",
        type=float,
        default=0.0,
        help="Weight of the vocoder template-positive CE term (0 = off).",
    )
    p.add_argument(
        "--acoustic-template-weight",
        type=float,
        default=0.0,
        help="Weight of the acoustic-model template-positive CE term (0 = off).",
    )
    p.add_argument(
        "--vocoder-template-prefix",
        choices=("query", "passage"),
        default="query",
        help="E5 role prefix for vocoder templates.",
    )
    p.add_argument(
        "--acoustic-template-prefix",
        choices=("query", "passage"),
        default="passage",
        help="E5 role prefix for acoustic templates.",
    )
    p.add_argument(
        "--attr-template-temperature",
        type=float,
        default=None,
        help="Fixed temperature for the template CE; default uses the "
        "model's learned logit scale (like InfoNCE).",
    )
    p.add_argument(
        "--attr-template-symmetric",
        action="store_true",
        help="Add the symmetric value->audio direction (ablation; more "
        "collapse pressure on system identity).",
    )
    p.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Save the best checkpoint then stop, skipping the built-in "
        "zero-shot eval (use external eval scripts instead; much "
        "faster for sweeps).",
    )
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())
