import torch
import torch.nn.functional as F

from tts_zs.model import ContrastiveModel

@torch.no_grad()
def project_centroids(
    model: ContrastiveModel, anchors: torch.Tensor, device
) -> torch.Tensor:
    if anchors.dim() == 2:
        return F.normalize(model.text_proj(anchors.to(device)), dim=-1)
    n, K, D = anchors.shape
    flat = anchors.view(n * K, D).to(device)
    proj = F.normalize(model.text_proj(flat), dim=-1)
    return F.normalize(proj.view(n, K, -1).mean(dim=1), dim=-1)

def _metrics(ranks: torch.Tensor) -> dict:
    return {
        "R@1": (ranks < 1).float().mean().item(),
        "R@5": (ranks < 5).float().mean().item(),
        "R@10": (ranks < 10).float().mean().item(),
        "MRR": (1.0 / (ranks.float() + 1.0)).mean().item(),
        "n_val": int(len(ranks)),
    }

@torch.no_grad()
def evaluate_closed_set(
    model: ContrastiveModel,
    loader,
    text_anchors: torch.Tensor,
    device,
) -> dict:
    model.eval()
    proj_centroids = F.normalize(model.text_proj(text_anchors.to(device)), dim=-1)
    ranks = []
    for audio, _text, model_idx in loader:
        audio = audio.to(device, non_blocking=True)
        model_idx = model_idx.to(device)
        a = model.audio_proj(model.layer_sum(audio))
        sim = a @ proj_centroids.T
        order = sim.argsort(dim=1, descending=True)
        rank_of_gt = (order == model_idx.unsqueeze(1)).nonzero()[:, 1]
        ranks.append(rank_of_gt.cpu())
    return _metrics(torch.cat(ranks))

@torch.no_grad()
def evaluate_seen_within_epoch(
    model: ContrastiveModel,
    loader,
    centroids: torch.Tensor,
    device,
) -> dict:
    model.eval()
    ranks = []
    for audio, _t, pair_idx in loader:
        audio = audio.to(device, non_blocking=True)
        pair_idx = pair_idx.to(device)
        a = F.normalize(model.audio_proj(model.layer_sum(audio)), dim=-1)
        sim = a @ centroids.T
        order = sim.argsort(dim=1, descending=True)
        rank_of_gt = (order == pair_idx.unsqueeze(1)).nonzero()[:, 1]
        ranks.append(rank_of_gt.cpu())
    return _metrics(torch.cat(ranks))

@torch.no_grad()
def evaluate_lc_ensemble(
    model: ContrastiveModel,
    loader,
    text_variants: torch.Tensor,
    device,
    gallery_pair_to_model: dict,
    query_pair_to_model: dict,
) -> dict:
    _, K, _ = text_variants.shape
    results = []
    for k in range(K):
        centroids = F.normalize(model.text_proj(text_variants[:, k, :].to(device)), dim=-1)
        results.append(evaluate_lc(model, loader, centroids, device,
                                   gallery_pair_to_model, query_pair_to_model))

    def _avg(level: str) -> dict:
        d = {}
        for key in results[0][level]:
            d[key] = results[0][level][key] if key == "n_val" else sum(r[level][key] for r in results) / K
        return d

    return {"pair": _avg("pair"), "model": _avg("model")}

@torch.no_grad()
def evaluate_lc(
    model: ContrastiveModel,
    loader,
    centroids: torch.Tensor,
    device,
    gallery_pair_to_model: dict,
    query_pair_to_model: dict,
) -> dict:
    model.eval()
    pair_ranks, model_ranks = [], []
    gallery_model_ids = torch.tensor(
        [gallery_pair_to_model[i] for i in range(len(centroids))],
        dtype=torch.long,
        device=device,
    )

    for audio, _t, pair_idx in loader:
        audio = audio.to(device, non_blocking=True)
        pair_idx = pair_idx.to(device)
        a = F.normalize(model.audio_proj(model.layer_sum(audio)), dim=-1)
        sim = a @ centroids.T
        order = sim.argsort(dim=1, descending=True)

        pair_rank = (order == pair_idx.unsqueeze(1)).nonzero()[:, 1]
        pair_ranks.append(pair_rank.cpu())

        query_model = torch.tensor(
            [query_pair_to_model[int(p)] for p in pair_idx.cpu()],
            dtype=torch.long,
            device=device,
        )
        ranked_models = gallery_model_ids[order]
        match = ranked_models == query_model.unsqueeze(1)
        model_rank = match.int().argmax(dim=1)
        model_ranks.append(model_rank.cpu())

    return {
        "pair": _metrics(torch.cat(pair_ranks)),
        "model": _metrics(torch.cat(model_ranks)),
    }

SIBLING_AXES = ("model", "arch_family", "acoustic_model", "vocoder", "speaker_type", "lang_family")

@torch.no_grad()
def evaluate_sibling_aware(
    model: ContrastiveModel,
    loader,
    centroids: torch.Tensor,
    device,
    gallery_attrs: list[dict],
    query_attrs: list[dict],
) -> dict:
    model.eval()
    n_gallery = len(gallery_attrs)
    assert centroids.shape[0] == n_gallery, (
        f"centroids({centroids.shape[0]}) must match gallery_attrs({n_gallery})"
    )

    axis_vocab: dict[str, dict[str, int]] = {}
    gallery_axis_ids: dict[str, torch.Tensor] = {}
    for axis in SIBLING_AXES:
        vocab: dict[str, int] = {}
        for g in gallery_attrs:
            v = g[axis]
            if v not in vocab:
                vocab[v] = len(vocab)
        axis_vocab[axis] = vocab
        gallery_axis_ids[axis] = torch.tensor(
            [vocab[g[axis]] for g in gallery_attrs], dtype=torch.long, device=device
        )

    pair_ranks: list[torch.Tensor] = []
    axis_ranks: dict[str, list[torch.Tensor]] = {axis: [] for axis in SIBLING_AXES}
    recalls: list[float] = []
    recall_axes = ("arch_family", "vocoder", "speaker_type", "lang_family")

    for audio, _t, pair_idx in loader:
        audio = audio.to(device, non_blocking=True)
        pair_idx = pair_idx.to(device)
        B = audio.shape[0]

        a = F.normalize(model.audio_proj(model.layer_sum(audio)), dim=-1)
        sim = a @ centroids.T
        order = sim.argsort(dim=1, descending=True)

        pair_rank = (order == pair_idx.unsqueeze(1)).nonzero()[:, 1]
        pair_ranks.append(pair_rank.cpu())

        pair_idx_cpu = pair_idx.cpu().tolist()
        for axis in SIBLING_AXES:
            vocab = axis_vocab[axis]
            q_ids = torch.tensor(
                [vocab.get(query_attrs[i][axis], -1) for i in pair_idx_cpu],
                dtype=torch.long, device=device,
            )
            g_in_order = gallery_axis_ids[axis][order]
            match = g_in_order == q_ids.unsqueeze(1)
            has_match = match.any(dim=1)
            rank = match.int().argmax(dim=1)
            rank = torch.where(has_match, rank, torch.full_like(rank, n_gallery))
            axis_ranks[axis].append(rank.cpu())

        top1 = order[:, 0].cpu().tolist()
        for bi, gi in enumerate(top1):
            q = query_attrs[pair_idx_cpu[bi]]
            g = gallery_attrs[gi]
            matches = sum(1 for ax in recall_axes if q[ax] == g[ax])
            recalls.append(matches / 4.0)

    out: dict = {"pair": _metrics(torch.cat(pair_ranks))}
    for axis in SIBLING_AXES:
        out[axis] = _metrics(torch.cat(axis_ranks[axis]))
    out["attribute_recall@1"] = {
        "mean": float(sum(recalls) / max(len(recalls), 1)),
        "n_val": int(len(recalls)),
    }
    return out
