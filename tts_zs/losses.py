import torch
import torch.nn.functional as F

def info_nce_loss(
    a: torch.Tensor, t: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    logits = scale * (a @ t.T)
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

def supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    N = embeddings.size(0)
    device = embeddings.device
    sim = (embeddings @ embeddings.T) / temperature
    sim = sim - sim.detach().max(dim=1, keepdim=True).values

    self_mask = torch.eye(N, dtype=torch.bool, device=device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    n_pos = pos_mask.sum(dim=1).float()
    has_pos = n_pos > 0

    exp_sim = torch.exp(sim) * (~self_mask).float()
    log_denom = torch.log(exp_sim.sum(dim=1))
    log_prob = sim - log_denom.unsqueeze(1)

    loss_per = -(pos_mask.float() * log_prob).sum(dim=1) / n_pos.clamp(min=1)
    return loss_per[has_pos].mean() if has_pos.any() else embeddings.sum() * 0.0

def cross_modal_supcon_loss(
    a: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    N = a.size(0)
    device = a.device
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    n_pos = pos_mask.sum(dim=1)
    has_pos = n_pos > 0

    sim_at = (a @ t.T) / temperature
    sim_at = sim_at - sim_at.detach().max(dim=1, keepdim=True).values
    log_prob_a = sim_at - torch.log(torch.exp(sim_at).sum(dim=1, keepdim=True))
    loss_a = -(pos_mask * log_prob_a).sum(dim=1) / n_pos.clamp(min=1)

    sim_ta = (t @ a.T) / temperature
    sim_ta = sim_ta - sim_ta.detach().max(dim=1, keepdim=True).values
    log_prob_t = sim_ta - torch.log(torch.exp(sim_ta).sum(dim=1, keepdim=True))
    loss_t = -(pos_mask * log_prob_t).sum(dim=1) / n_pos.clamp(min=1)

    if not has_pos.any():
        return a.sum() * 0.0
    return 0.5 * (loss_a[has_pos].mean() + loss_t[has_pos].mean())

def cross_modal_supcon_loss_bank(
    a: torch.Tensor,
    a_labels: torch.Tensor,
    t: torch.Tensor,
    t_labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    pos = (a_labels.unsqueeze(1) == t_labels.unsqueeze(0)).float()
    n_pos_a = pos.sum(dim=1)
    has_a = n_pos_a > 0

    sim_at = (a @ t.T) / temperature
    sim_at = sim_at - sim_at.detach().max(dim=1, keepdim=True).values
    log_prob_a = sim_at - torch.log(torch.exp(sim_at).sum(dim=1, keepdim=True))
    loss_a = -(pos * log_prob_a).sum(dim=1) / n_pos_a.clamp(min=1)

    pos_t = pos.T
    n_pos_t = pos_t.sum(dim=1)
    has_t = n_pos_t > 0

    sim_ta = (t @ a.T) / temperature
    sim_ta = sim_ta - sim_ta.detach().max(dim=1, keepdim=True).values
    log_prob_t = sim_ta - torch.log(torch.exp(sim_ta).sum(dim=1, keepdim=True))
    loss_t = -(pos_t * log_prob_t).sum(dim=1) / n_pos_t.clamp(min=1)

    if not has_a.any() and not has_t.any():
        return a.sum() * 0.0
    la = loss_a[has_a].mean() if has_a.any() else a.sum() * 0.0
    lt = loss_t[has_t].mean() if has_t.any() else a.sum() * 0.0
    return 0.5 * (la + lt)

def unified_supcon_loss(
    a: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    emb = torch.cat([a, t], dim=0)
    lbl = torch.cat([labels, labels], dim=0)
    return supcon_loss(emb, lbl, temperature=temperature)

def uniformity_loss(embeddings: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    sq_dist = torch.pdist(embeddings, p=2).pow(2)
    return sq_dist.mul(-t).exp().mean().log()

def same_lang_margin_loss(
    embeddings: torch.Tensor,
    lang_ids: torch.Tensor,
    pair_ids: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    sim = embeddings @ embeddings.T
    same_lang = lang_ids.unsqueeze(0) == lang_ids.unsqueeze(1)
    diff_pair = pair_ids.unsqueeze(0) != pair_ids.unsqueeze(1)
    mask = same_lang & diff_pair
    if not mask.any():
        return embeddings.sum() * 0.0
    violations = (sim[mask] - margin).clamp(min=0.0)
    return violations.mean()
