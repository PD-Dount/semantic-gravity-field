"""Portable field-dynamics simulator.

Vectorised end to end (no Python loop over tokens), device agnostic
(cpu / cuda / npu), resumable-friendly, so a long grid on a compute node can be
interrupted and restarted.  Everything the phase diagram needs lives in SimConfig.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------- device
def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:                                     # Ascend
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return torch.device("npu")
    except Exception:
        pass
    return torch.device("cpu")


# --------------------------------------------------------------------- config
@dataclass
class SimConfig:
    # data
    n: int = 256
    d: int = 32
    k_topics: int = 4
    noise_frac: float = 0.20
    topic_spread: float = 3.0
    seed: int = 0
    # dynamics
    steps: int = 120
    lam: float = 0.25
    beta: float = 6.0                 # plain content term
    kappa_a: float = 1.0              # attraction (gravity) strength
    kappa_r: float = 0.0              # repulsion (competitive load) strength
    p: float = 2.0                    # generalised-Gaussian exponent
    sigma_a: float = 0.5              # attraction range scale
    mass_mode: str = "density"        # density | uniform
    # sparsification / removal
    truncate_k: int = 0               # 0 = dense; else keep top-k keys per query
    escape_eps: float = 1e-3          # irreversible removal of starved keys
    escape_rate: float = 0.02         # at most this fraction removed per step (cascade guard)
    merge_every: int = 0              # 0 = never
    # bookkeeping
    device: str = "auto"
    dtype: str = "float32"
    cluster_tau: float = 0.9
    cluster_iters: int = 64

    def to_dict(self) -> dict:
        return asdict(self)

    def key(self) -> str:
        return "|".join(f"{k}={v}" for k, v in sorted(self.to_dict().items())
                        if k not in ("device", "dtype"))


# --------------------------------------------------------------------- helpers
def build_tokens(cfg: SimConfig, device: torch.device):
    g = torch.Generator().manual_seed(cfg.seed)
    cent = F.normalize(torch.randn(cfg.k_topics, cfg.d, generator=g), dim=-1)
    lab = torch.arange(cfg.n) % cfg.k_topics
    x = cent[lab] * cfg.topic_spread + torch.randn(cfg.n, cfg.d, generator=g)
    n_noise = int(cfg.n * cfg.noise_frac)
    noise_idx = torch.randperm(cfg.n, generator=g)[:n_noise]
    x[noise_idx] = torch.randn(n_noise, cfg.d, generator=g)
    x = F.normalize(x, dim=-1).to(device)
    return x, lab.to(device), noise_idx.to(device)


def density_mass(x, mode="density"):
    if mode == "uniform":
        return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    sim = x @ x.t()
    dens = (sim.sum(1) - 1.0) / max(1, x.shape[0] - 1)
    z = (dens - dens.mean()) / dens.std().clamp_min(1e-6)
    return F.softplus(2.0 * z).clamp_min(1e-3)


def attraction_logits(cfg, x, mass):
    sim = x @ x.t()
    if cfg.kappa_a == 0.0:
        return cfg.beta * sim
    d2 = (2.0 - 2.0 * sim).clamp_min(0.0)
    sigma = cfg.sigma_a * 0.5 * (mass[:, None] + mass[None, :])
    term = (d2 / (2.0 * sigma ** 2 + 1e-6)) ** (cfg.p / 2.0)
    return cfg.beta * sim + torch.log(mass)[None, :] - cfg.kappa_a * term


def topk_mask(logits, k):
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    thr = logits.topk(k, dim=-1).values[:, -1:]
    return logits.masked_fill(logits < thr, float("-inf"))


def cluster_labels(x, tau, max_iters=64):
    """Connected components of the cos>tau graph, fully vectorised."""
    n = x.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=x.device)
    adj = (x @ x.t()) > tau
    label = torch.arange(n, device=x.device, dtype=torch.float32)
    big = float(n + 1)
    full = torch.full((1, n), big, device=x.device, dtype=torch.float32)
    for _ in range(max_iters):
        nb = torch.where(adj, label[None, :].expand(n, n), full).amin(1)
        new = torch.minimum(label, nb)
        if torch.equal(new, label):
            break
        label = new
    return torch.unique(label, return_inverse=True)[1]


def nmi(a, b):
    a, b = a.view(-1).long(), b.view(-1).long()
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    tab = torch.zeros(na, nb, device=a.device, dtype=torch.float64)
    tab.index_put_((a, b), torch.ones_like(a, dtype=torch.float64), accumulate=True)
    p = tab / tab.sum()
    pa, pb = p.sum(1), p.sum(0)
    outer = pa[:, None] * pb[None, :]
    nz = p > 0
    mi = (p[nz] * (p[nz] / outer[nz]).log()).sum()
    ha = -(pa[pa > 0] * pa[pa > 0].log()).sum()
    hb = -(pb[pb > 0] * pb[pb > 0].log()).sum()
    return 0.0 if mi <= 0 else float(mi / (ha * hb).sqrt())


def mutual_best_merge(x, mass, lab, w):
    """Mass-conserving merge of mutual-nearest pairs; vectorised, no Python loop.

    Returns (x, mass, lab, n_merged).  Labels follow the surviving representative.
    """
    n = x.shape[0]
    if n < 4:
        return x, mass, lab, 0
    s = w + w.t()
    s.fill_diagonal_(-float("inf"))
    best = s.argmax(1)
    idx = torch.arange(n, device=x.device)
    mutual = best[best] == idx
    left = idx[mutual & (idx < best)]
    if left.numel() == 0:
        return x, mass, lab, 0
    right = best[left]
    new_x = F.normalize(x[left] + x[right], dim=-1)
    new_m = mass[left] + mass[right]                      # mass conservation
    keep = torch.ones(n, dtype=torch.bool, device=x.device)
    keep[left] = False
    keep[right] = False
    return (torch.cat([new_x, x[keep]], 0), torch.cat([new_m, mass[keep]], 0),
            torch.cat([lab[left], lab[keep]], 0), int(left.numel()))


# --------------------------------------------------------------------- main
def simulate(cfg: SimConfig) -> dict:
    dev = pick_device(cfg.device)
    dt = getattr(torch, cfg.dtype)
    torch.manual_seed(cfg.seed)

    x, lab, noise_idx = build_tokens(cfg, dev)
    x, lab = x.to(dt), lab.to(dev)
    is_noise = torch.zeros(n_total := int(x.shape[0]), dtype=torch.bool, device=dev)
    is_noise[noise_idx] = True
    traj, merged_total = [], 0
    snap = max(1, cfg.steps // 20)

    for t in range(cfg.steps + 1):
        mass = density_mass(x, cfg.mass_mode).to(dt)
        logits = attraction_logits(cfg, x, mass)
        if cfg.kappa_r > 0.0:
            # competitive / lateral inhibition: keys already carrying a lot of
            # attention get pushed down, so one attractor cannot swallow the field.
            base_w = torch.softmax(topk_mask(logits, cfg.truncate_k), dim=-1)
            load = base_w.sum(0)
            load = (load - load.mean()) / load.std().clamp_min(1e-6)
            logits = logits - cfg.kappa_r * load[None, :]
        w = torch.softmax(topk_mask(logits, cfg.truncate_k), dim=-1)

        if t % snap == 0 or t == cfg.steps:
            lb = cluster_labels(x, cfg.cluster_tau, cfg.cluster_iters)
            sizes = torch.bincount(lb)
            traj.append({
                "t": t, "units": int(x.shape[0]), "clusters": int(sizes.numel()),
                "top": int(sizes.max()), "median": float(sizes.median()),
                "eff_part": float(1.0 / w.pow(2).sum(1).mean()),
            })
        if t == cfg.steps:
            break

        x = F.normalize(x + cfg.lam * (w @ x - x), dim=-1)

        recv = w.sum(0)
        # 逃逸阈值必须对"初始"质量标度取，且每步限速。用当前均值做相对阈值会级联崩溃：
        # 删掉一批 key -> 剩下的分到更多质量 -> 均值升高 -> 阈值升高 -> 再删一批 -> 归零。
        # 本机实测 N>=1024 时原始写法会把 1024/8192 个 unit 全部吃光。
        if t == 0:
            recv_ref = recv.mean().clamp_min(1e-6)
        thr = cfg.escape_eps * recv_ref
        starved = recv < thr
        budget = max(1, int(cfg.escape_rate * x.shape[0]))
        if int(starved.sum()) > budget:                      # 只删最饿的那一批
            cut = torch.topk(recv, budget, largest=False).values.max()
            starved = recv <= cut
        keep = ~starved
        if not bool(keep.all()):
            idx = torch.nonzero(keep).squeeze(1)
            x, lab, mass, is_noise = x[idx], lab[idx], mass[idx], is_noise[idx]
            if x.shape[0] == 0:      # a strong enough field can eat the whole set
                break

        if cfg.merge_every and (t + 1) % cfg.merge_every == 0:
            m2 = density_mass(x, cfg.mass_mode).to(dt)
            w2 = torch.softmax(attraction_logits(cfg, x, m2), dim=-1)
            x, mass, lab, n_m = mutual_best_merge(x, mass, lab, w2)
            is_noise = is_noise[: x.shape[0]]
            merged_total += n_m

    lb = cluster_labels(x, cfg.cluster_tau, cfg.cluster_iters)
    if x.shape[0] == 0:
        return {"config": cfg.to_dict(), "device": str(dev), "final_clusters": 0,
                "final_top": 0, "final_units": 0, "merged_total": merged_total,
                "removed_total": n_total, "final_nmi": 0.0, "escaped_frac": 1.0,
                "noise_escaped": 0.0, "noise_left": 0, "collapsed": True, "traj": traj}
    sizes = torch.bincount(lb)
    return {
        "config": cfg.to_dict(),
        "device": str(dev),
        "final_clusters": int(sizes.numel()),
        "final_top": int(sizes.max()),
        "final_units": int(x.shape[0]),
        "merged_total": merged_total,
        "removed_total": n_total - int(x.shape[0]),
        "final_nmi": nmi(lb, lab),
        "collapsed": False,
        "escaped_frac": float((n_total - int(x.shape[0])) / max(1, n_total)),
        "noise_escaped": float(is_noise[: int(x.shape[0])].float().mean()) if int(x.shape[0]) else 0.0,
        "noise_left": int(is_noise[: int(x.shape[0])].sum()),
        "traj": traj,
    }
