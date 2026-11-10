"""Route A: does the field produce self-organization that vanilla attention does not?

Three maps, identical budget, identical inputs. No training: masses come from an
unsupervised density statistic, so nothing is tuned to the outcome.

  M0 vanilla : logit = beta * x_i . x_j
  M1 field   : logit = alpha * x_i . x_j + log m_j - kappa * ((d^2/2) / sigma^2)^(p/2)
  M2 field+  : M1 plus mass-conserving merge + escape, so N really shrinks

Metrics per step: live tokens, cluster count (union-find on cos > 0.9), escaped
fraction, cluster-size distribution (hierarchy probe).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N, D, K = 256, 32, 4          # tokens, dim, latent topics
NOISE_FRAC = 0.20
BETA, ALPHA, KAPPA, P, LAM = 6.0, 1.0, 0.35, 2.0, 0.35
STEPS, MERGE_EVERY, ESC_EPS = 120, 10, 1e-3
CLUST_COS = 0.9


def make_tokens(seed=0):
    g = torch.Generator().manual_seed(seed)
    cent = F.normalize(torch.randn(K, D, generator=g), dim=-1)
    lab = torch.arange(N) % K
    x = cent[lab] * 3.0 + torch.randn(N, D, generator=g)
    n_noise = int(N * NOISE_FRAC)
    noise_idx = torch.randperm(N, generator=g)[:n_noise]
    x[noise_idx] = torch.randn(n_noise, D, generator=g) * 1.0
    x = F.normalize(x, dim=-1)
    return x, lab, noise_idx


def density_mass(x):
    """Core-ness from an unsupervised statistic: mean cosine to all other units."""
    sim = x @ x.t()
    dens = (sim.sum(1) - 1.0) / (x.shape[0] - 1)
    z = (dens - dens.mean()) / dens.std().clamp_min(1e-6)
    return F.softplus(2.0 * z).clamp_min(1e-3), z


def logits_vanilla(x, mass=None):
    return BETA * (x @ x.t())


def logits_field(x, mass):
    d2 = (2.0 - 2.0 * (x @ x.t())).clamp_min(0.0)
    r = mass
    sigma = 0.7 * 0.5 * (r[:, None] + r[None, :])
    term = (d2 / (2.0 * sigma ** 2 + 1e-6)) ** (P / 2.0)
    return ALPHA * (x @ x.t()) + torch.log(mass)[None, :] - KAPPA * term


def clusters(x):
    """Union-find over cos > CLUST_COS; returns cluster count and size list."""
    n = x.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    adj = (x @ x.t() > CLUST_COS).triu(1)
    for i, j in zip(*torch.nonzero(adj, as_tuple=True)):
        ra, rb = find(int(i)), find(int(j))
        if ra != rb:
            parent[rb] = ra
    roots = {}
    for i in range(n):
        roots.setdefault(find(i), 0)
        roots[find(i)] += 1
    return len(roots), sorted(roots.values(), reverse=True)


def merge_top_pairs(x, w, keep_frac=0.25):
    """Mass-conserving greedy merge of the highest mutual-attention pairs."""
    n = x.shape[0]
    if n < 4:
        return x, None
    mut = w + w.t()
    mut.fill_diagonal_(-1.0)
    n_pairs = max(1, int(n * keep_frac / 2))
    flat = torch.topk(mut.flatten(), 2 * n_pairs).indices
    used, groups, plain = set(), [], []
    for f in flat.tolist():
        i, j = divmod(f, n)
        if i in used or j in used:
            continue
        used.update((i, j))
        groups.append((i, j))
    for i in range(n):
        if i not in used:
            plain.append(i)
    if not groups:
        return x, None
    new = [F.normalize(x[i] + x[j], dim=-1)[None, :] for i, j in groups]
    new += [x[i][None, :] for i in plain]
    return torch.cat(new, 0), (len(groups), n - len(new))


def run(kind, steps=STEPS):
    x, lab, noise_idx = make_tokens()
    traj = []
    for t in range(steps + 1):
        mass, _ = density_mass(x)
        w = torch.softmax(logits_vanilla(x) if kind == "M0" else logits_field(x, mass), dim=-1)
        nclus, sizes = clusters(x)
        recv = w.sum(0)
        escaped = float((recv < ESC_EPS * recv.mean()).float().mean())
        traj.append({"t": t, "n": int(x.shape[0]), "clusters": nclus,
                     "top_size": sizes[0], "median_size": float(np.median(sizes)),
                     "escaped": escaped, "eff_part": float(1.0 / w.pow(2).sum(1).mean())})
        if t == steps:
            break
        x = F.normalize(x + LAM * (w @ x - x), dim=-1)
        if kind == "M2" and (t + 1) % MERGE_EVERY == 0:
            x, _ = merge_top_pairs(x, w)
    return traj


def main():
    out = {k: run(k) for k in ("M0", "M1", "M2")}
    with open(os.path.join(RES, "selforg.json"), "w") as f:
        json.dump(out, f, indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    names = {"M0": "M0 vanilla softmax", "M1": "M1 field (no merge)", "M2": "M2 field + merge/escape"}
    col = {"M0": "#111111", "M1": "#d62728", "M2": "#1f77b4"}
    for k in out:
        ts = [r["t"] for r in out[k]]
        ax[0].plot(ts, [r["clusters"] for r in out[k]], color=col[k], label=names[k], lw=1.9)
        ax[1].plot(ts, [r["n"] for r in out[k]], color=col[k], label=names[k], lw=1.9)
        ax[2].plot(ts, [r["top_size"] / max(1, r["median_size"]) for r in out[k]],
                   color=col[k], label=names[k], lw=1.9)
    ax[0].set_title("A. cluster count (cos>0.9)")
    ax[1].set_title("B. live units (merge really removes them)")
    ax[2].set_title("C. hierarchy probe: max cluster / median cluster")
    for a in ax:
        a.set_xlabel("iteration")
        a.grid(alpha=0.25)
    ax[0].legend(fontsize=8)
    fig.suptitle("Route A: self-organization with a control arm (N=256, 4 topics + 20% noise, no training)")
    fig.tight_layout()
    p = os.path.join(RES, "selforg.png")
    fig.savefig(p, dpi=140)
    for k in out:
        r = out[k]
        print("%s  clusters %3d -> %3d | units %3d -> %3d | escaped %.3f -> %.3f | top/median %.1f -> %.1f"
              % (k, r[0]["clusters"], r[-1]["clusters"], r[0]["n"], r[-1]["n"],
                 r[0]["escaped"], r[-1]["escaped"],
                 r[0]["top_size"] / max(1, r[0]["median_size"]),
                 r[-1]["top_size"] / max(1, r[-1]["median_size"])))
    print("wrote", p)


if __name__ == "__main__":
    main()
