"""Route A, part 2: phase diagram over interaction range.

Global attention (vanilla or global field) collapses every token to one point, so
"self-organization happened" is not a discriminating observation. The question that
does discriminate: with a *finite* capture radius, does the field settle into several
persistent clusters -- and do those clusters recover the latent topics that the
vanilla arm cannot recover at all?
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N, D, K = 256, 32, 4
NOISE_FRAC = 0.20
BETA, ALPHA, KAPPA, P, LAM = 6.0, 1.0, 1.0, 2.0, 0.25
STEPS, ESC_EPS = 120, 1e-3
CLUST_COS = 0.9
SIGMAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.5]


def make_tokens(seed=0):
    g = torch.Generator().manual_seed(seed)
    cent = F.normalize(torch.randn(K, D, generator=g), dim=-1)
    lab = torch.arange(N) % K
    x = cent[lab] * 3.0 + torch.randn(N, D, generator=g)
    n_noise = int(N * NOISE_FRAC)
    noise_idx = torch.randperm(N, generator=g)[:n_noise]
    x[noise_idx] = torch.randn(n_noise, D, generator=g)
    return F.normalize(x, dim=-1), lab, noise_idx


def density_mass(x):
    sim = x @ x.t()
    dens = (sim.sum(1) - 1.0) / (x.shape[0] - 1)
    z = (dens - dens.mean()) / dens.std().clamp_min(1e-6)
    return F.softplus(2.0 * z).clamp_min(1e-3)


def logits(x, mass, sigma_scale):
    d2 = (2.0 - 2.0 * (x @ x.t())).clamp_min(0.0)
    sigma = sigma_scale * 0.5 * (mass[:, None] + mass[None, :])
    term = (d2 / (2.0 * sigma ** 2 + 1e-6)) ** (P / 2.0)
    return ALPHA * (x @ x.t()) + torch.log(mass)[None, :] - KAPPA * term


def labels_of(x):
    n = x.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(*torch.nonzero((x @ x.t() > CLUST_COS).triu(1), as_tuple=True)):
        ra, rb = find(int(i)), find(int(j))
        if ra != rb:
            parent[rb] = ra
    remap, out = {}, np.zeros(n, dtype=np.int64)
    for i in range(n):
        r = find(i)
        out[i] = remap.setdefault(r, len(remap))
    return out


def nmi(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = a.max() + 1, b.max() + 1
    tab = np.zeros((na, nb))
    for i in range(a.size):
        tab[a[i], b[i]] += 1
    p = tab / tab.sum()
    pa, pb = p.sum(1), p.sum(0)
    mi = 0.0
    for i in range(na):
        for j in range(nb):
            if p[i, j] > 0:
                mi += p[i, j] * math.log(p[i, j] / (pa[i] * pb[j]))
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return 0.0 if mi <= 0 else mi / math.sqrt(ha * hb)


def run(sigma_scale, vanilla=False):
    x, lab, noise_idx = make_tokens()
    alive = torch.ones(N, dtype=torch.bool)
    for t in range(STEPS):
        mass = density_mass(x)
        lg = BETA * (x @ x.t()) if vanilla else logits(x, mass, sigma_scale)
        w = torch.softmax(lg, dim=-1)
        recv = w.sum(0)
        alive &= recv >= ESC_EPS * recv.mean()          # irreversible escape
        if t == STEPS - 1:
            break
        x = F.normalize(x + LAM * (w @ x - x), dim=-1)
    lb = labels_of(x)
    keep = alive.numpy()
    nl = len(set(lb[keep].tolist()))
    noise_esc = float((~alive[noise_idx]).float().mean())
    return {"sigma": sigma_scale, "vanilla": vanilla, "clusters": nl,
            "alive": int(alive.sum()), "noise_escaped": noise_esc,
            "nmi": float(nmi(lb[keep], lab.numpy()[keep])),
            "biggest": int(np.bincount(lb[keep]).max())}


def main():
    rows = [run(None, vanilla=True)] + [run(s) for s in SIGMAS]
    with open(os.path.join(RES, "selforg_phase.json"), "w") as f:
        json.dump(rows, f, indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    v = rows[0]
    f_rows = rows[1:]
    xs = [r["sigma"] for r in f_rows]
    ax[0].plot(xs, [r["clusters"] for r in f_rows], "o-", c="#d62728", label="field: clusters")
    ax[0].axhline(v["clusters"], ls="--", c="#111111", label="vanilla: %d cluster" % v["clusters"])
    ax[0].axhline(K, ls=":", c="gray", label="ground-truth K=%d" % K)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("capture radius scale sigma")
    ax[0].set_ylabel("clusters at convergence")
    ax[0].set_title("A. bistable: all separate or all collapsed - nothing at K=4")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25)

    ax[1].plot(xs, [r["nmi"] for r in f_rows], "o-", c="#1f77b4", label="field: NMI vs topics")
    ax[1].axhline(v["nmi"], ls="--", c="#111111", label="vanilla NMI %.2f" % v["nmi"])
    ax[1].set_xscale("log")
    ax[1].set_xlabel("capture radius scale sigma")
    ax[1].set_ylabel("NMI with latent topics")
    ax[1].set_title("B. clusters do not recover the latent topics")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25)
    fig.suptitle("Route A phase diagram: N=256, 4 topics + 20% noise, 120 iterations, no training")
    fig.tight_layout()
    p = os.path.join(RES, "selforg_phase.png")
    fig.savefig(p, dpi=140)
    for r in rows:
        tag = "vanilla" if r["vanilla"] else "sigma=%.2f" % r["sigma"]
        print("%-12s clusters=%3d alive=%3d biggest=%3d noise_escaped=%.2f NMI=%.3f"
              % (tag, r["clusters"], r["alive"], r["biggest"], r["noise_escaped"], r["nmi"]))
    print("wrote", p)


if __name__ == "__main__":
    main()
