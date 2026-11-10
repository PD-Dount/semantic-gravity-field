"""Render the attraction x repulsion phase diagram as a categorical map.

Categories are the only thing that matters for the question: did the field settle
into a *few* attractors at all?
    0      : everything was eaten (no units left)
    1      : global collapse (the failure mode we are trying to escape)
    2..8   : K-like multi-attractor state  <- the thing we are looking for
    >=9    : dispersion (nothing merged)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CATS = [(0, "eaten", "#111111"), (1, "collapsed to 1", "#9e9e9e"),
        (2, "K-like 2-8", "#2ca02c"), (3, "dispersed >=9", "#ff7f0e")]


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def cat_of(c):
    if c == 0:
        return 0
    if c == 1:
        return 1
    return 2 if c <= 8 else 3


def grids(rows):
    ka = sorted({r["config"]["kappa_a"] for r in rows})
    kr = sorted({r["config"]["kappa_r"] for r in rows})
    cat = np.zeros((len(ka), len(kr)))
    nmi = np.zeros((len(ka), len(kr)))
    for i, a in enumerate(ka):
        for j, b in enumerate(kr):
            sel = [r for r in rows if r["config"]["kappa_a"] == a and r["config"]["kappa_r"] == b]
            if not sel:
                continue
            ids = [CATS.index(next(x for x in CATS if x[0] == cat_of(r["final_clusters"]))) for r in sel]
            cat[i, j] = float(np.mean(ids))
            nmi[i, j] = float(np.mean([r["final_nmi"] for r in sel]))
    return np.array(ka), np.array(kr), cat, nmi


def draw(ax, ka, kr, m, title, cmap, norm=None, fmt="%.2f", vmin=0, vmax=1):
    im = ax.imshow(m, aspect="auto", origin="lower", cmap=cmap, norm=norm, vmin=None if norm else vmin,
                   vmax=None if norm else vmax)
    ax.set_xticks(range(len(kr)))
    ax.set_xticklabels(["0" if x == 0 else "%g" % x for x in kr])
    ax.set_yticks(range(len(ka)))
    ax.set_yticklabels(["%g" % x for x in ka])
    ax.set_xlabel("repulsion strength  kappa_r")
    ax.set_ylabel("attraction strength  kappa_a")
    ax.set_title(title)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, fmt % m[i, j], ha="center", va="center", fontsize=8,
                    color="white" if norm is None and m[i, j] < 0.6 * vmax else "black")
    return im


def main():
    dense = load(os.path.join(RES, "phase_rep.jsonl"))
    sparse = load(os.path.join(RES, "phase_rep_sparse.jsonl"))
    cmap = ListedColormap([c[2] for c in CATS])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6))
    for row, (name, rows) in enumerate((("dense", dense), ("top-16 truncation", sparse))):
        ka, kr, cat, nmi = grids(rows)
        draw(axes[row][0], ka, kr, cat,
             "%s: outcome category (K=4 is the target)" % name, cmap, norm, fmt="%.1f")
        draw(axes[row][1], ka, kr, nmi, "%s: mean NMI with latent topics" % name, "magma", None, "%.2f")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c[2], label=c[1]) for c in CATS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Attraction x repulsion phase diagram - N=256, 4 topics + 20% noise, 120 iters, 3 seeds\n"
                 "(cell value = mean category id; 0=eaten 1=collapsed 2=K-like 3=dispersed)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = os.path.join(RES, "phase_repulsion.png")
    fig.savefig(p, dpi=140)
    print("wrote", p)

    for name, rows in (("dense", dense), ("top16", sparse)):
        good = [r for r in rows if r["config"]["kappa_r"] > 0]
        ok = [r for r in good if 2 <= r["final_clusters"] <= 8 and r["final_nmi"] >= 0.5]
        zero = [r for r in rows if r["config"]["kappa_r"] == 0]
        alive = [r for r in good if r["final_units"] >= 32]
        ok2 = [r for r in alive if 2 <= r["final_clusters"] <= 8 and r["final_nmi"] >= 0.5]
        print("%-6s kr=0: %d/%d collapsed  |  kr>0: K-state %d/%d (%.0f%%)  |  "
              "with >=32 units alive: %d/%d (%.0f%%)"
              % (name, len(zero), len(zero), len(ok), len(good), 100 * len(ok) / max(1, len(good)),
                 len(ok2), len(alive), 100 * len(ok2) / max(1, len(alive))))


if __name__ == "__main__":
    main()
