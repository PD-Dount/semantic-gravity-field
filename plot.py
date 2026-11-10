"""Plot the ablation: equal-step curves, equal-wallclock curves, cost bars. ASCII-only source."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
COLORS = {"A0": "#111111", "A1": "#1f77b4", "A2": "#2ca02c",
          "A3": "#d62728", "A4": "#9467bd", "A5": "#ff7f0e"}
STEP_MS = {"A0": 121.3, "A1": 127.2, "A2": 243.1, "A3": 317.7, "A4": 334.5, "A5": 420.9}


def load():
    rows = []
    with open(os.path.join(RES, "runs_merged.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                rows.append(r)
    return rows


def interp(xs, ys, grid):
    return np.interp(grid, xs, ys)


def band(ax, arms, rows, key, grid):
    for a in arms:
        sel = [r for r in rows if r["arm"] == a]
        if not sel:
            continue
        cur = np.stack([interp([p[0] for p in r[key]], [p[1] for p in r[key]], grid) for r in sel])
        ax.plot(grid, cur.mean(0), color=COLORS.get(a), label=a, lw=1.9)
        ax.fill_between(grid, cur.mean(0) - cur.std(0), cur.mean(0) + cur.std(0),
                        color=COLORS.get(a), alpha=0.12)


def main():
    rows = load()
    arms = sorted({r["arm"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))

    ax = axes[0][0]
    band(ax, arms, rows, "curve", np.arange(0, 1201, 20))
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("eval accuracy (masked)")
    ax.set_title("A. equal steps: every arm is the same curve")
    ax.axhline(1 / 16, ls=":", c="gray", lw=1)
    ax.text(20, 1 / 16 + 0.006, "chance 6.25%", fontsize=8, color="gray")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.25)

    ax = axes[0][1]
    band(ax, arms, rows, "loss_curve", np.arange(0, 1201, 20))
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("eval loss (masked)")
    ax.set_title("B. equal steps: A0 has the lowest loss")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.25)

    ax = axes[1][0]
    tgrid = np.arange(0, 620, 10.0)
    for a in arms:
        sel = [r for r in rows if r["arm"] == a]
        if not sel:
            continue
        cur = []
        for r in sel:
            xs = np.array([p[0] for p in r["curve"]]) * STEP_MS[a] / 1000.0
            ys = np.array([p[1] for p in r["curve"]])
            cur.append(interp(xs, ys, tgrid))
        cur = np.stack(cur)
        ax.plot(tgrid, cur.mean(0), color=COLORS.get(a), label=a, lw=1.9)
        ax.fill_between(tgrid, cur.mean(0) - cur.std(0), cur.mean(0) + cur.std(0),
                        color=COLORS.get(a), alpha=0.12)
    ax.set_xlabel("training wall-clock seconds (same 1-thread step cost)")
    ax.set_ylabel("eval accuracy")
    ax.set_title("C. equal budget: field arms spend 2-3.5x per step, gain nothing")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.25)

    ax = axes[1][1]
    accs = [np.mean([r["final_acc"] for r in rows if r["arm"] == a]) for a in arms]
    accs_sd = [np.std([r["final_acc"] for r in rows if r["arm"] == a]) for a in arms]
    steps = [STEP_MS[a] for a in arms]
    x = np.arange(len(arms))
    ax.bar(x - 0.2, steps, 0.4, color=[COLORS.get(a) for a in arms], alpha=0.85, label="step time (ms)")
    ax.set_ylabel("step time (ms)")
    ax2 = ax.twinx()
    ax2.errorbar(x + 0.2, accs, yerr=accs_sd, fmt="o", c="k", capsize=3, label="final acc")
    ax2.set_ylabel("final eval accuracy")
    ax2.set_ylim(0, 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    for i, (s, a) in enumerate(zip(steps, accs)):
        ax.text(i - 0.2, s + 6, "%.2fx" % (s / STEP_MS["A0"]), ha="center", fontsize=8)
    ax.set_title("D. cost vs accuracy: pay 2.6x, get 0.00")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Semantic gravity field v2 - MQAR V=32 N=64 d=48 L=3, 1200 steps, 3 seeds (mean +- sd)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(RES, "curves.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
