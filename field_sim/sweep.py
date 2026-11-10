"""Grid runner for the field phase diagram.

Resumable (append-only JSONL, keyed by config), device agnostic, with a
`--estimate` preflight so a compute node can be sized before the long run.

  python -m field_sim.sweep --preset repulsion --out results/phase.jsonl
  python -m field_sim.sweep --preset repulsion --estimate
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from field_sim.sim import SimConfig, simulate, pick_device  # noqa: E402

PRESETS = {
    # 2D: attraction strength x repulsion strength, with and without truncation
    "repulsion": {
        "kappa_a": [0.1, 0.3, 1.0, 3.0],
        "kappa_r": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
        "truncate_k": [0],
        "seeds": [0, 1, 2],
        "n": [256], "steps": [120],
    },
    # same grid but with hard top-k sparsification on
    "repulsion_sparse": {
        "kappa_a": [0.1, 0.3, 1.0, 3.0],
        "kappa_r": [0.0, 0.5, 1.0, 2.0],
        "truncate_k": [16],
        "seeds": [0, 1, 2],
        "n": [256], "steps": [120],
    },
    # scale check for the compute node
    "scale": {
        "kappa_a": [1.0], "kappa_r": [0.0, 0.5, 1.0], "truncate_k": [0],
        "seeds": [0], "n": [1024, 4096, 8192], "steps": [120],
    },
}


def build_grid(preset: str):
    p = PRESETS[preset]
    keys = [k for k in p if k not in ("seeds", "n", "steps")]
    base = SimConfig()
    out = []
    for combo in itertools.product(*[p[k] for k in keys]):
        for seed in p["seeds"]:
            for n in p["n"]:
                for steps in p["steps"]:
                    cfg = replace(base, seed=seed, n=n, steps=steps,
                                  **dict(zip(keys, combo)))
                    out.append(cfg)
    return out


def load_done(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                c = SimConfig(**{k: v for k, v in r["config"].items()})
                done.add(c.key())
            except Exception:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="repulsion", choices=sorted(PRESETS))
    ap.add_argument("--out", default="results/phase.jsonl")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0, help="0-based shard index")
    ap.add_argument("--shards", type=int, default=1, help="total shards (one process/GPU each)")
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()

    dev = pick_device(args.device)
    grid = build_grid(args.preset)
    done = load_done(args.out)
    todo = [c for c in grid if c.key() not in done]
    if args.shards > 1:
        todo = [c for i, c in enumerate(todo) if i % args.shards == args.shard]
    if args.limit:
        todo = todo[: args.limit]
    print(f"device={dev}  grid={len(grid)}  done={len(done)}  todo={len(todo)}"
          f"  shard={args.shard}/{args.shards}", flush=True)

    if args.estimate:
        for c in (todo[:1] or grid[:1]):
            t0 = time.perf_counter()
            c2 = replace(c, device=args.device)
            simulate(c2)
            dt = time.perf_counter() - t0
            print(f"one config (n={c.n}, steps={c.steps}) took {dt:.2f}s -> "
                  f"{len(todo)} configs ~= {dt * len(todo) / 60:.1f} min", flush=True)
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.perf_counter()
    with open(args.out, "a", encoding="utf-8") as f:
        for i, c in enumerate(todo, 1):
            r = simulate(replace(c, device=args.device))
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(todo)}  {el:.1f}s  eta {el / i * (len(todo) - i):.0f}s", flush=True)
    print("done", args.out)


if __name__ == "__main__":
    main()
