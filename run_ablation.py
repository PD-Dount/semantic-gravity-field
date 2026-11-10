"""消融编排器：python run_ablation.py --mode full

每个 (arm, task, seed) 独立子进程、1 线程，靠进程并行吃满 4 核。
结果写 results/runs.jsonl + results/summary_<mode>.md
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
PY = sys.executable


def job(spec):
    (arm, task, seed, steps, seq, batch, lr, d_model, n_layers, vocab, eval_every) = spec
    out = os.path.join(RES, f"{arm}_{task}_s{seed}.json")
    cmd = [PY, "-X", "utf8", "-m", "gravity_lab.train", "--arm", arm, "--task", task,
           "--seed", str(seed), "--steps", str(steps), "--seq", str(seq),
           "--batch", str(batch), "--lr", str(lr), "--d-model", str(d_model),
           "--layers", str(n_layers), "--vocab", str(vocab), "--threads", "1",
           "--eval-every", str(eval_every), "--eval-batches", "4",
           "--log-every", "1000", "--out", out]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t0
    if p.returncode != 0:
        return {"arm": arm, "task": task, "seed": seed, "error": p.stderr[-3000:], "wall_s": dt}
    with open(out) as f:
        r = json.load(f)
    r = r.get("result", r)
    curve = r.get("curve") or []
    r["steps_to_50pct"] = next((int(s) for s, a in curve if a >= 0.5), None)
    r.update({"arm": arm, "seed": seed, "task": task, "wall_s": dt})
    return r


def summarize(rows, path):
    arms = sorted({r["arm"] for r in rows})
    tasks = sorted({r.get("task", "?") for r in rows})
    L = ["# 消融结果\n"]
    for task in tasks:
        L.append(f"\n## task = {task}\n")
        L.append("| arm | acc mean±sd | loss mean±sd | 逐 seed acc | steps→50% | wall/run | eff_part | sparsity | kv_len | escape | p_mean | params |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in arms:
            sel = [r for r in rows if r["arm"] == arm and r.get("task") == task and "error" not in r]
            if not sel:
                L.append(f"| {arm} | — | — | — | — | — | — | — | — | — | — | — |")
                continue

            def col(k):
                return [r[k] for r in sel if isinstance(r.get(k), (int, float))]

            def ms(k):
                v = col(k)
                if not v:
                    return "—"
                m = sum(v) / len(v)
                sd = (sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5
                return "%.3f±%.3f" % (m, sd)

            accs = "/".join("%.2f" % x for x in col("final_acc"))
            s50 = [r.get("steps_to_50pct") for r in sel if isinstance(r.get("steps_to_50pct"), (int, float))]
            L.append("| %s | %s | %s | %s | %s | %.0fs | %.1f | %.3f | %.1f | %.3f | %s | %d |" % (
                arm, ms("final_acc"), ms("final_loss"), accs,
                ("%.0f" % (sum(s50) / len(s50))) if s50 else "—",
                sum(col("wall_s")) / max(1, len(col("wall_s"))),
                sum(col("eff_part")) / max(1, len(col("eff_part"))),
                sum(col("sparsity")) / max(1, len(col("sparsity"))),
                sum(col("kv_len")) / max(1, len(col("kv_len"))),
                sum(col("escape_frac")) / max(1, len(col("escape_frac"))),
                ("%.3f" % (sum(col("p_mean")) / len(col("p_mean")))) if col("p_mean") else "—",
                int(col("params")[0]) if col("params") else 0))
        L.append("")
    errs = [r for r in rows if "error" in r]
    if errs:
        L.append("\n**失败 %d 个**\n" % len(errs))
        for e in errs[:6]:
            tail = (e["error"].strip().splitlines() or ["?"])[-1][:180]
            L.append(f"- {e['arm']}/{e['task']}/s{e['seed']}: `{tail}`")
    txt = "\n".join(L)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A0,A1,A2,A3,A4,A5")
    ap.add_argument("--tasks", default="mqar")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    specs = [(a, t, s, args.steps, args.seq, args.batch, args.lr, args.d, args.layers,
              args.vocab, args.eval_every)
             for a in args.arms.split(",") for t in args.tasks.split(",")
             for s in [int(x) for x in args.seeds.split(",")]]
    print(f"{len(specs)} 个 run，{args.jobs} 路并行，每个 {args.steps} 步 "
          f"(seq={args.seq} d={args.d} L={args.layers} V={args.vocab} lr={args.lr})", flush=True)
    t0 = time.time()
    rows = []
    with Pool(args.jobs) as pool:
        for r in pool.imap_unordered(job, specs):
            rows.append(r)
            if "error" in r:
                print("  FAIL %-3s s%d  %s" % (r["arm"], r["seed"],
                                               (r["error"].strip().splitlines() or ["?"])[-1][:120]), flush=True)
            else:
                print("  done %-3s s%d  acc=%.3f loss=%.3f wall=%.0fs" % (
                    r["arm"], r["seed"], r.get("final_acc", -1), r.get("final_loss", -1),
                    r.get("wall_s", -1)), flush=True)
    with open(os.path.join(RES, f"runs_{args.tag}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summarize(rows, os.path.join(RES, f"summary_{args.tag}.md"))
    print("\n总耗时 %.1f 分钟" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
