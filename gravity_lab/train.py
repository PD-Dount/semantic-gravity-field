"""Training / eval loop for gravity_lab arms (CPU only).

模型只经 config.py 冻结的接口调用：
    from .model import TinyLM            -> TinyLM(mcfg, acfg); forward(idx) -> (logits, stats)
run_training() 不 import attention 内部类。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from .config import ARMS, AttnConfig, ModelConfig, TrainConfig
from .tasks import make_batch, split_seq

_RESULTS_DIR = "results"


# ------------------------------------------------------------------ helpers
def _finite(v):
    """JSON 安全：非有限浮点 -> None。"""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (int, bool)) or v is None:
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f if math.isfinite(f) else None


def _lr_factor(step: int, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / float(warmup)
    span = max(1, total - warmup)
    prog = min(1.0, max(0.0, (step - warmup) / float(span)))
    return 0.5 * (1.0 + math.cos(math.pi * prog))


def _model_forward(model, idx, return_stats: bool):
    out = model(idx, return_stats=return_stats)
    if isinstance(out, tuple):
        logits, stats = out[0], (out[1] if len(out) > 1 else {})
    else:
        logits, stats = out, {}
    return logits, (stats or {})


def _masked_loss_acc(logits, y, mask):
    """mask 位置的 cross-entropy（mean）、命中数、样本数。"""
    lg = logits[mask]
    tgt = y[mask]
    if lg.numel() == 0:
        raise ValueError("mask 全为 False，无法计算 loss")
    loss = F.cross_entropy(lg, tgt)
    correct = int((lg.argmax(dim=-1) == tgt).sum())
    return loss, correct, int(tgt.numel())


def _write_json(path: str, payload: dict) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


# ------------------------------------------------------------------ training
def run_training(tcfg: TrainConfig, mcfg: ModelConfig, acfg: AttnConfig,
                 out_path: str | None = None, verbose: bool = True) -> dict:
    torch.set_num_threads(max(1, int(tcfg.threads)))
    torch.manual_seed(int(tcfg.seed))          # 只用于模型初始化；batch 走独立 Generator

    from .model import TinyLM                  # 延迟导入: 由另一 teammate 维护

    parts = split_seq(tcfg.task, mcfg.seq_len)  # 早期暴露非法 seq/task 组合
    model = TinyLM(mcfg, acfg)
    n_params = int(sum(p.numel() for p in model.parameters()))

    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.lr), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_factor(s, int(tcfg.warmup), int(tcfg.steps))
    )

    # 训练 / 评估用两条独立随机流；评估集固定，跨 eval 可比
    trng = torch.Generator().manual_seed(int(tcfg.seed))
    erg = torch.Generator().manual_seed(int(tcfg.seed) + 10_007)
    n_eval = max(1, int(tcfg.eval_batches))
    eval_sets = [make_batch(tcfg.task, tcfg.batch_size, mcfg.seq_len, mcfg.vocab, erg)
                 for _ in range(n_eval)]

    def evaluate() -> tuple[float, float]:
        """返回 (mask 位置平均 loss, mask 位置 accuracy)。"""
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.inference_mode():
            for b in eval_sets:
                logits, _ = _model_forward(model, b["x"], False)
                loss, hit, n = _masked_loss_acc(logits, b["y"], b["mask"])
                loss_sum += float(loss) * n
                correct += hit
                total += n
        total = max(1, total)
        return loss_sum / total, correct / total

    curve: list[list] = []
    loss_curve: list[list] = []
    stats_last: dict = {}
    stats_mean: dict = {}
    n_stat_calls: dict = {}

    t0 = time.perf_counter()
    loss0, acc0 = evaluate()
    curve.append([0, acc0])
    loss_curve.append([0, loss0])
    best_acc = acc0
    if verbose:
        print(f"eval step     0  loss {loss0:.4f}  acc {acc0:.4f}")

    run_loss, run_n = 0.0, 0
    steps = max(1, int(tcfg.steps))
    for step in range(1, steps + 1):
        model.train()
        b = make_batch(tcfg.task, tcfg.batch_size, mcfg.seq_len, mcfg.vocab, trng)
        logits, stats = _model_forward(model, b["x"], True)
        loss, _, _ = _masked_loss_acc(logits, b["y"], b["mask"])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        run_loss += float(loss.detach())
        run_n += 1
        for k, v in stats.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            stats_last[k] = fv if math.isfinite(fv) else None   # NaN 落成 null，不进均值
            if math.isfinite(fv):
                stats_mean[k] = stats_mean.get(k, 0.0) + fv
                n_stat_calls[k] = n_stat_calls.get(k, 0) + 1

        if step % int(tcfg.eval_every) == 0 or step == steps:
            e_loss, e_acc = evaluate()
            curve.append([step, e_acc])
            loss_curve.append([step, e_loss])
            best_acc = max(best_acc, e_acc)
            if verbose:
                msg = f"eval step {step:5d}  loss {e_loss:.4f}  acc {e_acc:.4f}  best {best_acc:.4f}"
                if stats_last:
                    msg += f"  eff_part {stats_last.get('eff_part', float('nan')):.2f}"
                print(msg)
            run_loss, run_n = 0.0, 0
        elif verbose and int(tcfg.log_every) > 0 and step % int(tcfg.log_every) == 0:
            print(f"train step {step:5d}  loss {run_loss / max(1, run_n):.4f}"
                  f"  lr {sched.get_last_lr()[0]:.2e}")

    wall_s = time.perf_counter() - t0
    final_acc = curve[-1][1]
    final_loss = loss_curve[-1][1]
    s90 = next((int(s) for s, a in curve if a >= 0.9), None)
    s50 = next((int(s) for s, a in curve if a >= 0.5), None)
    stats_mean = {k: _finite(v / max(1, n_stat_calls.get(k, 1))) for k, v in stats_mean.items()}

    result = {
        "arm": acfg.arm,
        "seed": int(tcfg.seed),
        "task": tcfg.task,
        "seq": int(mcfg.seq_len),
        "batch": int(tcfg.batch_size),
        "steps": steps,
        "params": n_params,
        "final_acc": final_acc,
        "best_acc": best_acc,
        "final_loss": final_loss,
        "steps_to_90pct": s90,
        "steps_to_50pct": s50,
        "chance_acc": 1.0 / max(1, mcfg.vocab // 2),
        "wall_s": wall_s,
        "tok_per_s": steps * int(tcfg.batch_size) * int(mcfg.seq_len) / max(1e-9, wall_s),
        "curve": curve,
        "loss_curve": loss_curve,
        "stats": stats_last,
        "stats_mean": stats_mean,
        "layout": parts,
    }
    payload = {
        "config": {"train": asdict(tcfg), "model": asdict(mcfg), "attn": asdict(acfg)},
        "result": result,
    }
    if out_path:
        result["out"] = _write_json(out_path, payload)
    if verbose:
        print(f"final step {steps}  loss {final_loss:.4f}  acc {final_acc:.4f}"
              f"  best {best_acc:.4f}  steps_to_90pct {s90}  wall {wall_s:.2f}s")
        if out_path:
            print(f"wrote {result['out']}")
    return result


# ------------------------------------------------------------------ CLI
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gravity_lab.train")
    p.add_argument("--arm", default="A3", choices=sorted(ARMS))
    p.add_argument("--task", default="mqar", choices=("mqar", "noise_recall", "needle"))
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", default=None)
    return p


def main(argv=None) -> int:
    a = _build_parser().parse_args(argv)
    mcfg = ModelConfig(vocab=a.vocab, seq_len=a.seq, d_model=a.d_model,
                       n_layers=a.layers, n_heads=a.heads)
    acfg = AttnConfig.from_arm(a.arm, d_model=a.d_model, n_heads=a.heads)
    tcfg = TrainConfig(task=a.task, steps=a.steps, batch_size=a.batch, lr=a.lr,
                       warmup=a.warmup, seed=a.seed, eval_every=a.eval_every,
                       eval_batches=a.eval_batches, log_every=a.log_every,
                       threads=a.threads)
    out = a.out or os.path.join(_RESULTS_DIR, f"{a.arm}_{a.task}_s{a.seed}.json")
    try:
        run_training(tcfg, mcfg, acfg, out_path=out)
    except ValueError as e:                     # 非法 seq/task、形状错误等
        print(f"error: {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
