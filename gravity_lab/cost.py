"""CPU cost measurement per (arm, seq): wall time, peak memory, stats-derived FLOPs.

实测项：前向 wall time、前向+反向 wall time、进程 RSS 峰值。
折算项：按 forward stats 的 kv_len 折算的"块稀疏可达 FLOPs"——是模型侧估算，不是实测，
       结果里用 flops.note 明确标注。
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import statistics
import threading
import time

import torch
import torch.nn.functional as F

from .config import ARMS, AttnConfig, ModelConfig
from .tasks import make_batch

# ------------------------------------------------------------------ memory
_RSS_FN = None


def _win_rss_mb() -> float:
    import ctypes
    from ctypes import wintypes

    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kern = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kern.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    if not psapi.GetProcessMemoryInfo(kern.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return pmc.WorkingSetSize / 1e6


def _resolve_rss_fn():
    global _RSS_FN
    if _RSS_FN is not None:
        return _RSS_FN
    try:
        import psutil  # noqa: F401

        _RSS_FN = lambda: psutil.Process().memory_info().rss / 1e6  # noqa: E731
        return _RSS_FN
    except Exception:
        pass
    if os.name == "nt":
        try:
            _win_rss_mb()
            _RSS_FN = _win_rss_mb
            return _RSS_FN
        except Exception:
            pass
    _RSS_FN = lambda: None  # noqa: E731
    return _RSS_FN


def rss_mb():
    try:
        return _resolve_rss_fn()()
    except Exception:
        return None


@contextlib.contextmanager
def _rss_peak(sample_ms: float = 4.0):
    """采样进程 RSS，yield 一个 list，[0] 为采样到的峰值 MB（无可用实现时 None）。"""
    peak = [rss_mb()]
    if peak[0] is None:
        yield peak
        return
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            v = rss_mb()
            if v is not None and v > peak[0]:
                peak[0] = v
            stop.wait(sample_ms / 1e3)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    try:
        yield peak
    finally:
        stop.set()
        th.join(timeout=1.0)


# ------------------------------------------------------------------ flops
def _flops_estimate(mcfg: ModelConfig, batch: int, kv_len: float) -> dict:
    """模型侧 FLOPs 估算（单次前向），attn 部分按 kv_len/N 折算成可达块稀疏量。"""
    N, d, H, L = int(mcfg.seq_len), int(mcfg.d_model), int(mcfg.n_heads), int(mcfg.n_layers)
    dh = max(1, d // H)
    causal = 0.5
    attn = 4.0 * batch * H * N * N * dh * causal
    proj = 8.0 * batch * N * d * d
    mlp = 16.0 * batch * N * d * d
    io = 2.0 * 2.0 * batch * N * d * int(mcfg.vocab)
    dense = L * (attn + proj + mlp) + io
    r = 0.0 if not math.isfinite(kv_len) or N <= 0 else max(0.0, min(1.0, kv_len / N))
    sparse = L * (attn * r + proj + mlp) + io
    return {
        "note": "模型侧折算（按 stats.kv_len 折算 attn 项），非实测",
        "kv_ratio": r,
        "attn_dense_flops": attn * L,
        "model_dense_flops": dense,
        "model_sparse_flops": sparse,
        "reachable_ratio": (dense / sparse) if sparse > 0 else None,
    }


# ------------------------------------------------------------------ measure
def _finite_num(v, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _timeit_reps(fn, reps: int) -> list[float]:
    out = []
    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def profile_arm(acfgs, seq_lens, batch: int = 8, reps: int = 3,
                mcfg: ModelConfig | None = None, threads: int = 1,
                warmup: int = 1, task: str = "mqar", verbose: bool = True) -> list[dict]:
    """逐 (arm, seq) 测时间/内存/stats/折算 FLOPs，返回 list[dict]。"""
    torch.set_num_threads(max(1, int(threads)))
    from .model import TinyLM                       # 延迟导入: 由另一 teammate 维护

    if isinstance(acfgs, (AttnConfig, str)):
        acfgs = [acfgs]
    if isinstance(seq_lens, int):
        seq_lens = [seq_lens]
    reps = max(1, int(reps))
    base = mcfg or ModelConfig()
    results: list[dict] = []
    gen = torch.Generator().manual_seed(0)

    for spec in acfgs:
        for seq in seq_lens:
            mcfg_i = dataclasses.replace(base, seq_len=int(seq))
            try:
                acfg = (AttnConfig.from_arm(spec, d_model=mcfg_i.d_model,
                                            n_heads=mcfg_i.n_heads)
                        if isinstance(spec, str) else spec)
                torch.manual_seed(0)                 # 同一初始化, 跨 arm/seq 可比
                model = TinyLM(mcfg_i, acfg)
                model.eval()
                row = _measure_one(model, mcfg_i, acfg, batch, reps, warmup, task, gen)
            except Exception as e:                   # 单个配置失败不中断整套扫描
                row = {"arm": (spec if isinstance(spec, str) else acfg.arm),
                       "seq": int(seq), "batch": int(batch), "error": f"{type(e).__name__}: {e}"}
            results.append(row)
            if verbose:
                print(_fmt_row(row))
    return results


def _measure_one(model, mcfg: ModelConfig, acfg: AttnConfig, batch: int, reps: int,
                 warmup: int, task: str, gen) -> dict:
    b = make_batch(task, batch, mcfg.seq_len, mcfg.vocab, gen)
    x, y, mask = b["x"], b["y"], b["mask"]

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        logits, _ = model(x, return_stats=False)
        F.cross_entropy(logits[mask], y[mask]).backward()

    # 峰值内存：从"模型已就绪"基线起，采样首次前向+反向的高水位（激活首次分配）
    rss_base = rss_mb()
    with _rss_peak() as peak:
        fwd_bwd()
        for _ in range(max(0, warmup)):
            fwd_bwd()
    peak_mb = (peak[0] - rss_base) if (peak[0] is not None and rss_base is not None) else None
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        fwd_reps = _timeit_reps(lambda: model(x, return_stats=False), reps)
    fwb_reps = _timeit_reps(fwd_bwd, reps)
    model.zero_grad(set_to_none=True)

    with torch.inference_mode():
        logits, st = model(x, return_stats=True)
        loss = float(F.cross_entropy(logits[mask], y[mask]))
    st = {k: _finite_num(v) for k, v in (st or {}).items()}
    kv_len = _finite_num(st.get("kv_len"), float(mcfg.seq_len))
    flops = _flops_estimate(mcfg, batch, kv_len)

    fwb_med = statistics.median(fwb_reps)
    fwd_med = statistics.median(fwd_reps)
    row = {
        "arm": acfg.arm,
        "seq": int(mcfg.seq_len),
        "batch": int(batch),
        "reps": reps,
        "params": int(sum(p.numel() for p in model.parameters())),
        "fwd_ms": fwd_med * 1e3,
        "fwd_ms_min": min(fwd_reps) * 1e3,
        "fwd_bwd_ms": fwb_med * 1e3,
        "fwd_bwd_ms_min": min(fwb_reps) * 1e3,
        "fwd_bwd_ms_max": max(fwb_reps) * 1e3,
        "fwd_bwd_ms_std": (statistics.pstdev(fwb_reps) if len(fwb_reps) > 1 else 0.0) * 1e3,
        "peak_mem_mb": peak_mb,
        "rss_mb": rss_mb(),
        "mem_source": "host-rss-sampled" if peak_mb is not None else None,
        "mem_note": "host RSS 增量（含激活首次分配），非 allocator 精确值",
        "eval_loss": loss,
        "stats": st,
        "flops": flops,
        "tok_per_s": batch * mcfg.seq_len / max(1e-9, fwd_med),
        "steps_per_s": 1.0 / max(1e-9, fwb_med),
    }
    row["dense_gflops"] = flops["model_dense_flops"] / max(1e-9, fwd_med * 1e9)
    return row


def _fmt_row(r: dict) -> str:
    if "error" in r:
        return f"{r['arm']:>3} seq {r['seq']:>4}  ERROR {r['error']}"
    st = r["stats"]
    return (f"{r['arm']:>3} seq {r['seq']:>4}  fwd {r['fwd_ms']:8.2f}ms"
            f"  fwd+bwd {r['fwd_bwd_ms']:8.2f}ms  peak {_fmt_mem(r['peak_mem_mb'])}"
            f"  eff_part {_fmt_num(st.get('eff_part'))}"
            f"  kv_len {_fmt_num(st.get('kv_len'))}"
            f"  GFLOP/s {r['dense_gflops']:6.1f}")


def _fmt_num(v) -> str:
    return "   n/a" if v is None else f"{v:6.2f}"


def _fmt_mem(v) -> str:
    return "   n/a" if v is None else f"{v:6.1f}MB"


# ------------------------------------------------------------------ CLI
def _csv(v) -> list[str]:
    return [p.strip() for p in str(v).split(",") if p.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gravity_lab.cost")
    p.add_argument("--arms", default="A0,A3,A4,A5")
    p.add_argument("--seqs", default="64,128,256,512")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--task", default="mqar", choices=("mqar", "noise_recall"))
    p.add_argument("--out", default=os.path.join("results", "cost.json"))
    return p


def main(argv=None) -> int:
    a = _build_parser().parse_args(argv)
    arms = _csv(a.arms)
    bad = [x for x in arms if x not in ARMS]
    if bad:
        raise SystemExit(f"未知 arm {bad}，可选 {sorted(ARMS)}")
    seqs = [int(s) for s in _csv(a.seqs)]
    mcfg = ModelConfig(vocab=a.vocab, d_model=a.d_model, n_layers=a.layers, n_heads=a.heads)
    rows = profile_arm(arms, seqs, batch=a.batch, reps=a.reps, mcfg=mcfg,
                       threads=a.threads, task=a.task)
    payload = {"config": {"arms": arms, "seqs": seqs, "reps": a.reps, "batch": a.batch,
                          "threads": a.threads, "task": a.task,
                          "model": dataclasses.asdict(mcfg)},
               "results": rows}
    path = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    n_err = sum(1 for r in rows if "error" in r)
    print(f"wrote {path}  ({len(rows)} rows, {n_err} error)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
