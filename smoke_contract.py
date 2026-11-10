"""Contract smoke test. 任何人改完代码必须跑通：python smoke_contract.py"""
import math
import sys

import torch
import torch.nn.functional as F

from gravity_lab.config import ARMS, AttnConfig, ModelConfig
from gravity_lab.attention import build_attention
from gravity_lab.model import TinyLM
from gravity_lab.tasks import make_batch

FAIL = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ("  " + extra if extra else ""))
    if not cond:
        FAIL.append(name)


def manual_softmax_attn(x, Wq, Wk, Wv, n_heads, causal=True):
    B, N, D = x.shape
    dh = D // n_heads
    q, k, v = x @ Wq.T, x @ Wk.T, x @ Wv.T
    q, k, v = [t.view(B, N, n_heads, dh).transpose(1, 2) for t in (q, k, v)]
    s = q @ k.transpose(-1, -2) / math.sqrt(dh)
    if causal:
        m = torch.triu(torch.ones(N, N, dtype=torch.bool), 1)
        s = s.masked_fill(m, float("-inf"))
    w = s.softmax(-1)
    out = (w @ v).transpose(1, 2).reshape(B, N, D)
    return out


def test_a0_equivalence():
    torch.manual_seed(0)
    cfg = AttnConfig.from_arm("A0", d_model=32, n_heads=4)
    attn = build_attention(cfg)
    x = torch.randn(2, 16, 32)
    out, st = attn(x, return_stats=True)
    ref = manual_softmax_attn(x, attn.W_q.weight, attn.W_k.weight, attn.W_v.weight, 4)
    err = (out - ref).abs().max().item()
    check("A0 == 原生 softmax attention (1e-6)", err < 1e-6, f"maxerr={err:.2e}")
    check("A0 stats keys", {"eff_part", "sparsity", "kv_len"} <= set(st), str(sorted(st)))


def test_mass_is_key_side_only():
    """A1 的查询侧质量必须完全不影响输出（SPEC 物理约束 1）。"""
    torch.manual_seed(1)
    cfg = AttnConfig.from_arm("A1", d_model=32, n_heads=4)
    attn = build_attention(cfg)
    x = torch.randn(2, 12, 32)
    out0, _ = attn(x)
    with torch.no_grad():
        attn.mass_bias.add_(1.0)   # 只抬高 key 侧 log 质量 = 乘常数，softmax 不变
    out1, _ = attn(x)
    err = (out0 - out1).abs().max().item()
    check("A1 全局常数平移不改变输出", err < 1e-6, f"maxerr={err:.2e}")


def test_merge_conservation():
    try:
        from gravity_lab.merging import merge_kv_mass_conserving
    except Exception as e:  # noqa: BLE001
        check("merging 模块可导入", False, str(e))
        return
    torch.manual_seed(2)
    mass = torch.rand(2, 3, 16) + 0.1
    kv = torch.randn(2, 3, 16, 8)
    out = merge_kv_mass_conserving(kv, kv, mass, block=4)
    m_in = mass.sum(-1)
    m_out = out["mass"].sum(-1)
    err = (m_in - m_out).abs().max().item()
    check("合并质量守恒 <1e-5", err < 1e-5, f"relerr={err / m_in.mean():.2e}")


def test_model_and_task():
    mcfg = ModelConfig(vocab=64, seq_len=32, d_model=32, n_layers=2, n_heads=4)
    for arm in ARMS:
        acfg = AttnConfig.from_arm(arm, d_model=32, n_heads=4)
        model = TinyLM(mcfg, acfg)
        x = torch.randint(0, 64, (2, 32))
        logits, st = model(x, return_stats=True)
        check(f"{arm} forward", tuple(logits.shape) == (2, 32, 64), str(tuple(logits.shape)))
    b = make_batch("mqar", 4, 64, 64, torch.Generator().manual_seed(0))
    check("make_batch keys", {"x", "y", "mask"} <= set(b), str(sorted(b)))
    check("mqar loss mask 非空", bool(b["mask"].any()))


if __name__ == "__main__":
    test_a0_equivalence()
    test_mass_is_key_side_only()
    test_merge_conservation()
    test_model_and_task()
    print("\n%d failure(s)" % len(FAIL))
    sys.exit(1 if FAIL else 0)
