"""tests_merging.py — merging.py 的不变量测试。python tests_merging.py 全 PASS 才算通过。

断言的是不变量，不是注释：
  (a) 质量守恒，随机 20 组，相对误差 < 1e-5
  (b) g=1 退化为恒等
  (c) 全 0 质量输入不产生 NaN（assign 两路：默认与外部传入）
  (d) straight_through_gate 前向严格 {0,1}、反向梯度 1
  (e) commit_escape_mask：mass_ema 全相等全存活；单点远低于均值只死它
  (f) 与逐块手算参考一致（抓 reshape/块布局错误）
"""
import math
import sys
import time

import torch

from gravity_lab.merging import (
    apply_dead,
    commit_escape_mask,
    merge_kv_mass_conserving,
    straight_through_gate,
    update_ema,
)

FAIL = []
REL_TOL = 1e-5


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ("  " + extra if extra else ""))
    if not cond:
        FAIL.append(name)


def block_ref(k, v, mass, g):
    """逐块手算参考（仅测试用）。

    c_m = sum m k / sum m；pi = softmax(k.c/sqrt(dh))；
    k̄ = sum pi m k / sum pi m；块质量 = sum m（守恒）。
    """
    B, H, N, dh = k.shape
    M = N // g
    ks, vs, ms, ps = [], [], [], []
    for b in range(B):
        kb, vb, mb, pb = [], [], [], []
        for h in range(H):
            kbh, vbh, mbh, pbh = [], [], [], []
            for blk in range(M):
                sl = slice(blk * g, (blk + 1) * g)
                mj = mass[b, h, sl].double()
                kj = k[b, h, sl].double()
                vj = v[b, h, sl].double()
                c = (mj[:, None] * kj).sum(0) / mj.sum().clamp_min(1e-30)
                logits = (kj @ c) / math.sqrt(dh)
                pi = torch.softmax(logits, -1)
                w = pi * mj
                den = w.sum().clamp_min(1e-30)
                kbh.append((w[:, None] * kj).sum(0) / den)
                vbh.append((w[:, None] * vj).sum(0) / den)
                mbh.append(mj.sum())
                pbh.append(pi)
            kb.append(torch.stack(kbh)); vb.append(torch.stack(vbh))
            mb.append(torch.stack(mbh)); pb.append(torch.stack(pbh))
        ks.append(torch.stack(kb)); vs.append(torch.stack(vb))
        ms.append(torch.stack(mb)); ps.append(torch.stack(pb))
    return (torch.stack(ks).float(), torch.stack(vs).float(),
            torch.stack(ms).float(), torch.stack(ps).float())


# ---------------------------------------------------------------- (a) 守恒
def test_mass_conservation():
    worst = 0.0
    worst_case = ""
    cases = []
    for i in range(20):
        torch.manual_seed(1000 + i)
        B, H = 2, 3
        N = [4, 8, 16, 32][i % 4]
        g = [1, 2, 4, 8][i % 4]
        dh = [1, 4, 8, 16][i % 4]
        scale = [1.0, 1e-3, 1e3, 1e-6][(i // 4) % 4]
        mass = (torch.rand(B, H, N) + 0.05) * scale
        k = torch.randn(B, H, N, dh) * [1.0, 10.0, 1e-2, 1.0][(i // 2) % 4]
        v = torch.randn(B, H, N, dh)
        cases.append((B, H, N, g, dh, mass, k, v))

    for i, (B, H, N, g, dh, mass, k, v) in enumerate(cases):
        M = N // g
        # 默认 assign
        out = merge_kv_mass_conserving(k, v, mass, block=g)
        m_in = mass.sum(-1)
        m_out = out["mass"].sum(-1)
        rel = ((m_in - m_out).abs() / m_in.abs().clamp_min(1e-30)).max().item()
        if rel > worst:
            worst, worst_case = rel, f"default i={i} B{B} H{H} N{N} g{g} dh{dh}"
        # 外部 assign
        torch.manual_seed(2000 + i)
        pi = torch.softmax(torch.randn(B, H, M, g) * 3.0, -1)
        out2 = merge_kv_mass_conserving(k, v, mass, block=g, assign=pi)
        rel2 = ((m_in - out2["mass"].sum(-1)).abs() / m_in.abs().clamp_min(1e-30)).max().item()
        if rel2 > worst:
            worst, worst_case = rel2, f"assign i={i} B{B} H{H} N{N} g{g} dh{dh}"

    check("(a) 质量守恒 20 组 <1e-5", worst < REL_TOL, f"max_rel={worst:.3e} [{worst_case}]")

    out = merge_kv_mass_conserving(cases[0][6], cases[0][7], cases[0][5], block=1)
    abs_err = (cases[0][5].sum(-1) - out["mass"].sum(-1)).abs().max().item()
    check("(a) 绝对守恒 g=1", abs_err < 1e-6, f"abs={abs_err:.3e}")


# ---------------------------------------------------------------- (b) g=1 恒等
def test_identity_g1():
    torch.manual_seed(7)
    B, H, N, dh = 2, 4, 16, 8
    mass = torch.rand(B, H, N) + 0.1
    k = torch.randn(B, H, N, dh)
    v = torch.randn(B, H, N, dh)
    out = merge_kv_mass_conserving(k, v, mass, block=1)
    ek = (out["k"] - k).abs().max().item()
    ev = (out["v"] - v).abs().max().item()
    em = (out["mass"] - mass).abs().max().item()
    check("(b) g=1 k 恒等", torch.allclose(out["k"], k, rtol=1e-5, atol=1e-6), f"maxerr={ek:.2e}")
    check("(b) g=1 v 恒等", torch.allclose(out["v"], v, rtol=1e-5, atol=1e-6), f"maxerr={ev:.2e}")
    check("(b) g=1 mass 恒等", torch.allclose(out["mass"], mass, rtol=0, atol=1e-6), f"maxerr={em:.2e}")
    check("(b) g=1 assign 全 1", bool(torch.equal(out["assign"], torch.ones_like(out["assign"]))))
    check("(b) g=1 形状不变", tuple(out["k"].shape) == (B, H, N, dh), str(tuple(out["k"].shape)))


# ---------------------------------------------------------------- (c) 零质量
def test_zero_mass_no_nan():
    B, H, N, dh = 2, 3, 16, 8
    z = torch.zeros(B, H, N)
    k = torch.randn(B, H, N, dh)
    v = torch.randn(B, H, N, dh)

    out = merge_kv_mass_conserving(k, v, z, block=4)
    bad = [x for x in ("k", "v", "mass", "assign") if not torch.isfinite(out[x]).all()]
    check("(c) 全 0 质量默认 assign 无 NaN/Inf", not bad, f"bad={bad}")
    check("(c) 全 0 质量 mass 全 0", bool((out["mass"] == 0).all()))
    s = out["assign"].sum(-1)
    check("(c) 全 0 质量 assign 每块和为 1", bool(torch.allclose(s, torch.ones_like(s))),
          f"min={s.min().item():.6f} max={s.max().item():.6f}")

    pi = torch.zeros(B, H, N // 4, 4)
    out2 = merge_kv_mass_conserving(k, v, z, block=4, assign=pi)
    bad2 = [x for x in ("k", "v", "mass", "assign") if not torch.isfinite(out2[x]).all()]
    check("(c) 全 0 质量 + 全 0 assign 无 NaN/Inf", not bad2, f"bad={bad2}")

    # 单块质量为 0、其它块正常
    m = torch.rand(B, H, N) + 0.5
    m[:, :, :4] = 0.0
    out3 = merge_kv_mass_conserving(k, v, m, block=4)
    check("(c) 单块零质量不污染其它块",
          bool(torch.isfinite(out3["k"]).all()) and bool((out3["mass"][:, :, 0] == 0).all()))

    # 不整除 + 零质量
    out4 = merge_kv_mass_conserving(k[:, :, :10], v[:, :, :10], z[:, :, :10], block=4)
    check("(c) 不整除 + 零质量无 NaN", bool(torch.isfinite(out4["k"]).all()) and out4["pad"] == 2)


# ---------------------------------------------------------------- (d) STE
def test_straight_through_gate():
    prob = torch.tensor([0.0, 0.02, 0.49, 0.5, 0.51, 0.98, 1.0], requires_grad=True)
    y = straight_through_gate(prob).sum()
    fwd = straight_through_gate(prob).detach()
    check("(d) 前向严格 {0,1}", bool(((fwd == 0) | (fwd == 1)).all()), str(fwd.tolist()))
    check("(d) 前向 == round(prob)", bool(torch.equal(fwd, torch.round(prob.detach()))))
    y.backward()
    g = prob.grad
    check("(d) 反向梯度全 1", bool(torch.equal(g, torch.ones_like(g))), str(g.tolist()))

    # 4D 形状 + 非 0/1 输入
    torch.manual_seed(3)
    p = torch.rand(2, 3, 4, 5, requires_grad=True)
    out = straight_through_gate(p)
    check("(d) 4D 前向严格 {0,1}", bool(((out == 0) | (out == 1)).all()))
    out.sum().backward()
    check("(d) 4D 反向梯度全 1", bool(torch.equal(p.grad, torch.ones_like(p))))


# ---------------------------------------------------------------- (e) 逃逸
def test_commit_escape_mask():
    m = torch.ones(2, 3, 16)
    alive = commit_escape_mask(m, 0.5)
    check("(e) 全相等 -> 全存活", bool(alive.all()) and alive.dtype == torch.bool,
          f"alive={int(alive.sum())}/{alive.numel()}")

    m2 = torch.ones(2, 3, 16)
    m2[:, :, 5] = 1e-3
    alive2 = commit_escape_mask(m2, 0.5)
    dead = ~alive2
    check("(e) 单点远低于均值只死它",
          bool((dead.sum(-1) == 1).all()) and bool(dead[:, :, 5].all()),
          f"dead_per_row={dead.sum(-1).flatten().tolist()} idx={dead.nonzero()[:, -1].tolist()}")

    m3 = torch.zeros(2, 3, 16)
    check("(e) 全 0 EMA -> 全存活且无 NaN", bool(commit_escape_mask(m3, 0.5).all()))
    m4 = torch.ones(2, 3, 16)
    check("(e) tau=1.0 边界 -> 全存活", bool(commit_escape_mask(m4, 1.0).all()))
    m5 = torch.ones(2, 3, 16) * 4.0
    check("(e) 常数 EMA 与 tau 无关", bool(commit_escape_mask(m5, 0.9).all()))

    # 两个低点
    m6 = torch.ones(2, 3, 16)
    m6[:, :, 1] = 0.0
    m6[:, :, 9] = 0.0
    alive6 = commit_escape_mask(m6, 0.5)
    check("(e) 两个低点都死", int((~alive6).sum(-1).min()) == 2, f"dead={int((~alive6).sum())}")

    # 掩码维度写回 logits
    logits = torch.zeros(2, 3, 16, 16)
    out = apply_dead(logits, ~alive2)
    ok = (out[:, :, :, 5] == -1e9).all() and (out[:, :, :, 4] == 0).all()
    check("(e) apply_dead 沿键轴屏蔽", bool(ok))
    out2 = apply_dead(logits, torch.zeros(16, dtype=torch.bool), neg=-5.0)
    check("(e) apply_dead 无 dead 时不改", bool((out2 == 0).all()))
    out3 = apply_dead(logits, ~alive2[0:1])
    check("(e) apply_dead 3D dead 广播", tuple(out3.shape) == tuple(logits.shape)
          and bool((out3[0, 0, :, 5] == -1e9).all()))
    out4 = apply_dead(torch.zeros(16), torch.tensor([True, False] + [False] * 14))
    check("(e) apply_dead 1D dead 广播", out4[0] == -1e9 and out4[1] == 0.0)


# ---------------------------------------------------------------- (f) 逐块参考
def test_against_block_reference():
    torch.manual_seed(11)
    B, H, N, dh, g = 2, 2, 12, 4, 4
    mass = torch.rand(B, H, N) + 0.2
    k = torch.randn(B, H, N, dh)
    v = torch.randn(B, H, N, dh)
    out = merge_kv_mass_conserving(k, v, mass, block=g)
    rk, rv, rm, rp = block_ref(k, v, mass, g)
    ek = (out["k"] - rk).abs().max().item()
    ev = (out["v"] - rv).abs().max().item()
    em = (out["mass"] - rm).abs().max().item()
    ep = (out["assign"] - rp).abs().max().item()
    check("(f) 与逐块参考一致 k", ek < 1e-5, f"maxerr={ek:.2e}")
    check("(f) 与逐块参考一致 v", ev < 1e-5, f"maxerr={ev:.2e}")
    check("(f) 与逐块参考一致 mass", em < 1e-5, f"maxerr={em:.2e}")
    check("(f) 与逐块参考一致 assign", ep < 1e-5, f"maxerr={ep:.2e}")
    check("(f) assign 非负且每块和为 1",
          bool((out["assign"] >= 0).all())
          and bool(torch.allclose(out["assign"].sum(-1), torch.ones_like(out["mass"]))))


# ---------------------------------------------------------------- 形状/补齐/梯度
def test_pad_and_shapes():
    torch.manual_seed(13)
    B, H, N, dh, g = 2, 2, 10, 8, 4
    mass = torch.rand(B, H, N) + 0.1
    k = torch.randn(B, H, N, dh)
    v = torch.randn(B, H, N, dh)
    out = merge_kv_mass_conserving(k, v, mass, block=g)
    check("(f) 不整除 pad 记录", out["pad"] == 2, f"pad={out['pad']}")
    check("(f) 不整除 M=ceil(N/g)", tuple(out["mass"].shape) == (B, H, 3), str(tuple(out["mass"].shape)))
    tail = (mass[:, :, 8] + mass[:, :, 9])
    check("(f) 尾块质量 = 尾部实键之和", torch.allclose(out["mass"][:, :, 2], tail, rtol=1e-6),
          f"maxerr={(out['mass'][:, :, 2] - tail).abs().max().item():.2e}")
    check("(f) 输出 dtype float32",
          all(out[x].dtype == torch.float32 for x in ("k", "v", "mass", "assign")))

    kr = torch.randn(B, H, 8, dh, requires_grad=True)
    kr2 = kr.detach().clone().requires_grad_(True)
    mr = (torch.rand(B, H, 8) + 0.1).requires_grad_(True)
    o = merge_kv_mass_conserving(kr2, kr, mr, block=4)
    (o["k"].sum() + o["v"].sum() + o["mass"].sum()).backward()
    gk = kr2.grad is not None and bool(torch.isfinite(kr2.grad).all())
    gv = kr.grad is not None and bool(torch.isfinite(kr.grad).all())
    gm = mr.grad is not None and bool(torch.isfinite(mr.grad).all())
    check("(f) 梯度可达且有限", gk and gv and gm, f"k={gk} v={gv} m={gm}")
    check("(f) 无参数形状检查", _raises(lambda: merge_kv_mass_conserving(kr2, kr, mr[:, :, :3], block=4)))


def _raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def test_update_ema():
    prev = torch.tensor([1.0, 2.0])
    new = torch.tensor([0.0, 4.0])
    out = update_ema(prev, new, 0.9)
    exp = 0.9 * prev + 0.1 * new
    check("update_ema 公式", torch.allclose(out, exp, rtol=0, atol=1e-7), str(out.tolist()))
    check("update_ema decay=0 取 new", torch.equal(update_ema(prev, new, 0.0), new))
    p = torch.ones(3, requires_grad=True)
    check("update_ema 输出已 detach", not update_ema(p, new[:2].new_ones(3), 0.5).requires_grad)
    check("update_ema prev=None", torch.equal(update_ema(None, new, 0.9), new))


def bench_compression():
    torch.manual_seed(21)
    B, H, N, dh, g = 2, 4, 1024, 64, 4
    mass = torch.rand(B, H, N) + 0.1
    k = torch.randn(B, H, N, dh)
    v = torch.randn(B, H, N, dh)
    for _ in range(2):
        merge_kv_mass_conserving(k, v, mass, block=g)
    it = 10
    t0 = time.perf_counter()
    for _ in range(it):
        out = merge_kv_mass_conserving(k, v, mass, block=g)
    dt = (time.perf_counter() - t0) / it * 1e3
    ratio = N / out["mass"].shape[-1]
    print(f"\n[bench] B{B} H{H} N{N} dh{dh} g{g}: "
          f"kv_len {N} -> {out['mass'].shape[-1]}  compression={ratio:.1f}x  "
          f"{dt:.2f} ms/call")
    print(f"[bench] g=8: ", end="")
    o8 = merge_kv_mass_conserving(k, v, mass, block=8)
    print(f"kv_len {N} -> {o8['mass'].shape[-1]}  compression={N / o8['mass'].shape[-1]:.1f}x")
    m_in = mass.sum(-1)
    err = ((m_in - out["mass"].sum(-1)).abs() / m_in).max().item()
    print(f"[bench] 大 shape 守恒 rel_err={err:.3e}")
    conc = out["assign"].max(-1).values.mean().item()
    print(f"[bench] assign 平均最大权重 (g={g}) = {conc:.4f} (均匀={1.0 / g:.4f})")


if __name__ == "__main__":
    test_mass_conservation()
    test_identity_g1()
    test_zero_mass_no_nan()
    test_straight_through_gate()
    test_commit_escape_mask()
    test_against_block_reference()
    test_pad_and_shapes()
    test_update_ema()
    bench_compression()
    print("\n%d failure(s)" % len(FAIL))
    sys.exit(1 if FAIL else 0)
