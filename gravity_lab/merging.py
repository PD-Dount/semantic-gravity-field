"""A4 / A5 primitives: 不可逆逃逸 + 质量守恒块合并 + STE 门控。

全部纯函数：无 nn.Module、无全局可变状态、float32、批量张量运算。
块合并只做 reshape + einsum，不做 Python 逐块循环。

块布局约定（g = block）：键轴 N 重排为 [M, g]，M = ceil(N / g)；
N 不能整除 g 时在尾部补零质量键，补的个数记在返回值的 ``pad``。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# 质量归一化分母下界：保证全 0 质量输入不产生 NaN / Inf。
EPS = 1e-12

__all__ = [
    "merge_kv_mass_conserving",
    "update_ema",
    "commit_escape_mask",
    "apply_dead",
    "straight_through_gate",
]


def _normalize_block_assign(assign, B, H, M, g, dtype):
    """assign -> [B,H,M,g]，每块权重归一化到 1（退化行回退均匀分布）。

    归一化是质量守恒的必要条件：块内 sum_j pi_j == 1 时
    sum_m M_m == sum_j m_j 恒成立。
    """
    if assign.dim() == 4 and tuple(assign.shape) == (B, H, M, g):
        pi = assign
    elif assign.numel() == B * H * M * g:
        pi = assign.reshape(B, H, M, g)
    elif assign.numel() == M * g:
        pi = assign.reshape(1, 1, M, g).expand(B, H, M, g)
    else:
        raise ValueError(
            f"assign shape {tuple(assign.shape)} 与 (B,H,M,g)=({B},{H},{M},{g}) 不匹配"
        )
    pi = pi.to(dtype)
    s = pi.sum(-1, keepdim=True)
    uniform = torch.full_like(pi, 1.0 / g)
    return torch.where(s > EPS, pi / s.clamp_min(EPS), uniform)


def merge_kv_mass_conserving(k, v, mass, block=4, assign=None):
    """块内质量守恒合并（SPEC A5）。

    Args:
        k, v: [B, H, N, dh] float32。合并后为 [B, H, M, dh]。
        mass: [B, H, N] 非负。合并后为 [B, H, M]；N 不能整除 block 时尾部补 0 质量。
        block: 块大小 g >= 1。g == 1 时退化为恒等。
        assign: 可选的块内权重，[B,H,M,g] 或等元素数的可 reshape 形状；None 时
            用 c_m = 块内质量加权均值、pi = softmax_j(k_j·c_m / sqrt(dh))。
            传入的权重会被重新归一化。

    块质量取 M_m = sum_j m_j（守恒），assign 只决定 k̄/v̄ 的相对加权。
    SPEC A5 的 "M_m = sum_j pi_jm m_j" 与 "sum_m M_m = sum_j m_j" 在 pi 逐块
    归一化时不能同时成立（前者给的是块内质量均值），此处以守恒为准。

    Returns:
        dict:
            'k'      [B,H,M,dh]
            'v'      [B,H,M,dh]
            'mass'   [B,H,M]       块内质量之和，sum(mass,-1) 逐行守恒
            'assign' [B,H,M,g]     块内权重，每块和为 1
            'pad'    int           尾部补的零质量键个数 (M*g - N)

    不变量：sum(out['mass'], -1) == sum(mass, -1)。
    """
    if k.dim() != 4:
        raise ValueError(f"k 需为 [B,H,N,dh]，实际 {tuple(k.shape)}")
    if tuple(v.shape) != tuple(k.shape):
        raise ValueError(f"v {tuple(v.shape)} 与 k {tuple(k.shape)} 形状不一致")
    B, H, N, dh = (int(s) for s in k.shape)
    if tuple(mass.shape) != (B, H, N):
        raise ValueError(f"mass 需为 {(B, H, N)}，实际 {tuple(mass.shape)}")
    g = int(block)
    if g < 1:
        raise ValueError(f"block 需 >= 1，实际 {g}")

    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    mf = mass.to(torch.float32)

    M = (N + g - 1) // g
    pad = M * g - N
    if N == 0:
        return {
            "k": kf.reshape(B, H, 0, dh),
            "v": vf.reshape(B, H, 0, dh),
            "mass": mf.reshape(B, H, 0),
            "assign": mf.new_zeros(B, H, 0, g),
            "pad": 0,
        }
    if pad:
        kf = F.pad(kf, (0, 0, 0, pad))
        vf = F.pad(vf, (0, 0, 0, pad))
        mf = F.pad(mf, (0, pad))

    kb = kf.reshape(B, H, M, g, dh)
    vb = vf.reshape(B, H, M, g, dh)
    mb = mf.reshape(B, H, M, g)

    pi = None
    if assign is not None:
        pi = _normalize_block_assign(assign, B, H, M, g, mf.dtype)
    if pi is None:
        block_mass = mb.sum(-1, keepdim=True)
        centroid = torch.einsum("bhmg,bhmgd->bhmd", mb, kb) / block_mass.clamp_min(EPS)
        sim = torch.einsum("bhmgd,bhmd->bhmg", kb, centroid) * (dh ** -0.5)
        pi = _normalize_block_assign(sim.softmax(-1), B, H, M, g, mf.dtype)

    w = pi * mb                                   # [B,H,M,g]  pi_jm * m_j
    wsum = w.sum(-1)                              # [B,H,M]   sum_j pi_jm m_j
    denom = wsum.clamp_min(EPS).unsqueeze(-1)     # 全 0 质量块 -> 0/eps = 0，不出 NaN
    k2 = torch.einsum("bhmg,bhmgd->bhmd", w, kb) / denom
    v2 = torch.einsum("bhmg,bhmgd->bhmd", w, vb) / denom
    mass2 = mb.sum(-1)                            # 守恒：块质量 = 块内质量之和

    return {"k": k2, "v": v2, "mass": mass2, "assign": pi, "pad": pad}


def update_ema(prev, new, decay):
    """mass_ema <- decay * prev + (1 - decay) * new，detach 输出（SPEC A4）。"""
    new = new.detach()
    if prev is None:
        return new.clone()
    return (float(decay) * prev.detach() + (1.0 - float(decay)) * new)


def commit_escape_mask(mass_ema, tau):
    """存活判据：mass_ema >= tau * mean(mass_ema, -1)。True = 存活。"""
    if mass_ema.dim() < 1:
        raise ValueError("mass_ema 需至少 1 维")
    thr = float(tau) * mass_ema.mean(-1, keepdim=True)
    return mass_ema >= thr


def apply_dead(logits, dead, neg=-1e9):
    """把 dead=True 的键位置为 neg。末维必须是键轴，其余维度按广播对齐。"""
    mask = dead
    if not mask.dtype == torch.bool:
        raise TypeError(f"dead 需为 bool，实际 {mask.dtype}")
    if mask.dim() == logits.dim() - 1:
        mask = mask.unsqueeze(-2)
    elif mask.dim() < logits.dim() - 1:
        mask = mask.reshape((1,) * (logits.dim() - mask.dim()) + tuple(mask.shape))
    if mask.shape[-1] != logits.shape[-1]:
        raise ValueError(f"dead 末维 {mask.shape[-1]} 与 logits 键轴 {logits.shape[-1]} 不一致")
    return logits.masked_fill(mask.to(logits.device), neg)


def straight_through_gate(prob):
    """前向 round 到 {0,1}，反向传 prob 的梯度（d out/d prob == 1）。"""
    hard = torch.round(prob)
    return prob + (hard - prob).detach()
