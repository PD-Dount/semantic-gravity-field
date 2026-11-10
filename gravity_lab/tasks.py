"""MQAR / noise_recall batch generators (pure CPU, vectorized, rng-explicit).

布局（单行），N = seq_len：
    [ kv 区: (k1,v1) ... (kL,vL) 与 D 个噪声 token 交错 ] [ q1,a1 ] [ q2,a2 ] ...
    L = 关联对数，Q = 查询对数 (= L)，D = 噪声数（mqar: 0），N = 2L + D + 2Q

key 取 vocab 前半 [0, V/2)，value 取后半 [V/2, V)，两段 id 不重叠；q_i 是打乱顺序抽到的
key，a_i 是配对 value。noise_recall 的噪声 id 取自当行未用作 key/value 的空闲 id，
在 id 空间里与正常 token 无法区分，只能靠"从未被查询"淘汰。

mask 标在 q_i 所在位置：因果 LM 的 next-token 目标 y[pos] = x[pos+1] = a_i，
答案 token 不在输入前缀里，模型无法 self-attend 直接抄到答案。

随机数只来自入参 rng（torch.Generator 或 numpy Generator/RandomState），不读全局状态；
整个 batch 一次生成，无逐样本 Python 循环。
"""
from __future__ import annotations

import random

import numpy as np
import torch

TASKS = ("mqar", "noise_recall")
COVER_L_MAX = 32          # L <= 32 时保证每个 key 至少被查询一次
_NOISE_PER_PAIR = 1.5     # noise_recall: D ~= 1.5 * L
_KEY_FRAC = 0.5           # key 占 vocab 前 1/2


# ------------------------------------------------------------------ rng
def _coerce_rng(rng):
    if rng is None:
        return torch.Generator().manual_seed(random.SystemRandom().randrange(1 << 31))
    return rng


def _uniform(rng, shape) -> np.ndarray:
    """[0,1) float32 数组，只从 rng 取数。"""
    if isinstance(rng, torch.Generator):
        return torch.rand(shape, generator=rng, dtype=torch.float32).numpy()
    if type(rng).__module__.split(".")[0] == "numpy":
        draw = getattr(rng, "random", None) or rng.random_sample
        return np.asarray(draw(shape), dtype=np.float32)
    raise TypeError(
        f"rng 需为 torch.Generator 或 numpy Generator/RandomState，实际 {type(rng)!r}"
    )


def _block(u: np.ndarray, o: int, w: int) -> np.ndarray:
    """取 [o, o+w) 列并拷贝成连续块（argsort 在连续块上快 3x）。"""
    if w <= 0:
        return np.zeros((u.shape[0], 0), dtype=np.float32)
    return np.ascontiguousarray(u[:, o:o + w])


def _rand_idx(u: np.ndarray, n: int) -> np.ndarray:
    """由 [0,1) 均匀数组得到 [0,n) int64 下标。"""
    return np.minimum((u * n).astype(np.int64), n - 1)


# ------------------------------------------------------------------ layout
def split_seq(task: str, seq_len: int) -> dict:
    """返回 {'pairs': L, 'queries': Q, 'noise': D}，满足 2L + D + 2Q == seq_len。"""
    seq_len = int(seq_len)
    if seq_len < 8:
        raise ValueError(f"seq_len 需 >= 8，实际 {seq_len}")
    if task == "mqar":
        if seq_len % 4:
            raise ValueError(
                f"task=mqar 需 seq_len %% 4 == 0（N = 2L + 2Q, L = Q = N//4），实际 {seq_len}"
            )
        L = seq_len // 4
        return {"pairs": L, "queries": L, "noise": 0}
    if task == "noise_recall":
        L = max(1, int(seq_len // (4 + _NOISE_PER_PAIR)))
        D = seq_len - 4 * L
        if D <= 0:
            raise ValueError(f"task=noise_recall: seq_len={seq_len} 太短，无法容纳噪声 token")
        return {"pairs": L, "queries": L, "noise": D}
    if task == "needle":
        if (seq_len - 2) % 2 or seq_len < 8:
            raise ValueError(f"task=needle 需 seq_len 为 >=8 的偶数，实际 {seq_len}")
        return {"pairs": (seq_len - 2) // 2, "queries": 1, "noise": 0}
    raise ValueError(f"未知 task {task!r}，可选 {TASKS}")


def _pair_starts(pairs: int, noise: int) -> np.ndarray:
    """kv 区长度 2*pairs+noise 时每对 (k_i,v_i) 的起始位置；噪声均分到对与对之间。"""
    ar = np.arange(int(pairs), dtype=np.int64)
    per, rem = divmod(int(noise), int(pairs))
    return 2 * ar + per * ar + np.minimum(ar, rem)


# ------------------------------------------------------------------ batch
def make_batch(task: str, batch: int, seq_len: int, vocab: int, rng) -> dict:
    """生成一个 batch：{'x'[B,N] int64, 'y'[B,N] int64, 'mask'[B,N] bool}。"""
    batch, seq_len, vocab = int(batch), int(seq_len), int(vocab)
    if batch <= 0:
        raise ValueError(f"batch 需 >= 1，实际 {batch}")
    if vocab < 8 or vocab % 2:
        raise ValueError(f"vocab 需为 >= 8 的偶数（前一半 id 作 key，后一半作 value），实际 {vocab}")

    parts = split_seq(task, seq_len)
    L, Q, D = parts["pairs"], parts["queries"], parts["noise"]
    V2 = int(vocab * _KEY_FRAC)
    n_kv = 2 * L + D
    if n_kv + 2 * Q != seq_len:
        raise ValueError(
            f"内部布局不自洽: 2L+D+2Q = {n_kv + 2 * Q} != seq_len = {seq_len} (task={task})"
        )
    if D and L >= V2:
        raise ValueError(
            f"task={task}: L={L} >= vocab//2={V2} 时没有空闲 id 可做噪声 token；"
            f"请增大 vocab 或减小 seq_len"
        )
    rng = _coerce_rng(rng)

    if task == "needle":
        # 1 个 needle 对 + (L-1) 个只做干扰、永不被查询的对；干扰 key 永远不等于 needle key。
        V2 = int(vocab * _KEY_FRAC)
        if V2 < 2:
            raise ValueError("needle 需要 vocab//2 >= 2")
        u = _uniform(rng, (batch, 2 * L + 2))
        nk = (u[:, 0] * V2).astype(np.int64)                      # [B] needle key
        nv = (u[:, 1] * V2).astype(np.int64) + V2                 # [B] needle value
        # 干扰 key：在 [0,V2) 去掉 nk 再取模，保证 != nk
        dk = (u[:, 2:2 + L - 1] * (V2 - 1)).astype(np.int64)
        dk = dk + (dk >= nk[:, None])                             # 跳过 nk
        dv = (u[:, 2 + L - 1:2 + 2 * (L - 1)] * V2).astype(np.int64) + V2
        rows = np.arange(batch)
        key_tok = np.concatenate([nk[:, None], dk], axis=1)        # [B,L]
        val_tok = np.concatenate([nv[:, None], dv], axis=1)        # [B,L]
        perm = np.argsort(u[:, -1][:, None] * np.ones((1, L)) + 1e-9 * np.arange(L)[None, :], axis=1)
        key_tok = np.take_along_axis(key_tok, perm, 1)             # needle 放在随机位置
        val_tok = np.take_along_axis(val_tok, perm, 1)
        x = np.empty((batch, seq_len), dtype=np.int64)
        x[:, 0:2 * L:2] = key_tok
        x[:, 1:2 * L:2] = val_tok
        x[:, 2 * L] = key_tok[rows, np.argmax(perm == 0, axis=1)]  # 查询 = needle key
        x[:, 2 * L + 1] = nv                                       # 答案 = needle value
        y = np.empty_like(x)
        y[:, :-1] = x[:, 1:]
        y[:, -1] = 0
        mask = np.zeros((batch, seq_len), dtype=bool)
        mask[:, 2 * L] = True
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y), "mask": torch.from_numpy(mask)}

    cover = L <= COVER_L_MAX and Q >= L               # 每个 key 至少查询一次
    n_q = max(L, Q) if cover else Q

    # ---- 一次抽够随机数：kv 排列 2*V2 + 查询 + 噪声
    cols = 2 * V2 + n_q + D
    u = _uniform(rng, (batch, cols))
    o = 0
    u_k = _block(u, o, V2); o += V2
    u_v = _block(u, o, V2); o += V2
    u_q = u[:, o:o + n_q]; o += n_q
    u_n = u[:, o:o + D]

    # ---- key/value：两个独立随机排列（id 区间不重叠）；L > V2 时循环复用同一配对
    key_pool = np.argsort(u_k, axis=1)               # [B,V2] key id 随机顺序
    val_pool = np.argsort(u_v, axis=1) + V2          # [B,V2] value id 随机顺序
    if L <= V2:
        key_tok, val_tok = key_pool[:, :L], val_pool[:, :L]
    else:
        cyc = np.arange(L, dtype=np.int64) % V2      # 同一 key 的多次出现保持同一 value
        key_tok, val_tok = key_pool[:, cyc], val_pool[:, cyc]

    # ---- 查询：覆盖模式优先，其余随机
    if cover:
        sel = np.argsort(_block(u_q, 0, L), axis=1)  # [B,L] pair 下标的随机排列
        if Q > L:
            sel = np.concatenate([sel, _rand_idx(u_q[:, L:], L)], axis=1)
    else:
        sel = _rand_idx(u_q, L)
    rows = np.arange(batch)[:, None]
    q_tok = key_tok[rows, sel]                        # [B,Q]
    a_tok = val_tok[rows, sel]

    # ---- 组装 [kv 区][q,a 交替]
    x = np.empty((batch, seq_len), dtype=np.int64)
    if D == 0:
        x[:, 0:n_kv:2] = key_tok
        x[:, 1:n_kv:2] = val_tok
    else:
        starts = _pair_starts(L, D)
        pair_pos = np.empty(2 * L, dtype=np.int64)
        pair_pos[0::2] = starts
        pair_pos[1::2] = starts + 1
        kv_pairs = np.empty((batch, 2 * L), dtype=np.int64)
        kv_pairs[:, 0::2] = key_tok
        kv_pairs[:, 1::2] = val_tok
        x[:, pair_pos] = kv_pairs
        pool = np.concatenate([key_pool[:, L:], val_pool[:, L:]], axis=1)   # [B,F] 空闲 id
        F = pool.shape[1]
        if F <= 0:
            raise ValueError(f"task={task}: L={L} 无空闲 id 可做噪声 token")
        used_pos = np.zeros(n_kv, dtype=bool)
        used_pos[starts] = True
        used_pos[starts + 1] = True
        free_pos = np.flatnonzero(~used_pos)                                 # [D]
        if free_pos.size != D:
            raise ValueError(f"内部错误: 噪声槽位 {free_pos.size} != D={D}")
        x[:, free_pos] = np.take_along_axis(pool, _rand_idx(u_n, F), axis=1)
    x[:, n_kv::2] = q_tok
    x[:, n_kv + 1::2] = a_tok

    # ---- next-token 目标 + loss mask
    y = np.empty_like(x)
    y[:, :-1] = x[:, 1:]
    y[:, -1] = 0
    mask = np.zeros((batch, seq_len), dtype=bool)
    mask[:, n_kv::2] = True

    if x.shape != (batch, seq_len) or y.shape != (batch, seq_len) or mask.shape != (batch, seq_len):
        raise ValueError(f"内部错误: 生成形状 {x.shape} != ({batch}, {seq_len})")
    return {"x": torch.from_numpy(x), "y": torch.from_numpy(y), "mask": torch.from_numpy(mask)}


__all__ = ["TASKS", "COVER_L_MAX", "split_seq", "make_batch"]
