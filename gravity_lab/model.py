"""Tiny decoder-only LM that hosts any arm's attention.

Shapes: idx [B,N] int64 -> logits [B,N,V] float32. CPU / float32 only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import AttnBase
from .config import AttnConfig, ModelConfig


def _ranks(x):
    order = x.argsort()
    ranks = torch.empty(x.numel(), dtype=torch.float32, device=x.device)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float32, device=x.device)
    return ranks


def _spearman(a, b):
    """Tie-free rank correlation; a diagnostic, not a training signal."""
    if a.numel() < 3:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = ra.norm() * rb.norm()
    return float("nan") if float(den) == 0.0 else float((ra * rb).sum() / den)


class _Block(nn.Module):
    def __init__(self, mcfg: ModelConfig, acfg: AttnConfig):
        super().__init__()
        d = mcfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.attn = AttnBase(acfg)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mcfg.mlp_ratio * d),
            nn.GELU(),
            nn.Linear(mcfg.mlp_ratio * d, d),
        )
        for lin in self.mlp:
            if isinstance(lin, nn.Linear):
                nn.init.normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)

    def forward(self, x, layer_idx, ctx, return_stats):
        a, stats = self.attn(self.ln1(x), layer_idx=layer_idx, ctx=ctx,
                             return_stats=return_stats)
        x = x + a
        return x + self.mlp(self.ln2(x)), stats


class TinyLM(nn.Module):
    def __init__(self, mcfg: ModelConfig, acfg: AttnConfig):
        super().__init__()
        self.mcfg = mcfg
        self.acfg = acfg
        if acfg.d_model != mcfg.d_model or acfg.n_heads != mcfg.n_heads:
            raise ValueError(
                f"AttnConfig(d_model={acfg.d_model},n_heads={acfg.n_heads}) 与 "
                f"ModelConfig(d_model={mcfg.d_model},n_heads={mcfg.n_heads}) 不一致"
            )
        self.embed = nn.Embedding(mcfg.vocab, mcfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        # 位置信号必须显式给：attention 的 logit 是纯内容 q·k，没有任何位置项，
        # 而 MQAR 要求"取出匹配 key 的下一个 token"，是相对位置操作。无位置编码时
        # 所有 arm 都停在 chance（实测 600 步 acc 0.048，chance 0.031）。
        self.pos = nn.Embedding(mcfg.seq_len, mcfg.d_model)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [_Block(mcfg, acfg) for _ in range(mcfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(mcfg.d_model)
        self.head = nn.Linear(mcfg.d_model, mcfg.vocab, bias=False)
        if mcfg.tie_embed:
            # Tied after construction: a shared tensor must not be re-initialised.
            self.head.weight = self.embed.weight

    def forward(self, idx, return_stats=False):
        if idx.dim() != 2:
            raise ValueError(f"idx 需要 [B,N]，收到 {tuple(idx.shape)}")
        x = self.embed(idx)
        n = idx.shape[1]
        if n > self.pos.num_embeddings:
            raise ValueError(f"序列长度 {n} 超过 pos 表 {self.pos.num_embeddings}")
        x = x + self.pos(torch.arange(n, device=idx.device))
        # A4 keeps eviction state across layers; every other arm leaves ctx 空.
        ctx = {} if self.acfg.committed else None
        layer_stats = []
        for i, blk in enumerate(self.blocks):
            x, st = blk(x, i, ctx, return_stats)
            if return_stats:
                layer_stats.append(st)
        logits = self.head(self.ln_f(x))

        stats = {}
        if return_stats and layer_stats:
            key_mass = [s.pop("_key_mass") for s in layer_stats if "_key_mass" in s]
            keys = set().union(*(s.keys() for s in layer_stats))
            for k in sorted(keys):
                vals = [s[k] for s in layer_stats if k in s]
                stats[k] = sum(vals) / len(vals)
            if key_mass:
                freq = torch.bincount(idx.reshape(-1), minlength=self.mcfg.vocab).float()
                stats["mass_rho"] = _spearman(
                    torch.stack(key_mass).mean(0), freq[idx].mean(0))
            stats["params"] = int(sum(p.numel() for p in self.parameters()))
        return logits, stats


def build_model(mcfg: ModelConfig, acfg: AttnConfig) -> TinyLM:
    return TinyLM(mcfg, acfg)


__all__ = ["TinyLM", "build_model"]
