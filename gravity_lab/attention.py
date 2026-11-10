"""Unified gravity attention (SPEC.md).

One logit expression serves every arm; AttnConfig's booleans zero out terms.
Per-arm forward implementations are forbidden on purpose: the ablation only
means "which terms exist", never "how they are computed".
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AttnConfig

try:  # merging.py is written by another teammate and may land later
    from . import merging as _merging
    _MERGE_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    _merging = None
    _MERGE_IMPORT_ERROR = exc

_EPS = 1e-6          # SPEC's epsilon in the distance/bandwidth ratio
_NEG = float("-inf") # causal mask: exact zeros after softmax, matches SDPA numerics


def _merging_fn(name):
    if _merging is None:
        raise RuntimeError("merging.py 未就绪") from _MERGE_IMPORT_ERROR
    fn = getattr(_merging, name, None)
    if fn is None:
        raise RuntimeError(f"merging.py 未就绪：缺少 {name}")
    return fn


def _inv_softplus(y, floor=1e-6):
    """Parameter init that lands on the value the config names, not near it."""
    return math.log(math.expm1(max(float(y), floor)))


def _bview(vec, ndim):
    """Broadcast a per-head vector [H] onto a tensor whose head axis is dim 1."""
    return vec.view(1, -1, *([1] * (ndim - 2)))


class AttnBase(nn.Module):
    """Softmax attention with an augmented logit.

    l_ij = alpha_h * (q_i . k_j) / sqrt(dh)
           + [mass]  log m_j
           - [rad]   kappa_h * ((D2_ij + eps) / (2 sigma_ij^2 + eps)) ** (p_h / 2)
           + [sink]  b_h

    Every switch below is a config bool, so A0..A5 differ only in which terms
    are materialised -- and in A5's key set, which SPEC defines differently.
    """

    def __init__(self, cfg: AttnConfig):
        super().__init__()
        if cfg.committed and int(cfg.merge) > 0:
            raise ValueError("committed 与 merge 不能同时开启：key 集合语义冲突，先定一个")
        d_model, n_heads = int(cfg.d_model), int(cfg.n_heads)
        if n_heads <= 0 or d_model % n_heads:
            raise ValueError(f"d_model={d_model} 不能被 n_heads={n_heads} 整除")
        self.cfg = cfg
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self._sqrt_dh = math.sqrt(self.dh)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        for lin in (self.W_q, self.W_k, self.W_v):
            nn.init.normal_(lin.weight, std=0.02)
        # Identity out-projection keeps A0 numerically equal to the raw SDPA
        # reference, which does not have an output projection at all.
        with torch.no_grad():
            self.W_o.weight.copy_(torch.eye(d_model))

        self.alpha = nn.Parameter(torch.full((n_heads,), float(cfg.alpha_init)))

        if cfg.mass:
            self.w_m = nn.Parameter(torch.empty(n_heads, d_model))
            # Scalar in the *log* domain: it shifts every key's logit by the same
            # constant, which softmax cannot see. That is what makes mass
            # strictly key-side (SPEC 物理约束 1) and what smoke_contract tests.
            self.mass_bias = nn.Parameter(torch.zeros(()))
            nn.init.normal_(self.w_m, std=0.02)
        if cfg.radius:
            self.w_r = nn.Parameter(torch.empty(n_heads, d_model))
            self.radius_bias = nn.Parameter(torch.zeros(()))
            self.s_raw = nn.Parameter(torch.full((n_heads,), _inv_softplus(cfg.init_radius)))
            # SPEC 把 A2 钉成 kappa=1（无质量项的纯余弦+带宽臂）
            kappa0 = 1.0 if cfg.arm == "A2" else float(cfg.kappa_init)
            self.kappa = nn.Parameter(torch.full((n_heads,), kappa0))
            nn.init.normal_(self.w_r, std=0.02)
        if cfg.shape_p:
            self.p_raw = nn.Parameter(torch.full((n_heads,), _inv_softplus(float(cfg.p_init) - 1.0)))
        else:
            self.register_buffer("p_const", torch.ones(n_heads))
        if int(cfg.aniso) > 0:
            # 不能零初始化：d^2 是 U 的偶函数，U=0 处梯度精确为 0，一阶法永远出不来
            # （实测 U.grad.max()==0.0）。小随机初始化让各向异性真正可学。
            self.U = nn.Parameter(torch.empty(n_heads, self.dh, int(cfg.aniso)))
            nn.init.normal_(self.U, std=0.01)
        if cfg.vacuum:
            self.b_sink = nn.Parameter(torch.zeros(n_heads))
            self.v_sink = nn.Parameter(torch.zeros(n_heads, self.dh))

    # ------------------------------------------------------------------ terms
    def _heads(self, t):
        b, n, _ = t.shape
        return t.view(b, n, self.n_heads, self.dh).transpose(1, 2)

    def _log_mass(self, x):
        """Key-side log-mass; (None, None) when the arm has no mass term."""
        if not self.cfg.mass:
            return None, None
        z = x @ self.w_m.t()                                   # [B,N,H]
        logm = F.softplus(z).transpose(1, 2) + self.mass_bias  # [B,H,N]
        # exp(logm) feeds the merge weights; the clamp only guards e^b overflow.
        mass = logm.clamp(max=30.0).exp()
        return logm, mass

    def _radius(self, x):
        """Per-token bandwidth radius r_i (SPEC: softplus(x w_r + c))."""
        if not self.cfg.radius:
            return None
        return F.softplus(x @ self.w_r.t() + self.radius_bias).transpose(1, 2)

    def _p(self):
        return 1.0 + F.softplus(self.p_raw) if self.cfg.shape_p else self.p_const

    def _sigma(self):
        return F.softplus(self.s_raw)

    def _metric(self):
        """Returns M = L^2 with L = I + U U^T, so D^2 = z^T M z and no d_h x d_h
        expansion of the difference vectors is ever materialised."""
        u = self.U
        g = u @ u.transpose(-1, -2)                            # [H,dh,dh]
        eye = torch.eye(self.dh, dtype=g.dtype, device=g.device)
        return eye + 2.0 * g + g @ g

    def _apply_metric(self, x, m):
        shp = x.shape
        xf = x.reshape(shp[0], shp[1], -1, shp[-1])
        return torch.matmul(xf, m).reshape(shp)

    def _logits(self, q, k, logm=None, rq=None, rk=None):
        """Shared logit. Key axis is k.dim(-2); every term broadcasts on the
        query axis, so the row-wise (merged-KV) case reuses this untouched."""
        s = (q @ k.transpose(-1, -2)) / self._sqrt_dh
        s = s * _bview(self.alpha, s.dim())
        if logm is not None:
            s = s + logm.unsqueeze(-2)
        if rq is not None:
            sig, term = self._radius_term(q, k, rq, rk)
            s = s - _bview(self.kappa, s.dim()) * term
        return s

    def _radius_term(self, q, k, rq, rk):
        qh = q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        kh = k / k.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        if int(self.cfg.aniso) > 0:
            m = self._metric()
            qm = self._apply_metric(qh, m)
            km = self._apply_metric(kh, m)
        else:
            qm, km = qh, kh
        d2 = ((qh * qm).sum(-1).unsqueeze(-1)
              + (kh * km).sum(-1).unsqueeze(-2)
              - 2.0 * (qm @ kh.transpose(-1, -2)))
        d2 = d2.clamp_min(0.0)                       # squared norm, rounding can dip
        sig = 0.5 * _bview(self._sigma(), d2.dim()) * (rq.unsqueeze(-1) + rk.unsqueeze(-2))
        ratio = (d2 + _EPS) / (2.0 * sig * sig + _EPS)
        return sig, ratio.pow(_bview(self._p(), d2.dim()) * 0.5)

    def _softmax(self, logits):
        if self.cfg.vacuum:
            sink = _bview(self.b_sink, logits.dim())
            logits = torch.cat([logits, sink.expand(*logits.shape[:-1], 1)], dim=-1)
        return logits.softmax(-1)

    def _sink_values(self, v):
        if not self.cfg.vacuum:
            return v
        b, h, _, dh = v.shape
        return torch.cat([v, self.v_sink.view(1, h, 1, dh).expand(b, h, 1, dh)], dim=-2)

    # ------------------------------------------------------------------ forward
    def forward(self, x, layer_idx: int = 0, ctx: dict | None = None,
                return_stats: bool = False):
        if x.dim() != 3:
            raise ValueError(f"x 需要 [B,N,D]，收到 {tuple(x.shape)}")
        b, n, d = x.shape
        if d != self.d_model:
            raise ValueError(f"输入维度 {d} 与 d_model={self.d_model} 不一致")
        if x.dtype != torch.float32:
            x = x.float()

        q = self._heads(self.W_q(x))
        k = self._heads(self.W_k(x))
        v = self._heads(self.W_v(x))
        logm, mass = self._log_mass(x)
        rad = self._radius(x)

        if int(self.cfg.merge) > 0:
            out, stats = self._merged(q, k, v, logm, mass, rad, return_stats)
        else:
            out, stats = self._dense(q, k, v, logm, rad, layer_idx, ctx, return_stats)
        if stats and mass is not None:
            # Per-position key mass; TinyLM turns it into mass/frequency rho.
            stats["_key_mass"] = mass.detach().mean(dim=(0, 1))

        out = out.transpose(1, 2).reshape(b, n, d)
        return self.W_o(out), stats

    def _dense(self, q, k, v, logm, rad, layer_idx, ctx, return_stats):
        b, h, n, dh = q.shape
        dev = q.device
        logits = self._logits(q, k, logm, rad, rad)
        causal = torch.ones(n, n, dtype=torch.bool, device=dev).tril()
        logits = logits.masked_fill(~causal, _NEG)

        dead = None
        if self.cfg.committed:
            dead, logits, v = self._commit(logits, v, layer_idx, ctx, n)

        w = self._softmax(logits)
        out = w @ self._sink_values(v)

        stats = {}
        if return_stats:
            # KV *set* size, not the causal-visible prefix: causality is common to
            # every arm, so counting it would fake sparsity for A0. This is the
            # number cost.py divides by N.
            kv_len = torch.tensor(float(n), device=dev)
            if dead is not None:
                kv_len = kv_len - dead.float().sum(-1).mean()
            if self.cfg.vacuum:
                kv_len = kv_len + 1.0
            stats = self._stats(w, kv_len, dead, (rad, rad))
        return out, stats

    def _commit(self, logits, v, layer_idx, ctx, n):
        """A4: irreversible escape. Returns (dead, logits, v)."""
        if ctx is None:
            ctx = {}
        apply_dead = _merging_fn("apply_dead")
        prev_dead = ctx.get("dead")
        if prev_dead is not None:
            logits = apply_dead(logits, prev_dead)
        if layer_idx < int(self.cfg.commit_start_layer):
            return prev_dead, logits, v

        # Mass each key received; recompute w so the eviction takes effect now
        # rather than one layer later.
        w = self._softmax(logits)
        col = w[..., :n].sum(dim=-2).detach()                  # [B,H,N]
        prev_ema = ctx.get("mass_ema")
        ema = col if prev_ema is None else _merging_fn("update_ema")(
            prev_ema, col, float(self.cfg.commit_ema))
        ctx["mass_ema"] = ema

        alive = _merging_fn("commit_escape_mask")(ema, float(self.cfg.commit_tau))
        dead = ~alive if prev_dead is None else (prev_dead | ~alive)
        ctx["dead"] = dead
        logits = apply_dead(logits, dead)

        tau = float(self.cfg.commit_tau)
        temp = max(float(self.cfg.commit_temp), 1e-6)
        prob = torch.sigmoid((tau * ema.mean(-1, keepdim=True) - ema) / temp)
        gate = _merging_fn("straight_through_gate")(prob)
        return dead, logits, v * (1.0 - gate).unsqueeze(-1)

    def _merged(self, q, k, v, logm, mass, rad, return_stats):
        """A5: far field is merged to one key per block, own block stays raw."""
        b, h, n, dh = q.shape
        dev = q.device
        g = int(self.cfg.merge)
        nb = (n + g - 1) // g
        npad = nb * g
        rad_q = rad            # query radius stays unpadded; keys get padded below
        if npad != n:  # every block must be full; padded keys carry no mass
            pad = npad - n
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            if logm is not None:
                logm = F.pad(logm, (0, pad))
                mass = F.pad(mass, (0, pad))
            if rad is not None:
                rad = F.pad(rad, (0, pad))
        if mass is None:
            mass = torch.ones(b, h, npad, dtype=q.dtype, device=dev)

        merged = _merging_fn("merge_kv_mass_conserving")(k, v, mass, block=g)
        kb, vb, mb = merged["k"], merged["v"], merged["mass"]

        rbar = None
        if rad is not None:
            # SPEC fixes k̄, v̄ weighting but not a block radius; reuse the merge's
            # own pi so the bandwidth stays consistent with them.
            pi = merged.get("assign")
            if not (isinstance(pi, torch.Tensor) and pi.shape == (b, h, nb, g)):
                pi = torch.ones(b, h, nb, g, dtype=q.dtype, device=dev)
            wgt = pi * mass.reshape(b, h, nb, g)
            # 分母必须是 Σ_j π_jm m_j，不能是块总质量 Σ_j m_j：π 逐块归一化后
            # Σ_j π_jm m_j ≈ m̄，用总质量当分母会让 r̄ 偏小约 g 倍。
            rbar = (wgt * rad.reshape(b, h, nb, g)).sum(-1) / wgt.sum(-1).clamp_min(_EPS)
        logmb = mb.clamp_min(_EPS).log() if logm is not None else None

        rows = torch.arange(n, device=dev)
        blk_row = rows.div(g, rounding_mode="floor")           # [N]
        far = blk_row.unsqueeze(-1) > torch.arange(nb, device=dev).unsqueeze(0)
        s_far = self._logits(q, kb, logmb, rad_q, rbar)
        s_far = s_far.masked_fill(~far.view(1, 1, n, nb), _NEG)

        off = torch.arange(g, device=dev)
        lidx = blk_row.unsqueeze(-1) * g + off.unsqueeze(0)     # [N,g]
        keep = (lidx <= rows.unsqueeze(-1)) & (lidx < n)        # causal + real keys
        lidx = lidx.clamp(max=npad - 1)
        idx2 = lidx.reshape(1, n * g).expand(b * h, n * g)
        idx3 = lidx.reshape(1, n * g, 1).expand(b * h, n * g, dh)
        k_loc = torch.gather(k.reshape(b * h, npad, dh), 1, idx3).reshape(b, h, n, g, dh)
        v_loc = torch.gather(v.reshape(b * h, npad, dh), 1, idx3).reshape(b, h, n, g, dh)
        logm_loc = (torch.gather(logm.reshape(b * h, npad), 1, idx2).reshape(b, h, n, g)
                    if logm is not None else None)
        r_loc = (torch.gather(rad.reshape(b * h, npad), 1, idx2).reshape(b, h, n, g)
                 if rad is not None else None)
        # row-wise keys: q gets a singleton query axis, so does the query radius
        s_loc = self._logits(q.unsqueeze(-2), k_loc, logm_loc,
                             None if rad_q is None else rad_q.unsqueeze(-1), r_loc)
        s_loc = s_loc.masked_fill(~keep.view(1, 1, n, 1, g), _NEG).squeeze(-2)

        w = self._softmax(torch.cat([s_far, s_loc], dim=-1))
        out = w[..., :nb] @ vb
        out = out + (w[..., nb:nb + g].unsqueeze(-2) @ v_loc).squeeze(-2)
        if self.cfg.vacuum:
            out = out + w[..., -1:] * self.v_sink.view(1, h, 1, dh)

        stats = {}
        if return_stats:
            # own block is kept raw (g slots) plus one merged slot per past block
            kv_len = blk_row.float().mean() + float(g)
            if self.cfg.vacuum:
                kv_len = kv_len + 1.0
            stats = self._stats(w, kv_len, None, (rad_q, rbar))
        return out, stats

    def _stats(self, w, kv_len, dead, radii):
        w = w.detach()
        stats = {
            "eff_part": float(w.pow(2).sum(-1).clamp_min(1e-12).reciprocal().mean()),
            "sparsity": float((w < 1e-4).float().mean()),
            "kv_len": float(kv_len),
            "escape_frac": 0.0 if dead is None else float(dead.float().mean()),
            "kappa": 0.0 if not self.cfg.radius else float(self.kappa.detach().mean()),
            "p_mean": float(self._p().detach().mean()) if self.cfg.shape_p else float("nan"),
            "sigma_mean": float("nan"),
        }
        rq, rk = radii
        if self.cfg.radius and rq is not None and rk is not None:
            sig = 0.5 * _bview(self._sigma(), 4) * (rq.unsqueeze(-1) + rk.unsqueeze(-2))
            stats["sigma_mean"] = float(sig.detach().mean())
        return stats


def build_attention(cfg: AttnConfig) -> AttnBase:
    return AttnBase(cfg)


__all__ = ["AttnBase", "build_attention"]
