"""Frozen interface contract for gravity_lab. Nobody edits this file except the Lead."""
from dataclasses import dataclass, asdict, field

# ---------------------------------------------------------------- arms
ARMS = {
    # A0: standard softmax attention (baseline)
    "A0": dict(mass=False, radius=False, aniso=0, shape_p=False, vacuum=False,
               committed=False, merge=0),
    # A1: + key-side log-mass bias only (isolates "mass = importance prior")
    "A1": dict(mass=True, radius=False, aniso=0, shape_p=False, vacuum=False,
               committed=False, merge=0),
    # A2: + per-token bandwidth only (cosine attention with learned sigma)
    "A2": dict(mass=False, radius=True, aniso=0, shape_p=False, vacuum=False,
               committed=False, merge=0),
    # A3: full field (anisotropic metric + learnable shape + vacuum body)
    "A3": dict(mass=True, radius=True, aniso=4, shape_p=True, vacuum=True,
               committed=False, merge=0),
    # A4: A3 + irreversible committed escape (KV eviction)
    "A4": dict(mass=True, radius=True, aniso=4, shape_p=True, vacuum=True,
               committed=True, merge=0),
    # A5: A3 + mass-conserving KV merging (block soft assignment, block=4)
    "A5": dict(mass=True, radius=True, aniso=4, shape_p=True, vacuum=True,
               committed=False, merge=4),
}


@dataclass
class AttnConfig:
    arm: str = "A0"
    d_model: int = 64
    n_heads: int = 4
    # field terms
    mass: bool = False
    radius: bool = False
    aniso: int = 0          # low-rank rank r of the precision metric L = I + U U^T
    shape_p: bool = False   # learnable generalized-Gaussian exponent p
    vacuum: bool = False    # learnable null key ("vacuum body") + sink logit
    committed: bool = False # irreversible KV eviction across layers
    merge: int = 0          # block size for mass-conserving KV merge (0 = off)
    # committed-escape hyper params
    commit_tau: float = 0.5   # drop keys whose EMA mass < tau * batch mean
    commit_temp: float = 0.1  # STE sigmoid temperature
    commit_ema: float = 0.9
    commit_start_layer: int = 1
    # logit init
    kappa_init: float = 0.1   # weight of the distance/gravity term
    alpha_init: float = 1.0   # weight of the plain q.k term
    p_init: float = 1.2       # 1.0 = Gaussian, >1 sharper. 不能取 1.0：softplus 在 0 处斜率 ~0，
                              # 参数化 raw=inv_softplus(p-1) 会在 p=1 时退化成 sigma(-13.8)=1e-6 的死梯度
    init_radius: float = 0.7

    @staticmethod
    def from_arm(arm, **kw):
        cfg = AttnConfig(arm=arm, **ARMS[arm])
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg


@dataclass
class ModelConfig:
    vocab: int = 64
    seq_len: int = 128
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: int = 4
    tie_embed: bool = True


@dataclass
class TrainConfig:
    task: str = "mqar"          # "mqar" | "noise_recall"
    steps: int = 2000
    batch_size: int = 32
    lr: float = 3e-3
    warmup: int = 100
    seed: int = 0
    eval_every: int = 100
    eval_batches: int = 8
    log_every: int = 50
    threads: int = 1


# ---------------------------------------------------------------- attention API
# class AttnBase(nn.Module):
#     def __init__(self, cfg: AttnConfig): ...
#     # x: [B, N, D]; ctx: per-layer cross-layer state dict owned by the caller
#     # returns out [B, N, D], stats dict of python floats
#     def forward(self, x, layer_idx: int = 0, ctx: dict | None = None,
#                 return_stats: bool = False) -> tuple[Tensor, dict]:
#
# stats keys (all float, mean over batch/heads):
#   'eff_part'     effective participation  mean(1 / sum_j w_ij^2)
#   'sparsity'     frac of w < 1e-4
#   'escape_frac'  frac of keys evicted by committed escape (0 if off)
#   'kv_len'       effective KV length after merge/eviction
#   'p_mean'       mean learned shape exponent (NaN if off)
#   'mass_rho'     Spearman rho between mass and token frequency (filled by model)
#
# registry:
#   build_attention(cfg: AttnConfig) -> AttnBase
#
# ---------------------------------------------------------------- model API
# class TinyLM(nn.Module):
#     def __init__(self, mcfg: ModelConfig, acfg: AttnConfig): ...
#     def forward(self, idx, return_stats=False) -> (logits [B,N,V], stats dict)
#
# ---------------------------------------------------------------- task API
# make_batch(task: str, batch: int, seq_len: int, vocab: int, rng) -> dict
#   dict keys: 'x' [B,N] int64 input ids, 'y' [B,N] int64 next-token targets,
#              'mask' [B,N] bool loss mask (True = compute loss here)

__all__ = ["ARMS", "AttnConfig", "ModelConfig", "TrainConfig"]
