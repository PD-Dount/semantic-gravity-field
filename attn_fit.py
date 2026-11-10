"""Can the gravity form reconstruct real LLM attention?

Ground truth: distilgpt2 attention logits (6 layers x 12 heads), row-centred so the
softmax normaliser drops out.

The question is NOT "is R^2 high" -- a free per-key bias b_j already makes it high.
The question is the **incremental** R^2 over that bias, against parameter-matched
control bases.  At constant bandwidth the field reduces to
    logit_ij = b_j + alpha*cos + beta*cos^2
so if that basis is no better than cos^3 / cos^5 / a step function, then "gravity"
is a name for "log-attention is smooth in cosine similarity" and nothing more.

q and k are taken from the model itself (hooked c_attn), so `cos` is the real
cosine between the query and key vectors -- not something fitted from the target.
"""
import json
import os

import numpy as np
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

TEXT = (
    "The history of the river is the history of the city. For six hundred years the water "
    "carried grain, salt, timber and rumours, and the men who owned the barges owned the "
    "council too. When the railway came, the barges lost; when the road came, the railway "
    "lost; and the river, which had never competed with anything, kept flowing. Today the "
    "old warehouses hold apartments with very expensive windows, and the water carries "
    "tourists and the occasional dead bicycle. Nobody remembers the price of salt. "
    "A city that forgets its river is a city that has decided to become a rumour itself."
)


def load_qk(name="distilgpt2", max_tokens=192):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager")
    model.eval()
    ids = tok(TEXT, return_tensors="pt")["input_ids"][:, :max_tokens]
    n_embd = model.config.n_embd
    grabs = []
    handles = []

    def mk(buf):
        def hook(mod, inp, out):
            buf.append(out.detach())
        return hook

    for mod in model.modules():
        w = getattr(mod, "weight", None)
        fused = (type(mod).__name__ in ("Conv1D", "Linear")
                 and w is not None and w.dim() == 2 and w.shape[1] == 3 * n_embd)
        if fused:
            buf = []
            grabs.append(buf)
            handles.append(mod.register_forward_hook(mk(buf)))
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    for h in handles:
        h.remove()

    hs = model.config.n_head
    dh = n_embd // hs
    qk = []
    for buf, attn in zip(grabs, out.attentions):
        qkv = buf[0][0]                                   # [T, 3D]
        q, k = qkv[:, :n_embd], qkv[:, n_embd:2 * n_embd]
        q = q.view(-1, hs, dh).transpose(0, 1)            # [H,T,dh]
        k = k.view(-1, hs, dh).transpose(0, 1)
        logits = (q @ k.transpose(-1, -2)) / np.sqrt(dh)
        a = attn[0].float().numpy()                       # [H,T,T]
        # sanity: our reconstructed logits must reproduce the model's own softmax
        T = logits.shape[-1]
        mask = np.triu(np.ones((T, T), dtype=bool), 1)
        lg = logits.numpy().copy()
        lg[mask[None].repeat(hs, 0)] = -1e30
        p = np.exp(lg - lg.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        err = float(np.abs(p - a).max())
        qn = (q / q.norm(dim=-1, keepdim=True)).numpy()
        kn = (k / k.norm(dim=-1, keepdim=True)).numpy()
        qk.append((logits.numpy(), np.einsum("hid,hjd->hij", qn, kn), err))
    return qk, hs, ids.shape[1]


def fit_linear(basis, Y, tr, te):
    T = Y.shape[0]
    cols = [b.ravel() for b in basis] + [np.tile(np.eye(T)[:, j], T) for j in range(T)]
    A = np.stack(cols, 1)
    rows = np.zeros(T * T, dtype=bool)
    for i in tr:
        rows[i * T:(i + 1) * T] = True
    y = Y.ravel()
    coef, *_ = np.linalg.lstsq(A[rows], y[rows], rcond=1e-8)
    pred = (A @ coef).reshape(T, T)
    yy, pp = Y[te].ravel(), pred[te].ravel()
    return 1.0 - ((yy - pp) ** 2).sum() / max(((yy - yy.mean()) ** 2).sum(), 1e-12)


def fit_gravity_full(cos, Y, tr, te, steps=500, lr=0.08):
    T = Y.shape[0]
    ct = torch.tensor(cos, dtype=torch.float32)
    yt = torch.tensor(Y, dtype=torch.float32)
    b = torch.zeros(T, requires_grad=True)
    rq = torch.full((T,), 0.7, requires_grad=True)
    rk = torch.full((T,), 0.7, requires_grad=True)
    kap = torch.tensor(1.0, requires_grad=True)
    p_raw = torch.tensor(-1.0, requires_grad=True)
    opt = torch.optim.Adam([b, rq, rk, kap, p_raw], lr=lr)
    m = torch.zeros(T * T, dtype=torch.bool)
    for i in tr:
        m[i * T:(i + 1) * T] = True
    d1 = 1.0 - ct
    for _ in range(steps):
        sigma = 0.5 * (rq[:, None] + rk[None, :])
        p = 1.0 + torch.nn.functional.softplus(p_raw)
        term = (d1 / (2.0 * sigma ** 2 + 1e-6)) ** (p / 2.0)
        logit = b[None, :] - kap * term
        logit = logit - logit.mean(-1, keepdim=True)
        loss = ((logit - yt).reshape(-1)[m] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        sigma = 0.5 * (rq[:, None] + rk[None, :])
        p = 1.0 + torch.nn.functional.softplus(p_raw)
        term = (d1 / (2.0 * sigma ** 2 + 1e-6)) ** (p / 2.0)
        logit = b[None, :] - kap * term
        logit = logit - logit.mean(-1, keepdim=True)
        pred = logit.numpy()
    yy, pp = Y[te].ravel(), pred[te].ravel()
    return 1.0 - ((yy - pp) ** 2).sum() / max(((yy - yy.mean()) ** 2).sum(), 1e-12), float(p)


def main():
    qk, hs, T = load_qk()
    print(f"distilgpt2: layers={len(qk)} heads={hs} tokens={T}")
    print(f"sanity (reconstructed logits vs model attention): maxerr="
          f"{max(x[2] for x in qk):.2e}")

    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(T), 5)
    variants = ["bias_only", "cos", "cos2(gravity)", "cos3(ctrl)", "cos5(ctrl)", "step(ctrl)", "gravity_full"]
    per_layer = {v: [] for v in variants}

    for li, (logits, cos_h, _) in enumerate(qk):
        for h in range(hs):
            Y = logits[h] - logits[h].mean(-1, keepdims=True)
            cos = cos_h[h]
            bases = {
                "bias_only": [],
                "cos": [cos],
                "cos2(gravity)": [cos, cos ** 2],
                "cos3(ctrl)": [cos, cos ** 3],
                "cos5(ctrl)": [cos, cos ** 5],
                "step(ctrl)": [cos, (cos > np.median(cos)).astype(np.float64)],
            }
            for name, basis in bases.items():
                s = []
                for k in range(5):
                    te = folds[k]
                    tr = np.concatenate([folds[j] for j in range(5) if j != k])
                    s.append(fit_linear(basis, Y, tr, te))
                per_layer[name].append(float(np.mean(s)))
            s = []
            for k in range(5):
                te = folds[k]
                tr = np.concatenate([folds[j] for j in range(5) if j != k])
                r2, _ = fit_gravity_full(cos, Y, tr, te)
                s.append(r2)
            per_layer["gravity_full"].append(float(np.mean(s)))
        print(f"  layer {li}: " + "  ".join(f"{v}={np.mean(per_layer[v][-hs:]):.3f}" for v in variants),
              flush=True)

    mean = {k: float(np.mean(v)) for k, v in per_layer.items()}
    inc = {k: mean[k] - mean["bias_only"] for k in variants}
    print("\n== held-out R^2 (mean over 6 layers x 12 heads) ==")
    for k in variants:
        print(f"  {k:18s} R2={mean[k]:.4f}   (+{inc[k]:+.4f} over bias)")
    print("\n== does the gravity basis beat parameter-matched controls? ==")
    for k in ("cos3(ctrl)", "cos5(ctrl)", "step(ctrl)", "gravity_full"):
        print(f"  cos2(gravity) - {k:16s} = {mean['cos2(gravity)'] - mean[k]:+.4f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "attn_fit.json")
    with open(out, "w") as f:
        json.dump({"model": "distilgpt2", "tokens": int(T), "heads": int(hs),
                   "layers": len(qk), "r2": mean, "incremental": inc}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
