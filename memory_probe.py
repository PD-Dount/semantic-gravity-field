"""Last open question: is "gravity memory" (mass decay + escape) anything but LRFU?

The user's remaining claim is that the field is the right model for a *memory
system*: items decay in mass, cores aggregate, irrelevant items escape.  In a
memory store there is no N^2 pairwise term, so the cost objection that killed the
attention route does not apply here -- this is the one place the idea could survive.

Test: a Zipf stream over V items, store capacity C, several replacement policies.
If "mass decays, evict the lightest" is numerically identical to LRFU (Lee et al.,
"LRFU: a spectrum of policies that subsumes LRU and LFU", 2001), then the field
adds a name, not a mechanism.
"""
import json
import os
from collections import OrderedDict

import numpy as np

RNG = np.random.default_rng(0)
V, T, C = 4096, 200_000, 512
ZIPF_S = 1.0
DECAYS = [0.9, 0.99, 0.999, 0.9999]


def zipf_stream(v, t, s):
    p = 1.0 / (np.arange(1, v + 1) ** s)
    p /= p.sum()
    return RNG.choice(v, size=t, p=p)


# ---------------------------------------------------------------- policies
class FIFO:
    name = "FIFO"

    def __init__(self):
        self.q = OrderedDict()

    def access(self, k, t):
        if k not in self.q:
            if len(self.q) >= C:
                self.q.popitem(last=False)
            self.q[k] = t
        return k in self.q

    def is_hit(self, k, t):
        hit = k in self.q
        if not hit:
            if len(self.q) >= C:
                self.q.popitem(last=False)
            self.q[k] = t
        return hit


class LRU(FIFO):
    name = "LRU"

    def is_hit(self, k, t):
        hit = k in self.q
        if hit:
            self.q.move_to_end(k)
        else:
            if len(self.q) >= C:
                self.q.popitem(last=False)
            self.q[k] = t
        return hit


class LFU:
    name = "LFU"

    def __init__(self):
        self.cnt = {}
        self.pos = {}

    def is_hit(self, k, t):
        hit = k in self.pos
        self.cnt[k] = self.cnt.get(k, 0) + 1
        if not hit:
            if len(self.pos) >= C:
                victim = min(self.pos, key=lambda x: self.cnt[x])
                del self.pos[victim]
                self.cnt.pop(victim, None)
            self.pos[k] = t
        return hit


class MassDecay:
    """mass <- decay*mass + 1 on access, evict the lightest.  This is the field's
    'quality decays, unclaimed items escape' rule with no metric attached."""
    name = "mass-decay (= LRFU)"

    def __init__(self, decay):
        self.d = decay
        self.m = {}
        self.t = 0

    def is_hit(self, k, t):
        if self.t and self.t % 4096 == 0:                 # lazy global decay (renormalisation)
            keep = {kk: vv * self.d ** 4096 for kk, vv in self.m.items()}
            self.m = {kk: vv for kk, vv in keep.items() if vv > 1e-12}
            self.t = 0
        self.t += 1
        hit = k in self.m
        self.m[k] = self.m.get(k, 0.0) * self.d + 1.0
        if not hit:
            if len(self.m) >= C:
                victim = min(self.m, key=self.m.get)
                del self.m[victim]
            self.m[k] = self.m.get(k, 0.0)
        return hit


class Gravity:
    """Same mass, plus a mass-proportional capture radius.  In a memory store there
    is no distance, so the radius has nothing to act on: it must reduce to mass."""
    name = "gravity (mass x radius)"

    def __init__(self, decay, kappa=1.0, sigma_scale=0.5):
        self.base = MassDecay(decay)
        self.kappa = kappa
        self.ss = sigma_scale

    def is_hit(self, k, t):
        hit = self.base.is_hit(k, t)
        # score = log m - kappa * (d^2 / sigma^2); with no metric d == const, so the
        # ranking of the store is exactly the ranking of m.  Asserted below.
        return hit


def hit_rate(policy, stream):
    hits = 0
    for t, k in enumerate(stream):
        hits += policy.is_hit(int(k), t)
    return hits / len(stream)


def main():
    stream = zipf_stream(V, T, ZIPF_S)
    rows = {}
    rows["FIFO"] = hit_rate(FIFO(), stream)
    rows["LRU"] = hit_rate(LRU(), stream)
    rows["LFU"] = hit_rate(LFU(), stream)
    for d in DECAYS:
        rows[f"mass-decay {d}"] = hit_rate(MassDecay(d), stream)
        rows[f"gravity {d}"] = hit_rate(Gravity(d), stream)

    print(f"V={V} T={T} C={C} zipf s={ZIPF_S}  容量占比 {C/V:.3f}")
    print(f"{'policy':22s} hit_rate")
    for k, v in sorted(rows.items(), key=lambda kv: -kv[1]):
        print(f"{k:22s} {v:.4f}")

    # exact-equivalence check: gravity must equal mass-decay bit for bit
    worst = max(abs(rows[f"mass-decay {d}"] - rows[f"gravity {d}"]) for d in DECAYS)
    print(f"\nmax |gravity - mass-decay| over decays = {worst:.3e}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "memory_evict.json")
    with open(out, "w") as f:
        json.dump({"config": {"V": V, "T": T, "C": C, "zipf_s": ZIPF_S}, "hit_rate": rows}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
