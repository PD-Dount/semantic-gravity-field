"""Merge per-run jsons and runs_*.jsonl into results/runs_merged.jsonl (config-filtered)."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
SKIP = {"cost.json", "merge.json"}
# only the main sweep config; older tuning files must not leak in
WANT = {"task": "mqar", "steps": 1200, "seq": 64, "batch": 32}


def ok(r):
    return all(r.get(k) == v for k, v in WANT.items())


def key(r):
    return (r.get("arm"), r.get("task"), int(r.get("seed", -1)))


def main():
    rows = {}
    for p in glob.glob(os.path.join(RES, "*.json")):
        if os.path.basename(p) in SKIP:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        r = d.get("result", d) if isinstance(d, dict) else None
        if not isinstance(r, dict) or "final_acc" not in r or "arm" not in r:
            continue
        if "curve" not in r or not ok(r):
            continue
        rows[key(r)] = r
    for p in glob.glob(os.path.join(RES, "runs_*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "error" in r or not ok(r):
                    continue
                rows[key(r)] = r
    out = os.path.join(RES, "runs_merged.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in sorted(rows.values(), key=key):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    arms = {}
    for r in rows.values():
        arms.setdefault(r["arm"], []).append(int(r.get("seed", -1)))
    print("merged %d runs -> %s" % (len(rows), out))
    for a in sorted(arms):
        print("  %s: seeds %s" % (a, sorted(arms[a])))


if __name__ == "__main__":
    main()
