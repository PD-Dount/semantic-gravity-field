"""璇?results/runs.jsonl锛屾寜 PREREG.md 鐨勫垽鎹粰鍑?H1鈥揌4 瑁佸喅銆?""
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(tag="mqar"):
    rows = []
    p = os.path.join(ROOT, "results", "runs_merged.jsonl")
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") == tag and "error" not in r:
                rows.append(r)
    return rows


def agg(rows, arm, key):
    v = [r[key] for r in rows if r["arm"] == arm and isinstance(r.get(key), (int, float))]
    if not v:
        return float("nan"), float("nan"), []
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), v


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "mqar"
    rows = load(tag)
    if not rows:
        print("娌℃湁鍙敤鐨?%s 缁撴灉" % tag)
        return
    arms = sorted({r["arm"] for r in rows})
    print("## %s锛?d 涓?run\n" % (tag, len(rows)))
    print("| arm | acc | std | 閫?seed | loss | eff_part | sparsity | kv_len | escape | p_mean | params |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        m, s, v = agg(rows, a, "final_acc")
        print("| %s | %.3f | %.3f | %s | %.3f | %.1f | %.3f | %.1f | %.3f | %s | %d |" % (
            a, m, s, "/".join("%.2f" % x for x in v),
            agg(rows, a, "final_loss")[0], agg(rows, a, "eff_part")[0],
            agg(rows, a, "sparsity")[0], agg(rows, a, "kv_len")[0],
            agg(rows, a, "escape_frac")[0],
            ("%.3f" % agg(rows, a, "p_mean")[0]) if not (agg(rows, a, "p_mean")[0] != agg(rows, a, "p_mean")[0]) else "鈥?,
            agg(rows, a, "params")[0]))

    def d(x, y):
        if x not in arms or y not in arms:
            return float("nan")
        return agg(rows, x, "final_acc")[0] - agg(rows, y, "final_acc")[0]

    print("\n## 瑁佸喅\n")
    g13 = d("A3", "A0")
    ok1 = g13 >= 0.03 and all(
        (a > b) for a, b in zip([r["final_acc"] for r in rows if r["arm"] == "A3"],
                                [r["final_acc"] for r in rows if r["arm"] == "A0"]))
    print("- **H1 鍦烘湁鏁?*锛毼?A3鈭扐0) = %+.3f锛岄€?seed 鍚屽悜=%s 鈫?%s" % (g13, ok1, "鎴愮珛" if ok1 else "涓嶆垚绔?))
    g1 = d("A1", "A0")
    g2 = d("A2", "A0")
    ok2 = g13 > 0.5 * (g1 + g2) if not (g1 != g1 or g2 != g2) else None
    print("- **H2 褰掑洜**锛毼?A1鈭扐0)=%+.3f, 螖(A2鈭扐0)=%+.3f, 闃堝€?%.3f 鈫?%s" % (
        g1, g2, 0.5 * (g1 + g2), "鍦烘湁鑷繁鐨勮础鐚? if ok2 else "鍦虹殑澧炵泭琚?token 鍏堥獙瑙ｉ噴鎺?))
    g4 = d("A4", "A3")
    kv3, kv4 = agg(rows, "A3", "kv_len")[0], agg(rows, "A4", "kv_len")[0]
    ok3 = g4 >= -0.01 and kv4 <= 0.7 * kv3
    print("- **H3 閫冮€?*锛毼?A4鈭扐3)=%+.3f锛宬v %0.1f鈫?0.1f 鈫?%s" % (
        g4, kv3, kv4, "鎴愮珛" if ok3 else "涓嶆垚绔?))
    ep3, ep0 = agg(rows, "A3", "eff_part")[0], agg(rows, "A0", "eff_part")[0]
    ok4 = ep3 < 0.5 * ep0
    print("- **H4 鎴愭湰**锛歟ff_part A0=%.1f 鈫?A3=%.1f锛埫?.2f锛夆啋 %s" % (
        ep0, ep3, ep3 / ep0 if ep0 else float("nan"), "鎴愮珛" if ok4 else "涓嶆垚绔?))


if __name__ == "__main__":
    main()

