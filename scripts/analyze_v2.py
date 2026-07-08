import json
import sys
from collections import defaultdict

rs = [json.loads(l) for l in open(sys.argv[1])]

print("--- per-function hits for key cells ---")
cells = ["full/referent/s4", "full/refcoef/x1", "low/refcoef/x1"]
for c in cells:
    by = defaultdict(lambda: [0, 0])
    for r in rs:
        if r["exp"] == "inject" and r.get("cell") == c:
            by[r["func"]][0] += r["hit"]
            by[r["func"]][1] += 1
    print(c, {f: f"{h}/{t}" for f, (h, t) in sorted(by.items())})

print("\n--- low/refcoef/x1 all trials ---")
for r in rs:
    if r["exp"] == "inject" and r.get("cell") == "low/refcoef/x1":
        mark = "HIT " if r["hit"] else "miss"
        print(f"{mark} {r['func']:>9} {r['arg']:<7} gold={r['gold']:<8} rank={r['gold_rank']:<5} top5={r['top5'][:4]}")

print("\n--- floor per function ---")
by = defaultdict(lambda: [0, 0])
for r in rs:
    if r["exp"] == "floor":
        by[r["func"]][0] += r["hit"]
        by[r["func"]][1] += 1
print({f: f"{h}/{t}" for f, (h, t) in sorted(by.items())})
