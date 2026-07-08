import json
import statistics
import sys

rs = [json.loads(l) for l in open(sys.argv[1])]

print("--- baseline misses ---")
for r in rs:
    if r["exp"] == "baseline" and not r["hit"]:
        print(f"{r['func']:>9} {r['arg']:<7} gold={r['gold']:<8} rank={r['gold_rank']:<5} top5={r['top5']}")

print("--- text-RAG misses ---")
for r in rs:
    if r["exp"] == "text" and not r["hit"]:
        print(f"{r['func']:>9} {r['arg']:<7} gold={r['gold']:<8} rank={r['gold_rank']:<5} top5={r['top5']}")

print("--- floor gold ranks (sorted) ---")
print(sorted(r["gold_rank"] for r in rs if r["exp"] == "floor"))

print("--- injection: gold rank stats by strength ---")
for s in [0.5, 1.0, 2.0, 4.0, 8.0]:
    ranks = [r["gold_rank"] for r in rs if r["exp"] == "vector" and r.get("strength") == s]
    print(f"s={s:<4} median={statistics.median(ranks):<8} min={min(ranks):<6} max={max(ranks)}")

print("--- injection detail: capital @ s=2 and s=8 ---")
for s in (2.0, 8.0):
    for r in rs:
        if r["exp"] == "vector" and r.get("strength") == s and r["func"] == "capital":
            print(f"s={s} {r['arg']:<7} gold={r['gold']:<8} rank={r['gold_rank']:<6} top5={r['top5']}")

print("--- swap misses by func ---")
from collections import Counter
miss = Counter((r["func"]) for r in rs if r["exp"] == "swap" and not r["hit"])
tot = Counter((r["func"]) for r in rs if r["exp"] == "swap")
for f in tot:
    print(f"{f:>9}: {tot[f]-miss[f]}/{tot[f]}")
