"""Prove the engine works on this machine before the server depends on it.

Three checks, one per finding:
  1. read    — 'Italy' ignites on a prompt that never says Italy
  2. steer   — inject France at the referent, the model answers Paris
  3. break   — inject France everywhere, the model just parrots 'France'

Device is auto-picked, so this doubles as the MPS/CPU compatibility check.

    cd jscope && HF_HUB_DISABLE_XET=1 ../.venv/bin/python smoke.py
"""

import time

from engine import RealEngine, pick_device

BOOT = "Fact: The currency used in the country shaped like a boot is"
CAP = "The capital of Kevin's home country is the city of"

print(f"device: {pick_device()}")
t0 = time.time()
eng = RealEngine()
print(f"loaded in {time.time() - t0:.0f}s  dtype={eng.dtype}  device={eng.device}")

meta = eng.meta()
band = meta["default_band"]
print(f"{meta['model']}  {meta['n_layers']} layers  lens on {len(meta['source_layers'])}")
print(f"band (depth 0.35-0.75): L{band[0]}-L{band[-1]}")

# --- 1. read ------------------------------------------------------------
print(f"\n=== READ: {BOOT!r} ===")
t0 = time.time()
r = eng.readout(BOOT, ["Italy"])
toks = [t["text"] for t in r["tokens"]]
print(f"readout in {time.time() - t0:.1f}s over {len(toks)} tokens")

# The concept does not live at the last token. It lives where the riddle resolves.
hits = [(r["cells"][str(l)][p]["concepts"]["Italy"], l, p)
        for l in r["layers"] for p in range(len(toks))]
hits.sort(reverse=True)
print("\nwhere 'Italy' is loudest:")
for v, l, p in hits[:5]:
    top = [d["tok"] for d in r["cells"][str(l)][p]["top"][:3]]
    print(f"  L{l:>2} pos{p:>2} {toks[p]!r:<10} p={v:.4f}  lens reads: {top}")

best, bl, bp = hits[0]
ok_read = best > 0.05
print(f"\n{'OK  ' if ok_read else 'FAIL'} 'Italy' peaks at p={best:.4f} on {toks[bp]!r} (L{bl}), "
      f"and is never written in the prompt.")

# --- 2. steer -----------------------------------------------------------
print(f"\n=== STEER: {CAP!r} ===")
ids = eng.model.encode(CAP)[0].tolist()
ctry = eng.tid("country")
ref = [i for i, t in enumerate(ids) if t == ctry]
allpos = list(range(len(ids)))
nat = eng.natural_alpha(CAP, "France", band)
print(f"referent ' country' at {ref}   natural alpha = {nat['mean']:.2f}")

def line(sites, a):
    d = eng.inject(CAP, "France", band, sites, a, watch=["Paris"])
    paris = d["watch"][0]["inj"] if d["watch"] else 0.0
    return paris, d["p_parrot_inj"], d["kl"], d["top1_inj"]

print(f"\n  aimed at the referent:")
print(f"  {'alpha':>6} {'P(Paris)':>9} {'parrot':>8} {'KL':>6}  top-1")
steered = 0.0
for a in [0.0, nat["mean"], 0.5, 1.0, 2.0]:
    paris, parrot, kl, t1 = line(ref, a)
    steered = max(steered, paris)
    print(f"  {a:>6.2f} {paris:>9.4f} {parrot:>8.4f} {kl:>6.2f}  {t1!r}")

# --- 3. break -----------------------------------------------------------
print(f"\n  smeared over every position (the naive route that scored 0%):")
print(f"  {'alpha':>6} {'P(Paris)':>9} {'parrot':>8} {'KL':>6}  top-1")
parroted = 0.0
for a in [0.0, nat["mean"], 1.0]:
    paris, parrot, kl, t1 = line(allpos, a)
    parroted = max(parroted, parrot)
    print(f"  {a:>6.2f} {paris:>9.4f} {parrot:>8.4f} {kl:>6.2f}  {t1!r}")

ok_steer = steered > 0.10
ok_break = parroted > 0.5
print(f"\n{'OK  ' if ok_steer else 'FAIL'} aimed injection makes it answer Paris (peak {steered:.3f}) "
      f"without saying 'France'.")
print(f"{'OK  ' if ok_break else 'FAIL'} smeared injection makes it parrot 'France' (peak {parroted:.3f}) "
      f"instead of answering.")

print("\nSMOKE OK" if (ok_read and ok_steer and ok_break) else "\nSMOKE FAILED")
