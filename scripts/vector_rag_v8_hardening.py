"""v8: pre-publication hardening — precision cells + robust grading.

Battery: 4 names x 16 countries x 4 functions = 256 trials per cell.

Cells:
  floor        — indirection question only
  text         — fact sentence in prompt
  phantom      — bound cache entry " {name}: {c}."
  two_recall   — cache holds TWO bound entries; ask about the first person
  two_select   — same cache; ask about the OTHER person (precision: fetch the
                 right person's entity, not just "an" entity)
  leak         — cache " {name}: {c}." but the question is a direct factual
                 question about an unrelated country t (no referent). Measures
                 whether an irrelevant phantom corrupts unrelated questions.
  leak_base    — same direct question, no cache (paired baseline).

Graders recorded per trial:
  strict  — gold first-token is argmax over the full vocab (as v3-v7)
  hit5    — gold first-token in top-5
  cand    — argmax over the function's candidate answer tokens is gold's
Analysis: per-cell totals under all graders; exact McNemar phantom vs text
(strict); leak intrusion rate (top-1 equals the cached entity's answer).
"""

import json
import random
import zlib
from math import comb

import torch
import transformers

import jlens

MODEL_NAME = "Qwen/Qwen3.5-4B"
OUT = "results_v8_hardening.jsonl"
NAMES = ["Kevin", "Maria", "Omar", "Chen"]

DATA = {
    "France":   (["Paris"], ["French"], ["Europe"], ["Euro"]),
    "Canada":   (["Ottawa"], ["English", "French"], ["North"], ["Dollar"]),
    "China":    (["Beijing"], ["Chinese", "Mandarin"], ["Asia"], ["Yuan", "Renminbi"]),
    "Egypt":    (["Cairo"], ["Arabic"], ["Africa"], ["Pound"]),
    "Japan":    (["Tokyo"], ["Japanese"], ["Asia"], ["Yen"]),
    "Germany":  (["Berlin"], ["German"], ["Europe"], ["Euro"]),
    "Italy":    (["Rome"], ["Italian"], ["Europe"], ["Euro"]),
    "Spain":    (["Madrid"], ["Spanish"], ["Europe"], ["Euro"]),
    "Russia":   (["Moscow"], ["Russian"], ["Europe", "Asia"], ["Ruble", "Rouble"]),
    "Turkey":   (["Ankara"], ["Turkish"], ["Asia", "Europe"], ["Lira"]),
    "Greece":   (["Athens"], ["Greek"], ["Europe"], ["Euro"]),
    "Poland":   (["Warsaw"], ["Polish"], ["Europe"], ["Zloty"]),
    "Kenya":    (["Nairobi"], ["Swahili", "English"], ["Africa"], ["Shilling"]),
    "Peru":     (["Lima"], ["Spanish"], ["South"], ["Sol"]),
    "Thailand": (["Bangkok"], ["Thai"], ["Asia"], ["Baht"]),
    "Vietnam":  (["Hanoi"], ["Vietnamese"], ["Asia"], ["Dong"]),
}
FUNCS = [
    {"name": "capital", "idx": 0,
     "direct": "The capital of {c} is the city of",
     "indirect": "The capital of {name}'s home country is the city of"},
    {"name": "language", "idx": 1,
     "direct": "The official language of {c} is",
     "indirect": "The official language of {name}'s home country is"},
    {"name": "continent", "idx": 2,
     "direct": "{c} is a country on the continent of",
     "indirect": "{name}'s home country is on the continent of"},
    {"name": "currency", "idx": 3,
     "direct": "The currency used in {c} is called the",
     "indirect": "The currency used in {name}'s home country is called the"},
]
TEXT_PREFIX = "Fact: {name}'s home country is {c}."

jlens.configure_logging()
print(f"Loading {MODEL_NAME}...")
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16).cuda()
tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf, tok)

countries = list(DATA)


def toks(w):
    return tok.encode(" " + w, add_special_tokens=False)


def gold_id_set(answers):
    ids = set()
    for w in answers:
        for v in {w, w.lower(), w.capitalize(), w.upper()}:
            ids.add(toks(v)[0])
    return ids


# candidate token pool per function: first tokens of every country's answers
CAND = {f["name"]: sorted(set().union(
    *[gold_id_set(DATA[c][f["idx"]]) for c in countries])) for f in FUNCS}


def text_logits(prompt):
    ids = model.encode(prompt)
    with torch.no_grad():
        return hf(ids).logits[0, -1]


def phantom_logits(prefix, prompt):
    pre = model.encode(prefix)
    with torch.no_grad():
        past = hf(pre, use_cache=True).past_key_values
    cont = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        return hf(cont, past_key_values=past, use_cache=True).logits[0, -1]


results = []


def grade(logits, answers, meta, intrude_answers=None):
    logits = logits.float()
    gids = gold_id_set(answers)
    rank = min(int((logits > logits[g]).sum().item()) + 1 for g in gids)
    cand_ids = CAND[meta["func"]]
    cand_best = cand_ids[int(torch.argmax(logits[cand_ids]).item())]
    rec = dict(meta, gold=answers[0], gold_rank=rank,
               strict=rank == 1, hit5=rank <= 5, cand=cand_best in gids)
    if intrude_answers is not None:
        top1 = int(logits.argmax().item())
        rec["intrusion"] = top1 in gold_id_set(intrude_answers)
    results.append(rec)


for f in FUNCS:
    for c in countries:
        answers = DATA[c][f["idx"]]
        for name in NAMES:
            rng = random.Random(zlib.crc32(f"{f['name']}|{c}|{name}".encode()))
            other_name = rng.choice([x for x in NAMES if x != name])
            pool = [x for x in countries if x != c
                    and not (set(DATA[x][f["idx"]]) & set(answers))]
            other_c = rng.choice(pool)
            q = f["indirect"].format(name=name)
            q_other = f["indirect"].format(name=other_name)
            meta = dict(func=f["name"], arg=c, name=name)

            grade(text_logits(q), answers, dict(meta, exp="floor"))
            grade(text_logits(TEXT_PREFIX.format(name=name, c=c) + " " + q),
                  answers, dict(meta, exp="text"))
            grade(phantom_logits(f" {name}: {c}.", q), answers,
                  dict(meta, exp="phantom"))

            pair = [f" {name}: {c}.", f" {other_name}: {other_c}."]
            if rng.random() < 0.5:
                pair.reverse()
            cache2 = "".join(pair)
            grade(phantom_logits(cache2, q), answers,
                  dict(meta, exp="two_recall", other=other_c))
            grade(phantom_logits(cache2, q_other), DATA[other_c][f["idx"]],
                  dict(meta, exp="two_select", other=other_c),
                  intrude_answers=answers)

            t = other_c  # unrelated third country with disjoint answers
            dq = f["direct"].format(c=t)
            grade(text_logits(dq), DATA[t][f["idx"]],
                  dict(meta, exp="leak_base", other=t))
            grade(phantom_logits(f" {name}: {c}.", dq), DATA[t][f["idx"]],
                  dict(meta, exp="leak", other=t), intrude_answers=answers)

CELLS = ["floor", "text", "phantom", "two_recall", "two_select", "leak_base", "leak"]
print(f"\nn per cell: {len([r for r in results if r['exp']=='floor'])}")
for cell in CELLS:
    rs = [r for r in results if r["exp"] == cell]
    s = sum(r["strict"] for r in rs)
    h5 = sum(r["hit5"] for r in rs)
    cd = sum(r["cand"] for r in rs)
    extra = ""
    if any("intrusion" in r for r in rs):
        extra = f"  intrusion={sum(r.get('intrusion', False) for r in rs)}/{len(rs)}"
    print(f"[{cell:<10}] strict={s}/{len(rs)}  hit@5={h5}  cand={cd}{extra}")

# exact McNemar, phantom vs text, strict grading (paired by trial)
key = lambda r: (r["func"], r["arg"], r["name"])
ph = {key(r): r["strict"] for r in results if r["exp"] == "phantom"}
tx = {key(r): r["strict"] for r in results if r["exp"] == "text"}
b = sum(1 for k in ph if ph[k] and not tx[k])
c_ = sum(1 for k in ph if tx[k] and not ph[k])
n_disc = b + c_
if n_disc:
    p = sum(comb(n_disc, i) for i in range(0, min(b, c_) + 1)) / 2**n_disc * 2
    p = min(1.0, p)
else:
    p = 1.0
print(f"\nMcNemar phantom vs text (strict): phantom-only-correct={b}, "
      f"text-only-correct={c_}, exact p={p:.3f}")

with open(OUT, "w") as fh:
    for r in results:
        fh.write(json.dumps(r) + "\n")
print("RUN COMPLETE")
