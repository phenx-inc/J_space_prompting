"""Vector-RAG v5: noise robustness. Text-RAG vs phantom KV under distractors.

For each trial, k distractor facts about other people are generated
("Fact: Maria's home country is Japan.") and the target fact is placed at a
seeded-random position among them. Three delivery lanes share the same fact
order:

  text_k     — all facts as visible prompt text ("Fact: {name}'s home country is {c}.")
  unbound_k  — cache-only prefix of bare entities (" Japan. ... France.") — no names
  bound_k    — cache-only prefix with minimal binding (" Maria: Japan. ... Kevin: France.")

k in {0, 4, 16, 48}. Distractor countries are constrained so their accepted
answers for the trial's function do not overlap the target's. Grading as v3/v4.
"""

import argparse
import json
import random
import zlib

import torch
import transformers

import jlens

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--out", default="results_vector_rag_v5.jsonl")
args = ap.parse_args()
MODEL_NAME = args.model
OUT = args.out
KS = [0, 4, 16, 48]

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
     "indirect": "The capital of Kevin's home country is the city of"},
    {"name": "language", "idx": 1,
     "indirect": "The official language of Kevin's home country is"},
    {"name": "continent", "idx": 2,
     "indirect": "Kevin's home country is on the continent of"},
    {"name": "currency", "idx": 3,
     "indirect": "The currency used in Kevin's home country is called the"},
]
NAMES = ["Maria", "Omar", "Chen", "Anna", "David", "Yuki", "Ivan", "Sara",
         "Leila", "Marco", "Priya", "Tom", "Nadia", "Hugo", "Aisha", "Felix",
         "Ingrid", "Ravi", "Sofia", "Diego", "Emma", "Tariq", "Lena", "Pablo",
         "Mei", "Andre", "Zara", "Noah", "Fatima", "Lucas", "Hana", "Viktor",
         "Amara", "Jonas", "Keiko", "Samuel", "Elif", "Mateo", "Ania", "Kofi",
         "Elena", "Bruno", "Aiko", "Stefan", "Rosa", "Dmitri", "Alice", "Farid"]

jlens.configure_logging()
print(f"Loading {MODEL_NAME}...")
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16).cuda()
tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf, tok)

countries = list(DATA)


def toks(word):
    return tok.encode(" " + word, add_special_tokens=False)


def gold_id_set(answers):
    ids = set()
    for w in answers:
        for v in {w, w.lower(), w.capitalize(), w.upper()}:
            ids.add(toks(v)[0])
    return ids


def make_facts(func_idx, target, k, seed):
    """k distractor (name, country) pairs + the target, in seeded-random order."""
    rng = random.Random(seed)
    target_answers = set(DATA[target][func_idx])
    pool = [c for c in countries
            if c != target and not (set(DATA[c][func_idx]) & target_answers)]
    names = rng.sample(NAMES, k) if k <= len(NAMES) else [
        rng.choice(NAMES) + str(i) for i in range(k)]
    facts = [(names[i], rng.choice(pool)) for i in range(k)]
    pos = rng.randint(0, k)
    facts.insert(pos, ("Kevin", target))
    return facts, pos


results = []


def grade(logits, answers, meta):
    logits = logits.float()
    gids = gold_id_set(answers)
    rank = min(int((logits > logits[g]).sum().item()) + 1 for g in gids)
    results.append(dict(meta, gold=answers[0], gold_rank=rank, hit=rank == 1,
                        top5=[tok.decode([t]) for t in logits.topk(5).indices]))


def text_logits(prompt):
    ids = model.encode(prompt)
    with torch.no_grad():
        out = hf(ids)
    return out.logits[0, -1]


def phantom_logits(prefix_text, prompt):
    pre_ids = model.encode(prefix_text)
    with torch.no_grad():
        pre = hf(pre_ids, use_cache=True)
    cont = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        out = hf(cont, past_key_values=pre.past_key_values, use_cache=True)
    return out.logits[0, -1]


for k in KS:
    for f in FUNCS:
        for c in countries:
            answers = DATA[c][f["idx"]]
            seed = zlib.crc32(f"{f['name']}|{c}|{k}".encode())
            facts, pos = make_facts(f["idx"], c, k, seed)
            meta_base = dict(func=f["name"], arg=c, k=k, target_pos=pos)

            text_prefix = " ".join(
                f"Fact: {n}'s home country is {cc}." for n, cc in facts) + " "
            grade(text_logits(text_prefix + f["indirect"]), answers,
                  dict(meta_base, exp="text"))

            unbound_prefix = " " + " ".join(f"{cc}." for _, cc in facts)
            grade(phantom_logits(unbound_prefix, f["indirect"]), answers,
                  dict(meta_base, exp="unbound"))

            bound_prefix = " " + " ".join(f"{n}: {cc}." for n, cc in facts)
            grade(phantom_logits(bound_prefix, f["indirect"]), answers,
                  dict(meta_base, exp="bound"))

    for exp in ["text", "unbound", "bound"]:
        rs = [r for r in results if r["k"] == k and r["exp"] == exp]
        by_func = {}
        for f in FUNCS:
            fr = [r for r in rs if r["func"] == f["name"]]
            by_func[f["name"]] = f"{sum(r['hit'] for r in fr)}/{len(fr)}"
        print(f"[k={k:>2} {exp:<8}] total={sum(r['hit'] for r in rs)}/{len(rs):<4} {by_func}")

with open(OUT, "w") as fh:
    for r in results:
        fh.write(json.dumps(r) + "\n")
print("RUN COMPLETE")
