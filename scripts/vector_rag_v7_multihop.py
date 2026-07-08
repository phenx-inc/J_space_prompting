"""v7: can a phantom entity serve as a multi-hop chain intermediate?

The cached entity is a capital CITY (" Kevin: Paris."). Every question first
requires the silent hop city -> country, then zero or one more knowledge hop:

  depth1 country   — "The country whose capital is Kevin's favorite city is"        (Paris -> France)
  depth2 continent — "The country whose capital is Kevin's favorite city is on the continent of"
  depth2 language  — "The official language of the country whose capital is Kevin's favorite city is"
  depth2 currency  — "The currency of the country whose capital is Kevin's favorite city is called the"

Cells: direct (city in text, template health) / floor / text-RAG / bound phantom.
Multi-token cities are dropped at runtime. Forward-only; no lens needed.
"""

import json

import torch
import transformers

import jlens

MODEL_NAME = "Qwen/Qwen3.5-4B"
OUT = "results_v7_multihop.jsonl"

# city: (country, continents, languages, currencies)
DATA = {
    "Paris":   ("France", ["Europe"], ["French"], ["Euro"]),
    "Ottawa":  ("Canada", ["North"], ["English", "French"], ["Dollar"]),
    "Beijing": ("China", ["Asia"], ["Chinese", "Mandarin"], ["Yuan", "Renminbi"]),
    "Cairo":   ("Egypt", ["Africa"], ["Arabic"], ["Pound"]),
    "Tokyo":   ("Japan", ["Asia"], ["Japanese"], ["Yen"]),
    "Berlin":  ("Germany", ["Europe"], ["German"], ["Euro"]),
    "Rome":    ("Italy", ["Europe"], ["Italian"], ["Euro"]),
    "Madrid":  ("Spain", ["Europe"], ["Spanish"], ["Euro"]),
    "Moscow":  ("Russia", ["Europe", "Asia"], ["Russian"], ["Ruble", "Rouble"]),
    "Ankara":  ("Turkey", ["Asia", "Europe"], ["Turkish"], ["Lira"]),
    "Athens":  ("Greece", ["Europe"], ["Greek"], ["Euro"]),
    "Warsaw":  ("Poland", ["Europe"], ["Polish"], ["Zloty"]),
    "Nairobi": ("Kenya", ["Africa"], ["Swahili", "English"], ["Shilling"]),
    "Lima":    ("Peru", ["South"], ["Spanish"], ["Sol"]),
    "Bangkok": ("Thailand", ["Asia"], ["Thai"], ["Baht"]),
    "Hanoi":   ("Vietnam", ["Asia"], ["Vietnamese"], ["Dong"]),
}
FUNCS = [
    {"name": "country", "depth": 1,
     "direct": "The country whose capital is {city} is",
     "indirect": "The country whose capital is Kevin's favorite city is",
     "answers": lambda d: [d[0]]},
    {"name": "continent", "depth": 2,
     "direct": "The country whose capital is {city} is on the continent of",
     "indirect": "The country whose capital is Kevin's favorite city is on the continent of",
     "answers": lambda d: d[1]},
    {"name": "language", "depth": 2,
     "direct": "The official language of the country whose capital is {city} is",
     "indirect": "The official language of the country whose capital is Kevin's favorite city is",
     "answers": lambda d: d[2]},
    {"name": "currency", "depth": 2,
     "direct": "The currency of the country whose capital is {city} is called the",
     "indirect": "The currency of the country whose capital is Kevin's favorite city is called the",
     "answers": lambda d: d[3]},
]
TEXT_PREFIX = "Fact: Kevin's favorite city is {city}."

jlens.configure_logging()
print(f"Loading {MODEL_NAME}...")
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16).cuda()
tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf, tok)


def toks(w):
    return tok.encode(" " + w, add_special_tokens=False)


cities = [c for c in DATA if len(toks(c)) == 1]
dropped = [c for c in DATA if c not in cities]
print(f"battery: {len(cities)} single-token cities (dropped: {dropped})")


def gold_id_set(answers):
    ids = set()
    for w in answers:
        for v in {w, w.lower(), w.capitalize(), w.upper()}:
            ids.add(toks(v)[0])
    return ids


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


def grade(logits, answers, meta):
    logits = logits.float()
    gids = gold_id_set(answers)
    rank = min(int((logits > logits[g]).sum().item()) + 1 for g in gids)
    results.append(dict(meta, gold=answers[0], gold_rank=rank, hit=rank == 1,
                        top5=[tok.decode([t]) for t in logits.topk(5).indices]))


for f in FUNCS:
    for city in cities:
        answers = f["answers"](DATA[city])
        meta = dict(func=f["name"], depth=f["depth"], arg=city)
        grade(text_logits(f["direct"].format(city=city)), answers,
              dict(meta, exp="direct"))
        grade(text_logits(f["indirect"]), answers, dict(meta, exp="floor"))
        grade(text_logits(TEXT_PREFIX.format(city=city) + " " + f["indirect"]),
              answers, dict(meta, exp="text"))
        grade(phantom_logits(f" Kevin: {city}.", f["indirect"]), answers,
              dict(meta, exp="phantom"))

for exp in ["direct", "floor", "text", "phantom"]:
    rs = [r for r in results if r["exp"] == exp]
    by_func = {}
    for f in FUNCS:
        fr = [r for r in rs if r["func"] == f["name"]]
        by_func[f["name"]] = f"{sum(r['hit'] for r in fr)}/{len(fr)}"
    print(f"[{exp:<8}] total={sum(r['hit'] for r in rs)}/{len(rs):<4} {by_func}")

with open(OUT, "w") as fh:
    for r in results:
        fh.write(json.dumps(r) + "\n")
print("RUN COMPLETE")
