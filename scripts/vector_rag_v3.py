"""Vector-RAG v3: scaled battery. 16 countries x 4 functions x 6 conditions.

Conditions:
  direct   — real entity in the direct template, no intervention (template health)
  floor    — indirection prompt, no info
  text     — indirection prompt + fact sentence (text-RAG ceiling)
  ref_s4   — uniform-strength injection at referent tokens, full band
  refcoef_full / refcoef_low — coefficient-matched injection at referent tokens

Grading: greedy next token vs a set of accepted answers (case variants included),
first-token matching. Countries whose name is not a single token are dropped.
"""

import json

import torch
import transformers

import jlens
from jlens.hooks import ActivationRecorder

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
OUT = "results_vector_rag_v3.jsonl"

# country: (capital, languages, continents, currencies) — lists = accepted answers
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
     "direct": "The capital of {arg} is the city of",
     "indirect": "The capital of Kevin's home country is the city of"},
    {"name": "language", "idx": 1,
     "direct": "The official language of {arg} is",
     "indirect": "The official language of Kevin's home country is"},
    {"name": "continent", "idx": 2,
     "direct": "{arg} is a country on the continent of",
     "indirect": "Kevin's home country is on the continent of"},
    {"name": "currency", "idx": 3,
     "direct": "The currency used in {arg} is called the",
     "indirect": "The currency used in Kevin's home country is called the"},
]
TEXT_RAG_PREFIX = "Fact: Kevin's home country is {arg}. "


class Injector:
    def __init__(self, blocks, layers):
        self.blocks = blocks
        self.layers = layers
        self.plan = None
        self.handles = []

    def __enter__(self):
        for l in self.layers:
            self.handles.append(self.blocks[l].register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _hook(self, l):
        def hook(module, inputs, output):
            if self.plan is None or l not in self.plan:
                return output
            hidden = output if torch.is_tensor(output) else output[0]
            pos, vec = self.plan[l]
            hidden = hidden.clone()
            hidden[:, pos] += vec
            if torch.is_tensor(output):
                return hidden
            return (hidden,) + tuple(output[1:])
        return hook


jlens.configure_logging()
print(f"Loading {MODEL_NAME}...")
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16).cuda()
tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(
    LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)

n = model.n_layers
band_full = [l for l in lens.source_layers if 0.35 <= l / n <= 0.75]
band_low = [l for l in lens.source_layers if 0.35 <= l / n <= 0.55]


def toks(word):
    return tok.encode(" " + word, add_special_tokens=False)


# keep only single-token countries
countries = []
for c in DATA:
    if len(toks(c)) == 1:
        countries.append(c)
    else:
        print(f"DROPPED (multi-token): {c}")
print(f"battery: {len(countries)} countries x {len(FUNCS)} functions")

W_U = hf.lm_head.weight.detach().float().cpu()


def tid(word):
    return toks(word)[0]


def direction(l, word):
    v = lens.jacobians[l].T @ W_U[tid(word)]
    return (v / v.norm()).to("cuda", torch.bfloat16)


dirs = {c: {l: direction(l, c) for l in band_full} for c in countries}

# mean residual norms over indirection prompts
norms = {l: [] for l in band_full}
for f in FUNCS:
    ids = model.encode(f["indirect"])
    with ActivationRecorder(model.layers, band_full) as rec, torch.no_grad():
        model.forward(ids)
    for l in band_full:
        norms[l].append(rec.activations[l].float().norm(dim=-1).mean().item())
mean_norm = {l: sum(v) / len(v) for l, v in norms.items()}

# natural coefficient profiles
coef = {}
for c in countries:
    p = TEXT_RAG_PREFIX.format(arg=c) + FUNCS[0]["indirect"]
    ids = model.encode(p)
    pos = (ids[0] == tid(c)).nonzero()
    assert len(pos) > 0, c
    ent_pos = pos[0].item()
    with ActivationRecorder(model.layers, band_full) as rec, torch.no_grad():
        model.forward(ids)
    coef[c] = {l: (rec.activations[l][0, ent_pos].float() @ dirs[c][l].float()).item()
               for l in band_full}

COUNTRY_TOKEN_IDS = {tok.encode(" country", add_special_tokens=False)[0]}


def referent_positions(ids):
    pos = [i for i, t in enumerate(ids[0].tolist()) if t in COUNTRY_TOKEN_IDS]
    assert pos
    return torch.tensor(pos, device=ids.device)


results = []


def gold_id_set(answers):
    ids = set()
    for w in answers:
        for v in {w, w.lower(), w.capitalize(), w.upper()}:
            ids.add(toks(v)[0])
    return ids


def score(prompt, injector, plan, answers, meta):
    injector.plan = plan
    ids = model.encode(prompt)
    with torch.no_grad():
        out = hf(ids)
    injector.plan = None
    logits = out.logits[0, -1].float()
    gids = gold_id_set(answers)
    rank = min(int((logits > logits[g]).sum().item()) + 1 for g in gids)
    top1 = int(logits.argmax().item())
    results.append(dict(meta, gold=answers[0], gold_rank=rank, hit=rank == 1,
                        top1_is_entity=top1 == tid(meta["arg"]),
                        top5=[tok.decode([t]) for t in logits.topk(5).indices]))


def inject_plan(c, band, positions, strength=None, use_coef=False):
    plan = {}
    for l in band:
        mag = coef[c][l] if use_coef else strength * mean_norm[l]
        plan[l] = (positions, (mag * dirs[c][l].float()).to(torch.bfloat16))
    return plan


CELLS = [
    ("direct", None, None, None),
    ("floor", None, None, None),
    ("text", None, None, None),
    ("ref_s4", band_full, "referent", dict(strength=4.0)),
    ("refcoef_full", band_full, "referent", dict(use_coef=True)),
    ("refcoef_low", band_low, "referent", dict(use_coef=True)),
]

with Injector(model.layers, band_full) as inj:
    for cell, band, postype, kw in CELLS:
        for f in FUNCS:
            for c in countries:
                answers = DATA[c][f["idx"]]
                meta = dict(exp=cell, func=f["name"], arg=c)
                if cell == "direct":
                    score(f["direct"].format(arg=c), inj, None, answers, meta)
                elif cell == "floor":
                    score(f["indirect"], inj, None, answers, meta)
                elif cell == "text":
                    score(TEXT_RAG_PREFIX.format(arg=c) + f["indirect"], inj, None,
                          answers, meta)
                else:
                    ids = model.encode(f["indirect"])
                    plan = inject_plan(c, band, referent_positions(ids), **kw)
                    score(f["indirect"], inj, plan, answers, meta)
        rs = [r for r in results if r["exp"] == cell]
        by_func = {}
        for f in FUNCS:
            fr = [r for r in rs if r["func"] == f["name"]]
            by_func[f["name"]] = f"{sum(r['hit'] for r in fr)}/{len(fr)}"
        total = f"{sum(r['hit'] for r in rs)}/{len(rs)}"
        print(f"[{cell:<13}] total={total:<7} {by_func}")

with open(OUT, "w") as fh:
    for r in results:
        fh.write(json.dumps(r) + "\n")
print("RUN COMPLETE")
