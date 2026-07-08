"""v6: does phantom-KV retrieval route through the global workspace?

Part A — readout. With " Kevin: {c}." as a cache-only prefix, run the capital
question and read the J-lens over the continuation: does the country appear in
the workspace even though it exists only in cache? Controls: floor (no
prefix, country should be absent) and text-RAG (country in prompt, should be
present).

Part B — causal mediation. Grade the full 64-trial battery under:
  phantom          — bound phantom, no intervention (reference)
  phantom_erase    — erase the target country's J-direction at band layers,
                     question positions only
  phantom_ctrl     — same erasure with a different country's direction
                     (matched intervention control)
  text_erase       — text-RAG with target-direction erasure at question
                     positions (is text retrieval workspace-mediated too?)
  text             — text-RAG reference

If phantom_erase collapses while phantom_ctrl holds, cache retrieval routes
through the workspace directions the lens identifies.
"""

import json
import random
import zlib

import torch
import transformers

import jlens
from jlens.hooks import ActivationRecorder

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
OUT = "results_v6_mediation.jsonl"

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
TEXT_PREFIX = "Fact: Kevin's home country is {c}. "

jlens.configure_logging()
print(f"Loading {MODEL_NAME}...")
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16).cuda()
tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(
    LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)

n = model.n_layers
band = [l for l in lens.source_layers if 0.35 <= l / n <= 0.75]
countries = list(DATA)
W_U = hf.lm_head.weight.detach().float().cpu()


def toks(w):
    return tok.encode(" " + w, add_special_tokens=False)


def tid(w):
    return toks(w)[0]


def direction(l, w):
    v = lens.jacobians[l].T @ W_U[tid(w)]
    return (v / v.norm()).to("cuda", torch.bfloat16)


dirs = {c: {l: direction(l, c) for l in band} for c in countries}


class Eraser:
    def __init__(self, blocks, layers):
        self.blocks = blocks
        self.layers = layers
        self.plan = None  # {layer: unit_vec}; applied to ALL positions of the pass
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
            v = self.plan[l]
            coef = (hidden.float() @ v.float()).to(hidden.dtype)
            hidden = hidden - coef.unsqueeze(-1) * v
            if torch.is_tensor(output):
                return hidden
            return (hidden,) + tuple(output[1:])
        return hook


def prefix_cache(prefix_text):
    ids = model.encode(prefix_text)
    with torch.no_grad():
        return hf(ids, use_cache=True).past_key_values


def cont_forward(prompt, past=None, record=None):
    if past is not None:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            out = hf(ids, past_key_values=past, use_cache=True)
    else:
        ids = model.encode(prompt)
        with torch.no_grad():
            out = hf(ids)
    return out.logits[0, -1]


# ---------- Part A: J-space readout ----------
print("\n[A] J-lens readout during the capital question (min rank of country over band x positions)")
Q = FUNCS[0]["indirect"]
readout_rows = []
for cond in ["floor", "phantom", "text"]:
    ranks = []
    for c in countries:
        if cond == "phantom":
            past = prefix_cache(f" Kevin: {c}.")
            ids = tok(Q, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        elif cond == "text":
            past = None
            ids = model.encode(TEXT_PREFIX.format(c=c) + Q)
        else:
            past = None
            ids = tok(Q, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with ActivationRecorder(model.layers, band) as rec, torch.no_grad():
            if past is not None:
                hf(ids, past_key_values=past, use_cache=True)
            else:
                hf(ids)
        target = tid(c)
        best = None
        for l in band:
            h = rec.activations[l][0].float().cpu()          # [P, d]
            if cond == "text":
                h = h[-ids.shape[1]:]  # same span either way; keep all for text
            transported = (lens.jacobians[l] @ h.T).T          # [P, d]
            logits = model.unembed(transported.to("cuda", torch.bfloat16)).float()
            r = int((logits > logits[:, target:target + 1]).sum(-1).min().item()) + 1
            best = r if best is None else min(best, r)
        ranks.append(best)
        readout_rows.append(dict(part="readout", cond=cond, arg=c, min_rank=best))
    hits = sum(r <= 10 for r in ranks)
    med = sorted(ranks)[len(ranks) // 2]
    print(f"  {cond:<8} country in J-space top-10: {hits}/16, median best rank {med}")

# ---------- Part B: erasure mediation ----------
def gold_id_set(answers):
    ids = set()
    for w in answers:
        for v in {w, w.lower(), w.capitalize(), w.upper()}:
            ids.add(toks(v)[0])
    return ids


results = []
with Eraser(model.layers, band) as er:
    for f in FUNCS:
        for c in countries:
            answers = DATA[c][f["idx"]]
            rng = random.Random(zlib.crc32(f"{f['name']}|{c}".encode()))
            other = rng.choice([x for x in countries if x != c])
            text_prefix = TEXT_PREFIX.format(c=c).rstrip()
            cells = [
                ("phantom", f" Kevin: {c}.", None),
                ("phantom_erase", f" Kevin: {c}.", dirs[c]),
                ("phantom_ctrl", f" Kevin: {c}.", dirs[other]),
                ("text", text_prefix, None),
                ("text_erase", text_prefix, dirs[c]),
            ]
            for name, prefix, plan in cells:
                past = prefix_cache(prefix)
                prompt = f["indirect"]
                er.plan = plan
                logits = cont_forward(prompt, past=past).float()
                er.plan = None
                gids = gold_id_set(answers)
                rank = min(int((logits > logits[g]).sum().item()) + 1 for g in gids)
                results.append(dict(part="mediation", exp=name, func=f["name"], arg=c,
                                    gold=answers[0], gold_rank=rank, hit=rank == 1))

print("\n[B] erasure mediation (64 trials per cell)")
for name in ["phantom", "phantom_erase", "phantom_ctrl", "text", "text_erase"]:
    rs = [r for r in results if r["exp"] == name]
    by_func = {}
    for f in FUNCS:
        fr = [r for r in rs if r["func"] == f["name"]]
        by_func[f["name"]] = f"{sum(r['hit'] for r in fr)}/{len(fr)}"
    print(f"  [{name:<14}] total={sum(r['hit'] for r in rs)}/{len(rs):<4} {by_func}")

with open(OUT, "w") as fh:
    for r in readout_rows + results:
        fh.write(json.dumps(r) + "\n")
print("RUN COMPLETE")
