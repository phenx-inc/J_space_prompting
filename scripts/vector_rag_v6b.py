"""v6b: where does the causal locus live — the cache entry or the residual copy?

Cells (64 trials each, bound phantom " Kevin: {c}."):
  phantom             — reference, no intervention
  erase_question      — erase target J-direction during the question forward only (v6 result: ~no effect)
  erase_prefix        — erase during the prefix forward only (corrupts the stored KV/state)
  erase_both          — erase during both passes
Control: erase_prefix_ctrl — prefix-pass erasure with a different country's direction.
"""

import json
import random
import zlib

import torch
import transformers

import jlens

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
OUT = "results_v6b_prefix_erase.jsonl"

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
            v = self.plan[l]
            coef = (hidden.float() @ v.float()).to(hidden.dtype)
            hidden = hidden - coef.unsqueeze(-1) * v
            if torch.is_tensor(output):
                return hidden
            return (hidden,) + tuple(output[1:])
        return hook


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
            prefix = f" Kevin: {c}."
            for name, prefix_plan, cont_plan in [
                ("phantom", None, None),
                ("erase_question", None, dirs[c]),
                ("erase_prefix", dirs[c], None),
                ("erase_both", dirs[c], dirs[c]),
                ("erase_prefix_ctrl", dirs[other], None),
            ]:
                er.plan = prefix_plan
                pre_ids = model.encode(prefix)
                with torch.no_grad():
                    past = hf(pre_ids, use_cache=True).past_key_values
                er.plan = cont_plan
                cont = tok(f["indirect"], return_tensors="pt",
                           add_special_tokens=False).input_ids.cuda()
                with torch.no_grad():
                    logits = hf(cont, past_key_values=past,
                                use_cache=True).logits[0, -1].float()
                er.plan = None
                gids = gold_id_set(answers)
                rank = min(int((logits > logits[g]).sum().item()) + 1 for g in gids)
                results.append(dict(exp=name, func=f["name"], arg=c,
                                    gold=answers[0], gold_rank=rank, hit=rank == 1))

for name in ["phantom", "erase_question", "erase_prefix", "erase_both", "erase_prefix_ctrl"]:
    rs = [r for r in results if r["exp"] == name]
    by_func = {}
    for f in FUNCS:
        fr = [r for r in rs if r["func"] == f["name"]]
        by_func[f["name"]] = f"{sum(r['hit'] for r in fr)}/{len(fr)}"
    print(f"[{name:<18}] total={sum(r['hit'] for r in rs)}/{len(rs):<4} {by_func}")

with open(OUT, "w") as fh:
    for r in results:
        fh.write(json.dumps(r) + "\n")
print("RUN COMPLETE")
