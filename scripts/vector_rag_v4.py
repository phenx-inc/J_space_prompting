"""Vector-RAG v4: attention-guided injection sites + phantom KV entity slot.

New conditions vs v3:
  attn_refcoef — coefficient-matched injection at the top-2 prompt positions the
                 final position attends to (mean over heads at full-attention
                 layers 15/19/23, excluding BOS-sink and the final position).
  phantom      — entity token " {arg}" forwarded first to build a 1-token cache;
                 indirect prompt continues from that cache. Entity never in text.
  phantom_dot  — same with " {arg}." (2-token) prefix.

Reruns floor / text / refcoef_low for an apples-to-apples table.
Qwen3.5-4B is hybrid attention: full attention only at layers 3,7,11,...,31.
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
OUT = "results_vector_rag_v4.jsonl"
ATTN_LAYERS = [15, 19, 23]  # full-attention layers inside the workspace band
TOP_K_SITES = 2

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
    MODEL_NAME, dtype=torch.bfloat16, attn_implementation="eager").cuda()
tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(
    LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)

n = model.n_layers
band_low = [l for l in lens.source_layers if 0.35 <= l / n <= 0.55]
band_full = [l for l in lens.source_layers if 0.35 <= l / n <= 0.75]

W_U = hf.lm_head.weight.detach().float().cpu()


def toks(word):
    return tok.encode(" " + word, add_special_tokens=False)


def tid(word):
    return toks(word)[0]


countries = [c for c in DATA if len(toks(c)) == 1]
print(f"battery: {len(countries)} countries x {len(FUNCS)} functions")


def direction(l, word):
    v = lens.jacobians[l].T @ W_U[tid(word)]
    return (v / v.norm()).to("cuda", torch.bfloat16)


dirs = {c: {l: direction(l, c) for l in band_full} for c in countries}

# natural coefficient profiles (from text-RAG prompt, entity position)
coef = {}
for c in countries:
    p = TEXT_RAG_PREFIX.format(arg=c) + FUNCS[0]["indirect"]
    ids = model.encode(p)
    ent_pos = (ids[0] == tid(c)).nonzero()[0].item()
    with ActivationRecorder(model.layers, band_full) as rec, torch.no_grad():
        model.forward(ids)
    coef[c] = {l: (rec.activations[l][0, ent_pos].float() @ dirs[c][l].float()).item()
               for l in band_full}

# --- attention-guided site selection (one clean pass per indirect prompt) ---
attn_sites = {}
for f in FUNCS:
    ids = model.encode(f["indirect"])
    with torch.no_grad():
        out = hf(ids, output_attentions=True)
    att = out.attentions
    FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
    if len(att) == n:
        layer_map = {i: a for i, a in enumerate(att) if a is not None}
    else:
        layer_map = dict(zip(FULL_ATTN_LAYERS, [a for a in att if a is not None]))
    use = [layer_map[l] for l in ATTN_LAYERS if layer_map.get(l) is not None]
    assert use, f"no attention maps for layers {ATTN_LAYERS} (got {len(att)} entries)"
    # score source positions by the final position's attention, mean over heads/layers
    score_vec = torch.stack([a[0, :, -1, :].mean(0) for a in use]).mean(0)
    score_vec[0] = 0.0          # BOS / sink
    score_vec[-1] = 0.0         # self
    sites = score_vec.topk(TOP_K_SITES).indices.tolist()
    toks_dec = [tok.decode([t]) for t in ids[0].tolist()]
    attn_sites[f["name"]] = sites
    print(f"[sites] {f['name']:>9}: {[(s, toks_dec[s]) for s in sites]}")

results = []


def gold_id_set(answers):
    ids = set()
    for w in answers:
        for v in {w, w.lower(), w.capitalize(), w.upper()}:
            ids.add(toks(v)[0])
    return ids


def grade(logits, answers, meta):
    logits = logits.float()
    gids = gold_id_set(answers)
    rank = min(int((logits > logits[g]).sum().item()) + 1 for g in gids)
    results.append(dict(meta, gold=answers[0], gold_rank=rank, hit=rank == 1,
                        top5=[tok.decode([t]) for t in logits.topk(5).indices]))


def forward_logits(prompt, injector=None, plan=None):
    if injector is not None:
        injector.plan = plan
    ids = model.encode(prompt)
    with torch.no_grad():
        out = hf(ids)
    if injector is not None:
        injector.plan = None
    return out.logits[0, -1]


def phantom_logits(prefix_text, prompt):
    pre_ids = model.encode(prefix_text)
    with torch.no_grad():
        pre = hf(pre_ids, use_cache=True)
    cont = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        out = hf(cont, past_key_values=pre.past_key_values, use_cache=True)
    return out.logits[0, -1]


def inject_plan(c, band, positions):
    return {l: (positions, (coef[c][l] * dirs[c][l].float()).to(torch.bfloat16))
            for l in band}


with Injector(model.layers, band_full) as inj:
    for f in FUNCS:
        ids = model.encode(f["indirect"])
        ref_pos = torch.tensor(
            [i for i, t in enumerate(ids[0].tolist())
             if t == tok.encode(" country", add_special_tokens=False)[0]],
            device=ids.device)
        attn_pos = torch.tensor(attn_sites[f["name"]], device=ids.device)
        for c in countries:
            answers = DATA[c][f["idx"]]

            grade(forward_logits(f["indirect"]), answers,
                  dict(exp="floor", func=f["name"], arg=c))
            grade(forward_logits(TEXT_RAG_PREFIX.format(arg=c) + f["indirect"]),
                  answers, dict(exp="text", func=f["name"], arg=c))
            grade(forward_logits(f["indirect"], inj, inject_plan(c, band_low, ref_pos)),
                  answers, dict(exp="refcoef_low", func=f["name"], arg=c))
            grade(forward_logits(f["indirect"], inj, inject_plan(c, band_low, attn_pos)),
                  answers, dict(exp="attn_refcoef", func=f["name"], arg=c))
            grade(phantom_logits(" " + c, f["indirect"]), answers,
                  dict(exp="phantom", func=f["name"], arg=c))
            grade(phantom_logits(" " + c + ".", f["indirect"]), answers,
                  dict(exp="phantom_dot", func=f["name"], arg=c))

for cell in ["floor", "text", "refcoef_low", "attn_refcoef", "phantom", "phantom_dot"]:
    rs = [r for r in results if r["exp"] == cell]
    by_func = {}
    for f in FUNCS:
        fr = [r for r in rs if r["func"] == f["name"]]
        by_func[f["name"]] = f"{sum(r['hit'] for r in fr)}/{len(fr)}"
    print(f"[{cell:<13}] total={sum(r['hit'] for r in rs)}/{len(rs):<4} {by_func}")

with open(OUT, "w") as fh:
    for r in results:
        fh.write(json.dumps(r) + "\n")
print("RUN COMPLETE")
