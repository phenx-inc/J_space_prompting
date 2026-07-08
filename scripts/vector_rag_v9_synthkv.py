"""v9: synthetic KV entries — a phantom slot whose content never existed as text.

Mechanism: forward placeholder tokens as a cache prefix, but overwrite (or
perturb) the residual stream entering EVERY decoder block at the placeholder
positions with synthetic vectors. The model's own attention layers then project
those vectors into position-stamped K/V (and recurrent state) — a cache entry
built from a vector, not from text. The question runs on top of that cache with
no hooks.

Cells (16 countries x 4 functions = 64 trials each):
  floor         — question only
  text_unbound  — real-token prefix " France."           (reference, ~84% at 4B)
  text_bound    — real-token prefix " Kevin: France."    (reference, ~95%)
  embed_raw     — 1 slot clamped to E[" France"] (raw input embedding, all blocks)
  embed_scaled  — 1 slot clamped to E[" France"] rescaled to the layer's mean
                  residual norm (per block)
  lens_add      — 1 slot: placeholder's natural residual + coef(l)*v_lens(l)
                  at band layers (residual injection aimed at a dedicated,
                  re-readable cache slot)
  embed_scaled_bound — 2 slots: E[" Kevin"], E[" France"], both norm-matched
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
OUT = "results_v9_synthkv.jsonl"

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
E = hf.get_input_embeddings().weight.detach()


def toks(w):
    return tok.encode(" " + w, add_special_tokens=False)


def tid(w):
    return toks(w)[0]


def direction(l, w):
    v = lens.jacobians[l].T @ W_U[tid(w)]
    return (v / v.norm()).to("cuda", torch.bfloat16)


dirs = {c: {l: direction(l, c) for l in band} for c in countries}

# mean residual norm entering each block (approximate: recorded at block outputs
# of the previous layer; use block-output norms per layer over indirection prompts)
norms = {l: [] for l in range(n)}
for f in FUNCS:
    ids = model.encode(f["indirect"])
    with ActivationRecorder(model.layers, list(range(n))) as rec, torch.no_grad():
        model.forward(ids)
    for l in range(n):
        norms[l].append(rec.activations[l].float().norm(dim=-1).mean().item())
mean_norm = {l: sum(v) / len(v) for l, v in norms.items()}

# natural coefficient profile for lens_add (from bound text prefix, as v6)
coef = {}
for c in countries:
    p = f"Fact: Kevin's home country is {c}. " + FUNCS[0]["indirect"]
    ids = model.encode(p)
    ent_pos = (ids[0] == tid(c)).nonzero()[0].item()
    with ActivationRecorder(model.layers, band) as rec, torch.no_grad():
        model.forward(ids)
    coef[c] = {l: (rec.activations[l][0, ent_pos].float() @ dirs[c][l].float()).item()
               for l in band}


class SynthClamp:
    """Pre-hooks on every block: overwrite/perturb residual at given positions.

    plan: {layer: {pos: ("set", vec) or ("add", vec)}}; layer=-1 applies the
    same spec at every block.
    """

    def __init__(self, blocks):
        self.blocks = blocks
        self.plan = None
        self.handles = []

    def __enter__(self):
        for l, blk in enumerate(self.blocks):
            self.handles.append(
                blk.register_forward_pre_hook(self._hook(l), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _hook(self, l):
        def hook(module, args, kwargs):
            if self.plan is None:
                return args, kwargs
            spec = self.plan.get(l, self.plan.get(-1))
            if not spec:
                return args, kwargs
            if args and torch.is_tensor(args[0]):
                hidden = args[0].clone()
                for pos, (mode, vec) in spec.items():
                    if mode == "set":
                        hidden[:, pos, :] = vec.to(hidden.dtype)
                    else:
                        hidden[:, pos, :] += vec.to(hidden.dtype)
                return (hidden,) + tuple(args[1:]), kwargs
            if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
                hidden = kwargs["hidden_states"].clone()
                for pos, (mode, vec) in spec.items():
                    if mode == "set":
                        hidden[:, pos, :] = vec.to(hidden.dtype)
                    else:
                        hidden[:, pos, :] += vec.to(hidden.dtype)
                kwargs = dict(kwargs, hidden_states=hidden)
            return args, kwargs
        return hook


def gold_id_set(answers):
    ids = set()
    for w in answers:
        for v in {w, w.lower(), w.capitalize(), w.upper()}:
            ids.add(toks(v)[0])
    return ids


PLACEHOLDER = tok.encode(" a a", add_special_tokens=False)  # 2 neutral slots


def synth_cache(clamp, plan, n_slots):
    ids = torch.tensor([PLACEHOLDER[:n_slots]]).cuda()
    clamp.plan = plan
    with torch.no_grad():
        past = hf(ids, use_cache=True).past_key_values
    clamp.plan = None
    return past


def question_logits(prompt, past=None):
    if past is None:
        ids = model.encode(prompt)
        with torch.no_grad():
            return hf(ids).logits[0, -1]
    cont = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        return hf(cont, past_key_values=past, use_cache=True).logits[0, -1]


def text_cache(prefix):
    ids = model.encode(prefix)
    with torch.no_grad():
        return hf(ids, use_cache=True).past_key_values


results = []


def grade(logits, answers, meta):
    logits = logits.float()
    gids = gold_id_set(answers)
    rank = min(int((logits > logits[g]).sum().item()) + 1 for g in gids)
    results.append(dict(meta, gold=answers[0], gold_rank=rank, hit=rank == 1,
                        top5=[tok.decode([t]) for t in logits.topk(5).indices]))


with SynthClamp(model.layers) as clamp:
    for f in FUNCS:
        for c in countries:
            answers = DATA[c][f["idx"]]
            q = f["indirect"]
            meta = dict(func=f["name"], arg=c)
            e_c = E[tid(c)].float().cuda()
            e_k = E[tid("Kevin")].float().cuda()

            grade(question_logits(q), answers, dict(meta, exp="floor"))
            grade(question_logits(q, text_cache(f" {c}.")), answers,
                  dict(meta, exp="text_unbound"))
            grade(question_logits(q, text_cache(f" Kevin: {c}.")), answers,
                  dict(meta, exp="text_bound"))

            # embed_raw: slot0 = raw embedding, every block
            plan = {-1: {0: ("set", e_c)}}
            grade(question_logits(q, synth_cache(clamp, plan, 1)), answers,
                  dict(meta, exp="embed_raw"))

            # embed_scaled: per-block rescale to layer mean residual norm
            plan = {l: {0: ("set", e_c / e_c.norm() * mean_norm[l])} for l in range(n)}
            grade(question_logits(q, synth_cache(clamp, plan, 1)), answers,
                  dict(meta, exp="embed_scaled"))

            # lens_add: natural placeholder residual + coef(l)*v_lens(l) at band
            plan = {l: {0: ("add", coef[c][l] * dirs[c][l].float())} for l in band}
            grade(question_logits(q, synth_cache(clamp, plan, 1)), answers,
                  dict(meta, exp="lens_add"))

            # embed_scaled_bound: slot0 Kevin, slot1 country, both norm-matched
            plan = {l: {0: ("set", e_k / e_k.norm() * mean_norm[l]),
                        1: ("set", e_c / e_c.norm() * mean_norm[l])} for l in range(n)}
            grade(question_logits(q, synth_cache(clamp, plan, 2)), answers,
                  dict(meta, exp="embed_scaled_bound"))

CELLS = ["floor", "text_unbound", "text_bound", "embed_raw", "embed_scaled",
         "lens_add", "embed_scaled_bound"]
for cell in CELLS:
    rs = [r for r in results if r["exp"] == cell]
    by_func = {}
    for f in FUNCS:
        fr = [r for r in rs if r["func"] == f["name"]]
        by_func[f["name"]] = f"{sum(r['hit'] for r in fr)}/{len(fr)}"
    print(f"[{cell:<18}] total={sum(r['hit'] for r in rs)}/{len(rs):<4} {by_func}")

with open(OUT, "w") as fh:
    for r in results:
        fh.write(json.dumps(r) + "\n")
print("RUN COMPLETE")
