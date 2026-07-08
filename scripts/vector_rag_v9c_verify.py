"""v9c: did the injected cache entries get picked up? J-space readout + attention mass.

For each cache-delivery condition, run the capital question on the cache and
measure two things the behavioral scores can't separate:
  1. J-space broadcast: min rank of the country token under the Jacobian lens,
     over band layers x question positions (v6 protocol).
  2. Attention mass into the cache slots: attention weight from the final
     question position into the slot key columns, averaged over heads at the
     full-attention band layers (15/19/23).

Conditions (16 countries, capital question):
  floor, text_bound, embed_scaled (v9 synthetic), lens_add (v9 synthetic),
  replay_same (v9b recording, original positions), replay_offset (v9b, moved).
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
OUT = "results_v9c_verify.jsonl"
Q = "The capital of Kevin's home country is the city of"
FILLER = "The weather was pleasant that day."
ATTN_LAYERS = [15, 19, 23]

COUNTRIES = ["France", "Canada", "China", "Egypt", "Japan", "Germany", "Italy",
             "Spain", "Russia", "Turkey", "Greece", "Poland", "Kenya", "Peru",
             "Thailand", "Vietnam"]

jlens.configure_logging()
print(f"Loading {MODEL_NAME} (eager attention)...")
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, attn_implementation="eager").cuda()
tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(
    LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)

n = model.n_layers
band = [l for l in lens.source_layers if 0.35 <= l / n <= 0.75]
W_U = hf.lm_head.weight.detach().float().cpu()
E = hf.get_input_embeddings().weight.detach()


def toks(w):
    return tok.encode(" " + w, add_special_tokens=False)


def tid(w):
    return toks(w)[0]


def direction(l, w):
    v = lens.jacobians[l].T @ W_U[tid(w)]
    return (v / v.norm()).to("cuda", torch.bfloat16)


class BlockIO:
    def __init__(self, blocks):
        self.blocks = blocks
        self.mode = None
        self.captured = {}
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

    def _get_hidden(self, args, kwargs):
        if args and torch.is_tensor(args[0]):
            return args[0], "args"
        if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
            return kwargs["hidden_states"], "kwargs"
        return None, None

    def _hook(self, l):
        def hook(module, args, kwargs):
            if self.mode is None:
                return args, kwargs
            hidden, where = self._get_hidden(args, kwargs)
            if hidden is None:
                return args, kwargs
            if self.mode == "capture":
                self.captured[l] = hidden[0].detach().clone()
                return args, kwargs
            spec = self.plan.get(l) if self.plan else None
            if not spec:
                return args, kwargs
            hidden = hidden.clone()
            for pos, (m, vec) in spec.items():
                if m == "set":
                    hidden[:, pos, :] = vec.to(hidden.dtype)
                else:
                    hidden[:, pos, :] += vec.to(hidden.dtype)
            if where == "args":
                return (hidden,) + tuple(args[1:]), kwargs
            return args, dict(kwargs, hidden_states=hidden)
        return hook


def question_forward(past, n_slots):
    """Question forward with lens recorder + attentions. Returns (min_rank, attn_mass)."""
    cont = tok(Q, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with ActivationRecorder(model.layers, band) as rec, torch.no_grad():
        out = hf(cont, past_key_values=past, use_cache=True, output_attentions=True)
    # lens readout
    best = None
    for l in band:
        h = rec.activations[l][0].float().cpu()
        transported = (lens.jacobians[l] @ h.T).T
        logits = model.unembed(transported.to("cuda", torch.bfloat16)).float()
        r_ = int((logits > logits[:, TARGET:TARGET + 1]).sum(-1).min().item()) + 1
        best = r_ if best is None else min(best, r_)
    # attention mass into slot columns from the final question position
    att = [a for a in out.attentions if a is not None]
    FULL = [3, 7, 11, 15, 19, 23, 27, 31]
    lmap = dict(zip(FULL, att)) if len(att) != n else {i: a for i, a in enumerate(att)}
    masses = []
    for l in ATTN_LAYERS:
        a = lmap.get(l)
        if a is None:
            continue
        masses.append(a[0, :, -1, :n_slots].sum(-1).mean().item())
    return best, (sum(masses) / len(masses) if masses else None)


def text_cache(prefix):
    ids = model.encode(prefix)
    with torch.no_grad():
        return hf(ids, use_cache=True).past_key_values


results = []
with BlockIO(model.layers) as io:
    for c in COUNTRIES:
        globals()["TARGET"] = tid(c)
        fact = f" Kevin: {c}."
        fact_ids = tok.encode(fact, add_special_tokens=False)
        k = len(fact_ids)
        e_c = E[tid(c)].float().cuda()

        # mean norms for embed_scaled (cheap reuse: norms of question forward)
        # measured once per country from a plain question pass
        ids = model.encode(Q)
        with ActivationRecorder(model.layers, list(range(n))) as rec0, torch.no_grad():
            model.forward(ids)
        mean_norm = {l: rec0.activations[l].float().norm(dim=-1).mean().item()
                     for l in range(n)}

        # coef profile for lens_add
        p = f"Fact: Kevin's home country is {c}. " + Q
        pids = model.encode(p)
        ent_pos = (pids[0] == tid(c)).nonzero()[0].item()
        with ActivationRecorder(model.layers, band) as recc, torch.no_grad():
            model.forward(pids)
        dcache = {l: direction(l, c) for l in band}
        coef = {l: (recc.activations[l][0, ent_pos].float() @ dcache[l].float()).item()
                for l in band}

        def synth_cache(plan, n_slots):
            ph = tok.encode(" a" * n_slots, add_special_tokens=False)[:n_slots]
            sids = torch.tensor([ph]).cuda()
            io.plan = plan
            io.mode = "clamp"
            with torch.no_grad():
                past = hf(sids, use_cache=True).past_key_values
            io.mode = None
            io.plan = None
            return past

        cells = {}
        cells["floor"] = (None, 0)
        cells["text_bound"] = (text_cache(fact), k)
        cells["embed_scaled"] = (synth_cache(
            {l: {0: ("set", e_c / e_c.norm() * mean_norm[l])} for l in range(n)}, 1), 1)
        cells["lens_add"] = (synth_cache(
            {l: {0: ("add", coef[l] * dcache[l].float())} for l in band}, 1), 1)

        io.mode = "capture"; io.captured = {}
        with torch.no_grad():
            hf(torch.tensor([fact_ids]).cuda())
        io.mode = None
        donorA = io.captured
        cells["replay_same"] = (synth_cache(
            {l: {i: ("set", donorA[l][i]) for i in range(k)} for l in range(n)}, k), k)

        filler_ids = tok.encode(FILLER, add_special_tokens=False)
        off = len(filler_ids)
        io.mode = "capture"; io.captured = {}
        with torch.no_grad():
            hf(torch.tensor([filler_ids + fact_ids]).cuda())
        io.mode = None
        donorB = io.captured
        cells["replay_offset"] = (synth_cache(
            {l: {i: ("set", donorB[l][off + i]) for i in range(k)} for l in range(n)}, k), k)

        for name, (past, n_slots) in cells.items():
            if past is None:
                cont = model.encode(Q)
                with ActivationRecorder(model.layers, band) as rec, torch.no_grad():
                    hf(cont)
                best = None
                for l in band:
                    h = rec.activations[l][0].float().cpu()
                    tr = (lens.jacobians[l] @ h.T).T
                    lg = model.unembed(tr.to("cuda", torch.bfloat16)).float()
                    r_ = int((lg > lg[:, TARGET:TARGET + 1]).sum(-1).min().item()) + 1
                    best = r_ if best is None else min(best, r_)
                results.append(dict(exp=name, arg=c, lens_rank=best, attn_mass=None))
            else:
                rank, mass = question_forward(past, n_slots)
                results.append(dict(exp=name, arg=c, lens_rank=rank, attn_mass=mass))

CELLS = ["floor", "text_bound", "embed_scaled", "lens_add", "replay_same", "replay_offset"]
for cell in CELLS:
    rs = [r for r in results if r["exp"] == cell]
    ranks = sorted(r["lens_rank"] for r in rs)
    med = ranks[len(ranks) // 2]
    top10 = sum(r <= 10 for r in ranks)
    masses = [r["attn_mass"] for r in rs if r["attn_mass"] is not None]
    m = f"{sum(masses)/len(masses):.3f}" if masses else "n/a"
    print(f"[{cell:<14}] country-in-J-space top10={top10}/16 median_rank={med:<6} attn_mass_to_slots={m}")

with open(OUT, "w") as fh:
    for r in results:
        fh.write(json.dumps(r) + "\n")
print("RUN COMPLETE")
