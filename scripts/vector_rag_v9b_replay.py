"""v9b: donor-residual replay — precomputed contextualized residuals as cache content.

v9 showed vector-built cache entries fail when the vectors are embeddings or
lens directions. The missing variant: harvest the REAL per-layer residual
trajectory of " Kevin: France." from a donor text pass (offline), then clamp
those residuals into placeholder positions at query time, letting the model
project them into K/V + recurrent state. Text is used only offline.

Cells (64 trials each):
  floor          — question only
  text_bound     — real-token prefix " Kevin: France." (reference, ~95%)
  replay_same    — donor residuals harvested at positions 0..k-1, replayed at
                   the same positions (implementation sanity; should match text)
  replay_offset  — donor harvested at positions 8..8+k-1 (after filler text),
                   replayed at prefix positions 0..k-1 (position portability)
  replay_clip    — only the entity tokens' residuals (" France" + ".") replayed,
                   dropping the "Kevin:" slots (is binding baked into the
                   entity's contextualized residuals?)
"""

import json

import torch
import transformers

import jlens

MODEL_NAME = "Qwen/Qwen3.5-4B"
OUT = "results_v9b_replay.jsonl"
FILLER = "The weather was pleasant that day."

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
n = model.n_layers
countries = list(DATA)


def toks(w):
    return tok.encode(" " + w, add_special_tokens=False)


def gold_id_set(answers):
    ids = set()
    for w in answers:
        for v in {w, w.lower(), w.capitalize(), w.upper()}:
            ids.add(toks(v)[0])
    return ids


class BlockIO:
    """Pre-hooks on every block: capture block inputs, or clamp them.

    capture: dict layer -> tensor [seq, d] of block inputs (set mode='capture')
    clamp:   plan {layer: {pos: vec}} applied with mode='clamp'
    """

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
            for pos, vec in spec.items():
                hidden[:, pos, :] = vec.to(hidden.dtype)
            if where == "args":
                return (hidden,) + tuple(args[1:]), kwargs
            return args, dict(kwargs, hidden_states=hidden)
        return hook


def capture_donor(io, prefix_text):
    """Run donor text; return {layer: [seq, d]} of block inputs."""
    ids = tok(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    io.mode = "capture"
    io.captured = {}
    with torch.no_grad():
        hf(ids)
    io.mode = None
    return io.captured, ids[0].tolist()


def replay_cache(io, donor, src_positions, n_slots, slot_map=None):
    """Build a cache from placeholder tokens whose block inputs are clamped to
    donor residuals. slot_map: {slot_pos: src_pos}; default identity over
    src_positions -> 0..k-1."""
    if slot_map is None:
        slot_map = {i: p for i, p in enumerate(src_positions)}
    placeholder = tok.encode(" a" * n_slots, add_special_tokens=False)[:n_slots]
    ids = torch.tensor([placeholder]).cuda()
    io.plan = {l: {slot: donor[l][src] for slot, src in slot_map.items()}
               for l in range(n)}
    io.mode = "clamp"
    with torch.no_grad():
        past = hf(ids, use_cache=True).past_key_values
    io.mode = None
    io.plan = None
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


with BlockIO(model.layers) as io:
    for f in FUNCS:
        for c in countries:
            answers = DATA[c][f["idx"]]
            q = f["indirect"]
            meta = dict(func=f["name"], arg=c)
            fact = f" Kevin: {c}."
            fact_ids = tok.encode(fact, add_special_tokens=False)
            k = len(fact_ids)

            grade(question_logits(q), answers, dict(meta, exp="floor"))
            grade(question_logits(q, text_cache(fact)), answers,
                  dict(meta, exp="text_bound"))

            # donor A: fact alone at positions 0..k-1
            donorA, _ = capture_donor(io, fact)
            grade(question_logits(q, replay_cache(io, donorA, list(range(k)), k)),
                  answers, dict(meta, exp="replay_same"))

            # donor B: fact after filler; harvest at its offset positions,
            # replay at prefix positions 0..k-1
            filler_ids = tok.encode(FILLER, add_special_tokens=False)
            off = len(filler_ids)
            donorB, _ = capture_donor(io, FILLER + fact)
            grade(question_logits(q, replay_cache(
                io, donorB, list(range(off, off + k)), k)),
                answers, dict(meta, exp="replay_offset"))

            # clip: only the entity token(s) + final period from donor A
            ent_first = toks(c)[0]
            ent_positions = [i for i, t in enumerate(fact_ids) if t == ent_first]
            clip_src = list(range(ent_positions[0], k))  # entity tokens + "."
            grade(question_logits(q, replay_cache(
                io, donorA, clip_src, len(clip_src))),
                answers, dict(meta, exp="replay_clip"))

CELLS = ["floor", "text_bound", "replay_same", "replay_offset", "replay_clip"]
for cell in CELLS:
    rs = [r for r in results if r["exp"] == cell]
    by_func = {}
    for f in FUNCS:
        fr = [r for r in rs if r["func"] == f["name"]]
        by_func[f["name"]] = f"{sum(r['hit'] for r in fr)}/{len(fr)}"
    print(f"[{cell:<14}] total={sum(r['hit'] for r in rs)}/{len(rs):<4} {by_func}")

with open(OUT, "w") as fh:
    for r in results:
        fh.write(json.dumps(r) + "\n")
print("RUN COMPLETE")
