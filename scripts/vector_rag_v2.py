"""Vector-RAG v2: position-targeted and coefficient-matched J-space injection.

v1 finding: uniform injection at every position saturates output ("say France").
v2 variants (all on indirection prompts, graded case-insensitively):
  - all       : v1 reference (every position, uniform strength)
  - prompt    : every position EXCEPT final (no direct output bias)
  - referent  : only at the tokens of "country" in "Kevin's home country"
  - refcoef   : referent positions, per-layer magnitude matched to the entity's
                natural coefficient profile measured from the text-RAG prompt

Bands: full (0.35-0.75 depth) and low (0.35-0.55 depth).
"""

import argparse
import json

import torch
import transformers

import jlens
from jlens.hooks import ActivationRecorder

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"

COUNTRIES = {
    "args": ["France", "Canada", "China", "Egypt"],
    "funcs": [
        {"name": "capital", "indirect": "The capital of Kevin's home country is the city of",
         "answers": {"Canada": "Ottawa", "China": "Beijing", "Egypt": "Cairo", "France": "Paris"}},
        {"name": "language", "indirect": "Most people in Kevin's home country speak",
         "answers": {"Canada": "English", "China": "Chinese", "Egypt": "Arabic", "France": "French"}},
        {"name": "continent", "indirect": "Kevin's home country is on the continent of",
         "answers": {"Canada": "North", "China": "Asia", "Egypt": "Africa", "France": "Europe"}},
        {"name": "currency", "indirect": "The single-word name for the currency now used in Kevin's home country is the",
         "answers": {"Canada": "Dollar", "China": "Yuan", "Egypt": "Pound", "France": "Euro"}},
    ],
}
TEXT_RAG_PREFIX = "Fact: Kevin's home country is {arg}. "


class Injector:
    def __init__(self, blocks, layers):
        self.blocks = blocks
        self.layers = layers
        self.plan = None  # {layer: (positions_tensor_or_None_or_'not_last', vec[d])}
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
            if pos is None:
                hidden += vec
            elif pos == "not_last":
                hidden[:, :-1] += vec
            else:
                hidden[:, pos] += vec
            if torch.is_tensor(output):
                return hidden
            return (hidden,) + tuple(output[1:])
        return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_vector_rag_v2.jsonl")
    args = ap.parse_args()

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
    print(f"band_full={band_full} band_low={band_low}")

    W_U = hf.lm_head.weight.detach().float().cpu()

    def tid(word):
        return tok.encode(" " + word, add_special_tokens=False)[0]

    def gold_ids(word):
        ids = set()
        for w in {word, word.lower(), word.capitalize()}:
            ids.add(tok.encode(" " + w, add_special_tokens=False)[0])
        return ids

    def direction(l, word):
        v = lens.jacobians[l].T @ W_U[tid(word)]
        return (v / v.norm()).to("cuda", torch.bfloat16)

    dirs = {a: {l: direction(l, a) for l in band_full} for a in COUNTRIES["args"]}

    # mean residual norms (over indirection prompts)
    ind_prompts = [f["indirect"] for f in COUNTRIES["funcs"]]
    norms = {l: [] for l in band_full}
    for p in ind_prompts:
        ids = model.encode(p)
        with ActivationRecorder(model.layers, band_full) as rec, torch.no_grad():
            model.forward(ids)
        for l in band_full:
            norms[l].append(rec.activations[l].float().norm(dim=-1).mean().item())
    mean_norm = {l: sum(v) / len(v) for l, v in norms.items()}

    # natural coefficient profile: (h . v_hat) at the entity token in text-RAG prompts
    coef = {}
    for a in COUNTRIES["args"]:
        p = TEXT_RAG_PREFIX.format(arg=a) + COUNTRIES["funcs"][0]["indirect"]
        ids = model.encode(p)
        ent = tid(a)
        pos = (ids[0] == ent).nonzero()
        assert len(pos) > 0, f"entity token for {a} not found"
        ent_pos = pos[0].item()
        with ActivationRecorder(model.layers, band_full) as rec, torch.no_grad():
            model.forward(ids)
        coef[a] = {l: (rec.activations[l][0, ent_pos].float() @ dirs[a][l].float()).item()
                   for l in band_full}
    print("coef profile France:", {l: round(c, 1) for l, c in coef["France"].items()})

    def referent_positions(ids):
        # positions of the "country" token(s) in the prompt
        cids = {tok.encode(" country", add_special_tokens=False)[0],
                tok.encode(" Country", add_special_tokens=False)[0]}
        pos = [i for i, t in enumerate(ids[0].tolist()) if t in cids]
        assert pos, "no ' country' token found"
        return torch.tensor(pos, device=ids.device)

    results = []

    def score(prompt, injector, plan, gold_word, meta):
        injector.plan = plan
        ids = model.encode(prompt)
        with torch.no_grad():
            out = hf(ids)
        injector.plan = None
        logits = out.logits[0, -1].float()
        gids = gold_ids(gold_word)
        rank = min(int((logits > logits[g]).sum().item()) + 1 for g in gids)
        top1 = int(logits.argmax().item())
        top = [tok.decode([t]) for t in logits.topk(5).indices]
        results.append(dict(meta, prompt=prompt[:60], gold=gold_word, gold_rank=rank,
                            hit=rank == 1, top1_is_entity=top1 == tid(meta.get("arg", "")),
                            top5=top))
        return results[-1]

    def inject_plan(a, band, positions, strength=None, use_coef=False, scale=1.0):
        plan = {}
        for l in band:
            if use_coef:
                mag = coef[a][l] * scale
            else:
                mag = strength * mean_norm[l]
            plan[l] = (positions, (mag * dirs[a][l].float()).to(torch.bfloat16))
        return plan

    CELLS = []
    for band_name, band in [("full", band_full), ("low", band_low)]:
        CELLS += [(f"{band_name}/all/s2", band, "all", dict(strength=2.0))]
        for s in (2.0, 4.0):
            CELLS.append((f"{band_name}/prompt/s{s:g}", band, "not_last", dict(strength=s)))
        for s in (2.0, 4.0, 8.0, 16.0):
            CELLS.append((f"{band_name}/referent/s{s:g}", band, "referent", dict(strength=s)))
        for sc in (1.0, 2.0, 4.0):
            CELLS.append((f"{band_name}/refcoef/x{sc:g}", band, "referent", dict(use_coef=True, scale=sc)))

    with Injector(model.layers, band_full) as inj:
        # regraded floor and text ceiling
        for f in COUNTRIES["funcs"]:
            for a in COUNTRIES["args"]:
                gold = f["answers"][a]
                score(f["indirect"], inj, None, gold, dict(exp="floor", func=f["name"], arg=a))
                score(TEXT_RAG_PREFIX.format(arg=a) + f["indirect"], inj, None, gold,
                      dict(exp="text", func=f["name"], arg=a))

        for cell_name, band, postype, kw in CELLS:
            for f in COUNTRIES["funcs"]:
                for a in COUNTRIES["args"]:
                    ids = model.encode(f["indirect"])
                    if postype == "all":
                        positions = None
                    elif postype == "not_last":
                        positions = "not_last"
                    else:
                        positions = referent_positions(ids)
                    plan = inject_plan(a, band, positions, **kw)
                    score(f["indirect"], inj, plan, f["answers"][a],
                          dict(exp="inject", cell=cell_name, func=f["name"], arg=a))
            done = [r for r in results if r["exp"] == "inject" and r["cell"] == cell_name]
            hits = sum(r["hit"] for r in done)
            sat = sum(r["top1_is_entity"] for r in done)
            print(f"[{cell_name:<22}] hits={hits:>2}/16 entity-saturated={sat:>2}/16")

    with open(args.out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    def acc(exp, **kw):
        rs = [r for r in results if r["exp"] == exp
              and all(r.get(k) == v for k, v in kw.items())]
        return f"{sum(r['hit'] for r in rs)}/{len(rs)}"

    print("\n===== SUMMARY (case-insensitive grading) =====")
    print(f"floor: {acc('floor')}   text ceiling: {acc('text')}")
    for cell_name, *_ in CELLS:
        print(f"inject {cell_name:<22}: {acc('inject', cell=cell_name)}")
    print("RUN COMPLETE")


if __name__ == "__main__":
    main()
