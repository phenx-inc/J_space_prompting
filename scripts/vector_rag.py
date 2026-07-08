"""Vector-RAG probe: can J-space injection substitute for retrieved text?

Three experiments on Qwen3.5-4B + pre-fitted Neuronpedia Jacobian lens:
  A) baseline  — flexgen countries templates, no intervention (grading sanity)
  B) swap      — paper's flexible-generalization swap replication (harness sanity)
  C) vectorrag — indirection prompts ("Kevin's home country") under
                 floor / text-RAG / J-space injection at several strengths

Protocol follows jacobian-lens/data/experiments/README.md:
  - steering direction for token t at layer l: unit-normalized J_l^T @ W_U[t]
  - injection: h += strength * mean_resid_norm[l] * v_hat, every band layer,
    every prompt position
  - swap: coefficient transfer h += (h . v_a)(v_b - v_a)
  - grading: greedy next token at final position vs gold first token

Usage:
  HF_HOME=... .venv/bin/python vector_rag.py --out results_vector_rag.jsonl
"""

import argparse
import json

import torch
import transformers

import jlens

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
BAND_DEPTH = (0.35, 0.75)  # fractional-depth window for the workspace band
STRENGTHS = [0.5, 1.0, 2.0, 4.0, 8.0]

COUNTRIES = {
    "args": ["France", "Canada", "China", "Egypt"],
    "funcs": [
        {"name": "capital", "template": "The capital of {arg} is the city of",
         "indirect": "The capital of Kevin's home country is the city of",
         "answers": {"Canada": "Ottawa", "China": "Beijing", "Egypt": "Cairo", "France": "Paris"}},
        {"name": "language", "template": "Most people in {arg} speak",
         "indirect": "Most people in Kevin's home country speak",
         "answers": {"Canada": "English", "China": "Chinese", "Egypt": "Arabic", "France": "French"}},
        {"name": "continent", "template": "{arg} is a country on the continent of",
         "indirect": "Kevin's home country is on the continent of",
         "answers": {"Canada": "North", "China": "Asia", "Egypt": "Africa", "France": "Europe"}},
        {"name": "currency", "template": "The single-word name for the currency now used in {arg} is the",
         "indirect": "The single-word name for the currency now used in Kevin's home country is the",
         "answers": {"Canada": "Dollar", "China": "Yuan", "Egypt": "Pound", "France": "Euro"}},
    ],
}
TEXT_RAG_PREFIX = "Fact: Kevin's home country is {arg}. "


def first_token_id(tokenizer, word):
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0]


class Injector:
    """Forward hooks on band layers adding/steering along lens directions."""

    def __init__(self, blocks, band):
        self.blocks = blocks
        self.band = band
        self.mode = None  # None | ("inject", {l: vec}) | ("swap", {l: (va, vb)})
        self.handles = []

    def __enter__(self):
        for l in self.band:
            self.handles.append(self.blocks[l].register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _hook(self, l):
        def hook(module, inputs, output):
            if self.mode is None:
                return output
            hidden = output if torch.is_tensor(output) else output[0]
            kind, payload = self.mode
            if kind == "inject":
                vec = payload[l]  # already strength*norm-scaled, model dtype
                hidden = hidden + vec
            elif kind == "swap":
                va, vb = payload[l]  # unit vectors, model dtype
                coef = (hidden * va).sum(-1, keepdim=True)
                hidden = hidden + coef * (vb - va)
            if torch.is_tensor(output):
                return hidden
            return (hidden,) + tuple(output[1:])
        return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_vector_rag.jsonl")
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
    band = [l for l in lens.source_layers if BAND_DEPTH[0] <= l / n <= BAND_DEPTH[1]]
    print(f"n_layers={n} lens_layers={lens.source_layers} band={band}")

    W_U = hf.lm_head.weight.detach().float().cpu()  # [vocab, d]

    def direction(l, word):
        t = first_token_id(tok, word)
        v = lens.jacobians[l].T @ W_U[t]
        return (v / v.norm()).to("cuda", torch.bfloat16)

    dirs = {a: {l: direction(l, a) for l in band} for a in COUNTRIES["args"]}

    # mean residual norm per band layer over the baseline prompts
    base_prompts = [f["template"].format(arg=a)
                    for f in COUNTRIES["funcs"] for a in COUNTRIES["args"]]
    from jlens.hooks import ActivationRecorder

    norms = {l: [] for l in band}
    for p in base_prompts:
        ids = model.encode(p)
        with ActivationRecorder(model.layers, band) as rec, torch.no_grad():
            model.forward(ids)
        for l in band:
            norms[l].append(rec.activations[l].float().norm(dim=-1).mean().item())
    mean_norm = {l: sum(v) / len(v) for l, v in norms.items()}
    print("mean residual norms:", {l: round(x, 1) for l, x in mean_norm.items()})

    results = []

    def score(prompt, injector, mode, gold_word, meta):
        injector.mode = mode
        ids = model.encode(prompt)
        with torch.no_grad():
            out = hf(ids)
        injector.mode = None
        logits = out.logits[0, -1].float()
        gold = first_token_id(tok, gold_word)
        rank = int((logits > logits[gold]).sum().item()) + 1
        top = [tok.decode([t]) for t in logits.topk(5).indices]
        rec = dict(meta, prompt=prompt, gold=gold_word, gold_rank=rank,
                   hit=rank == 1, top5=top)
        results.append(rec)
        return rec

    with Injector(model.layers, band) as inj:
        # A) baseline
        print("\n[A] baseline (no intervention)")
        for f in COUNTRIES["funcs"]:
            for a in COUNTRIES["args"]:
                r = score(f["template"].format(arg=a), inj, None, f["answers"][a],
                          dict(exp="baseline", func=f["name"], arg=a))
                print(f"  {f['name']:>9} {a:<7} gold={r['gold']:<8} rank={r['gold_rank']:<4} top={r['top5'][:3]}")

        # B) swap replication
        print("\n[B] flexgen swap (coefficient transfer, every prompt position)")
        hits = 0; tot = 0
        for f in COUNTRIES["funcs"]:
            for a in COUNTRIES["args"]:
                for b in COUNTRIES["args"]:
                    if a == b:
                        continue
                    mode = ("swap", {l: (dirs[a][l], dirs[b][l]) for l in band})
                    r = score(f["template"].format(arg=a), inj, mode, f["answers"][b],
                              dict(exp="swap", func=f["name"], arg=a, swap_to=b))
                    hits += r["hit"]; tot += 1
        print(f"  swap success: {hits}/{tot}")

        # C) vector-RAG
        print("\n[C] vector-RAG (indirection prompts)")
        for f in COUNTRIES["funcs"]:
            for a in COUNTRIES["args"]:
                gold = f["answers"][a]
                score(f["indirect"], inj, None, gold,
                      dict(exp="floor", func=f["name"], arg=a))
                score(TEXT_RAG_PREFIX.format(arg=a) + f["indirect"], inj, None, gold,
                      dict(exp="text", func=f["name"], arg=a))
                for s in STRENGTHS:
                    mode = ("inject", {l: (s * mean_norm[l] * dirs[a][l].float()).to(torch.bfloat16)
                                       for l in band})
                    score(f["indirect"], inj, mode, gold,
                          dict(exp="vector", func=f["name"], arg=a, strength=s))

    with open(args.out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    def acc(exp, **kw):
        rs = [r for r in results if r["exp"] == exp
              and all(r.get(k) == v for k, v in kw.items())]
        return f"{sum(r['hit'] for r in rs)}/{len(rs)}"

    print("\n===== SUMMARY =====")
    print(f"baseline (direct prompts):        {acc('baseline')}")
    print(f"swap replication:                 {acc('swap')}")
    print(f"vector-RAG floor (no info):       {acc('floor')}")
    print(f"vector-RAG text ceiling:          {acc('text')}")
    for s in STRENGTHS:
        print(f"vector-RAG inject strength {s:<4}:   {acc('vector', strength=s)}")
    print("RUN COMPLETE")


if __name__ == "__main__":
    main()
