"""J-Scope engine: read concepts out of the workspace, write concepts into it.

Two implementations behind one interface.

RealEngine loads Qwen3.5 and Anthropic's pre-fitted Jacobian lens on CUDA. The
readout is `lens.apply()`. The injection is the hook math from
`scripts/vector_rag_v3.py`, unchanged: a concept word becomes a unit direction
via the lens Jacobian, and its amplitude is expressed as a multiple of the mean
residual norm at that layer, so alpha means the same thing at every depth.

MockEngine needs no torch and no GPU. It returns structurally identical payloads
built from a seeded RNG, so the front end can be developed and exercised away
from the lab network. Its numbers are fiction and it says so in `meta()`.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Iterable

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"

# v3's band: the middle of the stack, where the workspace is legible.
BAND_LO, BAND_HI = 0.35, 0.75

# v3 measured the natural amplitude of an entity by putting the fact in the text
# and reading the residual at the entity's own token. This carrier reproduces that.
NATURAL_CARRIER = "Fact: Kevin's home country is {concept}. "

MAX_PROMPT_TOKENS = 64


def band_layers(source_layers: Iterable[int], n_layers: int, lo=BAND_LO, hi=BAND_HI):
    return [l for l in source_layers if lo <= l / n_layers <= hi]


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RealEngine:
    mock = False

    def __init__(self, model_name=MODEL_NAME, device=None, dtype=None):
        import torch
        import transformers

        import jlens

        self.torch = torch
        jlens.configure_logging()

        device = device or pick_device()
        # bf16 everywhere it is supported; CPU matmuls in bf16 are painfully slow.
        if dtype is None:
            dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.dtype = dtype

        self.hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype
        ).to(device)
        self.tok = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = jlens.from_hf(self.hf, self.tok)
        self.lens = jlens.JacobianLens.from_pretrained(
            LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
        )

        self.device = device
        self.model_name = model_name
        self.n_layers = self.model.n_layers
        self.source_layers = sorted(int(l) for l in self.lens.source_layers)
        self.W_U = self.hf.lm_head.weight.detach().float().cpu()
        self.vocab_size = int(self.W_U.shape[0])

        self._dir_cache: dict[tuple[int, str], object] = {}
        self._norm_cache: dict[str, dict[int, float]] = {}

    # -- vocabulary helpers -------------------------------------------------

    def tid(self, word: str) -> int:
        """Leading-space convention, matching v3's `toks()`."""
        ids = self.tok.encode(" " + word.strip(), add_special_tokens=False)
        if not ids:
            raise ValueError(f"{word!r} does not tokenize")
        return ids[0]

    def is_single_token(self, word: str) -> bool:
        return len(self.tok.encode(" " + word.strip(), add_special_tokens=False)) == 1

    def meta(self):
        return {
            "model": self.model_name,
            "lens": f"{LENS_REPO}@{LENS_REVISION}",
            "n_layers": self.n_layers,
            "source_layers": self.source_layers,
            "default_band": band_layers(self.source_layers, self.n_layers),
            "vocab_size": self.vocab_size,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "natural_carrier": NATURAL_CARRIER,
            "mock": False,
        }

    def tokenize(self, prompt: str):
        ids = self.model.encode(prompt)[0].tolist()
        if len(ids) > MAX_PROMPT_TOKENS:
            raise ValueError(
                f"prompt is {len(ids)} tokens; the grid is capped at {MAX_PROMPT_TOKENS}"
            )
        return [
            {"i": i, "id": int(t), "text": self.tok.decode([t])}
            for i, t in enumerate(ids)
        ]

    # -- read ---------------------------------------------------------------

    def readout(self, prompt, concepts, layers=None, use_jacobian=True, topk=5):
        """The ignition grid: for every (layer, position), what the lens verbalizes.

        Returns the top-k tokens per cell plus, for each tracked concept, the
        probability the lens assigns to that concept's token there. That second
        number is what makes a cell light up.
        """
        torch = self.torch
        toks = self.tokenize(prompt)
        n_pos = len(toks)
        layers = sorted(layers or self.source_layers)
        positions = list(range(n_pos))

        concept_ids = {}
        for c in concepts:
            try:
                concept_ids[c] = self.tid(c)
            except ValueError:
                continue

        with torch.no_grad():
            jl, model_logits, _ = self.lens.apply(
                self.model, prompt, layers=layers, positions=positions,
                use_jacobian=use_jacobian,
            )

        cells = {}
        for l in layers:
            logits = jl[l].float()                      # [n_pos, vocab]
            probs = torch.softmax(logits, dim=-1)
            tv, ti = probs.topk(topk, dim=-1)
            row = []
            for p in range(n_pos):
                row.append({
                    "top": [
                        {"tok": self.tok.decode([int(ti[p, k])]), "p": float(tv[p, k])}
                        for k in range(topk)
                    ],
                    "concepts": {
                        c: float(probs[p, cid]) for c, cid in concept_ids.items()
                    },
                })
            cells[str(l)] = row

        nl = torch.softmax(model_logits.float(), dim=-1)
        nv, ni = nl[-1].topk(10)
        next_token = [
            {"tok": self.tok.decode([int(ni[k])]), "p": float(nv[k])}
            for k in range(10)
        ]

        return {
            "tokens": toks,
            "layers": layers,
            "cells": cells,
            "next_token": next_token,
            "use_jacobian": use_jacobian,
            "missing_concepts": [c for c in concepts if c not in concept_ids],
        }

    # -- write --------------------------------------------------------------

    def direction(self, layer: int, word: str):
        """v3: `v = lens.jacobians[l].T @ W_U[tid(word)]`, normalised."""
        key = (layer, word)
        if key not in self._dir_cache:
            v = self.lens.jacobians[layer].T @ self.W_U[self.tid(word)]
            self._dir_cache[key] = (v / v.norm()).to(self.device, self.dtype)
        return self._dir_cache[key]

    def mean_norms(self, prompt: str, layers):
        """Mean residual norm per layer, on this prompt. The unit of alpha."""
        from jlens.hooks import ActivationRecorder

        torch = self.torch
        key = f"{prompt}|{tuple(layers)}"
        if key not in self._norm_cache:
            ids = self.model.encode(prompt)
            with ActivationRecorder(self.model.layers, layers) as rec, torch.no_grad():
                self.model.forward(ids)
            self._norm_cache[key] = {
                l: float(rec.activations[l].float().norm(dim=-1).mean()) for l in layers
            }
        return self._norm_cache[key]

    def natural_coefs(self, prompt, concept, layers, carrier=NATURAL_CARRIER):
        """The amplitude the concept deposits when the fact is simply written down.

        This is v3's coefficient matching: state the fact in text, find the
        entity's own token, project its residual onto the injection direction.
        It is the honest reference point for the alpha slider — 'as loud as the
        real thing' rather than an arbitrary number.
        """
        from jlens.hooks import ActivationRecorder

        torch = self.torch
        probe = carrier.format(concept=concept) + prompt
        ids = self.model.encode(probe)
        t = self.tid(concept)
        hits = (ids[0] == t).nonzero()
        if len(hits) == 0:
            return None
        ent = int(hits[0].item())
        with ActivationRecorder(self.model.layers, layers) as rec, torch.no_grad():
            self.model.forward(ids)
        return {
            l: float(rec.activations[l][0, ent].float() @ self.direction(l, concept).float())
            for l in layers
        }

    def _hooked_logits(self, prompt, plan):
        """One forward pass, with `plan` added into the residual stream."""
        torch = self.torch
        handles = []
        if plan:
            for l in plan:
                def make(layer):
                    def hook(module, inputs, output):
                        pos, vec = plan[layer]
                        hidden = output if torch.is_tensor(output) else output[0]
                        hidden = hidden.clone()
                        hidden[:, pos] += vec
                        if torch.is_tensor(output):
                            return hidden
                        return (hidden,) + tuple(output[1:])
                    return hook
                handles.append(self.model.layers[l].register_forward_hook(make(l)))
        try:
            ids = self.model.encode(prompt)
            with torch.no_grad():
                out = self.hf(ids)
            return out.logits[0, -1].float()
        finally:
            for h in handles:
                h.remove()

    def _plan(self, prompt, concept, layers, positions, alpha, mode, carrier=NATURAL_CARRIER):
        torch = self.torch
        pos = torch.tensor(positions, device=self.device)
        if mode == "natural":
            coefs = self.natural_coefs(prompt, concept, layers, carrier)
            if coefs is None:
                raise ValueError(
                    f"cannot measure a natural amplitude for {concept!r}: it does not "
                    f"appear as a single token in the carrier sentence"
                )
            mags = {l: coefs[l] * alpha for l in layers}
        else:
            norms = self.mean_norms(prompt, layers)
            mags = {l: alpha * norms[l] for l in layers}
        return {
            l: (pos, (mags[l] * self.direction(l, concept).float()).to(self.dtype))
            for l in layers
        }, mags

    def inject(self, prompt, concept, layers, positions, alpha, mode="strength",
               watch=None, topk=10, carrier=NATURAL_CARRIER):
        """Push `concept` in; report what the model does about it.

        Two different things can happen, and conflating them is the easiest way to
        fool yourself. The model can *use* the concept — inject France, and it
        answers Paris — or it can simply *parrot* it, emitting the injected word
        itself. v3 tracks the second as `top1_is_entity`, because that is the
        degenerate outcome that looks like a win and isn't.

        So `watch` (what a correct answer would be) is separate from `concept`
        (what was injected). P(concept) is reported as the parrot signal.
        """
        torch = self.torch
        layers = sorted(layers)
        watch = watch or []

        base = torch.softmax(self._hooked_logits(prompt, None), dim=-1)
        plan, mags = self._plan(prompt, concept, layers, positions, alpha, mode, carrier)
        inj = torch.softmax(self._hooked_logits(prompt, plan), dim=-1)

        parrot = self.tid(concept)
        watch_ids = {}
        for w in watch:
            try:
                watch_ids[w] = self.tid(w)
            except ValueError:
                continue

        # A shared token set, so the paired bars line up and nothing can hide.
        shown = set(base.topk(topk).indices.tolist()) | set(inj.topk(topk).indices.tolist())
        shown.add(parrot)
        shown |= set(watch_ids.values())

        rows = sorted(
            (
                {
                    "tok": self.tok.decode([t]),
                    "id": int(t),
                    "base": float(base[t]),
                    "inj": float(inj[t]),
                    "is_watch": t in watch_ids.values(),
                    "is_parrot": t == parrot,
                }
                for t in shown
            ),
            key=lambda r: -r["inj"],
        )

        kl = float((inj * (inj.clamp_min(1e-12).log() - base.clamp_min(1e-12).log())).sum())

        delta = (inj - base).abs()
        delta[parrot] = 0
        for t in watch_ids.values():
            delta[t] = 0
        mv, mi = delta.topk(6)
        collateral = [
            {
                "tok": self.tok.decode([int(mi[k])]),
                "base": float(base[int(mi[k])]),
                "inj": float(inj[int(mi[k])]),
                "delta": float(inj[int(mi[k])] - base[int(mi[k])]),
            }
            for k in range(6)
            if float(mv[k]) > 1e-4
        ]

        return {
            "concept": concept,
            "alpha": alpha,
            "mode": mode,
            "layers": layers,
            "positions": positions,
            "magnitudes": {str(l): mags[l] for l in layers},
            "watch": [
                {"tok": w, "base": float(base[i]), "inj": float(inj[i])}
                for w, i in watch_ids.items()
            ],
            "p_parrot_base": float(base[parrot]),
            "p_parrot_inj": float(inj[parrot]),
            "kl": kl,
            "rows": rows[:14],
            "collateral": collateral,
            "top1_base": self.tok.decode([int(base.argmax())]),
            "top1_inj": self.tok.decode([int(inj.argmax())]),
            "missing_watch": [w for w in watch if w not in watch_ids],
        }

    def sweep(self, prompt, concept, layers, positions, alphas, mode="strength", watch=None,
              carrier=NATURAL_CARRIER):
        """The alpha curve: does the concept get used, or just repeated, or does it break?"""
        out = []
        for a in alphas:
            r = self.inject(prompt, concept, layers, positions, a, mode, watch=watch, topk=3,
                            carrier=carrier)
            row = {
                "alpha": a,
                "parrot": r["p_parrot_inj"],
                "kl": r["kl"],
                "top1": r["top1_inj"],
            }
            for w in r["watch"]:
                row["watch_" + w["tok"]] = w["inj"]
            out.append(row)
        return out

    def natural_alpha(self, prompt, concept, layers, carrier=NATURAL_CARRIER):
        """Where 'as loud as the real thing' sits on the strength slider.

        The natural coefficient differs by layer, so this is a range, not a tick.
        The carrier is the sentence the concept is measured in; it must contain
        `{concept}` and should read naturally for whatever you are probing. The
        default is v3's, which is about countries — change it if your prompt is not.
        """
        coefs = self.natural_coefs(prompt, concept, layers, carrier)
        if coefs is None:
            return None
        norms = self.mean_norms(prompt, layers)
        per = {l: coefs[l] / norms[l] for l in layers}
        vals = list(per.values())
        return {
            "per_layer": {str(l): per[l] for l in layers},
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }


class MockEngine:
    """Fiction with the right shape. For building the UI off the lab network."""

    mock = True
    model_name = MODEL_NAME + " (mock)"

    def __init__(self):
        self.n_layers = 32
        self.source_layers = list(range(1, 32, 2))
        self.vocab_size = 151_936
        self._vocab = [
            "Italy", "France", "Rome", "Paris", "Berlin", "Euro", "Lira", "the",
            "a", "capital", "city", "country", "Madrid", "Spain", "Japan", "Tokyo",
            "Yen", "German", "Italian", "pizza", "boot", "shaped", "and", "of",
        ]

    def _rng(self, *parts):
        seed = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]
        return random.Random(int(seed, 16))

    def meta(self):
        return {
            "model": self.model_name,
            "lens": f"{LENS_REPO}@{LENS_REVISION} (mock)",
            "n_layers": self.n_layers,
            "source_layers": self.source_layers,
            "default_band": band_layers(self.source_layers, self.n_layers),
            "vocab_size": self.vocab_size,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "natural_carrier": NATURAL_CARRIER,
            "mock": True,
        }

    def tokenize(self, prompt):
        words = prompt.split()
        if len(words) > MAX_PROMPT_TOKENS:
            raise ValueError(f"prompt is {len(words)} tokens; capped at {MAX_PROMPT_TOKENS}")
        return [{"i": i, "id": 1000 + i, "text": (" " if i else "") + w}
                for i, w in enumerate(words)]

    def is_single_token(self, word):
        return True

    def _ignition(self, layer, pos, n_pos, concept, prompt):
        """A concept catches mid-stack and late in the prompt. Bump in both axes."""
        d = layer / self.n_layers
        rel = (pos + 1) / n_pos
        depth = math.exp(-((d - 0.62) ** 2) / (2 * 0.16 ** 2))
        late = rel ** 2.2
        base = self._rng(prompt, concept).random() * 0.6 + 0.4
        noise = self._rng(prompt, concept, layer, pos).random() * 0.08
        return max(0.0, min(0.97, depth * late * base + noise))

    def readout(self, prompt, concepts, layers=None, use_jacobian=True, topk=5):
        toks = self.tokenize(prompt)
        n_pos = len(toks)
        layers = sorted(layers or self.source_layers)
        damp = 1.0 if use_jacobian else 0.28   # the logit lens sees far less

        cells = {}
        for l in layers:
            row = []
            for p in range(n_pos):
                r = self._rng(prompt, l, p)
                probs = {c: self._ignition(l, p, n_pos, c, prompt) * damp for c in concepts}
                # distinct tokens, descending — a top-k list that isn't sorted reads as a bug
                pool = r.sample(self._vocab, min(topk + len(concepts), len(self._vocab)))
                top = {w: 0.4 * (0.6 ** k) + r.random() * 0.05 for k, w in enumerate(pool)}
                for c, pr in probs.items():
                    if pr > 0.25:
                        top[c] = pr
                ranked = sorted(top.items(), key=lambda kv: -kv[1])[:topk]
                row.append({
                    "top": [{"tok": w, "p": round(v, 4)} for w, v in ranked],
                    "concepts": {c: round(v, 4) for c, v in probs.items()},
                })
            cells[str(l)] = row

        r = self._rng(prompt, "next")
        pool = r.sample(self._vocab, 10)
        next_token = [{"tok": w, "p": round(0.35 * (0.62 ** k), 4)} for k, w in enumerate(pool)]
        return {"tokens": toks, "layers": layers, "cells": cells,
                "next_token": next_token, "use_jacobian": use_jacobian,
                "missing_concepts": []}

    def _curve(self, concept, alpha, prompt, positions):
        """The two regimes the real model actually shows.

        Aimed at a referent token, the concept gets *used*: the watched answer
        rises and the injected word itself stays quiet. Smeared over every
        position, the model *parrots*: it emits the injected word and the
        distribution blows up. Mock data that only showed the first would teach
        the wrong lesson when the GPU is not around.
        """
        n = max(1, len(self.tokenize(prompt)))
        smear = len(positions) / n
        r = self._rng(prompt, concept)
        nat = 0.22 + r.random() * 0.08

        if smear > 0.5:                                   # the naive route
            parrot = 0.97 * (1 - math.exp(-alpha / (nat * 1.2)))
            used = 0.0
            kl = 11.0 * (1 - math.exp(-alpha / (nat * 1.1)))
        else:                                             # aimed at the referent
            rise = 1 - math.exp(-alpha / nat)
            decay = math.exp(-max(0.0, alpha - nat * 6) / 4)
            used = 0.02 + 0.16 * rise * decay
            parrot = 0.001 + 0.02 * min(1.0, alpha / 8)
            kl = 0.45 * (1 - math.exp(-alpha / (nat * 2)))
        return max(0.0, min(0.98, used)), max(0.0, min(0.98, parrot)), kl

    def inject(self, prompt, concept, layers, positions, alpha, mode="strength",
               watch=None, topk=10, carrier=NATURAL_CARRIER):
        watch = watch or []
        used, parrot, kl = self._curve(concept, alpha, prompt, positions)
        r = self._rng(prompt, concept, alpha)

        rows = [{"tok": concept, "id": 1, "base": 0.0002, "inj": round(parrot, 4),
                 "is_watch": False, "is_parrot": True}]
        for k, w in enumerate(watch):
            rows.append({"tok": w, "id": 50 + k, "base": 0.019, "inj": round(used, 4),
                         "is_watch": True, "is_parrot": False})
        left = max(0.0, 1 - parrot - used)
        others = [w for w in self._vocab if w != concept and w not in watch][:8]
        for k, w in enumerate(others):
            b = round(0.28 * (0.7 ** k) + r.random() * 0.02, 4)
            rows.append({"tok": w, "id": 100 + k, "base": b,
                         "inj": round(left * b * (0.9 + r.random() * 0.2), 4),
                         "is_watch": False, "is_parrot": False})
        rows.sort(key=lambda d: -d["inj"])
        collateral = sorted(
            ({"tok": d["tok"], "base": d["base"], "inj": d["inj"],
              "delta": round(d["inj"] - d["base"], 4)}
             for d in rows if not d["is_parrot"] and not d["is_watch"]),
            key=lambda d: -abs(d["delta"]),
        )[:5]
        norms = {l: 40 + l for l in layers}
        return {
            "concept": concept, "alpha": alpha, "mode": mode, "layers": sorted(layers),
            "positions": positions,
            "magnitudes": {str(l): alpha * norms[l] for l in layers},
            "watch": [{"tok": w, "base": 0.019, "inj": round(used, 4)} for w in watch],
            "p_parrot_base": 0.0002, "p_parrot_inj": round(parrot, 4), "kl": round(kl, 4),
            "rows": rows[:14], "collateral": collateral,
            "top1_base": "the", "top1_inj": rows[0]["tok"],
            "missing_watch": [],
        }

    def sweep(self, prompt, concept, layers, positions, alphas, mode="strength", watch=None,
              carrier=NATURAL_CARRIER):
        watch = watch or []
        out = []
        for a in alphas:
            used, parrot, kl = self._curve(concept, a, prompt, positions)
            row = {"alpha": a, "parrot": round(parrot, 4), "kl": round(kl, 4),
                   "top1": concept if parrot > 0.3 else "the"}
            for w in watch:
                row["watch_" + w] = round(used, 4)
            out.append(row)
        return out

    def natural_alpha(self, prompt, concept, layers, carrier=NATURAL_CARRIER):
        r = self._rng(prompt, concept, "nat")
        m = 1.6 + r.random() * 0.8
        return {"per_layer": {str(l): m + r.random() * 0.4 - 0.2 for l in layers},
                "mean": m, "min": m - 0.35, "max": m + 0.35}
