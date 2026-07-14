# J-Scope

An instrument for looking at the workspace directly: type a prompt, watch which
concepts light up inside the model, then push a concept in yourself and watch the
output probabilities move.

It is a UI over the code already in `scripts/`. The readout is `lens.apply()`. The
injection is the hook from `vector_rag_v3.py`, unchanged — same unit direction from
the lens Jacobian, same amplitude convention. Numbers here should agree with the
experiments, because it is the same arithmetic.

## Run it

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch "transformers>=5.5" numpy fastapi "uvicorn[standard]"
.venv/bin/pip install git+https://github.com/anthropics/jacobian-lens

# device is auto-picked: cuda, else mps, else cpu
HF_HUB_DISABLE_XET=1 .venv/bin/python jscope/server.py
```

Then open <http://127.0.0.1:7801>.

`HF_HUB_DISABLE_XET=1` matters on a machine with no HF token: the Xet transfer
backend returns 401 for anonymous downloads and the classic CDN path does not.

No GPU, or working away from the model? `--mock` serves synthetic data with the
same shape, so the front end runs anywhere:

```bash
.venv/bin/python jscope/server.py --mock
```

The badge in the top-right always says which engine you are looking at. Mock data
is fiction and labels itself as such.

Check the engine before trusting the UI:

```bash
cd jscope && HF_HUB_DISABLE_XET=1 ../.venv/bin/python smoke.py
```

That runs the paper's silent-`Italy` readout and a `France` injection sweep, and
prints both. If `Italy` does not ignite, the lens and the model are mismatched and
nothing downstream means anything.

## The two halves

**The ignition grid** is the workspace at every layer and every token. Each cell is
coloured by the probability the lens assigns to a concept you are tracking — and
the concept does not have to appear in the prompt. That is the point: ask for the
currency of "the country shaped like a boot" and `Italy` lights up mid-stack
without ever being written down. Toggle to the plain logit lens to see how much of
that is invisible without the Jacobian.

**The injection bench** goes the other way. A concept word becomes a direction:

```python
v = lens.jacobians[layer].T @ W_U[token_id(word)]   # then normalised
hidden[:, positions] += alpha * mean_residual_norm[layer] * v
```

α is expressed in multiples of the mean residual norm at that layer, so it means
the same thing at every depth. The bench shows the next-token distribution before
and after, as paired bars.

## Editing cells, and why the grid lies about erasure

Cells are editable one at a time: `cells=[[layer, position], ...]`, an arbitrary
set rather than a layers × tokens rectangle. Three operations:

```python
add:     h += alpha * v        # v is a unit vector
erase:   h -= (h @ v) * v      # project the concept out
replace: erase, then add
```

All three are **rank-1**. Nothing is flattened or zeroed. At (L11, `' boot'`) the
Italy direction accounts for **16% of the residual's norm** and erasing it rotates
the vector by about 9 degrees — cosine similarity 0.987, the other 2559 dimensions
untouched. (16% is a lot for one direction: a random one would capture ~2%.)

Erasing is the causal test, and it works — project Italy out of the `' boot'`
column and next-token goes from `' euro'` (0.143) to `'\n'`.

**But do not read the grid as proof the concept is gone.** The lens reads a
concept by projecting onto `v`; erase removes exactly that projection. So
P(concept) collapsing to 0.0000 after an erase is *partly tautological*, at the
edited cells and every cell above them.

The model disagrees with the grid:

| cut at `' boot'` | P(euro) | KL |
|---|---|---|
| nothing (baseline) | 0.143 | — |
| L0–L7, below where Italy forms | 0.131 | 0.00 |
| L8–L11, where it forms | 0.091 | 0.08 |
| L12–L21, after it has formed | 0.035 | 0.36 |
| L8–L21, the whole column | 0.023 | 0.46 |

After cutting L8–L11 the grid claims Italy is gone at every layer above — yet the
model still answers `euro` at 0.091, and cutting L12–L21 *as well* drops it to
0.023. It could not, if the concept were really gone. **`v` is a 1-D projection of
the concept, not the concept.** Italy-flavoured information survives in directions
the lens cannot see. Trust the next-token distribution for whether a concept is
really gone, and treat the grid as a view, not a ground truth.

Cutting L0–L7 does nothing at all (KL = 0.00), which independently confirms the
ignition timing the grid shows: Italy has not formed yet down there.

## Read the damage, not just the win

The bench reports KL divergence from the baseline distribution and lists the
tokens that moved most that you never asked for. This is deliberate. Naive
injection in `vector_rag.py` scored **0%** — pushing hard enough at every position
does not steer the model, it breaks it. Turn α up and you should watch the
distribution come apart. A tool that only showed the target probability rising
would be lying about what the experiments found.

The α sweep plots both curves together. The useful amplitude is the window where
P(concept) has risen and KL has not yet exploded.

The slider is annotated with the **natural amplitude**: how loud the concept is
when the fact is simply written in the text, measured the way `vector_rag_v3.py`
measures it — state the fact, find the entity's own token, project its residual
onto the injection direction. It is the honest reference point. "As loud as the
real word" is a more meaningful number than any α you would pick by hand.

## Freeze

**Freeze session** bakes every response you have looked at into a standalone HTML
file in `jscope/frozen/`. It is the same page, with the answers inlined and the
network never touched, so it opens on any machine with no model behind it. Use it
to keep a finding, or to hand someone a view they can poke at.

Frozen pages only know what you looked at. Anything you did not click is not in
there, and the page says so rather than pretending.

## Limits

- Concepts must be **single tokens** under the leading-space convention
  (`" France"`), and are rejected out loud when they are not. They must be, because
  the injection direction is built from exactly one token id. Taking the first
  piece of a multi-token word instead would be worse than useless: `" 1"` tokenizes
  to `[' ', '1']`, so its first piece is the bare **space** token, and the space
  token's probability is enormous. You would read a triumphant 0.98 and it would
  mean nothing.

- That convention is prose-shaped, and it is the real limit on what you can track.
  In code the interesting tokens have no leading space — the lens reads `.println`
  perfectly well at L29 of a Java prompt, but you cannot name that token as a
  "concept" here, so it scores ~0. A low peak means your probe missed, not
  necessarily that the model is not representing the thing.

- The lens is **not limited to the content it was fitted on.** Its vocabulary is
  the model's own unembedding (248,320 tokens), not the fitting corpus; the 1000
  wikitext samples only estimate the linear map. It reads `意大利` on an English
  prompt, and a clinical note (`polydipsia, polyuria, elevated HbA1c`) lights up
  `diabetes` at **p=0.50** — stronger than the boot riddle's `Italy` at 0.26. What
  the fitting distribution limits is the *fidelity of the linear approximation*,
  not the range of concepts.
- Prompts are capped at 64 tokens. The grid is layers × tokens and stops being
  readable well before that.
- One forward pass at a time, under a lock. It is an instrument, not a serving
  stack.
- Qwen3.5 is a hybrid: only 8 of its 32 layers have full softmax attention. The
  lens has entries for a subset of layers, and only those can be read or written.
