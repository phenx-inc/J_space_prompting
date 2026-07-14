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
  (`" France"`). Multi-token concepts are rejected, not silently truncated. This is
  the same restriction the experiments run under, which is why `Vietnam` and
  friends get dropped from the battery.
- Prompts are capped at 64 tokens. The grid is layers × tokens and stops being
  readable well before that.
- One forward pass at a time, under a lock. It is an instrument, not a serving
  stack.
- Qwen3.5 is a hybrid: only 8 of its 32 layers have full softmax attention. The
  lens has entries for a subset of layers, and only those can be read or written.
