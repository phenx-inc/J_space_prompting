# J-Space Prompting

Experiments on writing facts into a language model through its **J-space**
(the global workspace identified by Anthropic's Jacobian-lens work) and its
**KV cache**, instead of through the prompt. Companion code and raw data for
the article *"I Fed a Model Facts Through Its J-Space Instead of Its Prompt"*.

Everything runs on one consumer GPU (RTX 4090, 24 GB) against open-weight
Qwen3.5 models, using [Anthropic's open-source Jacobian lens](https://github.com/anthropics/jacobian-lens)
and Neuronpedia's pre-fitted lenses. No training anywhere; every experiment is
forward passes, hooks, and grading.

## The question

Anthropic's [global-workspace paper](https://transformer-circuits.pub/2026/workspace/index.html)
shows that swapping a concept's direction inside the workspace redirects the
model's downstream answers. Every swap edits a concept that is already there
because it appeared in the text. These experiments ask what happens when the
slot is empty: can retrieval deposit an entity directly — as a vector, as a
cache entry, as a recorded internal state — instead of spending prompt tokens?

The test battery: 16 countries x 4 questions (capital, official language,
continent, currency), asked about "Kevin's home country" with the country
never stated in the prompt. Floor (no information) ~6-9%; ceiling (fact
written in the prompt) ~92-95%.

## Findings, in one table

| Delivery route | Accuracy | Script |
|---|---|---|
| No information (floor) | 6-9% | any |
| Residual injection, naive (every position) | 0% | `vector_rag.py` |
| Residual injection, tuned (site + amplitude + attention-picked) | 41 → 53 → 64% | `vector_rag_v2.py`-`v4.py` |
| Phantom KV, bare entity `" France."` | 84% | `vector_rag_v4.py` |
| Phantom KV, bound `" Kevin: France."` | **95%** (= text, McNemar p=0.61 at n=256) | `v4`/`v5`/`v8` |
| Synthetic KV entries (embeddings / lens directions) | floor | `vector_rag_v9_synthkv.py` |
| Recorded internal states, replayed at original positions | 95% (lossless) | `vector_rag_v9b_replay.py` |
| Same recording, moved to new positions | 63% | `vector_rag_v9b_replay.py` |

Supporting results: the bound phantom tracks text-RAG under up to 48
distractor facts (`v5`); binding is the failure axis (bare entities collapse
to floor with 4 distractors); reading cache shorthand grows with scale
(2B trails, 9B ties prose — `v5 --model`); the fact is stored redundantly
(cache entry + workspace broadcast, 17% only when both are erased — `v6`/`v6b`);
phantom entities chain through multi-hop questions (`v7`); precision holds with
multiple cached facts, 1.6% worst-case intrusion (`v8`); and the failed no-text
routes fail at three different measured stages — never attended, attended but
unusable, attended-and-broadcast but unconsumed (`v9c`).

`RESULTS.md` is the full experiment record, version by version, with every
table and caveat. `results/` holds the raw per-trial JSONL behind every number.

## Setup

Python 3.12, CUDA GPU with ~10 GB free for the 4B model (18 GB for 9B).

```bash
python3.12 -m venv .venv
.venv/bin/pip install "torch" "transformers>=5.5" numpy
.venv/bin/pip install git+https://github.com/anthropics/jacobian-lens
```

Models (`Qwen/Qwen3.5-4B`, optionally `-2B`/`-9B`) and the pre-fitted lens
(`neuronpedia/jacobian-lens`, `qwen-n1000` revision) download automatically on
first run. Set `HF_HOME` if you want the cache somewhere specific.

## Running

Each script is self-contained: run it from the repo root, it prints a summary
table and writes raw per-trial JSONL next to it.

```bash
.venv/bin/python scripts/smoke_test.py                # lens readouts (silent "Italy"/"spider")
.venv/bin/python scripts/vector_rag_v3.py             # injection battery, 64 trials
.venv/bin/python scripts/vector_rag_v4.py             # attention-guided sites + phantom KV
.venv/bin/python scripts/vector_rag_v5.py --model Qwen/Qwen3.5-9B --out results_9b.jsonl   # noise + scale
.venv/bin/python scripts/vector_rag_v8_hardening.py   # n=256 power/precision/leakage cells
.venv/bin/python scripts/vector_rag_v9_synthkv.py     # synthetic cache entries (all floor)
.venv/bin/python scripts/vector_rag_v9b_replay.py     # recorded-state replay
.venv/bin/python scripts/vector_rag_v9c_verify.py     # attended? broadcast? used?
```

Notes:
- `vector_rag_v4.py` and `vector_rag_v9c_verify.py` load the model with
  `attn_implementation="eager"` because attention maps are read; slower, fine.
- Qwen3.5 is a hybrid: only 8 of 32 layers have full softmax attention
  (3, 7, ..., 31); the rest keep recurrent state. Several scripts depend on
  this layout — expect changes for other architectures.
- Grading is strict greedy next-token against accepted-answer sets unless a
  script says otherwise; `v8` also records hit@5 and candidate-set grading.

## Figures

`figures/` holds the article's charts and diagrams as PNGs; `figures/src/`
holds the HTML sources (self-contained React+Recharts pages — render with any
headless Chrome at 2x device scale to reproduce the PNGs).

## Credits

- Jacobian lens method and code: [Anthropic](https://github.com/anthropics/jacobian-lens) (Apache-2.0),
  from *Verbalizable Representations Form a Global Workspace in Language Models*.
- Pre-fitted lenses: [Neuronpedia](https://huggingface.co/neuronpedia/jacobian-lens).
- Models: Qwen3.5 family (Alibaba).

## License

MIT for the code and data in this repository. The Jacobian-lens dependency is
Apache-2.0 (Anthropic).
