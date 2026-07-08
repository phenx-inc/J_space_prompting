# Vector-RAG via the Jacobian lens — experiment record

**Date:** 2026-07-07 · **Hardware:** RTX 4090 (gpu-server, 192.168.31.242) · **Model:** Qwen/Qwen3.5-4B (bf16)
**Lens:** neuronpedia/jacobian-lens `qwen3.5-4b/.../Qwen3.5-4B_jacobian_lens_n1000.pt` (pre-fitted, wikitext n=1000)
**Code:** `smoke_test.py`, `vector_rag.py` (v1), `vector_rag_v2.py`, `vector_rag_v3.py` in this directory.
**Raw results:** `results_vector_rag*.jsonl` · **Logs:** `*.log`

## Question

Anthropic's global-workspace paper shows J-space *swaps* redirect downstream reasoning.
Can pure *injection* — writing an entity vector into the workspace with no text mention —
substitute for retrieved text? ("Vector-RAG": resolve references via the residual stream
instead of the context window.)

## Protocol

Follows `jacobian-lens/data/experiments/README.md` conventions:
steering direction for token t at layer l = unit-normalized `J_l^T @ W_U[t]`;
injection adds `magnitude * v_hat` at band layers via forward hooks; grading is
greedy next token vs accepted-answer first tokens (case variants allowed).

Indirection prompts of the form "The capital of Kevin's home country is the city of";
the country name never appears; conditions vary how the model learns it.
Battery: 16 single-token countries × 4 functions (capital / official language /
continent / currency). Workspace band: layers 12–24 of 32 ("full", 35–75% depth)
or 12–17 ("low", 35–55%).

## Results (v3, n=64 per condition)

| condition | total | capital | language | continent | currency |
|---|---|---|---|---|---|
| direct template (entity in text, health check) | 55/64 | 16/16 | 16/16 | 15/16 | 8/16 |
| floor (no info) | 6/64 | 0/16 | 4/16 | 2/16 | 0/16 |
| text-RAG ceiling (fact sentence) | 60/64 | 15/16 | 16/16 | 16/16 | 13/16 |
| inject: referent tokens, uniform s=4, full band | 26/64 | 11/16 | 9/16 | 6/16 | 0/16 |
| inject: referent tokens, coef-matched, full band | 33/64 | 13/16 | 6/16 | 14/16 | 0/16 |
| **inject: referent tokens, coef-matched, low band** | **34/64** | **14/16** | 6/16 | **14/16** | 0/16 |

Earlier iterations (v1/v2, 4-country battery):
- Swap replication (paper's flexible-generalization protocol): 34/36 = 94%
  excluding the currency template, whose grading was case-broken (fixed in v2+).
- Naive injection (every position, uniform magnitude): 0/16 with 16/16
  "entity saturation" — top-5 becomes the country name itself; the vector acts
  as an output bias ("say France"), not an argument.
- Ablation of the naive failure: skip-final-position 2–3/16 → referent-only
  6–7/16 (saturation 0) → +coefficient-matching 8–9/16.

## Findings

1. **J-space injection can substitute for retrieved text — partially.** Hero cell
   53% overall vs 9% floor and 94% text ceiling (≈52% of the retrieval gap
   recovered with zero context tokens). Capital and continent: 88%.
2. **How you write to the workspace matters more than how hard.** The gradient
   naive→targeted→coefficient-matched (0% → 41% → 53%) shows downstream circuits
   consume an injected vector only when it mimics a naturally occurring entity
   representation: right position (the referring noun), natural per-layer
   magnitude. Overdriving magnitude (×2, ×4 natural) *reduces* accuracy.
3. **Different functions read the workspace differently.** Uniform s=4 beats
   coef-matched on language (9 vs 6) but loses badly on continent (6 vs 14);
   currency never works via injection (0/16) and is weak even direct (8/16).
   Text in context serves all functions (94%); injection has no
   one-setting-fits-all. Working hypothesis: functions differ in where/at what
   strength they read entity representations.
4. **Failure modes are knowledge slips, not injection failures.** Hero-cell
   capital misses: Turkey→"Istanbul" (Ankara rank 2), Vietnam→"Ho…" — the model
   errs about the *correct* country, proving the entity arrived.

## Caveats / claim discipline

- One model (Qwen3.5-4B instruct), one lens corpus, completion-style prompts.
  No claims about Claude, other scales, or chat-formatted use.
- The floor is nonzero (9%) because some answers are guessable priors
  (Europe/Asia continents, English language).
- Currency template is unhealthy even with the entity in text (8/16 direct);
  its injection zero should not be read as a mechanism limit.
- "Coefficient matching" measures the entity's natural per-layer coefficient
  from a text prompt — a real retrieval system would need these calibrated
  per entity offline (cheap: one forward pass per entity).

## Next steps

- [ ] More functions (leader, flag color, borders) and non-country categories
      (flexgen has months/animals/numbers) to test generality.
- [ ] Scale axis: repeat on Qwen3.5-0.8B/2B/9B (fit lenses) — does injection
      consumability emerge with scale?
- [ ] Chat-template prompts (the deployment-realistic case).
- [ ] Multi-entity: inject two facts at two referents ("Kevin's and Maria's").
- [ ] Write up as blog post (ctlsurf page + charts).

## v4 — attention-guided sites + phantom KV slot (2026-07-07, later)

Script: `vector_rag_v4.py` (eager attention; Qwen3.5-4B is HYBRID: full softmax
attention only at layers 3,7,...,31; attention maps read at L15/19/23).
Raw: `results_vector_rag_v4.jsonl`.

| condition | total | capital | language | continent | currency |
|---|---|---|---|---|---|
| floor | 4/64 | 0/16 | 2/16 | 2/16 | 0/16 |
| text ceiling | 59/64 | 15/16 | 16/16 | 16/16 | 12/16 |
| refcoef_low (v3 reference) | 34/64 | 14/16 | 6/16 | 14/16 | 0/16 |
| attn-guided refcoef | 41/64 | 16/16 | 9/16 | 16/16 | 0/16 |
| phantom KV (" France") | 43/64 | 14/16 | 13/16 | 14/16 | 2/16 |
| phantom KV (" France.") | 54/64 | 15/16 | 15/16 | 16/16 | 8/16 |

Findings:
- Attention-chosen injection sites beat the hand-picked " country" referent
  (64% vs 53%; capital & continent perfect). The model fetches from " Kevin"
  (the unresolved variable) and the function word, not the referent noun.
- Phantom KV slot (entity forwarded as cache-only prefix, never in visible
  prompt) reaches 84% with a trailing period vs 92% text ceiling; currency
  hits its direct-template health ceiling (8/16). No binding text needed —
  the model links a dangling cache entity to the unresolved referent itself.
- Trailing period worth +17 points (67%->84%): punctuation-triggered
  consolidation of the entity representation.
- Honest framing: the phantom slot IS two tokens of (invisible, precomputable,
  constant-size) context built via cache prefix — the finding is that
  retrieval needs no fact sentence, not that information appears from nowhere.

## v5 — noise robustness: text-RAG vs phantom KV under distractors (2026-07-07)

Script: `vector_rag_v5.py` · Raw: `results_vector_rag_v5.jsonl`.
k distractor facts about other people ("Fact: Maria's home country is Japan."),
target at seeded-random position, same fact order across lanes, distractor
answers never collide with target answer for the trial's function.

| k | text-RAG | unbound phantom (" Japan. ... France.") | bound phantom (" Maria: Japan. ... Kevin: France.") |
|---|---|---|---|
| 0 | 60/64 (94%) | 54/64 (84%) | 61/64 (95%) |
| 4 | 60/64 (94%) | 4/64 (6%) | 56/64 (88%) |
| 16 | 57/64 (89%) | 5/64 (8%) | 55/64 (86%) |
| 48 | 53/64 (83%) | 3/64 (5%) | 54/64 (84%) |

Findings:
- Unbound phantom collapses to floor with as few as 4 distractor entities:
  bare cache entities carry no binding info, so the model cannot select
  Kevin's among candidates. Binding is the phantom slot's failure mode.
- One name token per fact restores full robustness: bound phantom tracks
  text-RAG at every k and equals it at k=48 (84% vs 83%) at ~half the cache
  tokens per fact (~4.5 vs ~9.5), still invisible in the prompt.
- Headline refinement: bound phantom at k=0 (" Kevin: France.", 3 tokens)
  scores 95% vs clean text-RAG 94% — the earlier 84%-vs-92% gap was a binding
  problem, not a phantom problem. The "error rate doubles" caveat is retired.
- Capital function: bound phantom 16/16 at every noise level.

## v5-scale — noise battery across model sizes (2026-07-07)

Same protocol as v5 on Qwen3.5-2B (`results_v5_2b.jsonl`) and 9B (`results_v5_9b.jsonl`).
Bound-phantom totals (text-RAG in parens):

| k | 2B | 4B | 9B |
|---|---|---|---|
| 0 | 91% (92%) | 95% (94%) | 95% (94%) |
| 4 | 67% (80%) | 88% (94%) | 94% (92%) |
| 16 | 45% (67%) | 86% (89%) | 91% (92%) |
| 48 | 42% (59%) | 84% (83%) | 83% (91%) |

- Reading name-colon-entity cache shorthand as reliably as prose grows with
  scale: 2B trails text badly under noise; 4B ties; 9B ties through k=16 and
  trails by 8 pts at k=48 (where 9B text is near-unshakeable).
- Unbound collapse replicates at all scales (2-11% with distractors).
- Curious: unbound CLEAN accuracy drops with scale (80% / 84% / 61%) — the
  larger model is more conservative about binding a dangling unattributed
  entity. Arguably the more correct behavior.

## v6 — workspace mediation on 4B (readout + erasure)

Script `vector_rag_v6_mediation.py`, raw `results_v6_mediation.jsonl`.

Part A (J-lens readout during the capital question, min rank over band x positions):
- floor: country in top-10 8/16 (median rank 13) · phantom: 16/16 (median 1) · text: 16/16 (median 1)
- A fact existing only in the KV cache is broadcast in the J-space exactly
  like a fact written in text.

Part B (erase target J-direction, question span only, 64 trials/cell):
- phantom 61/64 -> erase 56/64 (dip concentrated in capital 16->12);
  matched control 61/64; text 59/64 -> erase 61/64 (no effect).
- Dissociation: the workspace DISPLAYS the retrieved entity but the residual
  copy is not the load-bearing path — attention can re-read the untouched
  cache entry at every layer. v6b tests the complementary intervention
  (erase during the prefix forward = corrupt the stored entry).

## v6b — causal locus: cache entry vs residual copy (2026-07-07)

Script `vector_rag_v6b.py`, raw `results_v6b_prefix_erase.jsonl`. Bound phantom,
erase the target country J-direction at band layers during different passes:

| intervention | accuracy |
|---|---|
| none (reference) | 61/64 (95%) |
| erase residual copy (question pass) | 56/64 (88%) |
| erase stored cache entry (prefix pass) | 41/64 (64%) |
| erase both | 11/64 (17%) |
| prefix erasure, control direction | 61/64 |

Conclusion: redundant storage with joint necessity. Primary store = the cache
entry (−31 pts when corrupted); secondary = the workspace/residual copy
(−7 pts alone); removing both collapses retrieval to near-floor; control
erasure is a no-op, so the effect is entity-specific. The v6 near-null was
routing-around-damage, not absence of a causal role.

## v7 — multi-hop chains through phantom intermediates (2026-07-07)

Script `vector_rag_v7_multihop.py`, raw `results_v7_multihop.jsonl`.
Cached entity is a capital CITY (" Kevin: {city}."); every question needs the
silent hop city->country first. 15 single-token cities (Hanoi dropped).

| cell | total | country (d1) | continent (d2) | language (d2) | currency (d2) |
|---|---|---|---|---|---|
| direct (city in template) | 40/60 | 11/15 | 13/15 | 11/15 | 5/15 |
| floor | 4/60 | 0 | 2 | 2 | 0 |
| text | 56/60 | 15/15 | 15/15 | 15/15 | 11/15 |
| phantom | 44/60 | 4/15* | 15/15 | 13/15 | 12/15 |

- Two-hop chains through a cache-only intermediate run at 87-100% under
  phantom delivery; currency (12/15) beats its own direct-template health.
- *The depth-1 "4/15" is a grading artifact: on every miss the top token is
  " the"/" known" (periphrastic continuation) with the correct country at
  rank 2-9 in all 15 trials (top-10: 15/15, top-5: 11/15). No echo of the
  city, no wrong countries. Strict greedy top-1 grading undercounts here.
- Conclusion: phantom cache entities serve as multi-hop chain intermediates
  essentially as well as text facts; the gap does not widen with depth.

## v8 — pre-publication hardening: power, precision, leakage (2026-07-07)

Script `vector_rag_v8_hardening.py`, raw `results_v8_hardening.jsonl`.
n=256/cell (4 names x 16 countries x 4 functions); graders: strict top-1,
hit@5, candidate-set argmax.

| cell | strict | hit@5 | cand |
|---|---|---|---|
| floor | 20/256 | 90 | 26 |
| text | 236/256 | 256 | 256 |
| phantom | 233/256 | 256 | 256 |
| two_recall (2 facts, ask person A) | 219/256 | 253 | 239 |
| two_select (2 facts, ask person B) | 234/256 (intrusion 4/256) | 254 | 249 |
| leak_base (direct q, no cache) | 225/256 | 243 | 253 |
| leak (direct q + irrelevant phantom) | 234/256 (intrusion 2/256) | 256 | 254 |

- Phantom vs text: McNemar exact p=0.607 (6 vs 9 discordant) — statistically
  indistinguishable; both 256/256 under hit@5 and candidate grading.
- Precision: right-person retrieval from a 2-fact cache 91.4%, intrusion 1.6%.
- No leakage: irrelevant phantom does not hurt unrelated questions
  (higher than baseline; intrusion 0.8%).
- New caveat: a second cached fact costs ~5 pts on recall (233->219).
- Strict greedy grading undercounts informed cells by ~9% (style artifacts),
  consistent with the v7 periphrasis finding.

## v9 — synthetic KV entries: the no-text route into the cache (2026-07-07)

Script `vector_rag_v9_synthkv.py`, raw `results_v9_synthkv.jsonl`.
Placeholder prefix positions whose residual is overwritten at every block with
synthetic vectors, so the model projects OUR vectors into position-stamped
K/V + recurrent state; question runs on that cache unhooked.

| cell | accuracy |
|---|---|
| floor | 6/64 (9%) |
| text_unbound " France." | 54/64 (84%) — replicates |
| text_bound " Kevin: France." | 61/64 (95%) — replicates |
| embed_raw (clamped input embedding) | 4/64 (6%) |
| embed_scaled (norm-matched per layer) | 4/64 (6%) |
| lens_add (lens direction @ natural amplitude in a dedicated slot) | 6/64 (9%) |
| embed_scaled_bound (2 synthetic slots Kevin+France) | 1/64 (2%) |

- Every vector-built cache entry is at the no-information floor; outputs are
  IDENTICAL across countries within a cell (generic priors) — the slots carry
  no entity information at all, rather than wrong information.
- Interpretation: a retrievable cache entry is a projection of a
  CONTEXTUALIZED residual (the token actually processed through the stack).
  Static embeddings and readout directions are the wrong basis for the K/V
  projections, and clamping one vector at every layer removes the layer-wise
  evolution those projections were trained on. The write API is the full
  forward pass; there is no inference-time shortcut into memory without a
  trained bridge (which is what prefix-tuning / gist / xRAG learn).
- This answers the "phantom KV is just hidden-turn text" objection with data:
  yes — and the no-text alternatives all score at floor, so text-through-the-
  stack is not a lazy choice, it is the only working unlearned write path.

## v9b — donor-residual replay: positional portability (2026-07-08)

Script `vector_rag_v9b_replay.py`, raw `results_v9b_replay.jsonl`.
Harvest the real per-layer block-input residuals of " Kevin: {c}." from a donor
text pass; clamp them into placeholder prefix slots (pre-hooks on every block);
run the question on that cache.

| cell | accuracy |
|---|---|
| floor | 6/64 (9%) |
| text_bound | 61/64 (95%) |
| replay_same (original positions) | 61/64 (95%) — LOSSLESS |
| replay_offset (harvested at pos 8..11, replayed at 0..3) | 40/64 (63%); language 0/16, continent 16/16 |
| replay_clip (entity+period slots only) | 36/64 (56%); capital 14/16, continent 16/16, language 1/16 |

- replay_same = 95% validates the whole approach mechanically: cache content is
  fully captured by residual trajectories; v9 failed on content, not harness.
  Caveat: same-position replay is informationally equivalent to storing the KV
  itself — i.e., ordinary prompt caching in another coat.
- Position-portable precompute pays ~32 pts, function-lopsided (language dies,
  continent unaffected). Confound: the offset donor included filler text, so
  position shift and donor-context contamination are mixed — separate next.
- Binding is partly baked into the entity residuals: with only " France"+"."
  slots stored (Kevin never replayed), capital/continent are near-perfect.
- Final delivery spectrum: no cache (injection, 64% max) < vector-built cache
  (floor) < replayed contextualized residuals (95% same-pos, 63% offset) =
  text-built cache (95%). The unlearned write path is reading; the only
  lossless shortcut is caching what reading produced.

## v9c — verification: are injected cache slots attended and broadcast? (2026-07-08)

Script `vector_rag_v9c_verify.py`, raw `results_v9c_verify.jsonl`. Capital
question, 16 countries. Metrics: J-lens min-rank of the country over band x
question positions; attention mass from the final question position into the
slot key columns (mean over heads, layers 15/19/23, eager attention).

| cell | J-space top10 | median rank | attn mass to slots | behavioral |
|---|---|---|---|---|
| floor | 8/16 | 13 | n/a | 9% |
| text_bound | 16/16 | 1 | 0.405 | 95% |
| embed_scaled | 9/16 | 10 | 0.001 | floor |
| lens_add | 15/16 | 1 | 0.178 | floor |
| replay_same | 16/16 | 1 | 0.405 | 95% |
| replay_offset | 16/16 | 1 | 0.351 | 63% |

Findings — the three failures fail at three different stages:
- embed_scaled fails at ADDRESSING: keys computed from fake residuals are
  unreadable; the model never attends (0.001). Nothing fetched, nothing shown.
- lens_add fails at CONTENT: attended (0.178) and the country displays in the
  workspace at rank 1 — yet behavior is floor. The fetched value is the
  "disposed-to-say-France" axis, which the readout sees (partly circular) but
  downstream circuits cannot consume. IMPORTANT LENS CAVEAT: workspace readout
  presence is NOT sufficient evidence of usability.
- replay_offset fails at CONSUMPTION, selectively: fully attended (0.351) and
  fully broadcast (rank 1, 16/16) but 63% behavioral, language at 0 — specific
  reader circuits depend on position-sensitive features in the values.
- text_bound and replay_same agree on every metric (0.405 / rank 1 / 95%),
  as they must — the recording is the same computation.
