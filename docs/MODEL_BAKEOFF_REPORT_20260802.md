# Multi-Model Q&A Generation Bake-Off

**Date:** 2026-08-02
**Branch:** `experiment/multi-model-bakeoff-20260802`
**Artifacts:** `data/model_bakeoff/comparison.html`, `results.json`, `chains.json`, `pool_*.json`
**Design spec:** `docs/superpowers/specs/2026-08-02-model-bakeoff-design.md`

---

## 1. Question

Which large language model should generate the question-answer pairs in the
GRAFT ground-truth pipeline, and does paying for a larger model buy better
ground truth?

The pipeline had run on one model (`Qwen/Qwen3-235B-A22B-Instruct-2507`) without
any comparison against alternatives. Model choice was therefore an unexamined
assumption sitting underneath every dataset the pipeline had produced.

A second question was folded in. On 2026-08-01 the question-generation prompt was
changed to pick the interrogative form a fact affords rather than defaulting to
"What" (commit `21b8799`). That fix needed verification before any model
comparison could mean anything: if the prompt still forced every question into
one shape, the comparison would measure the prompt rather than the models.

---

## 2. Method

### 2.1 Hold the evidence constant

Running the whole pipeline once per model would let each model extract different
facts and sample different chains. The five outputs would then describe five
different experiments and could not be compared.

Instead, stages 0 through 6 ran **once** to produce a frozen set of 30 fact
chains. Only stage 7 — question generation and answer generation — varied by
model. Every model saw byte-identical evidence.

### 2.2 Rebuilding the chains

Neither existing artifact could drive stage 7 faithfully:

- `data/gt_final_20260801/gt_pairs.json` does not persist `Fact.role`, and
  `questions._format_role_shape_hint` steers the question form from it.
- `checkpoints/s5_graph.json` and `s6_chains.json` hold summaries
  (`{n_facts, edge_count}`), not the graph or the chains.

`extract._fact_to_cache` does persist `role` and `spans`, so the chains were
rebuilt from the stage-3 chunk cache. Extraction was a cache hit and cost
nothing; only stage-5 edge classification made fresh calls.

### 2.3 Corpus and chain composition

Two documents, 15 chains each:

| Document | Cached chunks | Facts |
|---|---|---|
| `din_iso_6507_1` (DIN EN ISO 6507-1, Vickers hardness) | 80 | 116 |
| `din_iso_3452_1` (DIN EN ISO 3452-1, penetrant testing) | 80 | 123 |

Final selection, seed 42:

| Property | Value |
|---|---|
| Chains | 30 (15 per document) |
| Depth | 23 single-fact, 7 two-fact |
| Fact roles | definition 10, condition 9, rule 6, example 5, exception 3, constraint 2, consequence 2 |
| Edge types | causal 6, mechanism 1 |

Two-fact chains were capped at one third of the panel. They exercise
`chain_edges` and the role-shape hint, but single-fact chains are roughly 99% of
what the pipeline emits, so a panel made entirely of two-fact chains would not
represent the output under comparison. The remainder was round-robined across
fact roles, which is why the panel is far more balanced than the raw corpus
(147 `definition` and 71 `rule` across the two documents).

### 2.4 Model roster

Nebius Token Factory hosts 27 models, all open-weight. Claude Sonnet and GPT-5.x
are not available there, so the roster was drawn entirely from what the key can
reach:

| Column | Model | Tier |
|---|---|---|
| 1 | `google/gemma-3-27b-it` | ~27B small |
| 2 | `meta-llama/Llama-3.3-70B-Instruct` | 70B mid |
| 3 | `Qwen/Qwen3-235B-A22B-Instruct-2507` | anchor, the pipeline's current model |
| 4 | `deepseek-ai/DeepSeek-V4-Pro` | frontier open-weight |
| 5 | `moonshotai/Kimi-K3` | frontier open-weight |

### 2.5 Measurement

Generation mirrors `pipeline._run_qgen`, including the compound-question reframe
and the abstention retry, with one deliberate difference: **gate verdicts are
recorded rather than applied**. The production path drops rejected pairs, which
would leave each model with a different row count and break the side-by-side
comparison. Recording instead of dropping turns the pass rate into a measurement.

Concurrency was 6 chains in flight; models ran sequentially so that a slow model
could not distort another's latency.

---

## 3. Prompt verification

Before spending anything on the comparison, 30 facts were drawn from the caches
and passed through the current prompt with the anchor model
(`scripts/probe_question_forms.py`).

| Stem | Count | Share |
|---|---|---|
| What | 14 | 46.7% |
| Why | 7 | 23.3% |
| Under what | 4 | 13.3% |
| How | 3 | 10.0% |
| Who | 2 | 6.7% |

Five distinct forms, "What" below half. The prompt is working. Cost: 34,774
tokens.

**A correction belongs here.** An earlier reading of this session concluded the
fix was under-covering, on the evidence that `gt_final_20260801/gt_pairs.json`
showed 89.7% "What" and that 66.1% of facts (158 of 239) match no affordance
regex. The regex-coverage figure is accurate. The conclusion drawn from it was
not. That JSON file was written at 23:08 and commit `21b8799` landed at 23:28,
so it measures the **pre-fix** prompt. In the probe, 9 of the 19 no-hint facts
still produced non-What stems, which shows the base instruction carries the facts
the regex table misses.

---

## 4. Results

### 4.1 Aggregate

| Model | Gate pass | Rejected | Errors | "What" | Stems | Q words | A words | Completion tok/call | Wall |
|---|---|---|---|---|---|---|---|---|---|
| Gemma 3 27B | 23/30 | 7 | 0 | 23.3% | 5 | 12.6 | 26.9 | 26.0 | 19.7s |
| Llama 3.3 70B | 23/30 | 7 | 0 | 70.0% | 4 | 12.9 | 27.6 | 26.2 | 32.3s |
| Qwen3 235B-A22B | 19/30 | 7 | 4 | 38.5% | 4 | 14.2 | 23.0 | 24.4 | 12.4s |
| DeepSeek V4 Pro | 19/30 | 8 | 3 | 29.6% | 5 | 12.8 | 14.2 | 19.3 | 17.3s |
| Kimi K3 | 19/30 | 11 | 0 | 43.3% | 5 | 12.8 | 23.2 | **403.6** | 62.6s |

Total: 362,223 tokens across 5 models and 30 chains.

### 4.2 Failure-mode counts

| Model | Template Qs | Compound Qs | Verbatim-copied As | Abstentions | Ungrounded |
|---|---|---|---|---|---|
| Gemma 3 27B | 0 | 4 | 0 | 0 | 3 |
| Llama 3.3 70B | **5** | **8** | 1 | 1 | 5 |
| Qwen3 235B-A22B | 0 | 4 | 2 | 1 | 4 |
| DeepSeek V4 Pro | 0 | 2 | 2 | **5** | 1 |
| Kimi K3 | 0 | 2 | 2 | 2 | **6** |

"Template" counts questions matching `property of .* and .* consequence`.
"Verbatim-copied" flags answers whose longest common run with the source fact
exceeds 65% of the answer.

---

## 5. Per-model assessment

Ranked on fitness for ground-truth generation, not general capability. Chain ids
refer to `comparison.html`.

### 5.1 DeepSeek V4 Pro — best answers

Answers are the tightest in the panel at 14.2 words against 23 to 28 for the
others: *"To prevent offsetting errors from becoming significant."* (c09),
*"When testing a coating cross-section."* (c10). An answer that is exactly the
span the question asked for is what ground truth wants. Its ungrounded rate is
the lowest at 1 of 30, and it produced the fewest compound questions.

The cost is yield. It abstained on 5 of 30, and some abstentions are wrong: at
c08 it asked *"Under what condition does the latest edition of a referenced
document apply?"* and then answered *"Insufficient information to answer"* — the
fact states the answer outright. Three generations returned empty.

**Use when answer precision matters more than volume. Budget for roughly 25%
loss and re-run the drops.**

### 5.2 Gemma 3 27B — best questions

The most varied and most natural questions in the panel: 23.3% "What", five
stems, zero template constructions, zero abstentions, zero errors, and the
highest gate pass rate. On chain c05 it asked *"How is the expanded uncertainty
adjusted when a measurement value is not corrected for bias?"* where Llama asked
*"What does the Xcorr measurement result correct for?"* — the first is an
information need, the second is a lookup.

Its weakness is answer fidelity. It re-words rather than tracking the source:
*"magnifying devices and reflective surfaces"* for *"magnification instruments or
mirrors"* (c20), *"diminished hardness readings"* for *"lower hardness values"*
(c04). Three ungrounded rejections trace to that drift. Paraphrase is a hazard
for ground truth because the answer should remain checkable against its span.

**Best choice for generating the questions. Pair it with a stricter model for
the answers.**

### 5.3 Qwen3 235B-A22B — solid, with one serious flaw

Balanced on every axis, no template questions, and the fastest wall time at
12.4s.

It hallucinated a domain. At c03 it produced *"Under what condition might
traceability not be ensured in the welding process due to machine component
issues?"* The source document is DIN EN ISO 6507-1, Vickers hardness testing.
Welding appears nowhere in it. A fabricated domain term is worse than a dropped
pair because it reads as correct and will survive review. It also produced the
most empty generations, 4 of 30.

**Defensible as the default, but questions need a term-grounding check against
the source before they ship.**

### 5.4 Kimi K3 — good output, unreliable process

When it works, its answers are the most fluent and complete in the panel (c06,
c07, c09, c14, c16). It recorded no hard errors.

It emits 403.6 completion tokens per call against 19 to 26 for every other
model, a factor of 16. That is chain-of-thought, and it escapes into the question
field in 3 of 30 cases:

- c00: raw planning text — *"which are terms from S1 and S2. Required anchors
  include Direct, Verification, Indirect and NMI, Vickers, National Level..."*
- c03: *"why tolerances matter; the answer is traceability. So don't mention
  traceability?"* — which leaks the answer into the question
- c17: *"What designation...?"* — truncated

This is the failure mode already recorded for `gpt-oss-120b` in the project
notes.

**Not usable for ground truth until the leak is filtered.**

### 5.5 Llama 3.3 70B — weakest

It collapsed into a single construction. Five of 30 questions are *"What property
of X and practical consequence of Y..."* (c01, c02, c03, c15, c16). That template
is why it sits at 70% "What", nearly three times Gemma's rate, and why it
produced the most compound questions at 8 of 30.

The template also backfires. At c04 its own question — *"What property of
hardness measurement and practical consequence of vibration are associated with
small forces?"* — was unanswerable, and the answer model returned *"Insufficient
information to answer"* on a fact that Gemma, DeepSeek and Kimi all answered
correctly. Where it avoids the template it tends toward shallow lookups, and at
c20 it copied the source sentence into the answer.

**Avoid.**

---

## 6. Findings

### 6.1 Model scale does not predict question quality

Gemma 3 27B, the smallest model in the panel, writes the most varied and most
natural questions. Llama 3.3 70B, more than twice its size, is the weakest
generator here. Both frontier models sit between them on stem diversity. What
separates the models on this task is whether they follow the affordance
instruction or fall back on a fixed construction, and that does not track
parameter count.

For the pipeline this means model spend is not a lever on question quality in
the way it is usually assumed to be.

### 6.2 The quality gate measures validity, not quality

Llama 3.3 70B and Gemma 3 27B tie on gate pass rate at 23 of 30. Llama's passing
set includes five copies of the same template question. Gemma's contains none.
The gate cannot separate them.

Two of Kimi's three leaked-reasoning outputs also passed, with
`reject_reason=None` and `is_grounded=True`. Chain c03 — *"why tolerances matter;
the answer is traceability. So don't mention traceability?"* — is grounded
against its fact, so the NLI check accepts it. It would have shipped into ground
truth.

The gate has no detector for reasoning leakage, truncated questions, or template
monotony.

### 6.3 The question-form fix works

46.7% "What" across five stems, verified on real facts with the real model. The
89.7% figure that motivated further concern came from a pre-fix run.

### 6.4 Question quality and answer quality are separable

The model that writes the best questions (Gemma) writes the least faithful
answers. The model that writes the best answers (DeepSeek) drops a quarter of
them. Nothing in the pipeline requires one model to do both, and the results
argue against it.

---

## 7. Recommendations

1. **Split the roles.** Use Gemma 3 27B for question generation and DeepSeek V4
   Pro for answer generation. This pairs the best question diversity with the
   tightest, best-grounded answers. `_run_qgen` already takes `llm` and
   `answer_llm` as separate arguments, so no structural change is needed. Token
   volumes were 63,524 for Gemma and 70,539 for DeepSeek against 75,994 for the
   anchor; whether the pairing is cheaper in currency depends on per-model
   pricing, which was not collected in this run.
2. **Add a reasoning-leak detector to the gate.** Reject a question when the
   generating call's completion-token count is far above the per-model median, or
   when the question contains second-person planning language or ends in an
   ellipsis. Both leaked Kimi questions would have been caught.
3. **Add a term-grounding check on questions.** Reject a question containing a
   domain noun absent from its supporting facts. This catches the Qwen "welding"
   hallucination, the single most damaging failure in the run because it is
   invisible to every existing check.
4. **Add a template-monotony check.** Flag when one syntactic construction
   accounts for more than roughly 15% of a run's questions.
5. **Investigate the empty generations.** Seven `qgen returned empty` failures
   across Qwen (4) and DeepSeek (3), on different chains each time, so they are
   not chain-specific. Neither model should be trusted for a production run until
   the cause is known.
6. **Do not use Llama 3.3 70B or Kimi K3** for ground-truth generation as
   configured.

---

## 8. Limitations

Thirty chains, one sample per cell, `temperature=0`. Differences in kind are
visible and reproducible: the Llama template, the Kimi leak, the Qwen
hallucination. Differences in degree are not distinguishable from noise, and the
middle three models should not be ranked against one another on these numbers.

Two documents, both DIN EN ISO standards in the same register. Behaviour on
prose, textbooks or papers is not covered.

The stage-5 candidate budget was reduced from the pipeline default of 400 to 150
to keep the graph build tractable. This narrows the pool of two-fact chains
available for selection. It does not bias the comparison, since all five models
received whichever chains were selected.

Claude Sonnet and GPT-5.x were requested for this comparison but are not
available on Nebius Token Factory, which serves open-weight models only. No
conclusion here extends to closed frontier models.

---

## 9. Reproducing

```bash
cd "D:/Mini Thesis/Rag_web_pipeline"

# Prompt gate (~35k tokens)
./venv/Scripts/python.exe scripts/probe_question_forms.py --n 30

# Stages 0-6, writes chains.json + pool_*.json  (~12 min)
./venv/Scripts/python.exe scripts/model_bakeoff.py --build-chains

# Re-select from saved pools without any LLM call
./venv/Scripts/python.exe scripts/model_bakeoff.py --select-only

# Stage 7 for all five models (~2.5 min, ~362k tokens)
./venv/Scripts/python.exe scripts/model_bakeoff.py --run

# Render
./venv/Scripts/python.exe scripts/build_bakeoff_html.py
```

The chain pools are committed, so swapping the roster or adding a sixth model
costs only the stage-7 calls.
