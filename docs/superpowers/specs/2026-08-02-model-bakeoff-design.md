# Multi-Model Q&A Bake-Off — Design

Date: 2026-08-02
Repo: `Rag_web_pipeline` (the only repo containing the question-framing fix, `21b8799`)

## Goal

Compare the *quality of generated questions and answers* across five LLMs of
differing capability, holding the evidence constant. Output is a single HTML
page showing all 30 questions with every model's Q&A visible side by side.

This is a qualitative comparison. Each cell is one sample at `temperature=0.0`.
It supports statements like "the 27B model writes vaguer questions". It does
**not** support a statistical ranking of the five models.

## Model roster

All five are served by Nebius Token Factory (`https://api.tokenfactory.nebius.com/v1`),
which hosts open-weight models only. Claude Sonnet and GPT-5.x are not available
there and are out of scope for this run.

| Column | Model ID | Tier |
|---|---|---|
| 1 | `google/gemma-3-27b-it` | ~27B small |
| 2 | `meta-llama/Llama-3.3-70B-Instruct` | 70B mid |
| 3 | `Qwen/Qwen3-235B-A22B-Instruct-2507` | anchor — the project's current model |
| 4 | `deepseek-ai/DeepSeek-V4-Pro` | frontier open-weight |
| 5 | `moonshotai/Kimi-K3` | frontier open-weight |

## Corpus

Two PDFs, 15 chains each:

- `data/gt_input_DINENISO65071ENG/DIN EN ISO 6507-1-ENG.pdf` (`din_iso_6507_1`)
- `data/gt_input_DINENISO34521ENG/DIN EN ISO 3452-1-ENG.pdf` (`din_iso_3452_1`)

Both have complete Stage-3 extraction caches (80 cached chunks each, 116 and 123
facts) at `data/reruns_postfix/<doc>/checkpoints/s3_chunk_cache/`.

## Architecture

### Why the evidence must be shared

Running the whole pipeline per model would let each model extract different
facts and sample different chains, so the columns would describe different
experiments. Instead, stages 0–6 run **once** to produce a frozen set of 30 fact
chains; only stage 7 (question generation + answer generation) varies by model.

### Why chains are rebuilt rather than loaded

Neither existing artifact can drive stage 7 faithfully:

- `gt_final_20260801/gt_pairs.json` omits `Fact.role`, which
  `questions._format_role_shape_hint` uses to steer the question form — the
  exact mechanism under test.
- `checkpoints/s5_graph.json` and `s6_chains.json` are summaries
  (`{n_facts, edge_count}`), not the graph or the chains.

`extract._fact_to_cache` **does** persist `role` and `spans`, so rebuilding from
the Stage-3 chunk cache is both faithful and nearly free: stages 0–3 are cache
hits, stage 4 is deterministic. Only stage 5 (TF-SFG edge classification) makes
fresh LLM calls, and only on the first run.

### Components

| Unit | Responsibility |
|---|---|
| `scripts/model_bakeoff.py` | Build or load the frozen chain set; run all five models over it; write `results.json` |
| `scripts/build_bakeoff_html.py` | `results.json` → one self-contained HTML page |

Model swapping is `APILLM(base_url, api_key, model_id)` — the class already
takes the model as a constructor argument, so no new abstraction is needed.

### Frozen chain artifact

`data/model_bakeoff/chains.json` persists each chain with its full fact payload
(`fact_id`, `text`, `canonical_form`, `role`, `weight`, `spans`) plus
`chain_edges`. Once written, adding a sixth model or re-running after a prompt
change costs only stage-7 calls.

### Chain selection

Deterministic, `seed=42`, 15 per document. Two-fact genuine-dependency chains
are capped at **one third** of the panel: they exercise `chain_edges` and the
role-shape hint and are the interesting ones, but single-fact chains are ~99% of
what the pipeline actually emits, so a panel made entirely of two-fact chains
would not represent the output under comparison. The rest are single-fact chains
round-robined across fact roles so the sample is not entirely `definition`.
Chains whose facts trip `_any_fragment_fact` are excluded during selection — the
pipeline skips them, so they would render as empty cells for every model.

The full per-document chain pool is persisted to `pool_<doc_id>.json`, and
`--select-only` re-runs selection from it with no LLM calls. Stages 0–6 are the
expensive part; saving only the 15 selected chains would make any change to the
selection rule cost another full graph build.

### Deliberate deviation from pipeline defaults

Stage 5's candidate budget is **150**, not the pipeline's
`min(400, n_facts * 3)` = 400. Edge classification is one LLM call per candidate
and measured ~7 pairs/min, so 400 costs ~55 min per document. A probe run had
accepted 41 edges by pair 125, and only ~5 two-fact chains per document are
needed. This narrows the two-fact *pool*; it does not bias the comparison, since
all five models see whichever chains are selected.

### Generation, per model per chain

Mirrors `pipeline._run_qgen` exactly, including the compound-question reframe
and abstention retry, with one deliberate difference: **gate verdicts are
recorded, not applied**. `_run_qgen` drops rejected pairs, which would leave
different models with different row counts and break the side-by-side alignment.
Here every model produces a value for all 30 rows, annotated with what the gate
said. "Gemma passed 19/30, Kimi 28/30" is itself a headline quality signal.

Recorded per cell: `question`, `answer`, `qgen_sec`, `agen_sec`, question stem,
`reject_reason` (or null), `is_abstention`, `is_grounded`, `error`.

### Error handling

A model that errors or times out yields a cell carrying the error text, rendered
visibly. A model that cannot reliably complete the task is reporting a quality
result, not a bug to be hidden.

Concurrency: 6 chains in flight per model, models run sequentially so a slow or
rate-limited model cannot distort another's latency measurements.

## HTML output

`data/model_bakeoff/comparison.html`, self-contained, no external assets.

- **Summary panel**: per model — question-stem distribution, gate pass rate,
  mean question and answer length, mean latency, error count.
- **30 row blocks**: shared evidence (fact text, role, page) across the top,
  then a five-column grid — model name, its question, its answer. All five
  visible at once; no accordions, no dropdowns. Columns sit in a horizontally
  scrolling container so nothing is crushed on a narrow viewport.
- Cells are tinted by gate verdict: pass, rejected (with reason), error.

## Cost

30 chains × 5 models × 2 calls = 300 generation calls, plus a one-time stage-5
edge-classification pass. Prompts are short. Real token usage is recorded via
`usage_summary()` and reported as measured cost, not estimated.

## Known limitation, stated up front

In the 2026-08-01 run, 89.7% of 438 questions began with "What". The cached role
distribution explains why: 147 of 239 facts are `definition`, and
`_format_role_shape_hint` only fires for specific role pairs such as
(`definition`, `condition`). A chain of two definitions affords "What" and
little else. This is a property of extraction, not of the question prompt, and
it will appear as a weakness shared by all five columns rather than as a
difference between them. The bake-off runs against the prompt as it stands so
the baseline is clean.
