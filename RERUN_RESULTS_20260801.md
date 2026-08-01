# Live rerun results — 5 DIN/ISO standards, 2026-08-01

Run after today's ingestion fixes, against Nebius
(`Qwen/Qwen3-235B-A22B-Instruct-2507`), via
`python -m rag_gt.allpdf.pipeline` in `Rag_web_pipeline` with
`PYTHONPATH` pinned to that repo's `src/` (see the import trap in
`HOW_TO_RUN_BOTH_PIPELINES.md` — without it the monorepo engine runs and
every call 404s).

Flags: `--multihop-chains 12 --score-necessity`.

## Results

| doc | result | QA pairs | pages | facts kept | failed gates |
|---|---|---|---|---|---|
| `din_iso_13919_1` | FAIL | 6 | **24/24** | 7/95 (7%) | stage4_filter, stage5_graph |
| `din_iso_3452_1` | PASS | 155 | **32/32** | 166/293 (57%) | — |
| `din_iso_3834_1` | FAIL | 11 | **15/15** | 11/48 (23%) | stage5_graph |
| `din_iso_4136` | PASS | 19 | **18/18** | 16/69 (23%) | — |
| `din_iso_6507_1` | PASS | 293 | **47/47** | 340/517 (66%) | — |

**484 QA pairs total. 3/5 PASS. 100% page coverage on all five documents.**

## What this validates

**The ingestion fixes work across the corpus, not just the one document.**
Every document reached full page coverage. `din_iso_13919_1` logged one
silent-Docling-drop repair (pages 18–20 recovered via PyMuPDF) and still
gated PASS at 24/24 — the exact failure that previously left it ingesting
8 of 24 pages, all front matter.

`din_iso_6507_1` is 47 pages, well past the old `docling_page_cap=8`
default, and covered all 47.

## The two failures

Both fail the same gate — `stage5_graph`, `edges>=1 -> got 0` — for
different reasons.

**`din_iso_3834_1` — legitimate.** `0/25 edges accepted from 25 attempted
pairs`. This matches the previously-recorded finding that this standard
consists of independent scope statements with no genuine cross-fact
relationships. The gate is doing its job; there is nothing to fabricate.
Not a bug, and it should not be "fixed" by loosening edge acceptance.

**`din_iso_13919_1` — cascade from Stage 4.** Only `0/12` pairs, because
Stage 4 had already starved it to 7 facts. The root cause is the table-row
fact representation gap documented in
`STAGE4_SELF_CONTAINMENT_FINDING_20260801.md`, not the graph stage.

## Third bug found by inspecting the run output: missing bbox provenance

The run surfaced a provenance gap that no test covered. Checking bboxes on
the Stage-4 facts of every document:

| doc | chunking | facts | with bbox |
|---|---|---|---|
| `din_iso_13919_1` | table_aware | 7 | **7** |
| `din_iso_3452_1` | clause | 166 | **0** |
| `din_iso_3834_1` | clause | 11 | **0** |
| `din_iso_4136` | clause | 16 | **0** |
| `din_iso_6507_1` | clause | 340 | **0** |

Only the table-aware document had any. bbox observability is a hard
requirement for this project, so 533 facts with no boxes is a real defect.

**It was never data loss — only a surfacing gap.** Stage 3's `_make_span()`
reads `chunk["bboxes"]`. `_pack_units()` (table_aware / ocr_block) rolls its
source units' boxes up into that key; the chunkers behind `chunk_document()`
(clause / heading / recursive) keep provenance nested in `source_units` and
emit no chunk-level key, so `_make_span()` read `[]`. Confirmed the boxes
were present the whole time: **all 1231 `source_units` of `din_iso_6507_1`
carried bboxes while 0/281 of its chunks did.**

Fixed with `_rollup_bboxes()`. Verified free of charge via `--dry-run`
(stages 0–4, no LLM):

```
din_iso_4136:   facts=66 with_bbox=66 (100%)     # was 0%
din_iso_3834_1: facts=33 with_bbox=33 (100%)     # was 0%
```

## Corpus-wide context for the Stage-4 finding

Keep rates: 7%, 57%, 23%, 23%, 66%. Only `din_iso_13919_1` is an outlier at
7%. The adaptive filter is healthy on prose-heavy standards; the collapse is
specific to the table/figure-dominated document, which supports the
diagnosis that the gap is a missing self-contained representation for table
rows rather than a mis-set threshold.

**No threshold was tuned to make anything pass.**
