# GT quality fixes — live rerun results, 5 DIN/ISO standards (2026-08-01)

Follows `RERUN_RESULTS_20260801.md`. That run validated the *ingestion* fixes.
This one validates the *ground-truth quality* fixes made after the user
reported vacuous pairs like:

```
F: Annex A lists criteria which assist in the selection of the appropriate
   part of the ISO 3834 series.
Q: What criteria does Annex A provide for selecting the appropriate part of
   the ISO 3834 series?
A: Annex A provides criteria that assist in the selection of the appropriate
   part of the ISO 3834 series.
```

The answer never names a criterion, because the fact never contained one.

## Why nothing caught it

The reported pair reproduces deterministically in two independent runs
(`f3161baa4666`, `fb0ee36df5d7`) from different chunk indices. Tested against
every tautology detector in the codebase, all returned False:

| gate | result |
|---|---|
| `Q3_answer_echoes_question` (gt_quality) | miss |
| `Q13_answer_tautology` (gt_quality) | miss |
| `_is_tautological` (allpdf pipeline) | miss |

Four independent root causes, none of them a threshold:

1. **Stage 3** — `_expand_short_fact` appended `". "` onto text already ending
   in `.`, because `rstrip(" ")` does not remove a period. The sentence
   splitter splits *after* `[.!?]`, so essentially every join it performed
   doubled the period.
2. **Stage 3** — the controlled-copy watermark is four lines; the regex matched
   only the last, so `Company name: RWTH Aachen University …` glued onto real
   sentences. A block straddling a chunk boundary also orphaned its tail, fixed
   separately via the truncated-field `..` signature.
3. **Stage 4** — no predicate existed for a **deferring fact**, one that asserts
   only that content lives elsewhere. Grounding cannot see this: a restatement
   of a deferring fact genuinely IS entailed by it. Grounding measures truth,
   not informativeness.
4. **Stage 7** — the 20-check `gt_quality` suite was never imported by
   `allpdf/pipeline.py` (only by the legacy `gt_pipeline.py`), and the tautology
   gate had been removed in "FIX 5", leaving `_is_tautological` callable from
   nothing but its own green test. That removal's reasoning held for two-fact
   pairs; the single-fact path — 99% of output — was never covered. No dedup of
   any kind ran, so byte-identical questions shipped.

## Deterministic-first validation

Before spending anything, the new gates were replayed over the recorded 453
pairs: **20.8% removed**, converging with an independent `gt_quality` replay
(20.5%). Every removal bucket was read by eye. Three rounds of false positives
were found that way and are now regression tests — most importantly the
tautology metric, which began as a novelty *ratio* and wrongly rejected
`"… is 10 minutes."` because its one new token was diluted by the question's
phrasing. Rewritten as novelty *presence*; a `len(w) > 2` filter was also
discarding `"2"`, `"10"`, `"5"` — usually the answer.

## Live rerun

Nebius `Qwen/Qwen3-235B-A22B-Instruct-2507`, fresh out-dirs so no cached
extraction masked the Stage-3 fixes.

| doc | before | after | result |
|---|---|---|---|
| `din_iso_3834_1` | 7 | 6 | FAIL (stage5_graph, 0 edges — legitimate) |
| `din_iso_4136` | 12 | 10 | PASS |
| `din_iso_13919_1` | **0** | **30** | PASS |
| `din_iso_3452_1` | 152 | 147 | PASS |
| `din_iso_6507_1` | 282 | 241 | PASS |
| **total** | **453** | **434** | |

Multi-hop pairs: **4 → 61**. The web path defaulted `multihop_chains` to 0 and
never exposed it.

## Quality, all 434 pairs

```
CLEAN (no defect flag):  77.6%     (was 56.7%)
FLAGGED:                 22.4%     (was 43.3%)
```

| defect | before | after |
|---|---|---|
| `mangled_fact_text` | 13.5% | 2.3% |
| `unresolved_anaphora` | 7.7% | **0%** |
| `watermark_bleed` | 6.4% | 2.1% |
| `near_duplicate` | 4.0% | 1.6% |
| `tautological_answer` | 2.2% | 0.7% |
| deferring facts | 8.4% | 4.6% |

Gates firing in the `6507_1` run log — every one of these was 0 before, because
the code was unreachable from this pipeline:

```
tautological_dropped=13  quality_dropped=45  duplicate_dropped=4
quality_reasons={'Q9_weak_required_evidence': 23, 'tautological_pair': 13,
                 'Q7_bad_fact_fragment': 20, 'Q11_answer_source_heading_leak': 2}
```

**No threshold was loosened or tightened.**

## The next bottleneck — `weak_self_containment`, untouched

`din_iso_3834_1` Stage 4:

```
in=54 kept=10 dropped=44
reasons={'weak_self_containment': 31, 'fragment': 4, 'iso_boilerplate': 2,
         'deferring_fact': 2, 'unresolved_deictic': 1, ...}
```

All the new rules together drop 3; `weak_self_containment` drops 31. This is
the pre-existing `_RELAXED_MIN_SELF_CONTAINMENT = 0.75` against a scorer that
emits quantized 1.0 / 0.5 / 0.2 / 0.0, so it behaves as "keep only exactly
1.0" — the degenerate distribution recorded in
`STAGE4_SELF_CONTAINMENT_FINDING_20260801.md`, which was resolved for the
table-heavy document but not for prose standards.

It has deliberately **not** been touched. Lowering it admits a mid-band that
is a genuine mix of content and boilerplate. The fix belongs in the scorer's
input or the judge itself, and should be scoped, not guessed.

`6507_1` is healthier (326/478 kept, 68%) — the collapse is specific to the
short prose standards.

## Still open

- 6.0% `meta_document_fact`, 4.6% deferring. The detector is a conjunction with
  deliberate escape hatches: precise, not exhaustive.
- 9.7% `near_tautological` under the audit's loose 80–95% overlap band, which
  is stricter than the shipped gate. Some of those are acceptable pairs.
- Generic running-header/repeated-line detection at ingest is the principled
  watermark fix, but it would rewrite text after char offsets and bboxes are
  computed. Left for a provenance-safe change rather than done blind.
