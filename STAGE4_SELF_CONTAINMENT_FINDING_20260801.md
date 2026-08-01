# Stage-4 self-containment is the new bottleneck (2026-08-01)

Written after the ingestion fixes landed and a real live rerun on
DIN EN ISO 13919-1. **Not fixed** — deliberately. Recording the evidence so
the fix can be designed rather than guessed.

## Why this only surfaced now

Before today, this document only ever ingested pages 1–8 (all front matter),
so there was nothing technical to filter. With ingestion fixed
(24/24 pages, 27 `table_aware` chunks) the real content finally reaches
Stage 3, and Stage 4 promptly discards most of it.

## What the live run shows

Stage 3 extracted **95 facts**. Stage 4 kept **7**.

```
[filter_adaptive] din_iso_13919_1: tier=relaxed doc_type=ISO_STANDARD
  in=95 kept=7 dropped=88
  reasons={'weak_self_containment': 76, 'iso_boilerplate': 5,
           'front_matter_artifact': 4, 'table_artifact_raw': 1,
           'reference_list_dump': 1, 'table_artifact': 1}
```

Gate `drop_rate<0.9` FAILED (7/95 kept). Downstream starved: 7 facts →
12 candidate pairs → `stage5_graph` FAIL → 6 QA pairs, `Pipeline FAIL`.

## The score distribution is degenerate

All 95 facts, by `self_containment_score`:

| score band | count |
|---|---|
| 1.0 | 10 |
| 0.75–0.99 | **0** |
| 0.5–0.74 | 30 |
| <0.5 | 55 |

`_RELAXED_MIN_SELF_CONTAINMENT = 0.75` therefore behaves as "keep only
exactly 1.0". Every one of the 7 survivors scored 1.0. The scorer emits
quantized values (1.0 / ~0.5 / ~0.2 / 0.0), not a continuous score, so the
threshold has no meaningful resolution where it is set.

**Do not just lower the threshold.** Dropping it to 0.5 would admit the 30
mid-band facts, but the sample below shows that band is a mix of real
content and real boilerplate — it would let junk in without getting the
tables out. The threshold is a symptom, not the cause.

## The actual cause

The surviving 7 facts are all *narrative* prose (pages 1, 9, 9, 9, 10, 12,
23) — the foreword and scope statements. **None come from the imperfection
tables**, which are the entire technical substance of this standard.

Sub-threshold examples show why:

```
0.00  ", 61 pores, d = 1 mm Figure A.7 - Surface percentage: 5"
0.20  "% , 122 pores, d = 1 mm Figure A.9 - Surface percent"
0.20  "45 pores, d = 1 mm DIN EN ISO 13919-1:2020-03 EN IS"
```

That *is* the technical content — arriving as table/figure cell fragments
with no sentence structure, so the self-containment scorer correctly rates
it near zero. `table_aware` chunking preserves tables as whole chunks, but
the Stage-3 SFU extractor still emits them as sentence fragments.

Genuine boilerplate is also in the sub-threshold band and *should* keep
being dropped:

```
0.00  "tifying any or all such patent rights. Details of any patent rights..."
0.20  "Any feedback or questions on this document should be directed to..."
```

So the filter is not malfunctioning. The gap is upstream: **there is no
representation for a table row as a self-contained fact.**

## What a real fix looks like

Give table-derived content a fact form that can stand alone — template a row
against its header, e.g.

> "For quality level B, the imperfection 'crack' is not permitted."

built from the table's header row + row label + cell value, rather than
emitting the raw cell text and hoping it reads as a sentence. That keeps the
self-containment floor honest and makes the tables reachable as ground
truth.

This is a Stage-3 extraction change, and it is the same conclusion as the
existing project guidance: fix garbage at the Stage-3/4 source, not with
output-stage band-aids, and do not remove fragment facts globally because
multi-hop chains need them.

## Status

- Ingestion: **fixed and verified** (8/24 → 24/24 pages).
- Stage 4: **diagnosed, not fixed.** Needs the table-row fact representation
  above. No threshold was touched.
