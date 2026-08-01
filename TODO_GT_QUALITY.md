# TODO — GT quality, next phase

Opened 2026-08-01, after the root-cause fixes in `d324d80..4f4e130`.
Baseline for every number here: `GT_QUALITY_FIX_RESULTS_20260801.md` and the
438-pair dataset in `data/gt_final_20260801/`.

Status key: `[ ]` open · `[~]` partially done · `[x]` done · `[!]` blocked/needs a decision

---

## P0 — the yield bottleneck

### [ ] T1. `weak_self_containment` drops more than every other rule combined

**Evidence.** `din_iso_3834_1` Stage 4: `in=54 kept=10`, of which
`weak_self_containment` accounts for **31**. Every rule added on 2026-08-01
drops 3 between them. `din_iso_6507_1` is healthier (326/478 kept) — the
collapse is specific to short prose standards.

**Cause.** `_RELAXED_MIN_SELF_CONTAINMENT = 0.75` in
`src/rag_gt/allpdf/filter_adaptive.py`, applied to a scorer
(`semantic_extraction.self_containment_score`) that emits **quantized** values
— 1.0 / 0.7 / 0.5 / 0.2 / 0.0, per its own prompt rubric. A 0.75 threshold on
that distribution means "keep only 1.0". The threshold has no resolution where
it is set.

**Do NOT** lower the threshold. `STAGE4_SELF_CONTAINMENT_FINDING_20260801.md`
already established the mid-band is a genuine mix of real content and real
boilerplate.

**Options to evaluate, cheapest first:**
- [ ] T1a. Re-score the 0.5 band with the surrounding chunk as context. Many
      facts score 0.5 for an unresolved referent the chunk would resolve.
      Free to prototype against `s4_facts.json` checkpoints — no regeneration.
- [ ] T1b. Replace the LLM judge with a deterministic self-containment signal
      (unresolved-pronoun count, undefined-acronym count, subject presence).
      Deterministic-first, per the project's own rule.
- [ ] T1c. Keep the judge but make the gate two-tier: 1.0 auto-keep, 0.5 band
      admitted only when it survives the other Stage-4 predicates. Measurable
      on recorded facts at zero cost.

**Acceptance.** `din_iso_3834_1` yields > 20 pairs without any threshold being
moved, and the 438-pair audit's clean rate does not drop below 78%.

---

## P1 — residual defects in the shipped dataset

Current audit of `data/gt_final_20260801/gt_pairs.jsonl` (438 pairs, 21.9% flagged):

### [ ] T2. `meta_document_fact` — 28 pairs (6.4%)
Facts about the document's own structure that the deferring-fact rule does not
reach (it is a deliberate conjunction: pointer AND no payload). Needs a
separate predicate for "describes the standard's organisation" rather than
widening the deferring rule, which would start eating real scope statements.

### [ ] T3. deferring facts — 27 pairs (6.2%)
`is_deferring_fact` in `facts/domain_filter.py` is precise, not exhaustive.
Collect the 27 survivors from the audit and extend the pattern set against
them, keeping the existing negative controls green.

### [ ] T4. `near_tautological_answer` — 43 pairs (9.8%)
This is the audit script's **loose** band (80–95% question/answer word
overlap), deliberately stricter than the shipped gate. Triage required: decide
per-pair whether these are genuinely circular or just well-formed answers that
echo the question's phrasing. Do not tighten the shipped gate before this
triage — the first version of it rejected `"... is 10 minutes."`

### [ ] T5. `fact_starts_midsentence` — 12 pairs (2.7%)
Stage-2 chunking splits mid-sentence and Stage 3 keeps the fragment.
`_is_fragment` only rejects a lower-case opener when the fact is ALSO short.

### [~] T6. watermark bleed — 3 pairs (0.7%)
Two of the three fixes landed (`945d828` handles the chunk-boundary orphan).
The remaining principled fix is **generic repeated-line / running-header
detection at ingest**, which is blocked below (T9).

### [ ] T7. `near_duplicate` — 7 pairs (1.6%)
Exact-match dedup ships. Semantic near-duplicate dedup does not:
`generation/dataset_budget.py::dedup_pairs` needs an `embed_fn` and **is dead
code called from nowhere in the repo**. Either wire it into `_run_qgen` with
the existing embedding model, or delete it.

---

## P2 — structural

### [!] T8. `allpdf/pipeline.py` re-implements `pipeline/gt_pipeline.py`

The root cause behind almost everything fixed on 2026-08-01: the quality
machinery existed but was imported only by the legacy pipeline.
`_relaxed_reject` re-implements only 5 of ~20 strict-tier predicates.
`dedup_pairs` / `allocate_singles` are dead. A green test suite proved nothing
because the tests exercised code the production path never reached.

**Needs a decision before any work starts:** converge the two pipelines, or
declare `gt_pipeline.py` legacy and delete it. Both are large. Until then,
**grep for a gate's importer before assuming it protects the allpdf path.**

### [!] T9. Generic running-header/footer detection at ingest

The correct general fix for watermarks (T6). Blocked: stripping text at Stage 1
rewrites char offsets after `char_start`/`char_end`/`bboxes` are computed, and
silently corrupting provenance is worse than 0.7% residue. Needs an
offset-preserving design (mask-and-record rather than delete).

### [ ] T10. Reconcile real token usage against the provider bill
`APILLM.usage_totals` now carries provider-reported counts (`4d9fb09`). Nothing
surfaces them yet at run level. Wire `usage_totals` into the run report so a
run can be checked against the Nebius dashboard — the point of
`docs/COST_TRACKING_IS_OPTIONAL.md`.

---

## P3 — verification debt

### [ ] T11. The Studio UX changes have not been exercised live
`questions_per_doc`, the per-run API key field, and the token-cap path are unit
tested (67 backend, 20 frontend) but no live run has gone through the web UI
since. Needs one end-to-end run through the browser, not the CLI.

### [ ] T12. Propagate the fixes to the other two repos
The engine is vendored three times. These fixes are in `Rag_web_pipeline` only.
`RAG_GT` (monorepo) and `Rag_software_stack` (Studio) still carry the
unfixed `_expand_short_fact`, the narrow watermark regex, and the unwired
`gt_quality`. Mind the import trap — verify `rag_gt.__file__` per repo.

### [ ] T13. `din_iso_3834_1` still fails `stage5_graph` (0 edges)
Recorded as legitimate in `RERUN_RESULTS_20260801.md` — the standard is
independent scope statements with no genuine cross-fact relationships. Re-confirm
after T1, since more surviving facts may create real edges. Do not loosen edge
acceptance to make it pass.

---

## Done on 2026-08-01

- [x] `..` doubled-period joiner (`extract.py:131`) — 13.5% → 1.1%
- [x] Four-line watermark strip + chunk-boundary orphan — 6.4% → 0.7%
- [x] Deferring-fact predicate — the reported defect class
- [x] ISO-register deictic opener, expletive "it" exempted — 7.7% → 0%
- [x] `gt_quality` 20-check suite wired into `_run_qgen`
- [x] Tautology gate restored, rewritten as novelty presence not ratio
- [x] Exact-duplicate question dedup
- [x] `questions_per_doc`, mode renames, per-run API key (never persisted)
- [x] Cost tracking demoted to optional; real provider token counts captured
