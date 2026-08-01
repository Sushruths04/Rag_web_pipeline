# GRAFT ingestion — how it works, what was broken, how it was fixed

2026-08-01. Written for whoever picks this up next, including future me.

---

## 1. What the pipeline does, stage by stage

Both spinout repos vendor the same engine (`src/rag_gt/`). The all-PDF path
is `rag_gt.allpdf.pipeline.run_doc_pipeline`:

| stage | module | what it produces |
|---|---|---|
| 0 preflight | `allpdf/preflight.py` | `DocProfile`: page count, scanned?, table density, doc type, **which extractor backend to use** |
| 1 ingest | `allpdf/ingest.py` | `SourceUnit[]` — text + page + bbox + char span |
| 2 chunk | `allpdf/chunk.py` | chunks, strategy chosen from the profile |
| 3 extract | `allpdf/extract.py` | `Fact[]` (SFUs) via LLM segmentation, each with a `Span` |
| 4 filter | `allpdf/filter_adaptive.py` | drops boilerplate / fragments / front matter |
| 5 graph | `graph/typed_sfg.py` | typed edges between facts (LLM classifier) |
| 6 chains | `allpdf/pipeline.py` | single- and multi-hop chains |
| 7 QGen | `generation/` | question/answer pairs |

Each stage has a `VerificationGate`. A failed gate does not stop the run; it
marks the pipeline FAIL and the run continues, which is why a stage-4 failure
still yields (bad) pairs at stage 7.

### The routing decision that matters

`preflight.select_backend` → `docling_table` when the document is
table-dense, else `legacy` (PyMuPDF), else `docling_ocr` when scanned.
`chunk.select_strategy` → `table_aware` / `clause` / `heading` / `recursive`
/ `ocr_block`.

**These two are chosen independently.** A document can be routed to
`docling_table` for ingestion and `clause` for chunking. That split is the
source of two of today's bugs.

---

## 2. The bugs, in causal order

They form a chain. Each one hid the next, which is why they had to be fixed
in this order.

### Bug 1 — `docling_page_cap` truncation (pages never reached the extractor)

Three orchestrators (`run.py`, `pipeline.py`, `eval_orchestrator.py`) each
redeclared `docling_page_cap` defaulting to **8**, while `ingest_document()`
itself defaulted to 40. Worse, the over-cap branch did not do what its own
docstring promised: instead of falling back to legacy extraction it *sliced
the PDF to pages 1..cap and still ran Docling on the slice*.

DIN EN ISO 13919-1 is 24 pages whose first 8 are title page, translation
notice, and foreword. The pipeline therefore ingested only front matter and
nothing else — and reported success.

**Fix:** the over-cap branch now falls back to full-document legacy
extraction and logs it; all three defaults raised to 60. A new stage-1 gate
(`pages_covered >= 90% of page_count`) fails loudly if this ever recurs.

### Bug 2 — silent Docling page drops

Docling swallows some native failures (`std::bad_alloc` in its preprocessor)
and returns a **partial document**. `convert()` reports success, so the
caller's `except` never fires and the pages simply vanish. Non-deterministic
across runs.

**Fix:** `_repair_dropped_pages()` — any page in a converted range that
produced no source unit is re-extracted via PyMuPDF and spliced back in page
order, with char offsets reassigned. Blank pages are self-correcting.

### Bug 3 — Docling tables silently dropped *(the real one)*

`_docling_units_to_text()` read only `item.text`:

```python
raw = getattr(item, "text", None)
cleaned = clean_text(str(raw or ""))
if not cleaned:
    continue
```

A Docling `TableItem` has `.text is None`. **Every table hit that `continue`
and disappeared** — on the backend chosen precisely *because* the document is
table-dense, with `do_table_structure=True` already paying to extract the
structure.

Pages 19–20 of 13919-1 contain `{SectionHeaderItem: 4, PictureItem: 2,
TableItem: 2}` and **zero** TextItems. All four text-bearing items were
tables/pictures, so those pages emitted nothing — which is what triggered
Bug 2's repair, which refilled them from PyMuPDF as a flat stream of cells
with row/column association destroyed. The `D C B` quality-level headers
landed in one chunk and their limit values in the next.

That flat stream is why the self-containment scorer rated the table content
0.0–0.2 and stage 4 dropped it. **The filter was never the problem.**

**Fix:** serialize tables with `export_to_markdown()`, preserving the header
row and cell alignment.

### Bug 4 — bbox provenance lost for non-table chunking

Stage 3's `_make_span()` builds a fact's provenance from `chunk["bboxes"]`.
`_pack_units()` (table_aware / ocr_block) rolls its source units' boxes up
into that key. The chunkers behind `chunk_document()` (clause / heading /
recursive) keep provenance nested in `source_units` and emit **no
chunk-level key at all**, so `_make_span()` read `[]`.

Result: 533 facts across four documents had no bboxes, while all 1231
`source_units` of one of them carried boxes the whole time. Not data loss —
a surfacing gap. bbox observability is a hard requirement here.

**Fix:** `_rollup_bboxes()` flattens `source_units` boxes onto the chunk for
the non-`_pack_units` strategies.

### Bug 5 — stale PDF slice cache

`_slice_pdf` keyed its cache on `sha1("{path}:{start}:{end}")` — nothing
about file contents. Replacing a PDF at a given path served the *previous*
document's pages while every provenance field pointed at the current file.

**Fix:** key includes `st_size` and `st_mtime_ns`. Also anchored the cache to
`data_dir()` instead of a bare relative path that followed the cwd.

### Bugs 6–8 — correctness-adjacent

- **QA-gen meta keys.** The stub blocks emitted `count`/`multi_hop`; the real
  engine-backed blocks that replaced them emit `n_qa`/`n_multihop`, so wiring
  in real blocks made the canvas show "? QA". Now emits both.
- **`cargo test` on a fresh clone.** `tauri_build::build()` validates the
  gitignored `python-runtime` resource even for `cargo test`, which never
  bundles. `build.rs` now creates the placeholder itself.
- **Misleading Ollama env.** `get_llm()` never reads `LLM_BACKEND`; setting
  `ollama` expecting free local inference silently bills the paid API. The
  var *is* still live for the RAGAS comparison subsystem, so it is documented
  rather than deleted.

---

## 3. The environment trap that cost the most time

All three repos install a package named `rag_gt`, only one editable install
can win per Python environment, and **neither spinout has a `venv`**. So

```bash
cd "D:/Mini Thesis/Rag_web_pipeline"
python -m rag_gt.allpdf.pipeline ...   # runs the MONOREPO's engine
```

The monorepo's older `get_llm()` ignores `RAG_LLM_CHAT_MODEL` and hardcodes
`gpt-4o`, so every call 404'd on Nebius. It presented as a config bug and was
actually the wrong code running.

The web app and Studio backend are **not** affected — both insert their own
`src/` before importing the engine (`_bootstrap()` in
`production/backend/app/stages/graft.py`, module header of
`studio/backend/adapters_live.py`).

**Always run `python -c "import rag_gt; print(rag_gt.__file__)"` before
trusting a CLI result.** Permanent fix: a venv per repo.

---

## 4. What is still open

### Stage-4 self-containment on table content

Bug 3's fix makes table rows arrive structured, which should let Stage 3
produce self-contained facts. If it does not fully, the remaining work is to
template a row against its header at extraction time:

> "For quality level B and thickness ≤ 0,5 mm, root concavity (ISO 6520-1
> reference 515) is limited to h ≤ 0,1 t, but max. 0,5 mm."

rather than emitting cells and hoping they read as sentences.

**Do not fix this by lowering `_RELAXED_MIN_SELF_CONTAINMENT`.** The score
distribution is degenerate (nothing between 0.75 and 0.99), the sub-threshold
band genuinely mixes table content with real boilerplate, and lowering the
floor admits junk without recovering the tables.

### Docling `bad_alloc` on pages 19–20

A native-layer failure inside Docling's preprocessor on this specific
document. The page-repair covers it, so no data is lost, but those pages come
back via PyMuPDF without table structure. Fixing it properly means either a
Docling upgrade or per-page conversion with a smaller memory footprint.

### `pdf_source` block in GRAFT Studio

Still a stub. It is the only thing between the Studio and the same one-click
batch experience the web pipeline already has.

### `din_iso_3834_1` stage-5 failure is legitimate

0/25 edges accepted. The standard is independent scope statements with no
genuine cross-fact relationships. **Do not "fix" this by loosening edge
acceptance** — the gate is correct and there is nothing to fabricate.

---

## 5. Principles this session confirmed

1. **Check what code is actually running.** `rag_gt.__file__` turned a
   "config bug" into a wrong-repo bug in one command.
2. **Check the schema before declaring a bug.** Two false alarms came from
   probing `page_no` instead of `page_start` and `gold_fact_ids` instead of
   `chain_fact_ids`.
3. **A fix that hides a symptom can hide a cause.** The page-repair (Bug 2)
   silently masked the table drop (Bug 3) by refilling those pages with flat
   text. Both were needed, but only fixing Bug 3 recovers the structure.
4. **Verify a guard actually guards.** Deleting the repair hook and watching
   a test fail proved the wiring; 10 unit tests passed against dead code.
5. **Don't tune a threshold to make a gate pass.** Every threshold in this
   run was left untouched.
