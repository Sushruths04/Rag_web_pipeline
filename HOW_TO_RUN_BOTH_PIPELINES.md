# How to open and work on the two GRAFT repos

Last updated: 2026-08-01

Two separate repos, same vendored engine (`src/rag_gt/`), different front ends.

| | `Rag_web_pipeline` | `Rag_software_stack` (GRAFT Studio) |
|---|---|---|
| Path | `D:\Mini Thesis\Rag_web_pipeline` | `D:\Mini Thesis\Rag_software_stack` |
| What it is | Always-on web app, fixed 19-stage pipeline | Drag-and-drop block/node graph builder |
| Raw PDFs → report in one go? | **Yes** — drag a folder onto the UI | **No** — canvas starts from already-chunked/extracted docs |
| Backend port | 8017 | 8100 |
| Frontend port | 5183 | 5190 |
| Needs Rust? | No | Only for the native desktop window |
| Use it when | You want ground truth out of a folder of PDFs | You want to inspect or rewire individual pipeline steps |

**If you just want ground truth from a folder of PDFs: use `Rag_web_pipeline`.**

---

## 1. One-time setup (per repo, identical)

Requires Python 3.11+ and Node.js 18+.

```bash
cd "D:\Mini Thesis\Rag_web_pipeline"     # or Rag_software_stack
python -m venv venv
venv\Scripts\activate                     # Windows; macOS/Linux: source venv/bin/activate
pip install -e .                          # installs the rag_gt engine package
```

Then the backend requirements — note the path differs per repo:

```bash
pip install -r production/backend/requirements.txt   # Rag_web_pipeline
pip install -r studio/backend/requirements.txt       # Rag_software_stack
```

### LLM credentials

Only needed to generate ground truth on your own PDFs. Copy `.env.example` to
`.env` in the repo root and fill in `API_BASE_URL`, `API_KEY`, `API_GT_MODEL`,
`API_ANSWER_MODEL`.

> **Gotcha:** if `RAG_LLM_CHAT_MODEL` is set, it overrides *every* role and
> `API_GT_MODEL`/`API_ANSWER_MODEL` are ignored. Leave it blank unless you
> want one model for everything. (This is intentional — see `get_llm()` in
> `src/rag_gt/core/llm.py`.)

Current setup uses Nebius: `Qwen/Qwen3-235B-A22B-Instruct-2507` for chat
(non-thinking, avoids a reasoning-leak issue this project hit before) and
`Qwen/Qwen3-Embedding-8B` for embeddings. The RWTH config is preserved
commented-out in both `.env` files.

---

## 2. Running `Rag_web_pipeline` (the one-click one)

Two terminals, both from the repo root with the venv active.

```bash
# Terminal 1 — backend
cd production/backend
python serve.py
# → http://127.0.0.1:8017

# Terminal 2 — frontend
cd production/frontend
npm install        # first time only
npm run dev
# → open http://localhost:5183
```

In the browser: pick **pipeline: graft**, drag your whole folder of PDFs onto
the upload area, hit **Run**. The graph lights up stage by stage (profile →
ingest → chunk → extract facts → fact graph → sample QA → index → evaluate →
report), and you get a self-contained `report.html` at the end.

**Try it with zero API cost first:** the repo ships a small pre-generated
dataset at `data/eval_results/allpdf_qa_pairs_verified_bbox.json`. Use
"import" mode to run the whole app against it without spending anything.

### Batch / scripted runs

```bash
curl -X POST http://127.0.0.1:8017/api/runs \
  -F "pipeline=graft" \
  -F 'config={"llm_mode":"live","max_cost_usd":5.0}' \
  -F files=@my_pdfs/doc1.pdf -F files=@my_pdfs/doc2.pdf
```

Always set `max_cost_usd` on live runs — it's the cost circuit-breaker.

---

## 3. Running `Rag_software_stack` (GRAFT Studio)

### Browser

```bash
# Terminal 1 — backend, from repo root
uvicorn studio.backend.api:app --reload --port 8100

# Terminal 2 — frontend
cd studio/core-ui
npm install        # first time only
npm run dev
# → open http://localhost:5190
```

### Desktop (native window)

```bash
cd studio/desktop
npm install        # first time only
npm run dev:desktop
```

This auto-spawns the Python backend — no separate uvicorn terminal. Needs the
Rust toolchain.

### The honest gap: raw PDFs

The `pdf_source` block (raw PDF → chunks+facts) is **still a stub**. Only some
of the 33 block types are wired to the real engine; the rest are marked
`(planned)` in the UI. So the canvas starts from already-chunked, already
fact-extracted documents.

To get PDFs into it, run the engine CLI first, then import the output:

```bash
rag-gt-generate --input_dir my_pdfs/ --output my_pdfs_gt.jsonl
```

That's **one command per document set**, not one click for a batch. If you
want raw-PDF-in/report-out with no wiring, use `Rag_web_pipeline` instead.

---

## 4. Running the tests

From each repo root with the venv active:

```bash
python -m pytest tests/ -q
```

Expected as of 2026-08-01:

| Repo | Result |
|---|---|
| `Rag_web_pipeline` | 768 passed, 4 skipped |
| `Rag_software_stack` | 846 passed, 4 skipped |
| `RAG_GT` (monorepo) | 540 passed, 1 skipped |

Anything less means something regressed — don't ignore it.

Frontend tests: `npm test` inside `production/frontend` or `studio/core-ui`.

---

## 5. Working on the code

- The **engine is vendored** into both repos at `src/rag_gt/`. It is *not* a
  shared install. A fix in one repo does **not** propagate to the other — it
  has to be ported by hand, and the two copies have already drifted slightly
  (e.g. `pipeline.py` in `Rag_software_stack` has `extract_only` /
  `extract_workers` args that `Rag_web_pipeline` lacks). Always diff before
  copying a file across.
- The monorepo `D:\Mini Thesis\RAG_GT` is the original lineage. Its `main`
  branch is an unrelated history; the real trunk is `v9 → v16 →
  multihop-bridge`.
- Work on a branch, never directly on `main`. For isolated work:
  `git worktree add .worktrees/<name> -b <branch>`.
- Nothing gets pushed to GitHub without an explicit per-instance request.

### Known gotcha: worktrees and Python imports

`python -m rag_gt` inside a worktree runs the **main repo's** editable
install, not the worktree's code. Set `PYTHONPATH` to the worktree's `src/`
and verify with `rag_gt.__file__` before trusting any result.

---

## 6. Current state of the fixes (2026-08-01)

Three ingestion bugs were found and fixed today. All are merged into each
repo's **local `main`** and verified. **Nothing has been pushed** — `origin/main`
is still at the pre-fix commit in both repos.

| Repo | local `main` | `origin/main` |
|---|---|---|
| `Rag_web_pipeline` | `8df38e8` | `9684c40` (behind) |
| `Rag_software_stack` | `3184094` | `b1fa245` (behind) |
| `RAG_GT` | `e04de22` on `hop-redesign-singlemulti-20260622` | n/a |

To publish, when you want to: `git push origin main` in each repo.
To undo instead: `git reset --hard <origin/main sha>`.

### What was fixed

1. **`docling_page_cap` truncation.** Three orchestrators defaulted the cap to
   8 while the engine's own default was 40, and the over-cap path sliced the
   PDF to pages 1–8 and *still ran Docling on the slice* instead of falling
   back to legacy extraction as its docstring promised. DIN EN ISO 13919-1
   ingested only its first 8 pages — all front matter. Now falls back to
   full-document legacy extraction, and defaults are 60.

2. **Silent Docling page drops.** Docling swallows `std::bad_alloc` from its
   own preprocessor and returns a *partial* document — `convert()` reports
   success, so the `except` never fires and pages vanish with no signal.
   Pages missing from a converted range are now re-extracted via PyMuPDF and
   spliced back in page order.

3. **Stale PDF slice cache.** `_slice_pdf` keyed its cache on
   `path:start:end` with nothing about file contents, so replacing a PDF at
   the same path served the *previous* document's pages. Key now includes
   size and mtime.

Net effect on DIN EN ISO 13919-1: **8/24 pages → 24/24**, and 12 chunks of
pure front matter → 27 `table_aware` chunks covering the actual technical
clauses and imperfection tables.

A new stage-1 gate (`pages_covered >= 90% of page_count`) now fails the
pipeline loudly if this class of silent truncation ever returns.

---

## 7. Known open items

- `pdf_source` block in GRAFT Studio is still a stub — this is the only thing
  standing between it and the same one-click batch experience the web
  pipeline has.
- `cargo test` fails out of the box in `Rag_software_stack`: Tauri's
  `build.rs` validates a gitignored `python-runtime` path even for tests. CI
  works around it with a placeholder dir; `main` is untouched.
- Meta-key naming mismatch (`n_qa`/`n_multihop` vs `count`/`multi_hop`) makes
  QA-gen nodes show "? QA" in the canvas. Cosmetic, data underneath is fine.
- `.env.example` in `Rag_web_pipeline` still lists Ollama vars that a code
  comment says were removed.
- `_slice_pdf` writes to the relative path `data/cache/allpdf_slices`, so the
  cache lands wherever you launched from. Run CLI commands from the repo root.
