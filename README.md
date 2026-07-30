# GRAFT — RAG Ground-Truth Web Pipeline

Upload a folder of PDFs → grounded ground-truth QA generation → retrieval
index build → LLM-free evaluation + auto-tune → an HTML report, all as one
19-stage pipeline you can watch run live as a graph in the browser.

This is a **web app**: a FastAPI backend that runs the pipeline and a React
(React Flow) frontend that visualizes it stage by stage — uploaded PDFs,
extracted facts, sampled QA pairs, retrieval indexes, evaluation scores, and
a final self-contained report, all with source/page/bbox provenance you can
click through to.

If you're looking for the desktop / drag-and-drop block-builder version of
this same pipeline, that's a separate repo:
[`Rag_software_stack`](https://github.com/Sushruths04/Rag_software_stack).
This repo is the always-on web app; that one is a visual pipeline *builder*.

---

## 1. What's in this repo

```
production/
  backend/    FastAPI app — orchestrates the pipeline, serves the REST + WS API
  frontend/   React + React Flow UI — upload PDFs, watch the graph run live
src/rag_gt/   the ground-truth generation engine itself (vendored, not a
              separate install — extraction, QA generation, retrieval,
              evaluation; the backend imports this directly)
data/eval_results/allpdf_qa_pairs_verified_bbox.json
              a small pre-generated GT dataset, used by the free "import"
              mode described below so you can try the whole app with zero
              API cost before pointing it at your own PDFs
pyproject.toml
              installs the `rag_gt` engine package (`pip install -e .`)
```

## 2. Requirements

- Python 3.11+
- Node.js 18+ (for the frontend)
- An OpenAI-compatible LLM API endpoint, **only if** you want to generate
  ground truth on your *own* PDFs (see "import vs. live mode" below). Trying
  the app on the bundled sample data needs nothing but Python and Node.

## 3. Install

```bash
# from the repo root
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e .                # installs the rag_gt engine package
pip install -r production/backend/requirements.txt

cd production/frontend
npm install
cd ../..
```

If you want to generate ground truth on your own PDFs (live mode), copy
`.env.example` to `.env` in the repo root and fill in your LLM endpoint:

```bash
cp .env.example .env
# edit .env: API_BASE_URL, API_KEY, API_GT_MODEL, API_ANSWER_MODEL (or the
# Ollama equivalents if you're running a local model)
```

## 4. Run it

Two processes, in two terminals.

**Backend:**

```bash
cd production/backend
python serve.py
# → Uvicorn running on http://127.0.0.1:8017
```

**Frontend:**

```bash
cd production/frontend
npm run dev
# → open http://localhost:5183
```

## 5. Using it on your own folder of PDFs

This is the main thing you'll actually want to do: point the pipeline at a
folder of documents and get grounded QA pairs, a retrieval index, and an
evaluation report out the other end.

**Option A — through the UI (easiest).** Open http://localhost:5183, pick
**pipeline: graft**, and drag your whole folder of PDFs onto the upload
area (or use the file picker and multi-select every PDF in the folder — the
browser lets you select a whole directory's worth of files at once). Hit
**Run**. You'll see the graph light up stage by stage: profile → ingest →
chunk → extract facts → build the fact graph → sample QA pairs → build
retrieval indexes → evaluate → report.

**Option B — via the API**, e.g. from a script or CI, pointing at a local
folder called `my_pdfs/`:

```bash
curl -X POST http://127.0.0.1:8017/api/runs \
  -F "pipeline=graft" \
  -F 'config={"llm_mode":"live","max_cost_usd":5.0}' \
  $(for f in my_pdfs/*.pdf; do printf ' -F files=@%q' "$f"; done)
```

or with Python:

```python
import requests
from pathlib import Path

pdf_dir = Path("my_pdfs")
files = [("files", (p.name, open(p, "rb"), "application/pdf")) for p in pdf_dir.glob("*.pdf")]
resp = requests.post(
    "http://127.0.0.1:8017/api/runs",
    data={"pipeline": "graft", "config": '{"llm_mode":"live","max_cost_usd":5.0}'},
    files=files,
)
run_id = resp.json()["run_id"]
print(run_id)
```

Then poll `GET /api/runs/{run_id}` (or watch `WS /ws/runs/{run_id}`) until
`status` is `completed`, and pull the results from
`GET /api/runs/{run_id}/artifacts` — this includes the generated GT dataset,
the retrieval leaderboard, and the final `report.html`.

### import mode vs. live mode

Every run's `config` sets `llm_mode`:

- **`"import"` (default, free)** — runs the real pipeline on your PDFs
  (profiling, ingestion, chunking, rule-based fact extraction, retrieval
  index build, evaluation) but instead of *generating* new QA pairs with an
  LLM, it pulls them from the bundled `data/eval_results/allpdf_qa_pairs_verified_bbox.json`,
  filtered down to whichever of your uploaded PDFs it has pre-generated
  pairs for. This is how you try the whole app, including a real evaluation
  report, without spending anything.
- **`"live"` (paid)** — actually generates ground truth on your PDFs with
  the LLM configured in `.env`. Every LLM call is recorded as a span
  (prompt, response, tokens, cost) and the run stops cleanly once total
  spend crosses `config.max_cost_usd` (default $5). Set
  `price_in_per_mtok` / `price_out_per_mtok` in `config` if you want the
  cost figures to reflect your endpoint's real pricing.

If you upload PDFs that aren't in the bundled sample set and run in
`import` mode, the run will fail cleanly with "no imported GT pairs match
the uploaded PDFs" — that's your signal to switch to `llm_mode=live`.

## 6. What you get out

Each run's artifacts (`GET /api/runs/{run_id}/artifacts`, or the **Report**
tab in the UI) include:

- the generated ground-truth QA pairs, each with source PDF / page / bbox
  provenance back to the exact text it came from,
- BM25 + dense + hybrid + hybrid-with-reranker retrieval indexes over your
  chunked PDFs,
- a leaderboard sweeping retrieval configs against the GT with LLM-free
  token-containment matching, auto-selecting the best (recall/precision/F1),
- a self-contained `report.html` with per-question drill-down you can hand
  to anyone without needing this app running.

The **Traces** tab shows every LLM call as an Opik-style span (prompt,
response, tokens, cost) when running in live mode.

## 7. Reference run (import mode, zero cost)

6 PDFs, 560 pages → 3,008 chunks → 8,496 facts kept → 492 GT pairs, all 19
stages in ~9 minutes on CPU. Winning retrieval config: hybrid+rerank@3 —
recall 0.896, precision 0.491, F1 0.635.

## 8. Troubleshooting

- **Frontend loads but nothing happens when you hit Run** — the backend
  isn't running, or isn't on port 8017. Start it first (§4).
- **`ModuleNotFoundError: rag_gt`** — run `pip install -e .` from the repo
  root (not from inside `production/backend/`).
- **Run fails with "no imported GT pairs match the uploaded PDFs"** — you're
  in `import` mode on PDFs that aren't in the bundled sample set; switch to
  `"llm_mode":"live"` and configure `.env`.
- **Live mode run stops early with a budget error** — that's
  `max_cost_usd` doing its job; raise it in `config` if you intended to
  spend more.
