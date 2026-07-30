# TODO / Known Gaps — Rag_web_pipeline

Audit date: 2026-07-30. Every number below was produced by actually running
the suite on this date, not recalled from memory — re-run the commands
yourself before trusting a count that looks stale.

---

## 1. Test status (verified 2026-07-30)

| Suite | Command | Result |
|---|---|---|
| Backend (`production/backend`) | `cd production/backend && python -m pytest -q` | **48 passed**, 55.3s |
| Frontend (`production/frontend`) | `cd production/frontend && npx vitest run` | **20 passed**, 4 files, 2.7s |
| Vendored engine (`src/rag_gt`) | — | **NO TESTS SHIPPED** — see §2, item P0-1 |

No `.github/workflows/` exist anywhere in this repo — there is **no CI**.
Every one of the above numbers is a manual local run; nothing enforces them
on push or PR.

## 2. P0 — fix before trusting this repo for anything real

1. **The vendored `src/rag_gt` engine has zero test coverage in this repo.**
   The engine's actual test suite (98 files, 742 passed / 3 skipped / 1
   deselected as of 2026-07-30 on the source branch) was deliberately left
   out of the export to keep the repo small — but that means nobody can
   verify the vendored copy still behaves correctly after any edit. Action:
   either (a) pull the `tests/` directory over from
   `github.com/Sushruths04/RAG_ground_truth` (branch `hop-evidence-v2`) and
   wire it into `pyproject.toml`'s `[project.optional-dependencies] test`
   group, or (b) write a smaller smoke suite scoped to just the modules
   `production/backend` actually imports (`rag_gt.allpdf.*`, `rag_gt.core.llm`,
   `rag_gt.rag.retriever`) if the full suite is too heavy to carry.
2. **No CI.** Add a GitHub Actions workflow that runs both test commands in
   §1 on every push/PR. Trivial to add, currently completely absent.
3. **`production/backend/requirements.txt` and the root `pyproject.toml`
   have never been pinned or dependency-audited as a pair** — install them
   both into a fresh venv on a clean machine (not just this dev box) and
   confirm there's no version conflict before calling this "installable."

## 3. P1 — real correctness/completeness gaps in the app itself

4. **The live-LLM acceptance run (`llm_mode=live`) has never actually been
   executed and verified end-to-end on this exact exported code.** The
   number quoted in the README (492 pairs, recall 0.896 / precision 0.491 /
   F1 0.635) is from the `import`-mode acceptance run on the source branch,
   before export — it has NOT been re-run against real API spend on this
   standalone repo. Before telling anyone "live mode works," run one small
   real live-mode job end to end (1-2 small PDFs, low `max_cost_usd`) and
   confirm the whole chain (extraction → generation → gates → verification
   → index → eval → report) actually completes and produces sane output.
5. **`RAG_LLM_CHAT_MODEL` silently overrides every role-specific model
   override** (`API_GT_MODEL`/`API_ANSWER_MODEL` in `.env.example` look like
   independent knobs but aren't, once `RAG_LLM_CHAT_MODEL` is set — see
   `src/rag_gt/core/llm.py`'s `get_llm()`). This is confusing and
   undocumented in the README; either fix the precedence so per-role models
   actually take effect, or update `.env.example`'s comments to say clearly
   that `RAG_LLM_CHAT_MODEL` wins over everything else.
6. **No rate/concurrency tuning guidance for a fresh LLM endpoint.** The
   engine defaults to 8 concurrent calls for API backends; nothing in this
   repo tells a new user how to find out if their own endpoint can handle
   that, or what error they'll see if it can't (timeouts, not outright
   failures, per prior engine-team experience — see the note in
   `src/rag_gt/core/llm.py` if present, or just test it).
7. **Cost estimation quality:** `price_in_per_mtok`/`price_out_per_mtok`
   default to `0.0` in `production/backend/app/stages/graft.py`'s
   `CostBudget` construction path — if a user doesn't set these explicitly
   in `config`, every live run "looks free" in the cost figures even though
   real money is being spent against `max_cost_usd`. Consider defaulting to
   a documented placeholder price or forcing the field to be set before a
   live run is allowed to start.

## 4. P2 — polish / nice-to-have

8. **Frontend test coverage is thin** (20 tests / 4 files for a full React
   Flow pipeline-visualization app). No tests were found covering the
   WebSocket live-update path (`WS /ws/runs/{id}`), the Traces tab, or the
   Report tab rendering — these are the most complex/stateful parts of the
   UI and currently have the least coverage.
9. **No Dockerfile / containerized dev setup.** Everything assumes a local
   Python venv + local Node install on the developer's machine. A
   `Dockerfile`/`docker-compose.yml` would make "try this on a machine that
   isn't mine" much less friction (matches the repo's own pitch of being a
   portable package).
10. **No `LICENSE` file.** Not a code defect, but worth deciding and adding
    before treating this as a distributable package — ask the repo owner
    which license they want (MIT, Apache-2.0, proprietary, etc.) rather than
    guessing.

## 5. What is genuinely solid (don't waste review time re-litigating)

- The 19-stage `graft` pipeline in `production/backend/app/stages/graft.py`
  wraps real, tested engine functions end to end for `import` mode — this
  path is exercised by the 48 passing backend tests and was the basis of
  the 492-pair acceptance run.
- `import` vs `live` mode separation, and the `CostBudget` hard-stop on
  `max_cost_usd`, are real and tested (`CostBudget.__init__`/`check` in
  `graft.py`), not just documented aspirations.
- Per-LLM-call span recording (prompt/response/tokens/cost) for the Traces
  tab is real, not simulated, when `llm_mode=live`.

## 6. Priority order for a fixing agent

1. P0-1 (ship engine tests or a scoped smoke suite) — without this, nobody
   can trust changes to the vendored engine copy.
2. P0-2 (CI) — cheap, and turns every future PR into a real gate.
3. P1-4 (one real live-mode run) — the single biggest "does this actually
   work as advertised" unknown.
4. P1-5 (model precedence) — small fix, currently actively misleading.
5. Everything else, in any order.
