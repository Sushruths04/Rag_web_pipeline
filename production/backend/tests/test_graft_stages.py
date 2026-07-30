"""Integration tests for the real GRAFT stage wrappers (no LLM, no cost).

Uses the small ecma404 PDF with the rule-based extraction path. Skipped when
the test PDF is not present.
"""
import shutil
from pathlib import Path

import pytest

from app.db.store import Store
from app.orchestrator.events import EventBus
from app.orchestrator.manager import RunManager

WORKTREE = Path(__file__).resolve().parents[3]
PDF = WORKTREE / "data" / "test_corpus_allpdf" / "ecma404_json.pdf"

pytestmark = pytest.mark.skipif(not PDF.exists(), reason=f"test PDF missing: {PDF}")


@pytest.fixture()
def mgr(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite")
    m = RunManager(store, EventBus(), tmp_path / "runs", mode="thread")
    yield m
    store.close()


def _start_with_pdf(mgr: RunManager, pipeline: str, config: dict) -> str:
    run_id = mgr.prepare_run(config, pipeline=pipeline)
    shutil.copy(PDF, mgr.data_dir / run_id / "uploads" / PDF.name)
    mgr.launch_run(run_id)
    return run_id


def test_graft_frontend_stages_on_real_pdf(mgr: RunManager):
    run_id = _start_with_pdf(mgr, "graft_p3", {"docling_page_cap": 4})
    mgr.wait(run_id, timeout=300)

    run = mgr.store.get_run(run_id)
    assert run["status"] == "completed", run
    stages = {s["name"]: s for s in run["stages"]}
    assert stages["profile"]["status"] == "completed"
    assert stages["ingest"]["status"] == "completed"
    assert stages["chunk"]["status"] == "completed"

    metrics = {m["name"]: m["value"] for m in mgr.store.list_metrics(run_id)}
    assert metrics["docs"] == 1
    assert metrics["total_pages"] > 0
    assert metrics["units"] > 0
    assert metrics["chunks"] > 0

    # chunks artifact is the RAG-track contract
    arts = {a["name"]: a for a in mgr.store.list_artifacts(run_id)}
    assert "chunks_ecma404_json.json" in arts
    assert arts["chunks_ecma404_json.json"]["size_bytes"] > 0

    # per-doc processing spans with outputs
    spans = mgr.store.list_spans(run_id, type="processing")
    names = {s["name"] for s in spans}
    assert {"profile_ecma404_json", "ingest_ecma404_json", "chunk_ecma404_json"} <= names
    prof_span = next(s for s in spans if s["name"] == "profile_ecma404_json")
    assert prof_span["output"]["pages"] > 0

    # resumable state persisted
    assert (mgr.data_dir / run_id / "state.pkl").exists()


def test_graft_extract_and_clean_rule_based(mgr: RunManager):
    run_id = _start_with_pdf(mgr, "graft_p3", {"docling_page_cap": 4, "llm_mode": "import"})
    mgr.wait(run_id, timeout=300)

    run = mgr.store.get_run(run_id)
    assert run["status"] == "completed", run
    stages = {s["name"]: s for s in run["stages"]}
    assert stages["extract"]["status"] == "completed"
    assert stages["clean"]["status"] == "completed"

    metrics = {m["name"]: m["value"] for m in mgr.store.list_metrics(run_id)}
    assert metrics["facts_extracted"] > 0
    assert metrics["facts_kept"] > 0
    assert metrics["facts_kept"] <= metrics["facts_extracted"]

    # provenance artifact with page-grounded facts
    arts = {a["name"]: a for a in mgr.store.list_artifacts(run_id)}
    art = arts["facts_ecma404_json.json"]
    import json as _json
    facts = _json.loads(Path(art["path"]).read_text(encoding="utf-8"))
    assert len(facts) == metrics["facts_kept"]
    assert all("fact_id" in f and "text" in f for f in facts)
    assert any(f.get("page_start") is not None for f in facts)

    # clean stage logged per-reason drops
    drop_logs = mgr.store.events_after(run_id, stage="clean", type="stage_log")
    assert any("dropped" in str(e["payload"].get("line", "")) for e in drop_logs)


def test_traced_llm_spans_and_budget(tmp_path):
    import threading

    from app.orchestrator.context import StageContext
    from app.stages.graft import BudgetExceeded, CostBudget, TracedLLM

    class FakeLLM:
        def generate(self, prompt, temperature=0.0, max_tokens=512):
            return "answer " * 100  # ~700 chars

    messages = []
    ctx = StageContext(
        "r1", "qagen", tmp_path,
        {"price_in_per_mtok": 0.5, "price_out_per_mtok": 0.5},
        messages.append, threading.Event(),
    )
    budget = CostBudget(max_cost_usd=0.0005)
    llm = TracedLLM(FakeLLM(), ctx, "gt", budget)

    out = llm.generate("what is X? " * 50)  # ~550 chars prompt
    assert out.startswith("answer")
    spans = [m["span"] for m in messages if m["kind"] == "span"]
    assert len(spans) == 1
    sp = spans[0]
    assert sp["type"] == "llm" and sp["name"] == "gt.generate"
    assert sp["tokens_in"] > 0 and sp["tokens_out"] > 0
    assert sp["cost_usd"] > 0
    assert budget.spent == sp["cost_usd"]

    # second call blows the tiny budget
    with pytest.raises(BudgetExceeded):
        for _ in range(10):
            llm.generate("more " * 200)


def test_reranker_orders_by_cross_encoder_score():
    from app.ragengine.rerank import Reranker

    class FakeCE:
        def predict(self, pairs):
            # score by chunk text length — deterministic, distinguishable
            return [len(text) for _, text in pairs]

    texts = {"c1": "short", "c2": "a much longer chunk text here", "c3": "medium text"}
    rr = Reranker(FakeCE())
    out = rr.rerank("q", [("c1", 0.9), ("c2", 0.1), ("c3", 0.5)], texts.__getitem__, top_k=2)
    assert [cid for cid, _ in out] == ["c2", "c3"]
    assert out[0][1] > out[1][1]
    assert rr.rerank("q", [], texts.__getitem__) == []


def test_full_graft_import_mode_end_to_end(mgr: RunManager):
    run_id = _start_with_pdf(
        mgr, "graft",
        {
            "docling_page_cap": 4,
            "llm_mode": "import",
            # keep the CPU cost of the test bounded; rerank mode is covered by
            # its unit test and the manual smoke
            "sweep_modes": ["bm25", "vector", "hybrid"],
            "sweep_sample": 40,
            "eval_span_cap": 25,
        },
    )
    mgr.wait(run_id, timeout=600)

    run = mgr.store.get_run(run_id)
    assert run["status"] == "completed", run
    stages = {s["name"]: s for s in run["stages"]}
    assert len(stages) == 19
    assert all(s["status"] == "completed" for s in stages.values())

    # GT artifact: imported pairs filtered to the uploaded doc (ecma404)
    arts = {a["name"]: a for a in mgr.store.list_artifacts(run_id)}
    gt = Path(arts["gt.jsonl"]["path"])
    lines = gt.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 10
    import json as _json
    pairs = [_json.loads(l) for l in lines]
    assert all(p["doc"].startswith("ecma404") for p in pairs)
    assert "question" in pairs[0] and "gold_chunk_ids" in pairs[0]

    metrics = {m["name"]: m["value"] for m in mgr.store.list_metrics(run_id)}
    assert metrics["pairs_kept"] == len(lines)

    # auto-tune leaderboard: 3 modes x 3 top_ks, exactly one winner
    lb = _json.loads(Path(arts["leaderboard.json"]["path"]).read_text(encoding="utf-8"))
    assert len(lb["rows"]) == 9
    winners = [r for r in lb["rows"] if r["winner"]]
    assert len(winners) == 1
    assert all(0.0 <= r["recall"] <= 1.0 for r in lb["rows"])

    # final eval ran at the winner config over ALL pairs with real numbers
    ev = _json.loads(Path(arts["eval_results.json"]["path"]).read_text(encoding="utf-8"))
    assert ev["winner"]["config"] == winners[0]["config"]
    assert ev["aggregate"]["n"] == len(lines)
    assert ev["aggregate"]["recall"] > 0.3  # sanity: retrieval finds evidence
    assert len(ev["rows"]) == len(lines)

    # retrieval spans with per-candidate verdicts (Opik drill-down)
    rspans = mgr.store.list_spans(run_id, type="retrieval")
    assert rspans
    cand = rspans[0]["output"]["candidates"][0]
    assert {"chunk_id", "score", "relevant", "text"} <= set(cand)
    assert metrics["final_f1"] > 0

    # self-contained HTML report
    report = Path(arts["report.html"]["path"]).read_text(encoding="utf-8")
    assert "Auto-tune leaderboard" in report
    assert "without an LLM judge" in report
    assert pairs[0]["question"][:40] in report


def test_graft_fails_cleanly_without_uploads(mgr: RunManager):
    run_id = mgr.start_run({}, pipeline="graft_p3")
    mgr.wait(run_id, timeout=60)
    run = mgr.store.get_run(run_id)
    assert run["status"] == "failed"
    stages = {s["name"]: s for s in run["stages"]}
    assert "no PDFs uploaded" in stages["profile"]["error"]
    assert stages["ingest"]["status"] == "skipped"
