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
    budget = CostBudget(max_cost_usd=0.0005, pricing_known=True)
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


def test_unpriced_run_reports_unknown_cost_rather_than_zero(tmp_path, monkeypatch):
    """A live run with unresolvable prices must not render as a free run.

    Replaces the old "must raise LivePricingRequired" pin. The guarantee being
    protected is unchanged -- an unpriced run may never look like $0.00 -- but
    it is now delivered by reporting cost as UNKNOWN and falling back to a
    token cap, instead of refusing to start and demanding two figures the user
    has no way to know.
    """
    import threading

    from app.orchestrator.context import StageContext
    from app.stages.graft import CostBudget, TracedLLM

    monkeypatch.setenv("API_BASE_URL", "https://unknown-provider.example.com/v1")
    monkeypatch.setenv("RAG_LLM_CHAT_MODEL", "some/unlisted-model")

    class FakeLLM:
        def generate(self, prompt, temperature=0.0, max_tokens=512):
            return "answer"

    ctx = StageContext(
        "r1", "qagen", tmp_path,
        {"llm_mode": "live"},  # no price_in_per_mtok / price_out_per_mtok
        lambda *a, **k: None, threading.Event(),
    )
    budget = CostBudget(max_cost_usd=5.0, max_tokens_total=1_000, pricing_known=False)
    llm = TracedLLM(FakeLLM(), ctx, "gt", budget)  # must NOT raise

    llm.generate("hello")

    assert budget.pricing_known is False
    assert budget.tokens > 0, "tokens must still be counted when price is unknown"


def test_unpriced_run_is_still_bounded_by_a_token_cap(tmp_path, monkeypatch):
    """Without prices the dollar cap cannot bite, so a token cap must."""
    import threading

    from app.orchestrator.context import StageContext
    from app.stages.graft import BudgetExceeded, CostBudget, TracedLLM

    monkeypatch.setenv("API_BASE_URL", "https://unknown-provider.example.com/v1")
    monkeypatch.setenv("RAG_LLM_CHAT_MODEL", "some/unlisted-model")

    class FakeLLM:
        def generate(self, prompt, temperature=0.0, max_tokens=512):
            return "answer"

    ctx = StageContext(
        "r1", "qagen", tmp_path, {"llm_mode": "live"},
        lambda *a, **k: None, threading.Event(),
    )
    budget = CostBudget(max_cost_usd=5.0, max_tokens_total=100, pricing_known=False)
    llm = TracedLLM(FakeLLM(), ctx, "gt", budget)

    with pytest.raises(BudgetExceeded, match="token budget exceeded"):
        for _ in range(20):
            llm.generate("more " * 200)


def test_cost_tracking_is_off_by_default():
    """The provider meters spend; a local estimate is opt-in, not the default.

    See docs/COST_TRACKING_IS_OPTIONAL.md.
    """
    from app.stages.pricing import cost_tracking_enabled, resolve_pricing

    assert cost_tracking_enabled({}) is False
    p = resolve_pricing(
        {},
        base_url="https://api.tokenfactory.nebius.com/v1",
        model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    )
    assert p.source == "off"
    assert p.is_known is False, "an untracked run must not report a dollar figure"


def test_supplying_a_price_is_itself_an_opt_in():
    from app.stages.pricing import cost_tracking_enabled

    assert cost_tracking_enabled({"price_in_per_mtok": 0.2}) is True


def test_opted_in_run_resolves_provider_prices_without_user_input():
    """Once opted in, the user still should not have to type rates."""
    from app.stages.pricing import resolve_pricing

    p = resolve_pricing(
        {"cost_tracking": "estimate"},
        base_url="https://api.tokenfactory.nebius.com/v1",
        model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    )
    assert p.source == "provider"
    assert p.is_known is True
    assert p.price_in_per_mtok > 0 and p.price_out_per_mtok > 0


def test_explicit_prices_override_the_provider_table():
    from app.stages.pricing import resolve_pricing

    p = resolve_pricing(
        {"price_in_per_mtok": 1.23, "price_out_per_mtok": 4.56},
        base_url="https://api.tokenfactory.nebius.com/v1",
        model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    )
    assert p.source == "explicit"
    assert (p.price_in_per_mtok, p.price_out_per_mtok) == (1.23, 4.56)


def test_unknown_provider_when_opted_in_is_unknown_not_free():
    from app.stages.pricing import resolve_pricing

    p = resolve_pricing(
        {"cost_tracking": "estimate"},
        base_url="https://mystery.example.org/v1",
        model="x",
    )
    assert p.source == "unknown"
    assert p.is_known is False


def test_self_hosted_zero_price_is_known_when_opted_in():
    """0.0 for an unmetered institutional endpoint is a fact, not a gap."""
    from app.stages.pricing import resolve_pricing

    p = resolve_pricing(
        {"cost_tracking": "estimate"},
        base_url="https://llm.hpc.itc.rwth-aachen.de/v1",
        model="x",
    )
    assert p.source == "provider"
    assert p.is_known is True
    assert (p.price_in_per_mtok, p.price_out_per_mtok) == (0.0, 0.0)


def test_real_provider_usage_is_preferred_over_the_char_estimate(tmp_path):
    """APILLM now records the API's own token counts; TracedLLM must use them."""
    import threading

    from app.orchestrator.context import StageContext
    from app.stages.graft import CostBudget, TracedLLM

    class UsageLLM:
        """Stands in for APILLM after it records a real usage block."""
        last_usage = {"prompt_tokens": 1234, "completion_tokens": 77,
                      "total_tokens": 1311}

        def generate(self, prompt, temperature=0.0, max_tokens=512):
            return "x"  # 1 char -> char-estimate would give 0 output tokens

    messages = []
    ctx = StageContext(
        "r1", "qagen", tmp_path, {}, messages.append, threading.Event(),
    )
    budget = CostBudget()
    TracedLLM(UsageLLM(), ctx, "gt", budget).generate("hi")  # 2 chars

    sp = [m["span"] for m in messages if m["kind"] == "span"][0]
    assert sp["tokens_in"] == 1234, "provider's prompt_tokens must win"
    assert sp["tokens_out"] == 77
    assert budget.tokens == 1311
    assert budget.tokens_are_estimated is False


def test_char_estimate_is_used_only_when_the_endpoint_omits_usage(tmp_path):
    import threading

    from app.orchestrator.context import StageContext
    from app.stages.graft import CostBudget, TracedLLM

    class NoUsageLLM:
        last_usage = None

        def generate(self, prompt, temperature=0.0, max_tokens=512):
            return "answer " * 100

    messages = []
    ctx = StageContext(
        "r1", "qagen", tmp_path, {}, messages.append, threading.Event(),
    )
    budget = CostBudget()
    TracedLLM(NoUsageLLM(), ctx, "gt", budget).generate("what is X? " * 50)

    assert budget.tokens > 0
    assert budget.tokens_are_estimated is True, "fallback must be flagged"


def test_untracked_run_reports_no_dollar_metric(tmp_path):
    """An untracked run must not emit llm_cost_usd=0.0."""
    import threading

    from app.orchestrator.context import StageContext
    from app.stages.graft import CostBudget, _report_budget

    messages = []
    ctx = StageContext(
        "r1", "qagen", tmp_path, {}, messages.append, threading.Event(),
    )
    budget = CostBudget()
    budget.add(0.0, 500)
    _report_budget(ctx, budget)

    names = {
        m["event"]["payload"]["name"]
        for m in messages
        if m["kind"] == "event" and m["event"]["type"] == "stage_metric"
    }
    assert "llm_tokens" in names
    assert not any("cost" in n for n in names), (
        "a run with cost tracking off must report no dollar figure at all"
    )


def test_traced_llm_live_mode_accepts_explicit_zero_price(tmp_path):
    """Explicit 0.0 (genuinely free endpoint) is a deliberate choice, not a default."""
    import threading

    from app.orchestrator.context import StageContext
    from app.stages.graft import CostBudget, TracedLLM

    class FakeLLM:
        def generate(self, prompt, temperature=0.0, max_tokens=512):
            return "answer"

    ctx = StageContext(
        "r1", "qagen", tmp_path,
        {"llm_mode": "live", "price_in_per_mtok": 0.0, "price_out_per_mtok": 0.0},
        lambda *a, **k: None, threading.Event(),
    )
    budget = CostBudget(max_cost_usd=5.0)
    llm = TracedLLM(FakeLLM(), ctx, "gt", budget)

    llm.generate("hello")
    assert budget.spent == 0.0


def test_construction_without_prices_never_blocks(tmp_path, monkeypatch):
    """Direct construction with an empty config must not raise for any mode."""
    import threading

    from app.orchestrator.context import StageContext
    from app.stages.graft import CostBudget, TracedLLM

    monkeypatch.setenv("API_BASE_URL", "https://unknown-provider.example.com/v1")
    monkeypatch.setenv("RAG_LLM_CHAT_MODEL", "some/unlisted-model")

    class FakeLLM:
        def generate(self, prompt, temperature=0.0, max_tokens=512):
            return "answer"

    for config in ({}, {"llm_mode": "import"}, {"llm_mode": "live"}):
        ctx = StageContext(
            "r1", "qagen", tmp_path, config,
            lambda *a, **k: None, threading.Event(),
        )
        budget = CostBudget(max_cost_usd=5.0, pricing_known=False)
        llm = TracedLLM(FakeLLM(), ctx, "gt", budget)  # must not raise
        llm.generate("hello")
        assert budget.spent == 0.0  # unknown pricing contributes no dollars


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
