from pathlib import Path

import pytest

from app.db.store import Store
from app.orchestrator.events import EventBus
from app.orchestrator.manager import RunManager


@pytest.fixture()
def mgr(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite")
    m = RunManager(store, EventBus(), tmp_path / "runs", mode="thread")
    yield m
    store.close()


def test_dummy_run_completes_and_persists_everything(mgr: RunManager):
    run_id = mgr.start_run({"sleep_scale": 0.0}, pipeline="dummy")
    mgr.wait(run_id)

    run = mgr.store.get_run(run_id)
    assert run["status"] == "completed"
    stages = {s["name"]: s for s in run["stages"]}
    assert len(stages) == 19  # full GRAFT topology
    assert all(s["status"] == "completed" for s in stages.values())
    assert stages["qagen"]["duration_s"] is not None

    evs = mgr.store.events_after(run_id)
    types = {e["type"] for e in evs}
    assert {"stage_started", "stage_progress", "stage_log", "stage_metric",
            "stage_completed", "run_completed"} <= types
    seqs = [e["seq"] for e in evs]
    assert seqs == sorted(seqs)

    # dummy qagen emits llm spans with cost; dummy vector_index emits embedding span
    spans = mgr.store.list_spans(run_id, stage="qagen", type="llm")
    assert len(spans) >= 3 and all(s["cost_usd"] > 0 for s in spans)
    assert mgr.store.list_spans(run_id, stage="vector_index", type="embedding")

    assert any(m["name"] == "pairs_kept" for m in mgr.store.list_metrics(run_id))
    assert any(a["name"] == "gt.jsonl" for a in mgr.store.list_artifacts(run_id))


def test_failure_marks_downstream_skipped(mgr: RunManager):
    run_id = mgr.start_run({"sleep_scale": 0.0, "fail_stage": "extract"}, pipeline="dummy")
    mgr.wait(run_id)
    run = mgr.store.get_run(run_id)
    assert run["status"] == "failed"
    stages = {s["name"]: s for s in run["stages"]}
    assert stages["extract"]["status"] == "failed"
    assert "simulated failure" in stages["extract"]["error"]
    assert stages["extract"]["traceback"]
    assert stages["clean"]["status"] == "skipped"
    assert stages["report"]["status"] == "skipped"
    # rag track does NOT depend on extract → still completes
    assert stages["bm25_index"]["status"] == "completed"


def test_retry_after_failure_resumes_from_completed(mgr: RunManager):
    run_id = mgr.start_run({"sleep_scale": 0.0, "fail_stage": "extract"}, pipeline="dummy")
    mgr.wait(run_id)

    # clear the failure flag, retry the failed stage
    mgr.store._conn.execute(  # test-only poke: update stored config
        "UPDATE runs SET config='{\"sleep_scale\": 0.0}' WHERE run_id=?", (run_id,)
    )
    mgr.store._conn.commit()
    mgr.retry_stage(run_id, "extract")
    mgr.wait(run_id)

    run = mgr.store.get_run(run_id)
    assert run["status"] == "completed"
    stages = {s["name"]: s for s in run["stages"]}
    assert all(s["status"] == "completed" for s in stages.values())
    # ingest ran once, not twice (resume skipped completed stages)
    started = mgr.store.events_after(run_id, stage="ingest", type="stage_started")
    assert len(started) == 1


def test_cancel_stops_run(mgr: RunManager):
    run_id = mgr.start_run({"sleep_scale": 0.5}, pipeline="dummy")
    mgr.cancel(run_id)
    mgr.wait(run_id)
    assert mgr.store.get_run(run_id)["status"] == "cancelled"
