import json
from pathlib import Path

import pytest

from app.db.store import Store


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "graft.db")


def test_run_lifecycle(store: Store):
    store.create_run("r1", {"top_k": 5}, pipeline="dummy")
    run = store.get_run("r1")
    assert run["status"] == "queued"
    assert run["config"] == {"top_k": 5}
    assert run["stages"] == []

    store.set_run_status("r1", "running")
    store.upsert_stage("r1", "ingest", status="running", started_at="t0")
    store.upsert_stage("r1", "ingest", status="completed", duration_s=1.5)
    run = store.get_run("r1")
    assert run["status"] == "running"
    assert run["stages"][0]["name"] == "ingest"
    assert run["stages"][0]["status"] == "completed"
    assert run["stages"][0]["duration_s"] == 1.5
    assert run["stages"][0]["started_at"] == "t0"  # preserved across upsert

    assert [r["run_id"] for r in store.list_runs()] == ["r1"]


def test_get_missing_run_returns_none(store: Store):
    assert store.get_run("nope") is None


def test_events_seq_and_filters(store: Store):
    store.create_run("r1", {}, pipeline="dummy")
    s1 = store.append_event({"run_id": "r1", "stage": "a", "type": "stage_started", "ts": "t", "payload": {}})
    s2 = store.append_event({"run_id": "r1", "stage": "a", "type": "stage_log", "ts": "t", "payload": {"line": "hi"}})
    s3 = store.append_event({"run_id": "r1", "stage": "b", "type": "stage_log", "ts": "t", "payload": {"line": "yo"}})
    assert s1 < s2 < s3

    evs = store.events_after("r1", after_seq=0)
    assert [e["seq"] for e in evs] == [s1, s2, s3]
    assert evs[1]["payload"] == {"line": "hi"}

    only_b_logs = store.events_after("r1", stage="b", type="stage_log")
    assert len(only_b_logs) == 1 and only_b_logs[0]["payload"]["line"] == "yo"

    assert store.events_after("r1", after_seq=s3) == []


def test_metrics_artifacts_spans(store: Store):
    store.create_run("r1", {}, pipeline="dummy")
    store.add_metric("r1", "extract", "facts", 123)
    assert store.list_metrics("r1")[0]["value"] == 123

    store.add_artifact("r1", "gt", "gt.jsonl", "runs/r1/artifacts/gt.jsonl", "jsonl", 456)
    assert store.list_artifacts("r1")[0]["size_bytes"] == 456

    store.add_span({
        "span_id": "sp1", "run_id": "r1", "stage": "qagen", "parent_id": None,
        "type": "llm", "name": "generate_pair", "status": "ok",
        "started_at": "t", "duration_ms": 900.0,
        "input": {"prompt": "Q?"}, "output": {"text": "A."},
        "tokens_in": 10, "tokens_out": 20, "cost_usd": 0.001, "extra": {"cache": False},
    })
    spans = store.list_spans("r1")
    assert spans[0]["input"] == {"prompt": "Q?"}
    assert spans[0]["cost_usd"] == 0.001
    assert store.list_spans("r1", type="retrieval") == []
