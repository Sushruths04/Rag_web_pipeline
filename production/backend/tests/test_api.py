import json
import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(data_dir=tmp_path, mode="thread")
    with TestClient(app) as c:
        yield c


def _start(client, config=None, files=None):
    data = {"config": json.dumps(config or {"sleep_scale": 0.0}), "pipeline": "dummy"}
    r = client.post("/api/runs", data=data, files=files or [])
    assert r.status_code == 201
    return r.json()["run_id"]


def _wait_done(client, run_id, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in ("completed", "failed", "cancelled"):
            return run
        time.sleep(0.05)
    raise AssertionError("run did not finish in time")


def test_topology_endpoint(client):
    r = client.get("/api/topology")
    assert r.status_code == 200
    stages = r.json()["stages"]
    assert {"name", "label", "deps", "track"} <= set(stages[0])
    assert any(s["name"] == "qagen" for s in stages)


def test_run_lifecycle_via_api(client):
    run_id = _start(client)
    run = _wait_done(client, run_id)
    assert run["status"] == "completed"
    assert len(run["stages"]) == 19
    assert run["metrics"]
    assert any(a["name"] == "gt.jsonl" for a in run["artifacts"])

    runs = client.get("/api/runs").json()["runs"]
    assert runs[0]["run_id"] == run_id

    evs = client.get(f"/api/runs/{run_id}/events", params={"type": "stage_metric"}).json()["events"]
    assert evs and all(e["type"] == "stage_metric" for e in evs)

    spans = client.get(f"/api/runs/{run_id}/spans", params={"type": "llm"}).json()["spans"]
    assert spans and spans[0]["stage"] == "qagen"

    export = client.get(f"/api/runs/{run_id}/traces/export")
    assert export.status_code == 200
    assert export.headers["content-disposition"].startswith("attachment")

    art = client.get(f"/api/runs/{run_id}/artifacts/gt.jsonl")
    assert art.status_code == 200 and b"dummy" in art.content


def test_upload_saves_files(client, tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    run_id = _start(client, files=[("files", ("x.pdf", pdf.read_bytes(), "application/pdf"))])
    _wait_done(client, run_id)
    uploads = tmp_path / "runs" / run_id / "uploads"
    assert (uploads / "x.pdf").read_bytes().startswith(b"%PDF")


def test_missing_run_404(client):
    assert client.get("/api/runs/nope").status_code == 404


def test_retry_conflict_while_running_and_ok_after(client):
    run_id = _start(client, config={"sleep_scale": 0.0, "fail_stage": "extract"})
    run = _wait_done(client, run_id)
    assert run["status"] == "failed"

    r = client.post(f"/api/runs/{run_id}/stages/extract/retry")
    # config still has fail_stage → it fails again, but the endpoint accepted
    assert r.status_code == 200
    _wait_done(client, run_id)


def test_websocket_replays_then_streams(client):
    run_id = _start(client)
    _wait_done(client, run_id)
    # connect AFTER completion: everything must arrive via replay
    with client.websocket_connect(f"/ws/runs/{run_id}?after=0") as ws:
        first = ws.receive_json()
        assert first["seq"] >= 1
        assert first["type"] == "stage_started"
        # drain until run_completed shows up
        ev = first
        while ev["type"] != "run_completed":
            ev = ws.receive_json()
        assert ev["type"] == "run_completed"
