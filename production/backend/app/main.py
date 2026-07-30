"""FastAPI app factory. All state (store/bus/manager) hangs off app.state."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from app.db.store import Store
from app.models import RunDetail, RunSummary, StageInfo
from app.orchestrator.events import EventBus
from app.orchestrator.manager import RunManager
from app.stages.topology import GRAFT_TOPOLOGY

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def create_app(data_dir: Optional[Path] = None, mode: str = "thread") -> FastAPI:
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = Store(base / "graft.db")
        app.state.bus = EventBus()
        app.state.bus.bind_loop(asyncio.get_running_loop())
        app.state.manager = RunManager(app.state.store, app.state.bus, base / "runs", mode=mode)
        yield
        app.state.store.close()

    app = FastAPI(title="GRAFT Studio", lifespan=lifespan)

    # -- topology -------------------------------------------------------------

    @app.get("/api/topology")
    def topology():
        return {
            "stages": [
                StageInfo(name=s.name, label=s.label, deps=list(s.deps), track=s.track).model_dump()
                for s in GRAFT_TOPOLOGY
            ]
        }

    # -- runs -------------------------------------------------------------

    @app.post("/api/runs", status_code=201)
    async def start_run(
        files: list[UploadFile] = [],
        config: str = Form("{}"),
        pipeline: str = Form("dummy"),
    ):
        try:
            cfg = json.loads(config)
        except json.JSONDecodeError:
            raise HTTPException(422, "config must be valid JSON")
        mgr: RunManager = app.state.manager
        # prepare -> save uploads -> launch, so the first stage never races
        # the file writes.
        run_id = mgr.prepare_run(cfg, pipeline=pipeline)
        updir = mgr.data_dir / run_id / "uploads"
        for f in files:
            (updir / Path(f.filename).name).write_bytes(await f.read())
        mgr.launch_run(run_id)
        return {"run_id": run_id}

    @app.get("/api/runs")
    def list_runs():
        runs = [
            RunSummary(**{k: r[k] for k in RunSummary.model_fields})
            for r in app.state.store.list_runs()
        ]
        return {"runs": [r.model_dump() for r in runs]}

    def _get_run_or_404(run_id: str) -> dict:
        run = app.state.store.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        return run

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        run = _get_run_or_404(run_id)
        store: Store = app.state.store
        detail = RunDetail(
            run_id=run["run_id"], pipeline=run["pipeline"], status=run["status"],
            created_at=run["created_at"], error=run["error"], config=run["config"],
            stages=[
                {k: s.get(k) for k in
                 ("name", "status", "started_at", "finished_at", "duration_s", "error", "traceback")}
                for s in run["stages"]
            ],
            metrics=store.list_metrics(run_id),
            artifacts=store.list_artifacts(run_id),
        )
        return detail.model_dump()

    @app.get("/api/runs/{run_id}/events")
    def run_events(
        run_id: str, after: int = 0, stage: Optional[str] = None,
        type: Optional[str] = None, limit: int = 2000,
    ):
        _get_run_or_404(run_id)
        return {"events": app.state.store.events_after(run_id, after, stage, type, limit)}

    @app.get("/api/runs/{run_id}/spans")
    def run_spans(
        run_id: str, stage: Optional[str] = None, type: Optional[str] = None,
        limit: int = 500, offset: int = 0,
    ):
        _get_run_or_404(run_id)
        return {"spans": app.state.store.list_spans(run_id, stage, type, limit, offset)}

    @app.get("/api/runs/{run_id}/traces/export")
    def traces_export(run_id: str):
        _get_run_or_404(run_id)
        spans = app.state.store.list_spans(run_id, limit=100000)
        return JSONResponse(
            {"run_id": run_id, "spans": spans},
            headers={"Content-Disposition": f'attachment; filename="{run_id}_traces.json"'},
        )

    @app.get("/api/runs/{run_id}/artifacts")
    def run_artifacts(run_id: str):
        _get_run_or_404(run_id)
        return {"artifacts": app.state.store.list_artifacts(run_id)}

    @app.get("/api/runs/{run_id}/artifacts/{name}")
    def artifact_file(run_id: str, name: str, download: bool = False):
        _get_run_or_404(run_id)
        for a in app.state.store.list_artifacts(run_id):
            if a["name"] == name:
                if download:
                    # filename= sets Content-Disposition: attachment
                    return FileResponse(a["path"], filename=name)
                # inline: lets the Report tab iframe actually render HTML
                return FileResponse(a["path"])
        raise HTTPException(404, f"artifact {name} not found")

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        _get_run_or_404(run_id)
        app.state.manager.cancel(run_id)
        return {"status": "cancelling"}

    @app.post("/api/runs/{run_id}/stages/{stage}/retry")
    def retry_stage(run_id: str, stage: str):
        _get_run_or_404(run_id)
        try:
            app.state.manager.retry_stage(run_id, stage)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"status": "retrying"}

    # -- websocket -------------------------------------------------------------

    @app.websocket("/ws/runs/{run_id}")
    async def ws_run(ws: WebSocket, run_id: str, after: int = 0):
        await ws.accept()
        store: Store = app.state.store
        bus: EventBus = app.state.bus
        # subscribe BEFORE replay so no event can fall between replay and live;
        # the seq > last guard deduplicates any overlap.
        q = bus.subscribe(run_id)
        try:
            last = after
            for ev in store.events_after(run_id, after_seq=after, limit=100000):
                await ws.send_json(ev)
                last = ev["seq"]
            while True:
                ev = await q.get()
                if ev["seq"] > last:
                    await ws.send_json(ev)
                    last = ev["seq"]
        except WebSocketDisconnect:
            pass
        finally:
            bus.unsubscribe(run_id, q)

    return app
