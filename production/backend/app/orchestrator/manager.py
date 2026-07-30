"""RunManager: owns worker lifecycles and the pump that persists messages.

mode="thread": worker runs in a daemon thread (tests, dummy pipeline).
mode="process": worker runs in a spawned subprocess (production; real
pipeline stages hold the GIL for seconds at a time). Same code path — the
queue type and cancel-event type are the only differences.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as thread_queue
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from app.db.store import Store
from app.orchestrator.dag import PipelineDAG
from app.orchestrator.events import EventBus
from app.orchestrator.runner import execute_run
from app.stages.dummy import build_dummy_dag
from app.stages.topology import GRAFT_TOPOLOGY

PIPELINES: dict[str, Callable[[], PipelineDAG]] = {
    "dummy": lambda: build_dummy_dag(GRAFT_TOPOLOGY),
}

from app.stages.graft import register as _register_graft  # noqa: E402

_register_graft(PIPELINES)


def _worker_entry(pipeline, run_id, run_dir, config, q, cancel, completed):
    """Top-level so it is picklable for mp spawn."""
    dag = PIPELINES[pipeline]()
    execute_run(dag, run_id, Path(run_dir), config, q.put, cancel, frozenset(completed))


class RunManager:
    def __init__(self, store: Store, bus: EventBus, data_dir: Path, mode: str = "thread") -> None:
        assert mode in ("thread", "process")
        self.store = store
        self.bus = bus
        self.data_dir = Path(data_dir)
        self.mode = mode
        self._cancels: dict[str, Any] = {}
        self._pumps: dict[str, threading.Thread] = {}

    # -- lifecycle -------------------------------------------------------------

    def prepare_run(self, config: dict, pipeline: str = "dummy") -> str:
        """Create the run row, stage rows, and run dirs — but do NOT launch.

        Callers that write uploads must do so between prepare and launch,
        otherwise the first stage races the file writes.
        """
        if pipeline not in PIPELINES:
            raise ValueError(f"unknown pipeline {pipeline!r}")
        run_id = uuid.uuid4().hex[:12]
        self.store.create_run(run_id, config, pipeline)
        dag = PIPELINES[pipeline]()
        for name in dag.topo_order():
            self.store.upsert_stage(run_id, name, status="queued")
        run_dir = self.data_dir / run_id
        for sub in ("uploads", "artifacts", "logs"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        return run_id

    def launch_run(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        self._launch(run_id, run["pipeline"], self.data_dir / run_id, run["config"], frozenset())

    def start_run(self, config: dict, pipeline: str = "dummy") -> str:
        run_id = self.prepare_run(config, pipeline)
        self.launch_run(run_id)
        return run_id

    def retry_stage(self, run_id: str, stage: str) -> None:
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        if run["status"] == "running":
            raise RuntimeError("run is still active")
        completed = frozenset(
            s["name"] for s in run["stages"] if s["status"] == "completed"
        ) - {stage}
        for s in run["stages"]:
            if s["name"] not in completed:
                self.store.upsert_stage(
                    run_id, s["name"], status="queued", error=None, traceback=None
                )
        self._launch(run_id, run["pipeline"], self.data_dir / run_id, run["config"], completed)

    def cancel(self, run_id: str) -> None:
        ev = self._cancels.get(run_id)
        if ev is not None:
            ev.set()

    def wait(self, run_id: str, timeout: float = 30.0) -> None:
        t = self._pumps.get(run_id)
        if t is not None:
            t.join(timeout)

    def is_active(self, run_id: str) -> bool:
        t = self._pumps.get(run_id)
        return t is not None and t.is_alive()

    # -- internals -------------------------------------------------------------

    def _launch(self, run_id, pipeline, run_dir, config, completed) -> None:
        self.store.set_run_status(run_id, "running")
        if self.mode == "process":
            q: Any = mp.Queue()
            cancel: Any = mp.Event()
            worker: Any = mp.Process(
                target=_worker_entry,
                args=(pipeline, run_id, str(run_dir), config, q, cancel, sorted(completed)),
                daemon=True,
            )
        else:
            q = thread_queue.Queue()
            cancel = threading.Event()
            worker = threading.Thread(
                target=_worker_entry,
                args=(pipeline, run_id, str(run_dir), config, q, cancel, sorted(completed)),
                daemon=True,
            )
        self._cancels[run_id] = cancel
        worker.start()
        pump = threading.Thread(target=self._pump, args=(run_id, q), daemon=True)
        self._pumps[run_id] = pump
        pump.start()

    def _pump(self, run_id: str, q: Any) -> None:
        while True:
            msg = q.get()
            kind = msg["kind"]
            if kind == "done":
                break
            if kind == "event":
                self._handle_event(msg["event"])
            elif kind == "span":
                self.store.add_span(msg["span"])
            elif kind == "artifact":
                a = msg["artifact"]
                self.store.add_artifact(
                    a["run_id"], a["stage"], a["name"], a["path"], a["kind"], a["size_bytes"]
                )

    def _handle_event(self, event: dict) -> None:
        seq = self.store.append_event(event)
        run_id, stage, etype, payload = (
            event["run_id"], event.get("stage"), event["type"], event["payload"]
        )
        if etype == "stage_started":
            self.store.upsert_stage(run_id, stage, status="running", started_at=event["ts"])
        elif etype == "stage_completed":
            self.store.upsert_stage(
                run_id, stage, status="completed",
                finished_at=event["ts"], duration_s=payload.get("duration_s"),
            )
        elif etype == "stage_failed":
            self.store.upsert_stage(
                run_id, stage, status="failed", finished_at=event["ts"],
                error=payload.get("error"), traceback=payload.get("traceback"),
                duration_s=payload.get("duration_s"),
            )
        elif etype == "stage_skipped":
            self.store.upsert_stage(run_id, stage, status="skipped")
        elif etype == "stage_metric":
            self.store.add_metric(run_id, stage, payload["name"], payload["value"])
        elif etype == "run_completed":
            self.store.set_run_status(run_id, "completed")
        elif etype == "run_failed":
            self.store.set_run_status(run_id, "failed")
        elif etype == "run_cancelled":
            self.store.set_run_status(run_id, "cancelled")
        self.bus.publish_threadsafe({**event, "seq": seq})
