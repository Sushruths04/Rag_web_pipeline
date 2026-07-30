"""StageContext: the only API stage functions get. Everything a stage does —
logs, progress, metrics, artifacts, spans, cancellation — flows through here
as queue messages, so stage code never touches the DB or the websocket.
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunCancelled(Exception):
    pass


class Span:
    def __init__(
        self,
        put: Callable[[dict], None],
        run_id: str,
        stage: str,
        type: str,
        name: str,
        input: Optional[dict],
        parent_id: Optional[str],
    ) -> None:
        self._put = put
        self.span_id = uuid.uuid4().hex[:16]
        self.run_id = run_id
        self.stage = stage
        self.type = type
        self.name = name
        self.input = input
        self.parent_id = parent_id
        self.output: Optional[dict] = None
        self.tokens_in: Optional[int] = None
        self.tokens_out: Optional[int] = None
        self.cost_usd: Optional[float] = None
        self.extra: Optional[dict] = None
        self.status = "ok"
        self.started_at = _now()
        self._t0 = time.perf_counter()

    @contextmanager
    def child(self, type: str, name: str, input: Optional[dict] = None) -> Iterator["Span"]:
        yield from _span_cm(
            self._put, self.run_id, self.stage, type, name, input, self.span_id
        )

    def _finish(self) -> None:
        self._put(
            {
                "kind": "span",
                "span": {
                    "span_id": self.span_id,
                    "run_id": self.run_id,
                    "stage": self.stage,
                    "parent_id": self.parent_id,
                    "type": self.type,
                    "name": self.name,
                    "status": self.status,
                    "started_at": self.started_at,
                    "duration_ms": (time.perf_counter() - self._t0) * 1000.0,
                    "input": self.input,
                    "output": self.output,
                    "tokens_in": self.tokens_in,
                    "tokens_out": self.tokens_out,
                    "cost_usd": self.cost_usd,
                    "extra": self.extra,
                },
            }
        )


def _span_cm(put, run_id, stage, type, name, input, parent_id) -> Iterator[Span]:
    sp = Span(put, run_id, stage, type, name, input, parent_id)
    try:
        yield sp
    except BaseException:
        sp.status = "error"
        raise
    finally:
        sp._finish()


class StageContext:
    def __init__(
        self,
        run_id: str,
        stage: str,
        run_dir: Path,
        config: dict,
        put: Callable[[dict], None],
        cancel: Any,  # threading.Event or multiprocessing.Event
    ) -> None:
        self.run_id = run_id
        self.stage = stage
        self.run_dir = Path(run_dir)
        self.config = config
        self._put = put
        self._cancel = cancel

    # -- events --------------------------------------------------------------

    def _event(self, type: str, payload: dict) -> None:
        self._put(
            {
                "kind": "event",
                "event": {
                    "run_id": self.run_id,
                    "stage": self.stage,
                    "type": type,
                    "ts": _now(),
                    "payload": payload,
                },
            }
        )

    def log(self, level: str, line: str) -> None:
        self._event("stage_log", {"level": level, "line": line})

    def progress(self, done: int, total: int, message: str = "") -> None:
        self._event("stage_progress", {"done": done, "total": total, "message": message})

    def metric(self, name: str, value: float) -> None:
        self._event("stage_metric", {"name": name, "value": value})

    def artifact(self, name: str, path: str | Path, kind: str = "file") -> None:
        p = Path(path)
        self._put(
            {
                "kind": "artifact",
                "artifact": {
                    "run_id": self.run_id,
                    "stage": self.stage,
                    "name": name,
                    "path": str(p),
                    "kind": kind,
                    "size_bytes": p.stat().st_size if p.exists() else 0,
                },
            }
        )

    # -- spans ---------------------------------------------------------------

    def span(self, type: str, name: str, input: Optional[dict] = None):
        return contextmanager(_span_cm)(
            self._put, self.run_id, self.stage, type, name, input, None
        )

    # -- cancel --------------------------------------------------------------

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise RunCancelled(f"run {self.run_id} cancelled")
