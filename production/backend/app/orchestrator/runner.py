"""Topological DAG execution with failure/skip propagation.

Runs stages sequentially in topo order (per-item parallelism belongs inside
stages). Pure function of its inputs: all effects go through `put`, so it
runs identically in a thread (tests) or a subprocess (production).
"""
from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.orchestrator.context import RunCancelled, StageContext
from app.orchestrator.dag import PipelineDAG


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execute_run(
    dag: PipelineDAG,
    run_id: str,
    run_dir: Path,
    config: dict,
    put: Callable[[dict], None],
    cancel: Any,
    completed: frozenset[str] = frozenset(),
) -> str:
    def event(stage: str | None, type: str, payload: dict) -> None:
        put(
            {
                "kind": "event",
                "event": {
                    "run_id": run_id,
                    "stage": stage,
                    "type": type,
                    "ts": _now(),
                    "payload": payload,
                },
            }
        )

    skipped: set[str] = set()
    failed_any = False
    cancelled = False

    for name in dag.topo_order():
        if name in completed:
            continue
        if name in skipped:
            event(name, "stage_skipped", {"reason": "upstream_failed"})
            continue
        if cancelled or cancel.is_set():
            cancelled = True
            event(name, "stage_skipped", {"reason": "run_cancelled"})
            continue

        sdef = dag.stage(name)
        ctx = StageContext(run_id, name, run_dir, config, put, cancel)
        event(name, "stage_started", {"label": sdef.label, "track": sdef.track})
        t0 = time.perf_counter()
        try:
            if sdef.fn is None:
                raise RuntimeError(f"stage {name!r} has no bound function")
            sdef.fn(ctx)
        except RunCancelled:
            cancelled = True
            event(name, "stage_skipped", {"reason": "run_cancelled"})
            continue
        except Exception as exc:  # noqa: BLE001 — every stage error must be captured
            failed_any = True
            event(
                name,
                "stage_failed",
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "duration_s": time.perf_counter() - t0,
                },
            )
            skipped |= dag.downstream(name) - completed
            continue
        event(name, "stage_completed", {"duration_s": time.perf_counter() - t0})

    if cancelled:
        status = "cancelled"
        event(None, "run_cancelled", {})
    elif failed_any:
        status = "failed"
        event(None, "run_failed", {})
    else:
        status = "completed"
        event(None, "run_completed", {})
    put({"kind": "done", "status": status})
    return status
