import threading
from pathlib import Path

from app.orchestrator.context import RunCancelled, StageContext
from app.orchestrator.dag import PipelineDAG, StageDef
from app.orchestrator.runner import execute_run


class Sink:
    """Collects queue messages for assertions."""

    def __init__(self):
        self.messages = []

    def put(self, msg):
        self.messages.append(msg)

    def events(self, type=None):
        evs = [m["event"] for m in self.messages if m["kind"] == "event"]
        return [e for e in evs if type is None or e["type"] == type]

    def spans(self):
        return [m["span"] for m in self.messages if m["kind"] == "span"]

    def done(self):
        return next(m for m in self.messages if m["kind"] == "done")


def _mkdag(fns: dict, deps: dict) -> PipelineDAG:
    return PipelineDAG(
        [StageDef(n, n, tuple(deps.get(n, ())), "gt", fn) for n, fn in fns.items()]
    )


def _run(dag, sink, cancel=None, completed=frozenset(), config=None, tmp=Path(".")):
    return execute_run(
        dag, "r1", tmp, config or {}, sink.put, cancel or threading.Event(), completed
    )


def test_happy_path_orders_and_completes(tmp_path):
    seen = []
    fns = {
        "a": lambda ctx: seen.append("a"),
        "b": lambda ctx: seen.append("b"),
        "c": lambda ctx: seen.append("c"),
    }
    sink = Sink()
    status = _run(_mkdag(fns, {"b": ("a",), "c": ("b",)}), sink, tmp=tmp_path)
    assert status == "completed"
    assert seen == ["a", "b", "c"]
    assert [e["stage"] for e in sink.events("stage_completed")] == ["a", "b", "c"]
    assert sink.events("run_completed")
    assert sink.done()["status"] == "completed"


def test_failure_skips_transitive_downstream_but_runs_independent(tmp_path):
    def boom(ctx):
        raise RuntimeError("kaput")

    seen = []
    fns = {
        "a": lambda ctx: seen.append("a"),
        "bad": boom,
        "child": lambda ctx: seen.append("child"),
        "island": lambda ctx: seen.append("island"),
    }
    sink = Sink()
    status = _run(
        _mkdag(fns, {"bad": ("a",), "child": ("bad",)}), sink, tmp=tmp_path
    )
    assert status == "failed"
    assert "child" not in seen and "island" in seen
    failed = sink.events("stage_failed")
    assert failed[0]["stage"] == "bad"
    assert "kaput" in failed[0]["payload"]["error"]
    assert "RuntimeError" in failed[0]["payload"]["traceback"]
    assert [e["stage"] for e in sink.events("stage_skipped")] == ["child"]
    assert sink.events("run_failed")


def test_completed_stages_are_not_rerun(tmp_path):
    seen = []
    fns = {"a": lambda ctx: seen.append("a"), "b": lambda ctx: seen.append("b")}
    sink = Sink()
    status = _run(_mkdag(fns, {"b": ("a",)}), sink, completed=frozenset({"a"}), tmp=tmp_path)
    assert status == "completed"
    assert seen == ["b"]  # a skipped silently (already done)
    assert [e["stage"] for e in sink.events("stage_started")] == ["b"]


def test_cancel_marks_run_cancelled(tmp_path):
    cancel = threading.Event()

    def first(ctx):
        cancel.set()

    def second(ctx):
        ctx.check_cancel()

    sink = Sink()
    fns = {"a": first, "b": second}
    status = _run(_mkdag(fns, {"b": ("a",)}), sink, cancel=cancel, tmp=tmp_path)
    assert status == "cancelled"
    assert sink.events("run_cancelled")


def test_context_emits_logs_progress_metrics_spans(tmp_path):
    def work(ctx: StageContext):
        ctx.log("info", "hello")
        ctx.progress(1, 2, "halfway")
        ctx.metric("facts", 42)
        with ctx.span("processing", "parse", input={"file": "x.pdf"}) as sp:
            sp.output = {"pages": 3}
            with sp.child("llm", "call") as ch:
                ch.tokens_in, ch.tokens_out, ch.cost_usd = 5, 7, 0.001

    sink = Sink()
    status = _run(_mkdag({"w": work}, {}), sink, tmp=tmp_path)
    assert status == "completed"
    assert sink.events("stage_log")[0]["payload"] == {"level": "info", "line": "hello"}
    assert sink.events("stage_progress")[0]["payload"] == {
        "done": 1, "total": 2, "message": "halfway",
    }
    assert sink.events("stage_metric")[0]["payload"] == {"name": "facts", "value": 42}
    spans = sink.spans()
    assert len(spans) == 2
    child = next(s for s in spans if s["type"] == "llm")
    parent = next(s for s in spans if s["type"] == "processing")
    assert child["parent_id"] == parent["span_id"]
    assert parent["output"] == {"pages": 3}
    assert child["cost_usd"] == 0.001
    assert parent["duration_ms"] >= 0


def test_span_error_status_on_exception(tmp_path):
    def work(ctx: StageContext):
        with ctx.span("processing", "parse"):
            raise ValueError("bad pdf")

    sink = Sink()
    status = _run(_mkdag({"w": work}, {}), sink, tmp=tmp_path)
    assert status == "failed"
    assert sink.spans()[0]["status"] == "error"
