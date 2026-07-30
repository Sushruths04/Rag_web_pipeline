"""Structured pipeline tracing for generation runs.

The tracer writes newline-delimited JSON events plus a compact summary file.
It is intentionally dependency-free so it can run inside the pipeline without
pulling in a web stack or changing normal console logging.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote


PIPELINE_STAGE_MAP: list[dict[str, Any]] = [
    {
        "id": "run_setup",
        "label": "Run setup",
        "outputs": ["config", "input_paths", "llm_handles"],
    },
    {
        "id": "ingestion",
        "label": "Document ingestion",
        "outputs": ["Document.text", "source_units", "source_backend"],
        "fallbacks": ["docling->pymupdf when ingestion.pdf_backend=auto"],
    },
    {
        "id": "profiling",
        "label": "Document profiling",
        "outputs": ["doc_type", "signal_scores", "fast_path_hit"],
    },
    {
        "id": "chunking",
        "label": "Chunking",
        "outputs": ["chunks", "chunk source ranges", "sentences", "tokens"],
    },
    {
        "id": "fact_extraction",
        "label": "Fact extraction and source mapping",
        "outputs": ["candidate_facts", "supporting_spans"],
        "gates": ["min/max fact length", "structural_quality", "span IoU"],
    },
    {
        "id": "fact_domain_filter",
        "label": "Fact domain filter",
        "outputs": ["content_facts", "drop reasons"],
    },
    {
        "id": "budget",
        "label": "Adaptive budget",
        "outputs": ["DocBudget"],
        "condition": "v16.2 only",
    },
    {
        "id": "vector_index",
        "label": "Embedding and vector index",
        "outputs": ["fact embeddings", "FactIndex"],
    },
    {
        "id": "tf_sfg",
        "label": "Typed Fact Sub-Graph",
        "outputs": ["typed edges", "L0 stats"],
        "condition": "v16 only",
    },
    {
        "id": "chain_sampling",
        "label": "Chain sampling",
        "outputs": ["FactChain candidates"],
        "fallbacks": ["legacy random sampler", "v16 single-hop fallback"],
    },
    {
        "id": "chain_filter",
        "label": "Chain filtering",
        "outputs": ["filtered chains", "reject reasons"],
        "gates": ["depth", "distinct facts/chunks/roles/pages", "char gap", "chain quality", "C3"],
    },
    {
        "id": "chain_scoring",
        "label": "Chain scoring and yield control",
        "outputs": ["ranked ChainScore", "yield category"],
        "condition": "v16.2 only",
    },
    {
        "id": "candidate_generation",
        "label": "Question and answer candidate generation",
        "outputs": ["question", "answer", "QA-NLI/QRSG/ARM extras"],
        "retries": ["question generation", "QA-NLI guided regeneration", "ARM step retries"],
    },
    {
        "id": "post_generation_gates",
        "label": "Post-generation gates",
        "outputs": ["accepted/rejected candidate rows"],
        "gates": ["QA-NLI", "QRSG", "provenance", "question-domain", "structure", "quality", "dedup"],
    },
    {
        "id": "answer_nli",
        "label": "Batch answer NLI",
        "outputs": ["survivors"],
    },
    {
        "id": "minimality",
        "label": "Minimality check",
        "outputs": ["minimal accepted rows"],
        "condition": "depth > 1 and not fast_mode",
    },
    {
        "id": "augmentation",
        "label": "Distractors and twins",
        "outputs": ["distractor_spans", "AT/CT twins"],
        "condition": "v16 only",
    },
    {
        "id": "persistence",
        "label": "Persistence",
        "outputs": ["partial JSONL", "final JSONL", "drop stats", "build summary"],
    },
]


def make_chain_id(fact_ids: list[str] | tuple[str, ...]) -> str:
    payload = "|".join(str(x) for x in fact_ids)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"chain_{digest}"


def fact_snapshot(fact: Any, *, include_text: bool = True) -> dict[str, Any]:
    spans = list(getattr(fact, "supporting_spans", []) or [])
    span_dicts: list[dict[str, Any]] = []
    bbox_summary: list[dict[str, Any]] = []
    first_source_path = ""
    first_page = None
    for span in spans:
        sd = span.to_dict() if hasattr(span, "to_dict") else dict(span)
        span_dicts.append(sd)
        if not first_source_path:
            first_source_path = str(sd.get("source_path", "") or "")
        if first_page is None and sd.get("page_start") is not None:
            first_page = int(sd.get("page_start"))
        for bbox in sd.get("bboxes", []) or []:
            bbox_summary.append(
                {
                    "page_no": bbox.get("page_no"),
                    "l": bbox.get("l"),
                    "t": bbox.get("t"),
                    "r": bbox.get("r"),
                    "b": bbox.get("b"),
                    "coord_origin": bbox.get("coord_origin"),
                }
            )
    pdf_page_link = ""
    if first_source_path and first_page is not None:
        pdf_page_link = (
            "file:///"
            + quote(str(Path(first_source_path).resolve()).replace("\\", "/"), safe="/:")
            + f"#page={first_page}"
        )
    pages = sorted(
        {
            int(p)
            for span in spans
            for p in (getattr(span, "page_start", None), getattr(span, "page_end", None))
            if p is not None
        }
    )
    out = {
        "fact_id": str(getattr(fact, "fact_id", "")),
        "role": str(getattr(fact, "role", "")),
        "weight": getattr(fact, "weight", None),
        "span_count": len(spans),
        "chunk_ids": [
            str(getattr(span, "chunk_id", ""))
            for span in spans
            if str(getattr(span, "chunk_id", ""))
        ],
        "pages": pages,
        "self_containment_known": bool(getattr(fact, "self_containment_known", False)),
        "self_containment_score": getattr(fact, "self_containment_score", None),
        "first_page": first_page,
        "source_path": first_source_path,
        "pdf_page_link": pdf_page_link,
        "bbox_count": len(bbox_summary),
        "bbox_summary": bbox_summary,
        "supporting_spans_full": span_dicts,
    }
    if include_text:
        out["text"] = str(getattr(fact, "text", "") or "")
    return out


def edge_snapshot(edge: Any) -> dict[str, Any]:
    """Compact TF-SFG edge snapshot for debugging pair/chain failures."""
    if hasattr(edge, "to_dict"):
        ed = edge.to_dict()
    elif isinstance(edge, dict):
        ed = dict(edge)
    else:
        ed = dict(getattr(edge, "__dict__", {}) or {})
    out = {
        "edge_id": str(ed.get("edge_id", "") or ""),
        "src": str(ed.get("src", "") or ""),
        "dst": str(ed.get("dst", "") or ""),
        "type": str(ed.get("type", "") or ""),
        "bridging_fact_id": str(ed.get("bridging_fact_id", "") or ""),
        "nli_score": ed.get("nli_score"),
        "single_src_score": ed.get("single_src_score"),
        "single_dst_score": ed.get("single_dst_score"),
        "joint_only_margin": ed.get("joint_only_margin"),
        "relation_claim": str(ed.get("relation_claim", "") or "")[:360],
        "bridging_quote": str(ed.get("bridging_quote", "") or "")[:240],
    }
    return out


def chain_snapshot(chain: Any, facts_by_id: dict[str, Any] | None = None) -> dict[str, Any]:
    fact_ids = [str(x) for x in getattr(chain, "fact_ids", []) or []]
    chain_edges = list(getattr(chain, "chain_edges", []) or [])
    out = {
        "chain_id": make_chain_id(fact_ids),
        "fact_ids": fact_ids,
        "depth": int(getattr(chain, "depth", len(fact_ids)) or len(fact_ids)),
        "anchor_id": str(getattr(chain, "anchor_id", "") or ""),
        "mean_cosine": getattr(chain, "mean_cosine", None),
        "role_path": [str(x) for x in getattr(chain, "role_path", []) or []],
        "edge_count": len(chain_edges),
        "edge_types": [
            str(edge.get("type", ""))
            for edge in chain_edges
            if isinstance(edge, dict)
        ],
        "edges": [edge_snapshot(edge) for edge in chain_edges],
    }
    if facts_by_id:
        out["facts"] = [
            fact_snapshot(facts_by_id[fid], include_text=True)
            for fid in fact_ids
            if fid in facts_by_id
        ]
    return out


def chunk_sample(chunks: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    sample = []
    for c in chunks[:limit]:
        sample.append(
            {
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "char_start": c.get("char_start"),
                "char_end": c.get("char_end"),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "n_chars": len(str(c.get("text", "") or "")),
                "n_sentences": len(c.get("sentences", []) or []),
                "preview": str(c.get("text", "") or "")[:240],
            }
        )
    return sample


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    return str(value)


class PipelineTracer:
    """Append-only JSONL event writer with lightweight aggregation."""

    def __init__(
        self,
        trace_path: str | os.PathLike[str],
        *,
        run_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.path = Path(trace_path)
        self.run_id = run_id or time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        self.started_at = time.time()
        self._event_seq = 0
        self._lock = threading.Lock()
        self._summary: dict[str, Any] = {
            "run_id": self.run_id,
            "trace_path": str(self.path),
            "started_at": self.started_at,
            "stage_counts": Counter(),
            "event_counts": Counter(),
            "status_counts": Counter(),
            "drop_reasons": Counter(),
            "drop_by_stage": defaultdict(Counter),
            "doc_counts": defaultdict(Counter),
            "stage_durations_ms": defaultdict(float),
            "pipeline_stage_map": PIPELINE_STAGE_MAP,
        }
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    @property
    def summary_path(self) -> Path:
        name = self.path.name
        if name.endswith(".jsonl"):
            return self.path.with_name(name[:-6] + "_summary.json")
        return self.path.with_suffix(self.path.suffix + ".summary.json")

    def emit(
        self,
        stage: str,
        event: str,
        *,
        status: str = "ok",
        doc_id: str | None = None,
        item_id: str | None = None,
        parent_ids: list[str] | None = None,
        reason: str | None = None,
        counts: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        thresholds: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        now = time.time()
        with self._lock:
            self._event_seq += 1
            record = {
                "schema_version": "rag_gt.trace.v1",
                "run_id": self.run_id,
                "event_id": self._event_seq,
                "ts": now,
                "elapsed_ms": round((now - self.started_at) * 1000, 3),
                "stage": stage,
                "event": event,
                "status": status,
                "doc_id": doc_id,
                "item_id": item_id,
                "parent_ids": parent_ids or [],
                "reason": reason,
                "counts": counts or {},
                "metrics": metrics or {},
                "thresholds": thresholds or {},
                "data": data or {},
            }
            record = _jsonable(record)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

            self._summary["stage_counts"][stage] += 1
            self._summary["event_counts"][event] += 1
            self._summary["status_counts"][status] += 1
            if doc_id:
                self._summary["doc_counts"][doc_id][stage] += 1
            if status == "dropped" or event.startswith("drop"):
                drop_reason = reason or "unspecified"
                self._summary["drop_reasons"][drop_reason] += 1
                self._summary["drop_by_stage"][stage][drop_reason] += 1
            return record

    @contextmanager
    def stage(
        self,
        stage: str,
        *,
        doc_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        start = time.time()
        self.emit(stage, "stage_start", doc_id=doc_id, data=data or {})
        try:
            yield
        except Exception as exc:
            elapsed_ms = (time.time() - start) * 1000
            self._summary["stage_durations_ms"][stage] += elapsed_ms
            self.emit(
                stage,
                "stage_error",
                status="error",
                doc_id=doc_id,
                reason=type(exc).__name__,
                metrics={"duration_ms": round(elapsed_ms, 3)},
                data={"error": str(exc)},
            )
            raise
        else:
            elapsed_ms = (time.time() - start) * 1000
            self._summary["stage_durations_ms"][stage] += elapsed_ms
            self.emit(
                stage,
                "stage_end",
                doc_id=doc_id,
                metrics={"duration_ms": round(elapsed_ms, 3)},
            )

    def drop(
        self,
        stage: str,
        reason: str,
        *,
        doc_id: str | None = None,
        item_id: str | None = None,
        parent_ids: list[str] | None = None,
        counts: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        thresholds: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.emit(
            stage,
            "drop",
            status="dropped",
            doc_id=doc_id,
            item_id=item_id,
            parent_ids=parent_ids,
            reason=reason,
            counts=counts,
            metrics=metrics,
            thresholds=thresholds,
            data=data,
        )

    def close(self, final_data: dict[str, Any] | None = None) -> Path | None:
        if not self.enabled:
            return None
        self.emit(
            "run",
            "run_end",
            data={
                "final": final_data or {},
                "python": sys.version,
                "platform": platform.platform(),
            },
        )
        summary = {
            "run_id": self.run_id,
            "trace_path": str(self.path),
            "summary_path": str(self.summary_path),
            "started_at": self.started_at,
            "ended_at": time.time(),
            "duration_ms": round((time.time() - self.started_at) * 1000, 3),
            "stage_counts": dict(self._summary["stage_counts"]),
            "event_counts": dict(self._summary["event_counts"]),
            "status_counts": dict(self._summary["status_counts"]),
            "drop_reasons": dict(self._summary["drop_reasons"]),
            "drop_by_stage": {
                stage: dict(counter)
                for stage, counter in self._summary["drop_by_stage"].items()
            },
            "doc_counts": {
                doc_id: dict(counter)
                for doc_id, counter in self._summary["doc_counts"].items()
            },
            "stage_durations_ms": dict(self._summary["stage_durations_ms"]),
            "pipeline_stage_map": PIPELINE_STAGE_MAP,
            "final": final_data or {},
        }
        self.summary_path.write_text(
            json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return self.summary_path
