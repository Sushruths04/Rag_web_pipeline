"""Question Relation-Support Gate (QRSG) — v13.

Runs between question generation and answer generation. Classifies the
question's relation type, grounds each named concept in a fact via a literal
substring quote, requires a single linking fact for relational types, and
Python-verifies the LLM's claims before accepting the row.

Trace contract (per reviewer guardrail): every verdict carries question,
fact ids, relation_type, unsupported_concepts, unsupported_relation,
risky_frame_hit, llm_raw, and the final Python acceptance reason — enough
to debug a false-positive reject without re-running the pipeline.

JSON parsing follows the repo pattern used in validation.atomic_decomp:
json.loads(repair_json(_extract_json(raw))). SQLite cache mirrors
validation.nli_check (thread-local connection, WAL mode, sha256 key).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from json_repair import repair_json
from loguru import logger

from rag_gt.core.config import load_config, repo_root
from rag_gt.core.llm import _extract_json
from rag_gt.core.types import Fact

if TYPE_CHECKING:
    from rag_gt.core.llm import LLM


GATE_VERSION = "v1"

# Relation type catalogue. Closed set — anything else from the LLM is rejected
# as "unknown_relation_type". `procedural` and `list` are distinct: procedural
# = step-by-step "how to do X", list = enumeration of items.
RELATION_TYPES: Tuple[str, ...] = (
    "descriptive",
    "definition",
    "comparison",
    "contrast",
    "causal",
    "mechanism",
    "intersection",
    "temporal",
    "quantitative",
    "procedural",
    "list",
)

# Relational types that require a single linking fact connecting both
# endpoints. Listed separately from RELATIONAL_NO_LINK_NEEDED so the rule
# is enumerable, not implicit.
RELATIONAL_REQUIRES_LINK: frozenset[str] = frozenset({
    "comparison", "contrast", "causal", "mechanism", "intersection",
})
RELATIONAL_NO_LINK_NEEDED: frozenset[str] = frozenset({
    "descriptive", "definition", "temporal", "list",
    "quantitative", "procedural",
})

_CAUSE_CUES = (
    "because", "cause", "causes", "caused", "leads to", "lead to",
    "results in", "result in", "due to", "therefore", "so that",
)
_MECHANISM_CUES = (
    " by ", " using ", " through ", " via ", " method", "process",
    "mechanism", "approximate", "approximating", "sample", "sampling",
    "mitigate", "handle", "handles", "address", "addresses",
)


# Risky-frame regex pre-filter (audit §9.1). A hit alone does NOT reject; it
# is recorded on the verdict and, for relational types, forces a hard fail
# when the LLM cannot point to a linking fact.
RISKY_FRAMES: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"^how (does|do|is|are|can) .+ (handle|address|deal with|cope)", re.I), "mechanism_handle"),
    (re.compile(r"^why (does|do|is|are)", re.I), "causal_why"),
    # compar\w* covers compare/compared/comparing/comparison; also catches differ(ence) between
    (re.compile(r"\bcompar\w*\b|\bdiff\w+\s+between\b|\bversus\b|\bvs\.?\b", re.I), "comparison"),
    # intersection/overlap/combined OR "both X and Y" phrasing (no "share" required)
    (re.compile(r"\bintersection\b|\boverlap\b|\bcombined\b|\bboth\b.+\band\b|\bin\s+common\b", re.I), "intersection"),
    (re.compile(r"\b(mechanism|process by which|method (used )?to)\b", re.I), "mechanism_process"),
    (re.compile(r"\b(cause|effect|leads to|results in|due to)\b", re.I), "causal_cause"),
    (re.compile(r"\b(before|after|first .+ then|sequence of)\b", re.I), "temporal"),
    (re.compile(r"\b(list|enumerate|what are the .+s)\b", re.I), "list"),
]


@dataclass(frozen=True)
class RelationVerdict:
    """The gate's decision plus enough context to debug a false reject."""

    accepted: bool
    relation_type: str
    unsupported_concepts: List[str]
    unsupported_relation: bool
    evidence_map: Dict[str, List[Dict[str, str]]]  # concept -> [{"fact_id","quote"}]
    linking_fact_id: Optional[str]
    reason: str
    risky_frame_hit: Optional[str]
    llm_raw: Optional[str]

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "relation_type": self.relation_type,
            "unsupported_concepts": list(self.unsupported_concepts),
            "unsupported_relation": self.unsupported_relation,
            "evidence_map": self.evidence_map,
            "linking_fact_id": self.linking_fact_id,
            "reason": self.reason,
            "risky_frame_hit": self.risky_frame_hit,
        }


# ---------- prompt ----------


_QRSG_PROMPT_TEMPLATE = """You validate whether a synthetic evaluation question is supported by its evidence facts.

For the question and facts below, return a single JSON object with these fields:

- "relation_type": exactly one of
    descriptive | definition | comparison | contrast | causal | mechanism |
    intersection | temporal | quantitative | procedural | list
- "concepts": list of central concepts named or implied by the question.
- "evidence_map": object mapping each concept to a list of
    {{"fact_id": "F...", "quote": "..."}} entries. The "quote" MUST appear
    verbatim (case-insensitive) inside the cited fact's text.
- "linking_fact_id": if relation_type is one of
    comparison | contrast | causal | mechanism | intersection, the fact_id
    whose own text connects both endpoints. Otherwise null.
- "unsupported_concepts": concepts no fact supports.
- "unsupported_relation": true iff relation_type is relational
    (comparison/contrast/causal/mechanism/intersection) and no single fact
    explicitly links the endpoints. False for descriptive/definition/list/
    temporal/quantitative/procedural questions.
- "reason": one short sentence summarising the verdict.

Return ONLY the JSON object. No prose, no markdown fences, no commentary.

Facts:
__FACTS_BLOCK__

Question: __QUESTION__
"""


def _build_facts_block(facts: List[Fact]) -> str:
    lines: List[str] = []
    for f in facts:
        text = (f.canonical_form or f.text).strip()
        text = text.replace("<<FACT", "").replace("<</FACT>>", "")
        lines.append(f"  <<FACT id={f.fact_id}>> {text} <</FACT>>")
    return "\n".join(lines)


def _build_prompt(question: str, facts: List[Fact]) -> str:
    # Avoid str.format because the prompt contains literal JSON braces.
    out = _QRSG_PROMPT_TEMPLATE.replace("{{", "{").replace("}}", "}")
    out = out.replace("__FACTS_BLOCK__", _build_facts_block(facts))
    out = out.replace("__QUESTION__", question.strip())
    return out


# ---------- risky-frame pre-filter ----------


def _pre_filter_risky_frame(question: str) -> Optional[str]:
    """First regex hit's name, or None."""
    for rx, name in RISKY_FRAMES:
        if rx.search(question):
            return name
    return None


# ---------- Python verification ----------


def _verify(
    data: dict,
    facts: List[Fact],
    risky_frame: Optional[str],
) -> Tuple[bool, str, Dict[str, List[Dict[str, str]]], Optional[str], List[str], bool, str]:
    """Returns (accepted, reason, evidence_map, linking_fact_id,
    unsupported_concepts, unsupported_relation, relation_type)."""

    relation_type = str(data.get("relation_type", "descriptive") or "descriptive")
    if relation_type not in RELATION_TYPES:
        return False, f"unknown_relation_type:{relation_type}", {}, None, [], False, relation_type

    unsupported_relation = bool(data.get("unsupported_relation", False))
    unsupported_concepts_raw = data.get("unsupported_concepts", []) or []
    unsupported_concepts = [str(x) for x in unsupported_concepts_raw if str(x).strip()]

    evidence_map_raw = data.get("evidence_map") or {}
    evidence_map: Dict[str, List[Dict[str, str]]] = {}
    if isinstance(evidence_map_raw, dict):
        for concept, items in evidence_map_raw.items():
            if not isinstance(items, list):
                continue
            cleaned: List[Dict[str, str]] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                cleaned.append({
                    "fact_id": str(it.get("fact_id", "")),
                    "quote": str(it.get("quote", "")),
                })
            evidence_map[str(concept)] = cleaned

    link_raw = data.get("linking_fact_id")
    linking_fact_id = str(link_raw) if link_raw not in (None, "", "null") else None

    fact_by_id = {f.fact_id: (f.canonical_form or f.text) for f in facts}

    def concepts_grounded_in_fact(fid: str) -> set[str]:
        grounded: set[str] = set()
        for concept, items in evidence_map.items():
            for item in items:
                if item.get("fact_id", "") == fid and item.get("quote", "").strip():
                    grounded.add(concept)
                    break
        return grounded

    # 1. LLM-asserted unsupported relation.
    if unsupported_relation:
        return False, "unsupported_relation", evidence_map, linking_fact_id, unsupported_concepts, True, relation_type

    # 2. LLM-asserted unsupported concepts.
    if unsupported_concepts:
        return (
            False,
            f"unsupported_concepts:{len(unsupported_concepts)}",
            evidence_map, linking_fact_id, unsupported_concepts, False, relation_type,
        )

    # 3. Risky-frame guard: keep this before the generic missing-link reject so
    # the histogram preserves the more specific failure mode.
    if risky_frame and relation_type in RELATIONAL_REQUIRES_LINK and not linking_fact_id:
        return (
            False, f"risky_frame_no_link:{risky_frame}",
            evidence_map, linking_fact_id, unsupported_concepts, False, relation_type,
        )

    # 4. Linking fact required for relational types.
    if relation_type in RELATIONAL_REQUIRES_LINK:
        if not linking_fact_id or linking_fact_id not in fact_by_id:
            return (
                False, "missing_linking_fact",
                evidence_map, linking_fact_id, unsupported_concepts, False, relation_type,
            )

    # 5. Substring quote verification — independent of LLM self-assessment.
    for concept, items in evidence_map.items():
        for item in items:
            fid = item.get("fact_id", "")
            quote = item.get("quote", "").strip()
            txt = fact_by_id.get(fid, "")
            if not quote:
                return (
                    False, f"empty_quote:{concept}:{fid}",
                    evidence_map, linking_fact_id, unsupported_concepts, False, relation_type,
                )
            if not txt:
                return (
                    False, f"unknown_fact_id:{concept}:{fid}",
                    evidence_map, linking_fact_id, unsupported_concepts, False, relation_type,
                )
            if quote.lower() not in txt.lower():
                return (
                    False, f"quote_mismatch:{concept}:{fid}",
                    evidence_map, linking_fact_id, unsupported_concepts, False, relation_type,
                )

    # 6. Linking fact must do more than exist. For comparison/contrast/
    # intersection, it must ground at least two distinct endpoint concepts.
    # For causal/mechanism questions, a single concept can be acceptable only
    # when the linking fact carries an explicit causal/mechanism lexical cue.
    if relation_type in RELATIONAL_REQUIRES_LINK and linking_fact_id:
        grounded = concepts_grounded_in_fact(linking_fact_id)
        linking_text = f" {fact_by_id.get(linking_fact_id, '').lower()} "
        if relation_type in {"comparison", "contrast", "intersection"}:
            if len(grounded) < 2:
                return (
                    False, "linking_fact_lacks_endpoints",
                    evidence_map, linking_fact_id, unsupported_concepts, False, relation_type,
                )
        elif relation_type == "causal":
            has_cue = any(cue in linking_text for cue in _CAUSE_CUES)
            if len(grounded) < 2 and not has_cue:
                return (
                    False, "linking_fact_lacks_causal_cue",
                    evidence_map, linking_fact_id, unsupported_concepts, False, relation_type,
                )
        elif relation_type == "mechanism":
            has_cue = any(cue in linking_text for cue in _MECHANISM_CUES)
            if len(grounded) < 2 and not has_cue:
                return (
                    False, "linking_fact_lacks_mechanism_cue",
                    evidence_map, linking_fact_id, unsupported_concepts, False, relation_type,
                )

    return True, "accepted", evidence_map, linking_fact_id, unsupported_concepts, False, relation_type


# ---------- cache ----------


_cfg = load_config()
_QRSG_CACHE_PATH = Path(
    os.getenv("QRSG_CACHE_PATH")
    or _cfg.get("v13_qrsg", {}).get("cache_path")
    or str(repo_root() / "data" / "cache" / "qrsg_cache.db")
)

_conn_lock = threading.Lock()
_thread_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    conn = getattr(_thread_local, "conn", None)
    if conn is not None:
        return conn
    _QRSG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_QRSG_CACHE_PATH), check_same_thread=False, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS qrsg_cache "
        "(key TEXT PRIMARY KEY, verdict_json TEXT, ts REAL)"
    )
    conn.commit()
    _thread_local.conn = conn
    return conn


def _model_id_of(llm: "LLM") -> str:
    for attr in ("model", "model_name", "model_id"):
        val = getattr(llm, attr, None)
        if val:
            return str(val)
    return llm.__class__.__name__


def _cache_key(
    question: str, facts: List[Fact], model_id: str, gate_version: str
) -> str:
    fact_payload = [
        {
            "fact_id": f.fact_id,
            "text_sha1": hashlib.sha1(
                (f.canonical_form or f.text).encode("utf-8")
            ).hexdigest(),
        }
        for f in sorted(facts, key=lambda x: x.fact_id)
    ]
    payload = json.dumps(
        {
            "gate_version": gate_version,
            "model_id": model_id,
            "question": question,
            "facts": fact_payload,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT verdict_json FROM qrsg_cache WHERE key=?", (key,)
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def _cache_set(key: str, verdict: RelationVerdict) -> None:
    conn = _get_conn()
    payload = json.dumps(
        {
            **verdict.to_dict(),
            "llm_raw": verdict.llm_raw,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    with _conn_lock:
        conn.execute(
            "INSERT OR REPLACE INTO qrsg_cache (key, verdict_json, ts) VALUES (?, ?, ?)",
            (key, payload, time.time()),
        )
        conn.commit()


def _verdict_from_cache(d: dict) -> RelationVerdict:
    return RelationVerdict(
        accepted=bool(d.get("accepted", False)),
        relation_type=str(d.get("relation_type", "descriptive")),
        unsupported_concepts=list(d.get("unsupported_concepts", []) or []),
        unsupported_relation=bool(d.get("unsupported_relation", False)),
        evidence_map=dict(d.get("evidence_map", {}) or {}),
        linking_fact_id=d.get("linking_fact_id"),
        reason=str(d.get("reason", "")),
        risky_frame_hit=d.get("risky_frame_hit"),
        llm_raw=d.get("llm_raw"),
    )


# ---------- public API ----------


def relation_support_gate(
    question: str,
    facts: List[Fact],
    llm: "LLM",
    *,
    gate_version: str = GATE_VERSION,
    max_tokens: int = 2048,
    use_cache: bool = True,
) -> RelationVerdict:
    """Single-LLM-call gate. Always returns a RelationVerdict (never raises
    for routine errors — parse failures and LLM transients reject with a
    reason)."""

    if not question or not question.strip():
        return RelationVerdict(
            accepted=False, relation_type="descriptive",
            unsupported_concepts=[], unsupported_relation=False,
            evidence_map={}, linking_fact_id=None,
            reason="empty_question", risky_frame_hit=None, llm_raw=None,
        )
    if not facts:
        return RelationVerdict(
            accepted=False, relation_type="descriptive",
            unsupported_concepts=[], unsupported_relation=False,
            evidence_map={}, linking_fact_id=None,
            reason="no_facts", risky_frame_hit=None, llm_raw=None,
        )

    risky_frame = _pre_filter_risky_frame(question)

    # Cache lookup
    model_id = _model_id_of(llm)
    key = _cache_key(question, facts, model_id, gate_version) if use_cache else ""
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return _verdict_from_cache(cached)

    prompt = _build_prompt(question, facts)

    # LLM call. APIError is unrecoverable (auth, etc.) — surface it; everything
    # else degrades to a structured reject so the pipeline can keep moving.
    from rag_gt.core.llm import APIError

    try:
        raw = llm.generate(prompt, temperature=0.0, max_tokens=max_tokens)
    except APIError:
        raise
    except Exception as e:
        logger.warning(f"[QRSG] LLM transient error: {type(e).__name__}: {e}")
        verdict = RelationVerdict(
            accepted=False, relation_type="descriptive",
            unsupported_concepts=[], unsupported_relation=False,
            evidence_map={}, linking_fact_id=None,
            reason=f"llm_error:{type(e).__name__}",
            risky_frame_hit=risky_frame, llm_raw=None,
        )
        if use_cache:
            _cache_set(key, verdict)
        return verdict

    raw_text = (raw or "").strip()

    try:
        data = json.loads(repair_json(_extract_json(raw_text)))
    except Exception as e:
        logger.warning(f"[QRSG] JSON parse failed: {type(e).__name__}: {e}; raw={raw_text[:200]!r}")
        verdict = RelationVerdict(
            accepted=False, relation_type="descriptive",
            unsupported_concepts=[], unsupported_relation=False,
            evidence_map={}, linking_fact_id=None,
            reason="parse_error", risky_frame_hit=risky_frame, llm_raw=raw_text,
        )
        if use_cache:
            _cache_set(key, verdict)
        return verdict

    if not isinstance(data, dict):
        verdict = RelationVerdict(
            accepted=False, relation_type="descriptive",
            unsupported_concepts=[], unsupported_relation=False,
            evidence_map={}, linking_fact_id=None,
            reason="non_object_json", risky_frame_hit=risky_frame, llm_raw=raw_text,
        )
        if use_cache:
            _cache_set(key, verdict)
        return verdict

    accepted, reason, evidence_map, linking_fact_id, unsupported_concepts, unsupported_relation, relation_type = _verify(
        data, facts, risky_frame
    )

    verdict = RelationVerdict(
        accepted=accepted,
        relation_type=relation_type,
        unsupported_concepts=unsupported_concepts,
        unsupported_relation=unsupported_relation,
        evidence_map=evidence_map,
        linking_fact_id=linking_fact_id,
        reason=reason,
        risky_frame_hit=risky_frame,
        llm_raw=raw_text,
    )
    if use_cache:
        _cache_set(key, verdict)
    return verdict


def batch_relation_support_gate(
    items: List[Tuple[str, List[Fact]]],
    llm: "LLM",
    *,
    gate_version: str = GATE_VERSION,
    max_tokens: int = 2048,
    use_cache: bool = True,
) -> List[RelationVerdict]:
    """Sequential batch. Concurrency is the caller's problem — the pipeline
    already runs inside a ThreadPoolExecutor."""
    return [
        relation_support_gate(
            q,
            f,
            llm,
            gate_version=gate_version,
            max_tokens=max_tokens,
            use_cache=use_cache,
        )
        for q, f in items
    ]


__all__ = [
    "GATE_VERSION",
    "RELATION_TYPES",
    "RELATIONAL_REQUIRES_LINK",
    "RELATIONAL_NO_LINK_NEEDED",
    "RISKY_FRAMES",
    "RelationVerdict",
    "relation_support_gate",
    "batch_relation_support_gate",
]
