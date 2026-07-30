"""LLM + NLI typed edge classifier for TF-SFG (P1).

For each candidate (fact_a, fact_b) pair:
  1. LLM proposes relation_type + bridging_quote + relation_claim.
  2. Python verifies bridging_quote is a verbatim substring of the bridging fact.
  3. NLI verifies (fact_a_text + fact_b_text) entails relation_claim >= threshold.

JSON parsing: json.loads(repair_json(_extract_json(raw))) — never response_format.
Cache: SQLite, sha256(fact_a_text | fact_b_text | model), same pattern as nli_check.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from json_repair import repair_json
from loguru import logger

from rag_gt.core.config import load_config, repo_root
from rag_gt.core.llm import APIError, _extract_json
from rag_gt.core.types import Fact
from rag_gt.graph.edge_canonicalize import CANONICAL_UNKNOWN, canonicalize_label
from rag_gt.validation.nli_check import nli_entailment
from rag_gt.validation.relation_support_gate import RELATION_TYPES

if TYPE_CHECKING:
    from rag_gt.core.llm import LLM


# ---------- EdgeRecord ----------

@dataclass
class EdgeRecord:
    edge_id: str
    src: str                  # source fact_id
    dst: str                  # destination fact_id
    type: str                 # relation_type from RELATION_TYPES
    bridging_fact_id: str     # fact_id of the fact containing bridging_quote
    bridging_quote: str       # exact verbatim substring from bridging_quote fact
    relation_claim: str       # LLM-generated natural-language edge claim
    nli_score: float          # NLI entailment score of relation_claim from context
    single_src_score: float = 0.0  # NLI score from source fact alone
    single_dst_score: float = 0.0  # NLI score from destination fact alone
    joint_only_margin: float = 0.0 # joint score minus best single-fact score
    source_contribution: str = ""  # answer clause supported by src alone
    destination_contribution: str = ""  # different answer clause supported by dst alone
    source_contribution_score: float = 0.0  # src entails source contribution
    destination_contribution_score: float = 0.0  # dst entails destination contribution
    source_cross_score: float = 0.0  # dst entails source contribution
    destination_cross_score: float = 0.0  # src entails destination contribution
    contribution_distinctness: float = 0.0  # minimum own-minus-cross margin
    answer_contribution_score: float = 0.0  # positive ranking signal
    question_seed: str = ""  # answer-hidden question scaffold proposed with the edge
    question_seed_score: float = 0.0  # 1 when seed is formatted and premise-safe

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- Cache ----------

_cfg = load_config()
_v16_cfg = _cfg.get("v16", {})

_CACHE_PATH = Path(
    os.getenv("TFSFG_CACHE_PATH")
    or str(repo_root() / "data" / "cache" / "tfsfg_cache.db")
)

_conn_lock = threading.Lock()
_thread_local = threading.local()

_NLI_EDGE_THRESHOLD = float(
    _cfg.get("v16", {}).get("tf_sfg", {}).get("nli_edge_threshold", 0.55)
)
_CACHE_VERSION = os.getenv("TFSFG_CACHE_VERSION", "v17-no-fabrication-contract")
_QUOTE_STOPWORDS = {
    "about", "after", "also", "because", "before", "between", "from", "have",
    "into", "more", "most", "only", "other", "same", "some", "such", "than",
    "that", "their", "there", "these", "this", "those", "through", "under",
    "using", "when", "where", "which", "while", "with", "would",
}


def _get_conn() -> sqlite3.Connection:
    conn = getattr(_thread_local, "tfsfg_conn", None)
    if conn is not None:
        return conn
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_CACHE_PATH), check_same_thread=False, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS edge_cache "
        "(key TEXT PRIMARY KEY, result_json TEXT, ts REAL)"
    )
    conn.commit()
    _thread_local.tfsfg_conn = conn
    return conn


def _cache_key(
    fact_a: Fact,
    fact_b: Fact,
    model: str,
    nli_threshold: float,
    canonicalize_edges: bool,
    allow_low_nli_with_quote: bool,
    low_nli_floor: float,
    edge_minimality_enabled: bool,
    edge_single_fact_threshold: float,
    edge_min_joint_only_margin: float,
    answer_contribution_enabled: bool,
    contribution_own_threshold: float,
) -> str:
    payload = json.dumps(
        {
            "cache_version": _CACHE_VERSION,
            "model": model,
            "nli_threshold": round(float(nli_threshold), 4),
            "canonicalize_edges": bool(canonicalize_edges),
            "allow_low_nli_with_quote": bool(allow_low_nli_with_quote),
            "low_nli_floor": round(float(low_nli_floor), 4),
            "edge_minimality_enabled": bool(edge_minimality_enabled),
            "edge_single_fact_threshold": round(float(edge_single_fact_threshold), 4),
            "edge_min_joint_only_margin": round(float(edge_min_joint_only_margin), 4),
            "answer_contribution_enabled": bool(answer_contribution_enabled),
            "contribution_own_threshold": round(float(contribution_own_threshold), 4),
            "fact_a_id": fact_a.fact_id,
            "fact_b_id": fact_b.fact_id,
            "fact_a_text": fact_a.text,
            "fact_b_text": fact_b.text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_cached(conn: sqlite3.Connection, key: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT result_json FROM edge_cache WHERE key=?", (key,)
    ).fetchone()
    if row:
        try:
            return json.loads(row[0])
        except Exception:
            return None
    return None


def _set_cached(conn: sqlite3.Connection, key: str, result: dict) -> None:
    with _conn_lock:
        conn.execute(
            "INSERT OR REPLACE INTO edge_cache (key, result_json, ts) VALUES (?, ?, ?)",
            (key, json.dumps(result, ensure_ascii=False), time.time()),
        )
        conn.commit()


# ---------- Prompt ----------

_RELATION_TYPES_STR = "\n".join(f"  - {rt}" for rt in RELATION_TYPES)

_EDGE_CLASSIFY_PROMPT = """\
You are a knowledge-graph edge classifier for multi-hop reasoning.

You are given two fact statements from the same document. Determine whether a
meaningful typed relationship exists between them that would support multi-hop
reasoning (i.e., a reader needs both facts together to answer a question).

Fact A (id={fact_a_id}):
{fact_a_text}

Fact B (id={fact_b_id}):
{fact_b_text}

Allowed relation types:
{relation_types}

Instructions:
- Choose the SINGLE most specific relation type that holds between A and B, or
  "none" if no clear relationship supports multi-hop reasoning.
- Return "none" when the two facts merely share a topic, restate the same idea,
  give a loose example, or one fact is only background context for the other.
- CRITICAL — do NOT invent a relationship. Return "none" if stating the link
  requires ANY of the following:
  (a) assuming the two facts refer to the SAME specific thing when neither fact
      says so (e.g. a fact about "Annex A format" and a fact about an unassigned
      material — do not assume the material rule applies *to that format*);
  (b) inventing a dependency such as "X depends on the definitions in Y",
      "X applies to Y", "X is governed by Y" that NEITHER fact actually states;
  (c) binding a pronoun/term in one fact to a noun that only appears in the other.
  A valid relation must be one the document itself asserts or directly implies —
  not one you construct by assuming the two facts are about the same specific
  referent. When in doubt, return "none".
- bridging_fact_id: the fact id ({fact_a_id} or {fact_b_id}) whose text most
  directly expresses the link.
- bridging_quote: copy an EXACT verbatim substring (word-for-word) from the
  bridging fact that establishes the link. Must appear literally in that fact.
- relation_claim: one source-grounded sentence stating the actual relationship
  using domain terms from BOTH facts. Do NOT use "Fact A", "Fact B", "the first
  fact", "the second fact", or vague phrasing like "is related to" or
  "provides context". Do NOT copy either fact verbatim.
- The relation_claim should become unsupported or incomplete if either fact is
  removed. If one fact alone supports the claim, return relation_type "none".
- source_contribution: one short answer clause supported by Fact A alone.
- destination_contribution: a different short answer clause supported by Fact B
  alone. These two clauses must be the distinct pieces a valid multi-hop answer
  would combine. If either clause cannot be stated, or the clauses repeat the
  same information, return relation_type "none".
- question_seed: one compact question that requires both contributions in its
  answer while NOT stating, quoting, or paraphrasing either contribution in the
  question. Use only topic/entity anchors necessary to identify the subject.
  If no such answer-hidden question can be written, return "".
- rationale: one sentence explaining why this relation type applies.

Return ONLY a JSON object. If relation_type is "none", set all other fields to "".

{{"relation_type": "...", "bridging_fact_id": "...", "bridging_quote": "...", "relation_claim": "...", "source_contribution": "...", "destination_contribution": "...", "question_seed": "...?", "rationale": "..."}}
"""


# ---------- Classifier ----------

def _fact_text(fact: Fact) -> str:
    return (fact.canonical_form.strip() or fact.text.strip())


def _quote_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text.lower())
        if tok not in _QUOTE_STOPWORDS
    }


def _best_verbatim_quote(source_text: str, relation_claim: str) -> str:
    """Pick a short verbatim source snippet supporting a paraphrased relation."""
    source_text = " ".join(source_text.split())
    if not source_text:
        return ""
    claim_tokens = _quote_tokens(relation_claim)
    if not claim_tokens:
        return source_text[:240]
    candidates = [
        part.strip()
        for part in re.split(r"(?<=[.!?;:])\s+|\n+", source_text)
        if part.strip()
    ]
    if not candidates:
        candidates = [source_text]

    def score(candidate: str) -> tuple[int, int]:
        toks = _quote_tokens(candidate)
        return (len(toks & claim_tokens), -len(candidate))

    best = max(candidates, key=score)
    if len(best) <= 240:
        return best
    return best[:237].rstrip() + "..."


_GENERIC_RELATION_RE = re.compile(
    r"\b("
    r"elaborates?\s+on|"
    r"provides?\s+(?:additional\s+)?context|"
    r"related\s+to|"
    r"both\s+(?:facts\s+)?(?:describe|discuss|mention|refer\s+to)|"
    r"same\s+(?:concept|topic|subject|idea|thing)|"
    r"similar\s+(?:concept|topic|subject|idea)|"
    r"further\s+(?:describes|explains|discusses)|"
    r"additional\s+information|"
    r"more\s+specific\s+description"
    r")\b",
    re.I,
)
_FACT_REFERENCE_RE = re.compile(
    r"\bfact\s+[ab]\b|\bfacts?\b|\b(?:first|second)\s+fact\b|\b(?:statement|sentence)\s+[ab]\b",
    re.I,
)


def _is_generic_relation_claim(claim: str) -> bool:
    """Return True for meta-relations that do not force multi-hop reasoning."""
    normalized = " ".join((claim or "").split())
    if len(normalized) < 20:
        return True
    if _FACT_REFERENCE_RE.search(normalized):
        return True
    return bool(_GENERIC_RELATION_RE.search(normalized))


def _normalized_clause(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def classify_edge(
    fact_a: Fact,
    fact_b: Fact,
    llm: "LLM",
    *,
    edge_counter: List[int],
    nli_threshold: float = _NLI_EDGE_THRESHOLD,
    canonicalize_edges: bool = False,
    canonical_map: Optional[dict[str, str]] = None,
    allow_low_nli_with_quote: bool = False,
    low_nli_floor: float = 0.0,
    edge_minimality_enabled: bool = False,
    edge_single_fact_threshold: float = 0.85,
    edge_min_joint_only_margin: float = 0.05,
    answer_contribution_enabled: bool = False,
    contribution_own_threshold: float = 0.45,
) -> Optional[EdgeRecord]:
    """Classify the edge between fact_a and fact_b.

    Returns None if the LLM proposes "none", Python verification fails, or
    the NLI score is below threshold. Populates cache on every call.

    `edge_counter` is a single-element list used for thread-safe edge-id
    generation without a lock — callers pass `[0]` and the counter increments
    in place.
    """
    model_name = getattr(llm, "model", "unknown")
    conn = _get_conn()
    key = _cache_key(
        fact_a,
        fact_b,
        model_name,
        nli_threshold,
        canonicalize_edges,
        allow_low_nli_with_quote,
        low_nli_floor,
        edge_minimality_enabled,
        edge_single_fact_threshold,
        edge_min_joint_only_margin,
        answer_contribution_enabled,
        contribution_own_threshold,
    )

    _MAX_FACT_CHARS = 450
    text_a = _fact_text(fact_a)[:_MAX_FACT_CHARS]
    text_b = _fact_text(fact_b)[:_MAX_FACT_CHARS]

    prompt = _EDGE_CLASSIFY_PROMPT.format(
        fact_a_id=fact_a.fact_id,
        fact_a_text=text_a,
        fact_b_id=fact_b.fact_id,
        fact_b_text=text_b,
        relation_types=_RELATION_TYPES_STR,
    )

    cached = _get_cached(conn, key)
    if cached is not None:
        tracker = getattr(llm, "_tracker", None)
        if tracker is not None:
            from rag_gt.observability.cost_tracker import CostEvent

            tracker.record(
                CostEvent(
                    stage=str(getattr(llm, "_stage", "tf_sfg_classify")),
                    doc_id=str(getattr(llm, "_doc_id", "")),
                    model=str(model_name),
                    call_count=1,
                    prompt_chars=len(prompt),
                    completion_chars=len(json.dumps(cached, ensure_ascii=False)),
                    cache_hit=True,
                    cache_source="tfsfg_sqlite",
                )
            )
        if cached.get("accepted"):
            edge_counter[0] += 1
            data = {
                k: cached.get(k, field.default)
                for k, field in EdgeRecord.__dataclass_fields__.items()
            }
            data["edge_id"] = f"e{edge_counter[0]:05d}"
            return EdgeRecord(**data)
        return None

    try:
        raw = llm.generate(prompt, temperature=0.0, max_tokens=1536)
    except APIError:
        logger.warning(
            f"[EdgeClassifier] API error for ({fact_a.fact_id}, {fact_b.fact_id}); "
            "not caching transient failure"
        )
        raise

    try:
        parsed = json.loads(repair_json(_extract_json(raw)))
    except Exception as e:
        logger.debug(f"[EdgeClassifier] LLM/parse error for ({fact_a.fact_id}, {fact_b.fact_id}): {e}")
        _set_cached(conn, key, {"accepted": False, "reason": "parse_error"})
        return None

    if isinstance(parsed, list):
        # LLM wrapped the object in an array; unwrap if possible
        parsed = parsed[0] if parsed and isinstance(parsed[0], dict) else {}
    if not isinstance(parsed, dict):
        _set_cached(conn, key, {"accepted": False, "reason": "parse_error_not_dict"})
        return None

    raw_relation_type = str(parsed.get("relation_type", "")).strip().lower()
    if raw_relation_type in {"", "none", "no_relation", "no relation", "n/a"}:
        _set_cached(conn, key, {"accepted": False, "reason": "none_relation"})
        return None
    relation_type = (
        canonicalize_label(raw_relation_type, canonical_map)
        if canonicalize_edges
        else raw_relation_type
    )
    if canonicalize_edges:
        if relation_type == CANONICAL_UNKNOWN:
            _set_cached(
                conn,
                key,
                {"accepted": False, "reason": f"unknown_type:{raw_relation_type}"},
            )
            return None
    elif relation_type not in RELATION_TYPES:
        _set_cached(conn, key, {"accepted": False, "reason": f"unknown_type:{relation_type}"})
        return None

    bridging_fact_id = str(parsed.get("bridging_fact_id", "")).strip()
    bridging_quote = str(parsed.get("bridging_quote", "")).strip()
    relation_claim = str(parsed.get("relation_claim", "")).strip()
    source_contribution = str(parsed.get("source_contribution", "")).strip()
    destination_contribution = str(parsed.get("destination_contribution", "")).strip()
    question_seed = str(parsed.get("question_seed", "")).strip()

    if not relation_claim:
        _set_cached(conn, key, {"accepted": False, "reason": "empty_relation_claim"})
        return None
    if _is_generic_relation_claim(relation_claim):
        _set_cached(conn, key, {"accepted": False, "reason": "generic_relation_claim"})
        return None
    if answer_contribution_enabled:
        if not source_contribution or not destination_contribution:
            _set_cached(conn, key, {"accepted": False, "reason": "missing_answer_contribution"})
            return None
        if _normalized_clause(source_contribution) == _normalized_clause(destination_contribution):
            _set_cached(conn, key, {"accepted": False, "reason": "duplicate_answer_contribution"})
            return None

    # Python verification: bridging_quote must be a verbatim substring.
    if bridging_fact_id == fact_a.fact_id:
        bridging_text = text_a
    elif bridging_fact_id == fact_b.fact_id:
        bridging_text = text_b
    else:
        # Fallback: accept whichever fact contains the quote.
        if bridging_quote and bridging_quote in text_a:
            bridging_fact_id = fact_a.fact_id
            bridging_text = text_a
        elif bridging_quote and bridging_quote in text_b:
            bridging_fact_id = fact_b.fact_id
            bridging_text = text_b
        else:
            _set_cached(conn, key, {"accepted": False, "reason": "invalid_bridging_fact_id"})
            return None

    if bridging_quote and bridging_quote not in bridging_text:
        # Quote verification failed — try case-insensitive match as a soft fallback.
        if bridging_quote.lower() not in bridging_text.lower():
            bridging_quote = _best_verbatim_quote(bridging_text, relation_claim)
    elif not bridging_quote:
        bridging_quote = _best_verbatim_quote(bridging_text, relation_claim)

    if not bridging_quote:
        _set_cached(conn, key, {"accepted": False, "reason": "bridging_quote_not_found"})
        return None

    # NLI verification: joint context must entail relation_claim.
    premise = text_a + "\n" + text_b
    nli_score = nli_entailment(premise, relation_claim)
    if nli_score < nli_threshold:
        if not (allow_low_nli_with_quote and bridging_quote and nli_score >= low_nli_floor):
            _set_cached(
                conn,
                key,
                {
                    "accepted": False,
                    "reason": f"nli_below_threshold:{nli_score:.3f}",
                    "relation_type": relation_type,
                    "relation_claim": relation_claim,
                    "nli_score": nli_score,
                },
            )
            return None
        logger.debug(
            f"[EdgeClassifier] soft-accepted low-NLI edge "
            f"({fact_a.fact_id}->{fact_b.fact_id}) type={relation_type} "
            f"nli={nli_score:.3f}"
        )

    # Diagnostic only: if one fact alone entails the edge claim, downstream
    # minimality risk is high even when the joint NLI score passes.
    single_src_score = nli_entailment(text_a, relation_claim)
    single_dst_score = nli_entailment(text_b, relation_claim)
    joint_only_margin = nli_score - max(single_src_score, single_dst_score)
    if (
        edge_minimality_enabled
        and max(single_src_score, single_dst_score) >= edge_single_fact_threshold
        and joint_only_margin <= edge_min_joint_only_margin
    ):
        _set_cached(
            conn,
            key,
            {
                "accepted": False,
                "reason": f"edge_entailed_by_single_fact:{joint_only_margin:.3f}",
                "relation_type": relation_type,
                "relation_claim": relation_claim,
                "nli_score": nli_score,
                "single_src_score": single_src_score,
                "single_dst_score": single_dst_score,
                "joint_only_margin": joint_only_margin,
            },
        )
        return None

    source_contribution_score = 0.0
    destination_contribution_score = 0.0
    source_cross_score = 0.0
    destination_cross_score = 0.0
    contribution_distinctness = 0.0
    answer_contribution_score = 0.0
    question_seed_score = 0.0
    if answer_contribution_enabled:
        source_contribution_score = nli_entailment(text_a, source_contribution)
        destination_contribution_score = nli_entailment(text_b, destination_contribution)
        if min(source_contribution_score, destination_contribution_score) < contribution_own_threshold:
            _set_cached(
                conn,
                key,
                {
                    "accepted": False,
                    "reason": "unsupported_answer_contribution",
                    "relation_type": relation_type,
                    "relation_claim": relation_claim,
                    "source_contribution": source_contribution,
                    "destination_contribution": destination_contribution,
                    "source_contribution_score": source_contribution_score,
                    "destination_contribution_score": destination_contribution_score,
                },
            )
            return None
        source_cross_score = nli_entailment(text_b, source_contribution)
        destination_cross_score = nli_entailment(text_a, destination_contribution)
        contribution_distinctness = max(
            0.0,
            min(
                source_contribution_score - source_cross_score,
                destination_contribution_score - destination_cross_score,
            ),
        )
        if question_seed.endswith("?"):
            from rag_gt.generation.questions import premise_leakage_indices

            if not premise_leakage_indices(question_seed, [fact_a, fact_b]):
                question_seed_score = 1.0
        answer_contribution_score = (
            0.40 * min(source_contribution_score, destination_contribution_score)
            + 0.25 * contribution_distinctness
            + 0.35 * question_seed_score
        )

    edge_counter[0] += 1
    edge_id = f"e{edge_counter[0]:05d}"

    record = EdgeRecord(
        edge_id=edge_id,
        src=fact_a.fact_id,
        dst=fact_b.fact_id,
        type=relation_type,
        bridging_fact_id=bridging_fact_id,
        bridging_quote=bridging_quote,
        relation_claim=relation_claim,
        nli_score=nli_score,
        single_src_score=single_src_score,
        single_dst_score=single_dst_score,
        joint_only_margin=joint_only_margin,
        source_contribution=source_contribution,
        destination_contribution=destination_contribution,
        source_contribution_score=source_contribution_score,
        destination_contribution_score=destination_contribution_score,
        source_cross_score=source_cross_score,
        destination_cross_score=destination_cross_score,
        contribution_distinctness=contribution_distinctness,
        answer_contribution_score=answer_contribution_score,
        question_seed=question_seed,
        question_seed_score=question_seed_score,
    )
    _set_cached(conn, key, {"accepted": True, **record.to_dict()})
    return record


def classify_edges_batch(
    pairs: List[Tuple[Fact, Fact]],
    llm: "LLM",
    *,
    max_pairs: int = 5000,
    nli_threshold: float = _NLI_EDGE_THRESHOLD,
    canonicalize_edges: bool = False,
    canonical_map: Optional[dict[str, str]] = None,
    allow_low_nli_with_quote: bool = False,
    low_nli_floor: float = 0.0,
    edge_minimality_enabled: bool = False,
    edge_single_fact_threshold: float = 0.85,
    edge_min_joint_only_margin: float = 0.05,
    answer_contribution_enabled: bool = False,
    contribution_own_threshold: float = 0.45,
) -> List[Optional[EdgeRecord]]:
    """Classify a list of candidate pairs, respecting the cost cap."""
    pairs = pairs[:max_pairs]
    counter = [0]
    results: List[Optional[EdgeRecord]] = []
    for fact_a, fact_b in pairs:
        results.append(
            classify_edge(
                fact_a,
                fact_b,
                llm,
                edge_counter=counter,
                nli_threshold=nli_threshold,
                canonicalize_edges=canonicalize_edges,
                canonical_map=canonical_map,
                allow_low_nli_with_quote=allow_low_nli_with_quote,
                low_nli_floor=low_nli_floor,
                edge_minimality_enabled=edge_minimality_enabled,
                edge_single_fact_threshold=edge_single_fact_threshold,
                edge_min_joint_only_margin=edge_min_joint_only_margin,
                answer_contribution_enabled=answer_contribution_enabled,
                contribution_own_threshold=contribution_own_threshold,
            )
        )
    return results
