"""Layer-2 NLI anti-leakage check. Batched + SQLite-cached.

Always look up the entailment label index via `NLI_LABEL_INDEX["entailment"]` —
never hard-code slot 1. Different cross-encoder NLI checkpoints use different
orderings, and `core.models._verify_nli_labels` already proves the resolved
indices match a known entailment + contradiction example.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from rag_gt.core.config import load_config, repo_root
from rag_gt.core.models import MM, NLI_LABEL_INDEX, NLI_MODEL_NAME
from rag_gt.core.types import Fact

_cfg = load_config()
NLI_THRESHOLD = _cfg["validation"]["nli_entailment_threshold"]
GT_CONSISTENCY_THRESHOLD = _cfg["validation"]["nli_gt_consistency_threshold"]
BATCH_SIZE = int(_cfg["validation"].get("nli_batch_size", 16))

# Truncation limits applied to the *model input*. The cache key hashes the
# FULL premise+hypothesis so two long pairs with the same prefix do not collide.
PREMISE_MAX = 1200
HYPOTHESIS_MAX = 400

_CACHE_PATH = Path(
    os.getenv("NLI_CACHE_PATH")
    or str(repo_root() / "data" / "cache" / "nli_cache.db")
)

_conn_lock = threading.Lock()
_thread_local = threading.local()

# The shared CrossEncoder is NOT safe under concurrent .predict() calls. Extraction
# (Stage 3) now runs chunk workers in parallel, and canonical_form_rewrite calls
# nli_batch from those threads, so every model inference is serialized behind this
# lock (BUG-I). The lock is uncontended in single-threaded callers, so it is
# effectively free there; under threads the local predict is fast and threads spend
# most of their time on network LLM calls anyway.
_model_lock = threading.Lock()

# BUG-J: premise/hypothesis truncation is silent; a joint premise of two long
# facts can exceed PREMISE_MAX so its entailment is scored on a prefix. Count the
# events so a run can report them instead of hiding the loss.
_trunc_lock = threading.Lock()
_trunc_stats = {"premise_truncated": 0, "hypothesis_truncated": 0, "total_inputs": 0}


def truncation_stats() -> dict:
    """Return a copy of the running NLI-input truncation counters (BUG-J)."""
    with _trunc_lock:
        return dict(_trunc_stats)


def reset_truncation_stats() -> None:
    with _trunc_lock:
        for key in _trunc_stats:
            _trunc_stats[key] = 0


def _truncate(premise: str, hypothesis: str) -> tuple[str, str]:
    """Truncate to model limits and record truncation events (BUG-J)."""
    p_over = len(premise) > PREMISE_MAX
    h_over = len(hypothesis) > HYPOTHESIS_MAX
    with _trunc_lock:
        _trunc_stats["total_inputs"] += 1
        if p_over:
            _trunc_stats["premise_truncated"] += 1
        if h_over:
            _trunc_stats["hypothesis_truncated"] += 1
    return premise[:PREMISE_MAX], hypothesis[:HYPOTHESIS_MAX]


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection (created on first use)."""
    conn = getattr(_thread_local, "conn", None)
    if conn is not None:
        return conn
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_CACHE_PATH), check_same_thread=False, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS nli_cache "
        "(key TEXT PRIMARY KEY, score REAL, ts REAL)"
    )
    conn.commit()
    _thread_local.conn = conn
    return conn


def _cache_key(premise: str, hypothesis: str) -> str:
    """Hash the FULL premise + hypothesis. Truncation is for the model, not the key."""
    payload = json.dumps(
        {
            "model": NLI_MODEL_NAME,
            "premise": premise,
            "hypothesis": hypothesis,
            "premise_len": len(premise),
            "hypothesis_len": len(hypothesis),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_cached(conn: sqlite3.Connection, key: str) -> Optional[float]:
    row = conn.execute("SELECT score FROM nli_cache WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _set_cached(conn: sqlite3.Connection, key: str, score: float) -> None:
    with _conn_lock:
        conn.execute(
            "INSERT OR REPLACE INTO nli_cache (key, score, ts) VALUES (?, ?, ?)",
            (key, score, time.time()),
        )
        conn.commit()


def _set_cached_many(conn: sqlite3.Connection, rows: List[Tuple[str, float]]) -> None:
    """Batch-write cache rows with a single commit (BUG-I).

    A cold nli_batch used to commit once per score under the global lock; this
    replaces thousands of commits per run with one commit per predict batch.
    """
    if not rows:
        return
    now = time.time()
    with _conn_lock:
        conn.executemany(
            "INSERT OR REPLACE INTO nli_cache (key, score, ts) VALUES (?, ?, ?)",
            [(key, score, now) for key, score in rows],
        )
        conn.commit()


def _entailment_index() -> int:
    """Resolve the entailment label index, loading the NLI model if needed."""
    if "entailment" not in NLI_LABEL_INDEX:
        MM.load_nli()  # populates NLI_LABEL_INDEX
    idx = NLI_LABEL_INDEX.get("entailment")
    if idx is None:
        raise RuntimeError("NLI label index for 'entailment' is unknown.")
    return int(idx)


def nli_entailment(premise: str, hypothesis: str) -> float:
    key = _cache_key(premise, hypothesis)
    conn = _get_conn()
    cached = _get_cached(conn, key)
    if cached is not None:
        return cached
    model = MM.get_nli()
    ent_idx = _entailment_index()
    truncated = _truncate(premise, hypothesis)
    with _model_lock:
        score = model.predict([truncated], apply_softmax=True)[0]
    entailment_score = float(score[ent_idx])
    _set_cached(conn, key, entailment_score)
    return entailment_score


def nli_batch(pairs: List[Tuple[str, str]]) -> List[float]:
    if not pairs:
        return []
    conn = _get_conn()
    scores_by_index: List[Optional[float]] = [None] * len(pairs)

    # Deduplicate identical (premise, hypothesis) inputs so a repeated pair is
    # predicted (and cached) once per call (BUG-I). Stage C/D inputs contain
    # repeats (e.g. the same fact text scored against several clauses).
    key_to_indices: dict[str, List[int]] = {}
    key_to_input: dict[str, Tuple[str, str]] = {}
    for i, (premise, hypothesis) in enumerate(pairs):
        key = _cache_key(premise, hypothesis)
        cached = _get_cached(conn, key)
        if cached is not None:
            scores_by_index[i] = cached
            continue
        key_to_indices.setdefault(key, []).append(i)
        if key not in key_to_input:
            key_to_input[key] = _truncate(premise, hypothesis)

    if key_to_input:
        model = MM.get_nli()
        ent_idx = _entailment_index()
        unique_keys = list(key_to_input.keys())
        unique_inputs = [key_to_input[k] for k in unique_keys]
        for batch_start in range(0, len(unique_inputs), BATCH_SIZE):
            batch = unique_inputs[batch_start : batch_start + BATCH_SIZE]
            with _model_lock:
                scores = model.predict(batch, apply_softmax=True)
            write_rows: List[Tuple[str, float]] = []
            for j, score_arr in enumerate(scores):
                key = unique_keys[batch_start + j]
                s = float(score_arr[ent_idx])
                for idx in key_to_indices[key]:
                    scores_by_index[idx] = s
                write_rows.append((key, s))
            _set_cached_many(conn, write_rows)

    # Every index is filled: cached or freshly predicted.
    return [0.0 if s is None else s for s in scores_by_index]


def nli_contradiction(premise: str, hypothesis: str) -> float:
    """Return the NLI contradiction score for (premise, hypothesis).

    Not cached — intended for low-volume CT twin checks (~100 per run).
    Loads the NLI model on first call (same instance as nli_entailment).
    """
    from rag_gt.core.models import MM, NLI_LABEL_INDEX

    if "contradiction" not in NLI_LABEL_INDEX:
        MM.load_nli()
    model = MM.get_nli()
    contr_idx = NLI_LABEL_INDEX.get("contradiction")
    if contr_idx is None:
        raise RuntimeError("NLI label index for 'contradiction' is unknown.")
    truncated = _truncate(premise, hypothesis)
    with _model_lock:
        scores = model.predict([truncated], apply_softmax=True)[0]
    return float(scores[int(contr_idx)])


def check_answer_entailment(answer: str, facts: List[Fact]) -> bool:
    if "insufficient information" in answer.lower():
        return True
    context = " ".join(f.text for f in facts)
    return nli_entailment(context, answer) >= GT_CONSISTENCY_THRESHOLD


def atomic_clause_entailment(
    clauses: List[str],
    support_texts: List[str],
    threshold: Optional[float] = None,
) -> List[Tuple[bool, float, int]]:
    """For each clause, return (passes, best_score, best_support_index).

    Phase 2 (plan v6) — used by the Constructive Gold Answer (CGA) guard.
    A clause "passes" iff at least one of `support_texts` NLI-entails it at
    or above `threshold` (default: `GT_CONSISTENCY_THRESHOLD`). The best
    support index lets callers report which SFU grounded the clause; -1
    means no support reached threshold.

    Uses the existing batched `nli_batch` cache so repeated checks across
    candidate gold answers are cheap.
    """
    if not clauses or not support_texts:
        return [(False, 0.0, -1) for _ in clauses]
    if threshold is None:
        threshold = GT_CONSISTENCY_THRESHOLD

    pairs: List[Tuple[str, str]] = []
    for clause in clauses:
        for support in support_texts:
            pairs.append((support, clause))

    scores = nli_batch(pairs)
    out: List[Tuple[bool, float, int]] = []
    n_support = len(support_texts)
    for i, _clause in enumerate(clauses):
        clause_scores = scores[i * n_support : (i + 1) * n_support]
        if not clause_scores:
            out.append((False, 0.0, -1))
            continue
        best_idx = max(range(n_support), key=lambda j: clause_scores[j])
        best_score = float(clause_scores[best_idx])
        out.append((best_score >= threshold, best_score, best_idx))
    return out


def batch_answer_entailment(
    pairs: List[Tuple[str, List[Fact]]],
) -> List[bool]:
    """Batched version: `pairs` is list of (answer, facts) two-tuples.

    (Renamed parameter from `triples` to `pairs` to match the actual arity;
    the alias `batch_answer_entailment(triples=...)` is no longer accepted.)
    """
    inputs: List[Tuple[str, str]] = []
    predict_indices: List[int] = []
    out: List[bool] = [True] * len(pairs)
    for i, (answer, facts) in enumerate(pairs):
        if not answer or "insufficient information" in answer.lower():
            continue
        context = " ".join(f.text for f in facts)
        inputs.append((context, answer))
        predict_indices.append(i)
    scores = nli_batch(inputs)
    for i, score in zip(predict_indices, scores):
        out[i] = score >= GT_CONSISTENCY_THRESHOLD
    return out
