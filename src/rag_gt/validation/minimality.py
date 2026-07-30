"""Layer-3 minimality checks.

A (q, answer, facts) tuple passes minimality iff dropping any single fact
produces a materially different answer (fuzz.ratio < FUZZ_THRESHOLD).

For v16 PASS-GT rows, a stricter support-based minimality check is available:
dropping any required fact must not leave the original question assumptions and
gold answer still supported by the remaining facts.
"""

from __future__ import annotations

import threading
import unicodedata
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from rapidfuzz import fuzz

from rag_gt.core.config import load_config
from rag_gt.core.llm import LLM, model_identifier
from rag_gt.core.types import Fact
from rag_gt.generation.answers import generate_answer
from rag_gt.storage import cache as answer_cache
from rag_gt.validation.nli_check import batch_answer_entailment, nli_batch

_cfg = load_config()
FUZZ_THRESHOLD = _cfg["validation"]["fuzz_match_threshold"]
MAX_LOO_WORKERS = int(
    _cfg.get("performance", {}).get("max_concurrent_minimality_loo", 4)
)
PERSIST_CACHE = bool(_cfg.get("validation", {}).get("persist_answer_cache", True))


@dataclass
class RemovedFactSupport:
    fact_id: str
    question_supported_without: bool = False
    answer_entailed_without: bool = False
    has_question_contribution: bool = False
    has_reasoning_contribution: bool = False
    reason: str = ""


@dataclass
class SupportMinimalityVerdict:
    passed: bool
    reason: str = "accepted"
    removed: List[RemovedFactSupport] = field(default_factory=list)

    @property
    def redundant_fact_ids(self) -> List[str]:
        return [r.fact_id for r in self.removed if r.reason]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "redundant_fact_ids": self.redundant_fact_ids,
            "removed": [
                {
                    "fact_id": r.fact_id,
                    "question_supported_without": r.question_supported_without,
                    "answer_entailed_without": r.answer_entailed_without,
                    "has_question_contribution": r.has_question_contribution,
                    "has_reasoning_contribution": r.has_reasoning_contribution,
                    "reason": r.reason,
                }
                for r in self.removed
            ],
        }


def _normalize_answer(text: str) -> str:
    """NFKC + collapse whitespace + lowercase. Same normalization for both sides
    of the fuzz.ratio comparison so the test is symmetric."""
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.split()).lower()


def _cached_answer(question: str, facts: List[Fact], llm: LLM) -> str:
    model = model_identifier(llm)
    # Sort fact_ids so reorderings of the same set hit the same cache entry.
    fact_ids = sorted({f.fact_id for f in facts})
    key = answer_cache.make_key(model, question, fact_ids)

    if PERSIST_CACHE:
        hit = answer_cache.get(key)
        if hit is not None:
            return hit

    with answer_cache.key_lock(key):
        if PERSIST_CACHE:
            hit = answer_cache.get(key)
            if hit is not None:
                return hit
        answer = generate_answer(question, facts, llm)
        if PERSIST_CACHE and answer:
            answer_cache.put(key, answer, model)
        return answer


def minimal_evidence_check(
    question: str, facts: List[Fact], answer: str, llm: LLM
) -> bool:
    if len(facts) <= 1:
        return True
    baseline = _normalize_answer(answer)
    stop_event = threading.Event()

    def _loo(i: int) -> bool:
        if stop_event.is_set():
            return True  # short-circuit; main thread is exiting anyway
        reduced = [facts[j] for j in range(len(facts)) if j != i]
        if not reduced:
            return True
        alt = _normalize_answer(_cached_answer(question, reduced, llm))
        similarity = fuzz.ratio(alt, baseline)
        return similarity < FUZZ_THRESHOLD

    workers = min(MAX_LOO_WORKERS, len(facts))
    if workers <= 1:
        return all(_loo(i) for i in range(len(facts)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_loo, i) for i in range(len(facts))]
        try:
            for fut in as_completed(futures):
                if not fut.result():
                    stop_event.set()
                    for f in futures:
                        f.cancel()
                    return False
        finally:
            stop_event.set()
    return True


def _profile_float(profile: Optional[Dict], key: str, default: float) -> float:
    if not profile:
        return default
    try:
        return float(profile.get(key, default))
    except Exception:
        return default


def _fact_text(fact: Fact) -> str:
    return fact.canonical_form.strip() or fact.text.strip()


def _assumption_supported_by_facts(
    assumptions: List[dict],
    facts: List[Fact],
    *,
    threshold: float,
    max_rel_fail_rate: float,
    max_total_fail_rate: float,
) -> bool:
    """Check whether already-decomposed question assumptions are supported.

    This intentionally avoids another decomposition LLM call during
    leave-one-out minimality. It reuses the hypotheses produced by QA-NLI for
    the accepted question and checks them against the reduced fact set.
    """
    usable = [
        a for a in assumptions
        if str(a.get("hypothesis", "")).strip()
    ]
    if not usable or not facts:
        return False

    support_texts = [_fact_text(f) for f in facts]
    pairs = []
    for a in usable:
        hyp = str(a.get("hypothesis", "")).strip()
        for support in support_texts:
            pairs.append((support, hyp))

    scores = nli_batch(pairs)
    n_support = len(support_texts)
    failed = 0
    rel_total = 0
    rel_failed = 0
    for i, a in enumerate(usable):
        clause_scores = scores[i * n_support : (i + 1) * n_support]
        best = max(clause_scores) if clause_scores else 0.0
        passes = best >= threshold
        atype = str(a.get("type", "")).upper()
        if atype == "REL":
            rel_total += 1
            if not passes:
                rel_failed += 1
        if not passes:
            failed += 1

    fail_rate = failed / len(usable) if usable else 1.0
    rel_fail_rate = rel_failed / rel_total if rel_total else 0.0
    return fail_rate <= max_total_fail_rate and rel_fail_rate <= max_rel_fail_rate


def support_minimality_check(
    question: str,
    facts: List[Fact],
    answer: str,
    *,
    question_assumptions: Optional[List[dict]] = None,
    reasoning_trace: Optional[List[dict]] = None,
    qa_nli_profile: Optional[Dict] = None,
) -> SupportMinimalityVerdict:
    """Support-based minimality for accepted v16 rows.

    A row fails if removing a fact still leaves the original gold answer
    entailed by the remaining facts. It also fails when the accepted question's
    QA-NLI assumptions remain supported without a fact that made no direct
    contribution to the original QA-NLI or ARM reasoning trace.
    """
    if len(facts) <= 1:
        return SupportMinimalityVerdict(passed=True)

    assumptions = list(question_assumptions or [])
    trace = list(reasoning_trace or [])
    threshold = _profile_float(qa_nli_profile, "threshold", 0.55)
    max_rel_fail_rate = _profile_float(qa_nli_profile, "max_rel_fail_rate", 0.10)
    max_total_fail_rate = _profile_float(qa_nli_profile, "max_total_fail_rate", 0.25)

    reduced_sets: List[List[Fact]] = []
    for i in range(len(facts)):
        reduced_sets.append([facts[j] for j in range(len(facts)) if j != i])

    answer_supported = batch_answer_entailment(
        [(answer, reduced) for reduced in reduced_sets]
    )

    question_supported = [False] * len(facts)
    if assumptions:
        for i, reduced in enumerate(reduced_sets):
            question_supported[i] = _assumption_supported_by_facts(
                assumptions,
                reduced,
                threshold=threshold,
                max_rel_fail_rate=max_rel_fail_rate,
                max_total_fail_rate=max_total_fail_rate,
            )

    question_contrib_indices = {
        int(a.get("entails_from_sfu"))
        for a in assumptions
        if bool(a.get("passes")) and str(a.get("entails_from_sfu", "")).lstrip("-").isdigit()
    }
    reasoning_contrib_ids = {
        str(fid)
        for step in trace
        if bool(step.get("passes", True))
        for fid in (step.get("fact_ids") or [])
    }

    removed: List[RemovedFactSupport] = []
    first_reason = ""
    for i, fact in enumerate(facts):
        has_question_contribution = i in question_contrib_indices
        has_reasoning_contribution = fact.fact_id in reasoning_contrib_ids
        reason = ""
        if answer_supported[i]:
            reason = "answer_entailed_without_fact"
        elif (
            question_supported[i]
            and assumptions
            and not has_question_contribution
            and not has_reasoning_contribution
        ):
            reason = "question_supported_without_noncontributing_fact"

        if reason and not first_reason:
            first_reason = reason
        removed.append(
            RemovedFactSupport(
                fact_id=fact.fact_id,
                question_supported_without=question_supported[i],
                answer_entailed_without=answer_supported[i],
                has_question_contribution=has_question_contribution,
                has_reasoning_contribution=has_reasoning_contribution,
                reason=reason,
            )
        )

    return SupportMinimalityVerdict(
        passed=not first_reason,
        reason=first_reason or "accepted",
        removed=removed,
    )
