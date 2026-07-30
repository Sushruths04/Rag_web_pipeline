"""Question-Assumption NLI Gate (QA-NLI) — v16 P2.

Decomposes a question into typed atomic assumptions and NLI-verifies each
against the chain SFUs. A REL assumption that cannot be grounded in any SFU
is a vacuous chain signal — the gate rejects the question and returns a
guidance string the caller can inject into a regeneration prompt.

JSON parsing: json.loads(repair_json(_extract_json(raw))) — never response_format.
NLI: reuses nli_check.nli_batch (batched, cached).

Three assumption types:
  EXIST(concept_x)             — "concept_x is mentioned in the source"
  REL(concept_x, concept_y, relation_type) — "relation holds between x and y"
  QUAL(concept_x, qualifier)   — "qualifier applies to concept_x"
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from json_repair import repair_json
from loguru import logger

from rag_gt.core.llm import _extract_json
from rag_gt.core.types import Fact
from rag_gt.graph.edge_canonicalize import canonicalize_label
from rag_gt.validation.nli_check import nli_batch

if TYPE_CHECKING:
    from rag_gt.core.llm import LLM


# Default per-assumption NLI threshold.
_DEFAULT_THRESHOLD = 0.55
# Maximum fraction of REL assumptions allowed to fail before the gate rejects.
_DEFAULT_MAX_REL_FAIL_RATE = 0.10
# NLI floor when lexical rescue passes — assumption accepted at reduced confidence.
_LEXICAL_RESCUE_NLI_FLOOR = 0.30

_LEX_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "not", "with",
    "this", "that", "it", "its", "by", "from", "as", "into",
}
_COVERAGE_GENERIC_TOKENS = _LEX_STOPWORDS | {
    "state", "states", "action", "actions", "policy", "policies", "value",
    "values", "function", "functions", "equation", "equations", "method",
    "methods", "problem", "problems", "source", "fact", "question",
}


def _lex_tokens(text: str) -> Set[str]:
    """Tokenize for lexical rescue — preserves symbols like V*, R_t, π(s,a)."""
    normalized = unicodedata.normalize("NFKD", text or "")
    words = re.findall(r"[\w*(),.]+", normalized.lower())
    return {w for w in words if len(w) >= 2 and w not in _LEX_STOPWORDS}


def _lexically_supported(args: List[str], chain_facts: List[Fact]) -> bool:
    """Return True if all content tokens from the assumption args appear in chain facts."""
    if not args:
        return False
    query_tokens = _lex_tokens(" ".join(str(a) for a in args))
    if not query_tokens:
        return False
    fact_tokens = _lex_tokens(" ".join(
        (f.canonical_form.strip() or f.text.strip()) for f in chain_facts
    ))
    return query_tokens.issubset(fact_tokens)


def _fact_text(fact: Fact) -> str:
    return fact.canonical_form.strip() or fact.text.strip()


def _strong_edge_involving_fact(fact: Fact, chain_edges: Optional[List[dict]]) -> bool:
    if not chain_edges:
        return False
    fact_id = str(fact.fact_id)
    for edge in chain_edges:
        if fact_id not in {str(edge.get("src", "")), str(edge.get("dst", ""))}:
            continue
        try:
            margin = float(edge.get("joint_only_margin", 0.0) or 0.0)
            nli = float(edge.get("nli_score", 0.0) or 0.0)
        except Exception:
            margin, nli = 0.0, 0.0
        if margin >= 0.35 or nli >= 0.75:
            return True
    return False


def _question_anchor_covered_indices(
    question: str,
    chain_facts: List[Fact],
    missing_fact_indices: List[int],
    chain_edges: Optional[List[dict]],
) -> Set[int]:
    """Rescue per-fact coverage when decomposition omits an explicit clause.

    The QA decomposition model sometimes turns a multi-hop question into
    assumptions for only one SFU even when the question text explicitly names
    anchors from the other SFU. This does not accept unsupported questions; it
    only credits coverage when the question shares concrete source anchors and
    the fact participates in a strong TF-SFG edge. Minimality still runs later.
    """
    q_tokens = _lex_tokens(question) - _COVERAGE_GENERIC_TOKENS
    if not q_tokens:
        return set()

    covered: Set[int] = set()
    for idx in missing_fact_indices:
        if idx < 0 or idx >= len(chain_facts):
            continue
        fact = chain_facts[idx]
        fact_tokens = _lex_tokens(_fact_text(fact)) - _COVERAGE_GENERIC_TOKENS
        if not fact_tokens:
            continue
        overlap = q_tokens & fact_tokens
        if len(overlap) >= 3:
            covered.add(idx)
        elif len(overlap) >= 2 and _strong_edge_involving_fact(fact, chain_edges):
            covered.add(idx)
    return covered


# ---------- Dataclasses ----------

@dataclass
class AssumptionVerdict:
    assumption_id: str
    assumption_type: str        # EXIST | REL | QUAL
    args: List[str]
    hypothesis: str
    nli_score: float
    passes: bool
    best_sfu_index: int         # index into chain_facts; -1 if none passed


@dataclass
class QANLIGateVerdict:
    accepted: bool
    assumption_verdicts: List[AssumptionVerdict]
    fail_rate: float            # fraction of all assumptions that failed
    rel_fail_rate: float        # fraction of REL assumptions that failed
    reason: str                 # human-readable rejection reason
    guidance: str               # injected into regeneration prompt when accepted=False
    missing_fact_indices: List[int] = field(default_factory=list)


# ---------- Prompt ----------

_DECOMPOSE_PROMPT = """\
You are a question-analysis assistant for multi-hop QA evaluation.

Break the question below into its atomic factual assumptions — the claims a
reader must verify against the source document to answer the question correctly.

Classify each assumption as one of:
  EXIST  — a concept or entity is mentioned in the source
  REL    — a specific relationship holds between two concepts
  QUAL   — a property or qualifier applies to a concept

Question:
{question}

Source facts available (for context, not for answering):
{facts_context}

Return ONLY a JSON object:
{{"assumptions": [
  {{"id": "a1", "type": "EXIST|REL|QUAL", "args": ["..."], "hypothesis": "one-sentence claim to verify against source"}}
]}}

Rules:
- Each assumption must be independently verifiable from a single source fact.
- For REL, args = [concept_x, concept_y, relation_type] (relation_type from: \
definition, comparative, causal, intersection, temporal, quantitative, \
procedural, descriptive, rule).
- For EXIST, args = [concept_name].
- For QUAL, args = [concept_name, qualifier].
- Hypothesis must be a complete sentence that NLI can check against source text.
- Produce 2–6 assumptions; stop when the question is fully decomposed.
"""


# ---------- Gate ----------

def _build_facts_context(chain_facts: List[Fact], max_chars: int = 800) -> str:
    parts = []
    total = 0
    for i, f in enumerate(chain_facts):
        text = (f.canonical_form.strip() or f.text.strip())
        line = f"[SFU{i + 1}] {text}"
        if total + len(line) > max_chars:
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts)


def _parse_assumptions(raw: str) -> List[dict]:
    try:
        parsed = json.loads(repair_json(_extract_json(raw)))
        return list(parsed.get("assumptions", []) or [])
    except Exception:
        return []


def _hypothesis_for(assumption: dict) -> str:
    """Return the pre-formed hypothesis, or a fallback generated from type+args."""
    hyp = str(assumption.get("hypothesis", "")).strip()
    if hyp:
        return hyp
    atype = str(assumption.get("type", "")).upper()
    args = assumption.get("args", [])
    if atype == "EXIST" and args:
        return f"The source mentions {args[0]}."
    if atype == "REL" and len(args) >= 2:
        rel = args[2] if len(args) > 2 else "related to"
        return f"{args[0]} is {rel} {args[1]}."
    if atype == "QUAL" and len(args) >= 2:
        return f"{args[0]} is {args[1]}."
    return " ".join(str(a) for a in args)


def check_question_assumptions(
    question: str,
    chain_facts: List[Fact],
    llm: "LLM",
    *,
    chain_edges: Optional[List[dict]] = None,
    threshold: float = _DEFAULT_THRESHOLD,
    max_rel_fail_rate: float = _DEFAULT_MAX_REL_FAIL_RATE,
    max_total_fail_rate: float = 0.25,
    fail_open_on_decomposition_error: bool = True,
    enforce_edge_relation_match: bool = False,
    max_rel_mismatch_rate: float = 0.5,
    qa_nli_profile: Optional[Dict] = None,
    require_per_fact_coverage: bool = False,
) -> QANLIGateVerdict:
    """Decompose `question` into assumptions and NLI-verify against chain SFUs.

    Args:
        question: the generated question to verify.
        chain_facts: SFUs forming the chain (used as NLI premises).
        llm: model used for assumption decomposition.
        chain_edges: TF-SFG edge records for this chain; used to cross-check
            REL assumptions against the edge relation_type.
        threshold: per-assumption NLI entailment threshold.
        max_rel_fail_rate: maximum fraction of REL assumptions allowed to fail.
        max_rel_mismatch_rate: when enforce_edge_relation_match is True, the
            maximum fraction of REL assumptions whose relation type may mismatch
            the TF-SFG edge types before rejecting. Fraction-based to avoid
            hard-rejecting on a single mismatch. Default 0.5.
    """
    # Apply doc-type profile overrides if provided.
    if qa_nli_profile:
        threshold = float(qa_nli_profile.get("threshold", threshold))
        max_rel_fail_rate = float(qa_nli_profile.get("max_rel_fail_rate", max_rel_fail_rate))
        max_total_fail_rate = float(qa_nli_profile.get("max_total_fail_rate", max_total_fail_rate))
        enforce_edge_relation_match = bool(
            qa_nli_profile.get("enforce_edge_relation_match", enforce_edge_relation_match)
        )
        max_rel_mismatch_rate = float(qa_nli_profile.get("max_rel_mismatch_rate", max_rel_mismatch_rate))
        require_per_fact_coverage = bool(
            qa_nli_profile.get("require_per_fact_coverage", require_per_fact_coverage)
        )

    if not chain_facts:
        return QANLIGateVerdict(
            accepted=False,
            assumption_verdicts=[],
            fail_rate=1.0,
            rel_fail_rate=1.0,
            reason="empty_chain_facts",
            guidance="",
        )

    facts_context = _build_facts_context(chain_facts)
    prompt = _DECOMPOSE_PROMPT.format(
        question=question,
        facts_context=facts_context,
    )

    from rag_gt.observability.cost_tracker import LiveCallBudgetExceeded

    try:
        raw = llm.generate(prompt, temperature=0.0, max_tokens=512)
        raw_assumptions = _parse_assumptions(raw)
    except LiveCallBudgetExceeded:
        raise
    except Exception as e:
        logger.debug(f"[QA-NLI] decomposition LLM error: {type(e).__name__}: {e}")
        return QANLIGateVerdict(
            accepted=bool(fail_open_on_decomposition_error),
            assumption_verdicts=[],
            fail_rate=0.0 if fail_open_on_decomposition_error else 1.0,
            rel_fail_rate=0.0 if fail_open_on_decomposition_error else 1.0,
            reason=(
                "decomposition_failed_passthrough"
                if fail_open_on_decomposition_error
                else "decomposition_failed"
            ),
            guidance="Regenerate a simpler question whose assumptions can be decomposed cleanly.",
        )

    if not raw_assumptions:
        logger.debug(f"[QA-NLI] no assumptions extracted for: {question[:80]!r}")
        return QANLIGateVerdict(
            accepted=bool(fail_open_on_decomposition_error),
            assumption_verdicts=[],
            fail_rate=0.0 if fail_open_on_decomposition_error else 1.0,
            rel_fail_rate=0.0 if fail_open_on_decomposition_error else 1.0,
            reason=(
                "no_assumptions_passthrough"
                if fail_open_on_decomposition_error
                else "no_assumptions"
            ),
            guidance="Regenerate a question with explicit concepts and relations grounded in the source facts.",
        )

    # Build NLI batch: each assumption hypothesis vs each chain SFU text.
    support_texts = [_fact_text(f) for f in chain_facts]
    hypotheses = [_hypothesis_for(a) for a in raw_assumptions]
    n_sfus = len(support_texts)

    pairs: List[Tuple[str, str]] = []
    for hyp in hypotheses:
        for sfu_text in support_texts:
            pairs.append((sfu_text, hyp))

    nli_scores = nli_batch(pairs)

    # Gather per-assumption results.
    verdicts: List[AssumptionVerdict] = []
    for i, (assumption, hyp) in enumerate(zip(raw_assumptions, hypotheses)):
        clause_scores = nli_scores[i * n_sfus : (i + 1) * n_sfus]
        if not clause_scores:
            best_idx, best_score = -1, 0.0
        else:
            best_idx = max(range(n_sfus), key=lambda j: clause_scores[j])
            best_score = float(clause_scores[best_idx])

        atype = str(assumption.get("type", "EXIST")).upper()
        args = list(assumption.get("args", []))
        passes = best_score >= threshold
        # Lexical rescue for EXIST/QUAL: if NLI is above a lower floor AND all
        # content tokens appear verbatim in chain-fact texts, accept the assumption.
        # This catches textbook symbols (V*, π, R_t) that NLI cannot handle well.
        lex_rescued = False
        if not passes and atype in ("EXIST", "QUAL") and best_score >= _LEXICAL_RESCUE_NLI_FLOOR:
            lex_rescued = _lexically_supported(args, chain_facts)
            if lex_rescued:
                passes = True
                logger.debug(
                    f"[QA-NLI] lexical rescue: {atype} {args} "
                    f"(nli={best_score:.3f}, all tokens in chain facts)"
                )
        verdicts.append(
            AssumptionVerdict(
                assumption_id=str(assumption.get("id", f"a{i + 1}")),
                assumption_type=atype,
                args=args,
                hypothesis=hyp,
                nli_score=best_score,
                passes=passes,
                best_sfu_index=best_idx if passes else -1,
            )
        )

    # REL-edge cross-check: canonicalize both sides before comparing so that
    # synonym labels (mechanism→causal, comparison→comparative, list→descriptive)
    # do not cause spurious mismatches. Gate rejects only when the fraction of
    # mismatched REL assumptions exceeds max_rel_mismatch_rate.
    rel_mismatch_count = 0
    n_rel_checked = 0
    if chain_edges:
        edge_relation_types = {
            canonicalize_label(str(e.get("type", ""))) for e in chain_edges
        }
        for v in verdicts:
            if v.assumption_type == "REL" and len(v.args) > 2:
                n_rel_checked += 1
                claimed_raw = str(v.args[2]).lower()
                claimed_rel = canonicalize_label(claimed_raw)
                if claimed_rel not in edge_relation_types:
                    rel_mismatch_count += 1
                    logger.debug(
                        f"[QA-NLI] REL assumption {v.assumption_id} claims "
                        f"'{claimed_raw}' (canonical: '{claimed_rel}') "
                        f"but TF-SFG edges are {edge_relation_types}"
                    )

    rel_mismatch_rate = rel_mismatch_count / n_rel_checked if n_rel_checked else 0.0
    relation_mismatch_over_threshold = (
        enforce_edge_relation_match and rel_mismatch_rate > max_rel_mismatch_rate
    )

    n_total = len(verdicts)
    n_failed = sum(1 for v in verdicts if not v.passes)
    rel_verdicts = [v for v in verdicts if v.assumption_type == "REL"]
    n_rel = len(rel_verdicts)
    n_rel_failed = sum(1 for v in rel_verdicts if not v.passes)

    fail_rate = n_failed / n_total if n_total else 0.0
    rel_fail_rate = n_rel_failed / n_rel if n_rel else 0.0

    covered_fact_indices = {
        int(v.best_sfu_index)
        for v in verdicts
        if v.passes and int(v.best_sfu_index) >= 0
    }
    required_fact_indices = set(range(len(chain_facts)))
    initial_missing_fact_indices = sorted(required_fact_indices - covered_fact_indices)
    anchor_covered_indices = _question_anchor_covered_indices(
        question,
        chain_facts,
        initial_missing_fact_indices,
        chain_edges,
    )
    covered_fact_indices |= anchor_covered_indices
    missing_fact_indices = sorted(required_fact_indices - covered_fact_indices)
    fact_coverage_ok = (
        not require_per_fact_coverage
        or len(chain_facts) <= 1
        or not missing_fact_indices
    )

    accepted = (
        rel_fail_rate <= max_rel_fail_rate
        and fail_rate <= max_total_fail_rate
        and not relation_mismatch_over_threshold
        and fact_coverage_ok
    )

    failed_verdicts = [v for v in verdicts if not v.passes]
    guidance = ""
    if not fact_coverage_ok:
        guidance = (
            "The question did not require every source fact. Rewrite it so the "
            "answer changes if any one fact is removed, especially the uncovered "
            f"fact indices: {missing_fact_indices}."
        )
    elif not accepted and failed_verdicts:
        guidance = (
            "The following question assumptions could not be grounded in the source facts: "
            + "; ".join(
                f"[{v.assumption_id}] {v.hypothesis} (score={v.nli_score:.2f})"
                for v in failed_verdicts[:3]
            )
            + ". Rewrite the question so it only asserts relationships that are "
            "explicitly stated in the source facts."
        )

    if accepted:
        reason = "accepted"
    elif not fact_coverage_ok:
        reason = "missing_required_fact_coverage:" + ",".join(
            str(i) for i in missing_fact_indices
        )
    elif relation_mismatch_over_threshold:
        reason = (
            f"relation_mismatch_with_tf_sfg"
            f"({rel_mismatch_count}/{n_rel_checked}>{max_rel_mismatch_rate:.2f})"
        )
        if not guidance:
            guidance = (
                "Rewrite the question so its relation type matches the verified "
                "TF-SFG edge relation connecting the source facts."
            )
    elif fail_rate > max_total_fail_rate:
        reason = f"fail_rate={fail_rate:.3f}>{max_total_fail_rate}"
    else:
        reason = f"rel_fail_rate={rel_fail_rate:.3f}>{max_rel_fail_rate}"

    return QANLIGateVerdict(
        accepted=accepted,
        assumption_verdicts=verdicts,
        fail_rate=fail_rate,
        rel_fail_rate=rel_fail_rate,
        reason=reason,
        guidance=guidance,
        missing_fact_indices=missing_fact_indices if not fact_coverage_ok else [],
    )


def verdicts_to_dicts(verdicts: List[AssumptionVerdict]) -> List[dict]:
    """Convert AssumptionVerdict list to the QuestionGT.question_assumptions format."""
    return [
        {
            "assumption_id": v.assumption_id,
            "type": v.assumption_type,
            "args": v.args,
            "hypothesis": v.hypothesis,
            "nli_score": round(v.nli_score, 4),
            "passes": v.passes,
            "entails_from_sfu": v.best_sfu_index,
        }
        for v in verdicts
    ]
