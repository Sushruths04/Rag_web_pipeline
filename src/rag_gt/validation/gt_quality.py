"""Cheap quality checks for generated ground-truth QA rows.

These checks are intentionally heuristic. They catch cases that are bad for a
universal RAG benchmark even when the answer is technically entailed by the
facts, for example questions with unresolved "this distribution" references or
facts that are PDF line-break fragments.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List


DEICTIC_RE = re.compile(
    r"\b(this|that|these|those|it|they|them|its|their|such)\b", re.I,
)
GENERIC_RE = re.compile(
    r"\b(?:according to\s+)?(?:the|this|that)\s+(?:[a-z]+\s+){0,2}"
    r"(description|result|figure|table|example|equation|method|approach|"
    r"algorithm|theorem|chapter|section|paragraph|case|paper|study|"
    r"distribution|formula|expression|graph|plot|diagram|model|"
    r"process|procedure|operation|operator|function)\b",
    re.I,
)
GENERIC_SOURCE_RE = re.compile(
    r"\b(according to|based on|from|in)\s+(the\s+)?"
    r"(description|text|passage|statement|provided information|facts?|given facts?|given statements?)\b|"
    r"\b(in|from)\s+the\s+given\s+statements?\b",
    re.I,
)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
NUMBER_WORD_RE = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"single|a|an|both)\b",
    re.I,
)
TWO_ITEM_QUESTION_RE = re.compile(r"\b(what|which|list|name|state)\s+two\b", re.I)
HOW_MANY_QUESTION_RE = re.compile(r"\bhow many\b|\bwhat number\b", re.I)
BROKEN_HYPHEN_RE = re.compile(r"\b\w{3,}-\s*$")
ANSWER_LEAK_RE = re.compile(
    r"\b(we need to answer|wait the facts|the facts\s*:|the facts are|the instruction:|"
    r"the question asks|however we can infer|not explicitly|so we might|"
    r"need to explain|i think|let's)\b",
    re.I,
)
WEAK_FACT_START_RE = re.compile(
    r"^(before we are done|in other words|moreover|however|of course|"
    r"as a hint|these methods|this method|this problem|this number|"
    r"this is|such methods|we consider|we use this problem)\b",
    re.I,
)
WEAK_FACT_END_RE = re.compile(
    r"\b(and|or|of|the|with|during|to|from|by|as|"
    r"provide|include|suggest|describe|discuss|enable|allow|require|"
    r"use|make|give|show|demonstrate|indicate|identify|present|"
    r"produce|create|support|apply|involve|contain|represent|define)\.?$",
    re.I,
)
BIBLIOGRAPHY_RE = re.compile(
    r"\b(proceedings|workshop|conference|pp\.|eds?\.\)|\bed\.\)|"
    r"technical report|tech\.?\s+report|dissertation|ph\.?\s*d\.?|"
    r"\bthesis\b|university press|morgan kaufmann|springer|elsevier|"
    r"journal|transactions|"
    r"advances in neural information processing systems)\b|"
    r"\b\d+\s*:\s*\d+\s*(?:--|-|\u2013|\u2014)\s*\d+\b",
    re.I,
)
TABLE_OR_FIGURE_ARTIFACT_RE = re.compile(
    r"\b(figure|chapter)\s+\d+(?:\.\d+)?\b|"
    r"^[A-Z][A-Z\s]{8,}\s+\d+\s*!?$|"
    r"\b\d+\s*!$",
    re.I,
)
ANSWER_IS_QUESTION_RE = re.compile(
    r"\?\s*$|\b(what|why|how|which|when|where|who)\b.+\?",
    re.I,
)
ANSWER_SOURCE_HEADING_RE = re.compile(
    r"\b(chapter|section|figure|table|appendix)\s+\d+(?:\.\d+)*\b|"
    r"^\s*[*∗]?\s*\d+(?:\.\d+)+\s+[A-Z]|"
    r"^\s*(?:[A-Z][A-Z0-9()-]*\s+){1,5}[A-Z][A-Z0-9()-]*\s*[:.-]\s+",
    re.I,
)
# No re.I — only catches actual ALL-CAPS headings, not normal lowercase answers.
ANSWER_ALL_CAPS_HEADING_RE = re.compile(r"^\s*[A-Z][A-Z ]{8,}\s*$")
ANSWER_ABSTENTION_RE = re.compile(
    r"\b(insufficient information|not enough information|cannot be answered|"
    r"cannot answer|no information is provided)\b",
    re.I,
)
ANSWER_TAUTOLOGY_RE = re.compile(
    r"\b(can now be explained|is explained by the facts|the facts explain|"
    r"the answer is that|this means that it means)\b",
    re.I,
)
QUESTION_SOURCE_ARTIFACT_RE = re.compile(
    r"\b("
    r"page ranges?|references? titled|which chapter is titled|"
    r"page numbers?|on which pages?|entries titled|exercise numbers?|"
    r"publishing house|publisher|city for the \d{4} work|"
    r"work authored by|work titled|title of the review|"
    r"publications? mentioned|university'?s department|"
    r"on pages?\s+\d+|journal|volume|conference|proceedings|"
    r"provided notation|specific headings?|terms follow|given excerpt|"
    r"mentioned in the statement|statement about|sentence stating|"
    r"in the source|source text|listed in the source|as listed in the source|"
    r"provided definition|provided fact|according to the definition"
    r")\b",
    re.I,
)
QUESTION_PROMPT_LEAK_RE = re.compile(
    r"\b(check length|count words|meets min(?:imum)?|word count|"
    r"that's \d+ words|output exactly one question|"
    r"maybe we can ask|we could ask|we can ask|we should ask|"
    r"we have a conflict|the user first gave|the instruction says|"
    r"higher-level instruction|intent constraint|single-fact question|"
    r"there'?s only one fact|there is only one fact|"
    r"unable to create|could you please supply|only a single fact|"
    r"we need to rewrite|only assert relationships|no second entity|"
    r"answer unknown|return only|do not include|"
    r"provided facts|given facts|source-grounding rules|"
    r"we need a specific answer|combine with (?:second|first|the)|"
    r"that's from (?:first|second|the)\s+fact|so question:|that's a phrase|"
    r"so maybe:|we need to name|we need to identify|we need to specify|"
    r"Count:\s*[\"']?\w|word\s+\d+\s*[\"'(])\b|"
    r"\bAnswer:\s*['\"]",
    re.I,
)
UNRESOLVED_PLACEHOLDER_RE = re.compile(
    r"(?<!\w)(?:\?{2,}|[A-Za-z]*\?{2,}|chapter[s]?\s+\?|section[s]?\s+\?|"
    r"figure[s]?\s+\?|table[s]?\s+\?|appendix\s+\?)(?!\w)",
    re.I,
)
PHRASE_LOOKUP_QUESTION_RE = re.compile(
    r"\bdescribed by the phrase\b|"
    r"\bwhat is said to\b|"
    r"\bthe technique described\b|"
    r"\bthe method described\b|"
    r"\bthat situation\b|"
    r"\bwhat (?:was|is|are|were) (?:said to|described as|referred to as)\b|"
    r"\b(?:referred|referenced)\s+(?:to\s+)?in the phrase\b|"
    r"\bin the phrase\s*[\"']|"
    r"\bthe phrase\s*[\"']",
    re.I,
)

_QUESTION_STARTERS = {
    "why", "what", "how", "where", "when", "who", "whom", "whose", "which",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "may",
    "might", "must", "shall", "should", "will", "would", "has", "have", "had",
    "explain", "describe", "state", "list", "name", "give", "show", "in", "of",
}


@dataclass(frozen=True)
class QualityCheck:
    name: str
    fn: Callable[[str, str, List[dict]], bool]
    weight: float


def _fact_text(fact: Any) -> str:
    if isinstance(fact, dict):
        return str(fact.get("text", "") or "")
    return str(getattr(fact, "text", "") or "")


def _fact_spans(fact: Any) -> list:
    if isinstance(fact, dict):
        return list(fact.get("supporting_spans") or [])
    return list(getattr(fact, "supporting_spans", []) or [])


def _span_doc_id(span: Any) -> str:
    if isinstance(span, dict):
        return str(span.get("doc_id", "") or "")
    return str(getattr(span, "doc_id", "") or "")


def _span_chunk_id(span: Any) -> str:
    if isinstance(span, dict):
        return str(span.get("chunk_id", "") or "")
    return str(getattr(span, "chunk_id", "") or "")


def _q1_deictic_without_antecedent(question: str) -> bool:
    m = DEICTIC_RE.search(question)
    if not m:
        return False
    pre = question[: m.start()]
    tokens = re.findall(r"\w+", pre)
    nonfunc = [t for t in tokens if t.lower() not in _QUESTION_STARTERS]
    if not nonfunc:
        return True
    return not any(re.match(r"^[A-Z][a-z]+$", t) or len(t) >= 5 for t in nonfunc)


def _q2_answer_too_short(answer: str) -> bool:
    return len(answer.strip()) < 40


def _question_expects_short_entity(question: str) -> bool:
    q = " ".join((question or "").lower().split())
    return bool(
        re.match(r"^(what|which|who|when|where)\b", q)
        and not re.search(r"\b(why|how|explain|describe|compare|relate)\b", q)
    )


def _q3_answer_echoes_question(question: str, answer: str) -> bool:
    qw = set(re.findall(r"\w+", question.lower()))
    aw = set(re.findall(r"\w+", answer.lower()))
    if not qw or not aw:
        return False
    return len(qw & aw) / max(1, len(qw | aw)) > 0.7


_CONTRASTIVE_PRE_RE = re.compile(
    r"\b(rather than|instead of|other than|as opposed to|not the)\s*$", re.I
)


def _q4_generic_referent(question: str) -> bool:
    if GENERIC_SOURCE_RE.search(question):
        return True
    for m in GENERIC_RE.finditer(question):
        tail = question[m.end(): m.end() + 30]
        if re.match(r"\s*[A-Z0-9]", tail) or re.match(r"\s+(in|of|from)\s+[A-Z0-9]", tail):
            continue
        # "rather than the method" / "instead of the approach" → conceptual, not doc-ref
        pre = question[: m.start()].rstrip()
        if _CONTRASTIVE_PRE_RE.search(pre):
            continue
        return True
    return False


def _q5_scattered_facts(facts: List[dict]) -> bool:
    if not facts:
        return False
    docs = {
        _span_doc_id(sp)
        for f in facts
        for sp in _fact_spans(f)
        if _span_doc_id(sp)
    }
    if len(docs) > 1:
        return True
    idxs = []
    for f in facts:
        for sp in _fact_spans(f):
            m = re.match(r".+_c0*(\d+)$", _span_chunk_id(sp))
            if m:
                idxs.append(int(m.group(1)))
    return bool(idxs) and (max(idxs) - min(idxs)) > 5


def _q6_unsupported_number(answer: str, facts: List[dict]) -> bool:
    if not facts:
        return False
    nums_a = set(NUMBER_RE.findall(answer))
    if not nums_a:
        return False
    fact_text = " ".join(_fact_text(f) for f in facts)
    nums_f = set(NUMBER_RE.findall(fact_text))
    return bool(nums_a - nums_f)


def _q7_bad_fact_fragment(_: str, __: str, facts: List[dict]) -> bool:
    for fact in facts:
        text = " ".join(_fact_text(fact).split())
        spans = _fact_spans(fact)
        if not spans or not any(_span_chunk_id(sp) for sp in spans):
            return True
        if not text:
            return True
        if BROKEN_HYPHEN_RE.search(text):
            return True
        if WEAK_FACT_END_RE.search(text):
            return True
        if re.match(r"^\d+\s+\d+\s+[A-Z]", text):
            return True
        if len(re.findall(r"\w+", text)) < 5:
            return True
        if _q1_deictic_without_antecedent(text):
            return True
    return False


def _q8_answer_leaks_reasoning(_: str, answer: str, __: List[dict]) -> bool:
    return bool(ANSWER_LEAK_RE.search(answer or ""))


def _q10_answer_is_question(_: str, answer: str, __: List[dict]) -> bool:
    return bool(ANSWER_IS_QUESTION_RE.search(answer or ""))


def _q11_answer_source_heading_leak(_: str, answer: str, __: List[dict]) -> bool:
    a = answer or ""
    return bool(ANSWER_SOURCE_HEADING_RE.search(a) or ANSWER_ALL_CAPS_HEADING_RE.search(a))


def _q12_answer_abstains(_: str, answer: str, __: List[dict]) -> bool:
    return bool(ANSWER_ABSTENTION_RE.search(answer or ""))


def _q13_answer_tautology(_: str, answer: str, __: List[dict]) -> bool:
    return bool(ANSWER_TAUTOLOGY_RE.search(answer or ""))


def _q14_answer_fragment(question: str, answer: str, __: List[dict]) -> bool:
    a = " ".join((answer or "").split())
    if not a:
        return True
    if _question_expects_short_entity(question):
        return False
    return len(re.findall(r"\w+", a)) < 6


def _q15_source_artifact_question(question: str, _: str, __: List[dict]) -> bool:
    return bool(QUESTION_SOURCE_ARTIFACT_RE.search(question or ""))


def _q16_question_prompt_leak(question: str, _: str, __: List[dict]) -> bool:
    return bool(QUESTION_PROMPT_LEAK_RE.search(question or ""))


def _q17_cardinality_answer_mismatch(question: str, answer: str, __: List[dict]) -> bool:
    if not TWO_ITEM_QUESTION_RE.search(question or ""):
        return False
    a = " ".join((answer or "").split())
    if re.search(r"\b(two|both)\b", a, re.I):
        return False
    item_like_parts = [p for p in re.split(r"\s+(?:and|plus)\s+|[.;]", a) if p.strip()]
    return len(item_like_parts) < 2


def _q18_numeric_answer_mismatch(question: str, answer: str, __: List[dict]) -> bool:
    if not HOW_MANY_QUESTION_RE.search(question or ""):
        return False
    a = answer or ""
    return not (NUMBER_RE.search(a) or NUMBER_WORD_RE.search(a))


def _q19_unresolved_placeholder(question: str, answer: str, facts: List[dict]) -> bool:
    text = " ".join([question or "", answer or ""] + [_fact_text(f) for f in facts])
    return bool(UNRESOLVED_PLACEHOLDER_RE.search(text))


def _q20_phrase_lookup_question(question: str, _: str, __: List[dict]) -> bool:
    return bool(PHRASE_LOOKUP_QUESTION_RE.search(question or ""))


def _q9_weak_required_evidence(_: str, __: str, facts: List[dict]) -> bool:
    weak = 0
    for fact in facts:
        text = " ".join(_fact_text(fact).split())
        if WEAK_FACT_START_RE.search(text):
            weak += 1
        elif BIBLIOGRAPHY_RE.search(text):
            weak += 1
        elif TABLE_OR_FIGURE_ARTIFACT_RE.search(text):
            weak += 1
    return weak > 0


CHECKS: List[QualityCheck] = [
    QualityCheck("Q1_deictic_no_antecedent", lambda q, a, f: _q1_deictic_without_antecedent(q), 0.25),
    QualityCheck("Q2_answer_too_short", lambda q, a, f: _q2_answer_too_short(a), 0.10),
    QualityCheck("Q3_answer_echoes_question", lambda q, a, f: _q3_answer_echoes_question(q, a), 0.10),
    QualityCheck("Q4_generic_referent", lambda q, a, f: _q4_generic_referent(q), 0.35),
    QualityCheck("Q5_scattered_facts", lambda q, a, f: _q5_scattered_facts(f), 0.05),
    QualityCheck("Q6_unsupported_number", lambda q, a, f: _q6_unsupported_number(a, f), 0.10),
    QualityCheck("Q7_bad_fact_fragment", _q7_bad_fact_fragment, 0.15),
    QualityCheck("Q8_answer_leaks_reasoning", _q8_answer_leaks_reasoning, 0.85),
    QualityCheck("Q9_weak_required_evidence", _q9_weak_required_evidence, 0.85),
    QualityCheck("Q10_answer_is_question", _q10_answer_is_question, 0.85),
    QualityCheck("Q11_answer_source_heading_leak", _q11_answer_source_heading_leak, 0.85),
    QualityCheck("Q12_answer_abstains", _q12_answer_abstains, 0.85),
    QualityCheck("Q13_answer_tautology", _q13_answer_tautology, 0.85),
    QualityCheck("Q14_answer_fragment", _q14_answer_fragment, 0.45),
    QualityCheck("Q15_source_artifact_question", _q15_source_artifact_question, 0.85),
    QualityCheck("Q16_question_prompt_leak", _q16_question_prompt_leak, 0.85),
    QualityCheck("Q17_cardinality_answer_mismatch", _q17_cardinality_answer_mismatch, 0.85),
    QualityCheck("Q18_numeric_answer_mismatch", _q18_numeric_answer_mismatch, 0.85),
    QualityCheck("Q19_unresolved_placeholder", _q19_unresolved_placeholder, 0.85),
    QualityCheck("Q20_phrase_lookup_question", _q20_phrase_lookup_question, 0.85),
]
TOTAL_WEIGHT = sum(c.weight for c in CHECKS)
HARD_FAIL_FLAGS = {
    "Q1_deictic_no_antecedent",
    "Q4_generic_referent",
    "Q7_bad_fact_fragment",
    "Q8_answer_leaks_reasoning",
    "Q9_weak_required_evidence",
    "Q10_answer_is_question",
    "Q11_answer_source_heading_leak",
    "Q12_answer_abstains",
    "Q13_answer_tautology",
    "Q14_answer_fragment",
    "Q15_source_artifact_question",
    "Q16_question_prompt_leak",
    "Q17_cardinality_answer_mismatch",
    "Q18_numeric_answer_mismatch",
    "Q19_unresolved_placeholder",
    "Q20_phrase_lookup_question",
}


def score_generated_pair(question: str, answer: str, facts: List[Any]) -> Dict[str, Any]:
    flags: Dict[str, bool] = {}
    penalty = 0.0
    for check in CHECKS:
        try:
            flag = bool(check.fn(question or "", answer or "", facts or []))
        except Exception:
            flag = False
        flags[check.name] = flag
        if flag:
            penalty += check.weight
    quality = max(0.0, 1.0 - penalty / TOTAL_WEIGHT)
    if any(flags.get(name) for name in HARD_FAIL_FLAGS):
        quality = min(quality, 0.0)
    return {
        "quality": round(quality, 3),
        "flags": flags,
        "n_facts": len(facts or []),
        "n_words_q": len((question or "").split()),
        "n_words_a": len((answer or "").split()),
    }


def score_question(row: dict) -> Dict[str, Any]:
    scored = score_generated_pair(
        str(row.get("question", "") or ""),
        str(row.get("gold_answer", "") or ""),
        list(row.get("required_facts") or []),
    )
    return {"q_id": row.get("q_id"), **scored}
