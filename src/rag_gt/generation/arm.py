"""ARM: Abstractive Reasoning Map — P5 for PASS-GT v16.

Replaces CGA's paraphrase-only step with decompose-then-synthesize:
  1. Build one reasoning step per chain fact (skeleton from TF-SFG edges when available).
  2. Synthesize each step: ≤15 words, ≤35% 4-gram overlap with source (compression check).
  3. NLI-verify: each synthesized claim must entail from its source fact.
  4. Compose claims into final gold_answer; store reasoning_trace.

JSON parsing: json.loads(repair_json(_extract_json(raw))) — never response_format.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from json_repair import repair_json
from loguru import logger

from rag_gt.core.llm import _extract_json
from rag_gt.core.types import Fact
from rag_gt.validation.nli_check import nli_entailment

if TYPE_CHECKING:
    from rag_gt.core.llm import LLM

ARM_NLI_THRESHOLD = 0.55
ARM_MAX_NP_OVERLAP = 0.35
ARM_MAX_STEP_RETRIES = 3


_SYNTHESIS_PROMPT = """Paraphrase the key claim from the source fact in 15 words or fewer.
{retry_feedback}
Source fact:
<<SOURCE>>
{source}
<</SOURCE>>

Rules:
- Write a GENUINE PARAPHRASE: change the sentence structure and most of the vocabulary.
- Do NOT copy any phrase of 4 or more consecutive words from the source verbatim.
- Do NOT preserve the original word order or sentence structure.
- If the source has N words, your claim must use a clearly different grammatical structure.
- Do NOT add information not present in the source.
- Express the claim directly in your own phrasing, without hedges like "it states".
- Preserve all numbers and named entities.
- Preserve explicit quantifiers such as a, an, one, each, every, all, both, two.

Return JSON only — no other text:
{{"claim": "your compressed claim here", "rationale": "how it avoids verbatim copying"}}"""

_COMPOSE_PROMPT = """Answer the question using all reasoning steps.

Question:
{question}

Steps:
{steps}
{relation_context}

Rules:
- Preserve EVERY step's claim; do not drop any step's content.
- Each step MUST contribute a distinct, irreplaceable claim to your answer.
  If removing any single step would leave the answer still correct and complete,
  your answer is wrong — revise until every step is strictly necessary.
- If verified relation evidence is provided, preserve the relation in the answer.
- Connect the steps directly; do not merely list two independent facts when the
  question asks why/how/under what condition.
- Do not explain causes, mechanisms, or implications unless a step explicitly states them.
- Prefer one or two concise sentences that together require all steps.
- If the question asks "how many", state the count only when entailed by the steps.
- Write as a direct answer, not as numbered steps.
- Output ONLY the paragraph — no preamble.

Answer:"""


# ---------- helpers ----------


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def ngram_overlap(synthesis: str, source: str, n: int = 4) -> float:
    """Fraction of n-grams in synthesis that appear verbatim in source."""
    syn_tokens = _tokenize(synthesis)
    src_tokens = _tokenize(source)
    if len(syn_tokens) < n:
        return 0.0
    syn_ngrams = [tuple(syn_tokens[i : i + n]) for i in range(len(syn_tokens) - n + 1)]
    if not syn_ngrams:
        return 0.0
    src_ngram_set = {tuple(src_tokens[i : i + n]) for i in range(len(src_tokens) - n + 1)}
    shared = sum(1 for ng in syn_ngrams if ng in src_ngram_set)
    return shared / len(syn_ngrams)


def _span_bbox_ref(span) -> dict:
    """Compact provenance reference for one reasoning-trace source span."""
    entry: dict = {
        "doc_id": getattr(span, "doc_id", ""),
        "chunk_id": getattr(span, "chunk_id", ""),
    }
    optional = {
        "page_start": getattr(span, "page_start", None),
        "page_end": getattr(span, "page_end", None),
        "char_start": getattr(span, "char_start", None),
        "char_end": getattr(span, "char_end", None),
        "source_path": getattr(span, "source_path", ""),
        "source_sha1": getattr(span, "source_sha1", ""),
        "source_text_sha1": getattr(span, "source_text_sha1", ""),
        "extractor": getattr(span, "extractor", ""),
    }
    for key, value in optional.items():
        if value not in (None, "", []):
            entry[key] = value

    bboxes = [b.to_dict() for b in getattr(span, "bboxes", [])]
    if bboxes:
        entry["bboxes"] = bboxes
    return entry


def _synthesize_step(
    fact: Fact,
    llm: "LLM",
    *,
    max_np_overlap: float,
    max_retries: int,
    nli_threshold: float,
    cost_tracker: object = None,
    doc_id: str = "",
) -> Tuple[str, float, float]:
    """Return (claim, np_overlap, nli_score) for one fact.

    Retries up to max_retries if NLI or overlap constraints fail.
    Falls back to a truncated fact text on total failure.
    """
    source_text = (fact.canonical_form or fact.text).strip()
    if cost_tracker is not None:
        from rag_gt.observability.cost_tracker import TrackedLLM

        llm = TrackedLLM(
            llm, cost_tracker, stage="arm_step_synth", doc_id=doc_id
        )

    from rag_gt.observability.cost_tracker import LiveCallBudgetExceeded

    best_claim: Optional[str] = None
    best_overlap = 1.0
    best_nli = 0.0

    for attempt in range(max_retries):
        # Escalate temperature on retries to escape verbatim-copy local minima.
        temperature = 0.3 + 0.2 * attempt
        retry_feedback = ""
        if attempt > 0 and best_claim is not None:
            if best_overlap >= 0.95:
                source_first = source_text.split()[0] if source_text.split() else ""
                retry_feedback = (
                    f"\n⚠ RETRY {attempt}: previous claim was nearly verbatim "
                    f"({best_overlap:.0%} 4-gram overlap). You MUST restructure completely.\n"
                    f"- Do NOT start with '{source_first}'.\n"
                    "- Try: passive voice, flip subject/object, or lead with the outcome.\n"
                    "- Example: 'X causes Y' → 'Y results from X' or 'The key limit is Z'.\n"
                )
            else:
                retry_feedback = (
                    f"\n⚠ RETRY {attempt}: previous claim had {best_overlap:.0%} "
                    "n-gram overlap — still too literal. Use completely different vocabulary.\n"
                )
        prompt = _SYNTHESIS_PROMPT.format(source=source_text, retry_feedback=retry_feedback)
        try:
            raw = llm.generate(prompt, temperature=temperature, max_tokens=256)
            parsed = json.loads(repair_json(_extract_json(raw)))
            claim = str(parsed.get("claim", "")).strip()
        except LiveCallBudgetExceeded:
            raise
        except Exception as e:
            logger.debug(f"[ARM] step synthesis parse error attempt {attempt + 1}: {e}")
            continue

        if not claim:
            continue

        overlap = ngram_overlap(claim, source_text)
        nli_score = nli_entailment(source_text, claim)

        if best_claim is None or nli_score > best_nli:
            best_claim = claim
            best_overlap = overlap
            best_nli = nli_score

        if overlap <= max_np_overlap and nli_score >= nli_threshold:
            return claim, overlap, nli_score

    if best_claim:
        return best_claim, best_overlap, best_nli

    # Hard fallback: first 15 words of source
    words = source_text.split()
    fallback = " ".join(words[:15]) + ("…" if len(words) > 15 else "")
    return fallback, ngram_overlap(fallback, source_text), 1.0


def _final_answer_covers_steps(
    answer: str,
    steps: List["ARMStep"],
    *,
    threshold: float,
) -> bool:
    if not answer.strip():
        return False
    for step in steps:
        try:
            if nli_entailment(answer, step.claim) < threshold:
                return False
        except Exception:
            return False
    return True


def _answer_entailed_by_sources(
    answer: str,
    chain_facts: List[Fact],
    *,
    threshold: float,
) -> bool:
    support = "\n".join((f.canonical_form or f.text).strip() for f in chain_facts)
    if not answer.strip() or not support.strip():
        return False
    try:
        return nli_entailment(support, answer) >= threshold
    except Exception:
        return False


def _edge_value(edge: dict, key: str) -> str:
    return str(edge.get(key, "") or "").strip()


def _relation_claims(tf_sfg_edges: List[dict]) -> List[str]:
    claims: List[str] = []
    seen: set[str] = set()
    for edge in tf_sfg_edges:
        claim = _edge_value(edge, "relation_claim")
        if not claim:
            continue
        key = claim.lower()
        if key in seen:
            continue
        seen.add(key)
        claims.append(claim)
    return claims


def _format_relation_context(tf_sfg_edges: List[dict]) -> str:
    lines: List[str] = []
    for edge in tf_sfg_edges[:3]:
        relation_type = _edge_value(edge, "type") or "related"
        claim = _edge_value(edge, "relation_claim")
        quote = _edge_value(edge, "bridging_quote")
        parts = [f"type={relation_type}"]
        if claim:
            parts.append(f"claim={claim}")
        if quote:
            parts.append(f"bridge={quote}")
        lines.append("- " + "; ".join(parts))
    if not lines:
        return ""
    return "\n\nVerified relation evidence:\n" + "\n".join(lines)


def _compose_fallback_answer(
    steps: List["ARMStep"],
    relation_claims: List[str],
    chain_facts: List[Fact],
    *,
    threshold: float,
) -> str:
    joined_steps = " ".join(s.claim for s in steps)
    if not relation_claims:
        return joined_steps
    relation_joined = " ".join(relation_claims + [joined_steps])
    if _answer_entailed_by_sources(relation_joined, chain_facts, threshold=threshold):
        return relation_joined
    return joined_steps


# ---------- dataclasses ----------


@dataclass
class ARMStep:
    step_id: str
    claim: str
    fact_ids: List[str]
    relation_type: str
    bbox_refs: List[dict]
    nli_score: float
    np_overlap: float
    passes: bool

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "claim": self.claim,
            "fact_ids": self.fact_ids,
            "relation_type": self.relation_type,
            "bbox_refs": self.bbox_refs,
            "nli_score": round(self.nli_score, 4),
            "np_overlap": round(self.np_overlap, 4),
            "passes": self.passes,
        }


@dataclass
class ARMResult:
    gold_answer: str
    reasoning_trace: List[dict] = field(default_factory=list)
    all_steps_pass: bool = True
    retries: int = 0
    construction_method: str = "SFU+ARM"

    @property
    def all_clauses_pass(self) -> bool:
        return self.all_steps_pass


# ---------- public API ----------


def build_abstractive_answer(
    question: str,
    chain_facts: List[Fact],
    llm: "LLM",
    *,
    tf_sfg_edges: Optional[List[dict]] = None,
    threshold: float = ARM_NLI_THRESHOLD,
    max_np_overlap: float = ARM_MAX_NP_OVERLAP,
    max_step_retries: int = ARM_MAX_STEP_RETRIES,
    cost_tracker: object = None,
    doc_id: str = "",
) -> ARMResult:
    """Build an abstractive gold answer with a per-step reasoning trace.

    Each chain fact becomes one reasoning step. Steps are NLI-verified
    (claim entails from source) and compression-checked (≤35% 4-gram overlap).
    Steps are composed into the final gold_answer by the LLM.
    """
    if not chain_facts:
        raise ValueError("ARM requires at least one chain fact")

    tf_sfg_edges = tf_sfg_edges or []
    edge_by_dst: dict = {e.get("dst"): e for e in tf_sfg_edges if e.get("dst")}

    steps: List[ARMStep] = []
    for i, fact in enumerate(chain_facts):
        edge = edge_by_dst.get(fact.fact_id, {})
        relation_type = edge.get("type", "definition")

        bbox_refs: List[dict] = []
        for span in (fact.supporting_spans or []):
            bbox_refs.append(_span_bbox_ref(span))

        claim, overlap, nli_score = _synthesize_step(
            fact,
            llm,
            max_np_overlap=max_np_overlap,
            max_retries=max_step_retries,
            nli_threshold=threshold,
            cost_tracker=cost_tracker,
            doc_id=doc_id,
        )

        passes = overlap <= max_np_overlap and nli_score >= threshold
        step = ARMStep(
            step_id=f"s{i + 1}",
            claim=claim,
            fact_ids=[fact.fact_id],
            relation_type=relation_type,
            bbox_refs=bbox_refs,
            nli_score=nli_score,
            np_overlap=overlap,
            passes=passes,
        )
        steps.append(step)
        if not passes:
            logger.debug(
                f"[ARM] step {step.step_id} marginal: "
                f"nli={nli_score:.3f} overlap={overlap:.3f}"
            )

    all_pass = all(s.passes for s in steps)

    # Compose steps into a final answer.
    steps_text = "\n".join(f"{s.step_id}. {s.claim}" for s in steps)
    relation_context = _format_relation_context(tf_sfg_edges)
    relation_claims = _relation_claims(tf_sfg_edges)
    compose_prompt = _COMPOSE_PROMPT.format(
        question=question,
        steps=steps_text,
        relation_context=relation_context,
    )
    compose_llm = llm
    if cost_tracker is not None:
        from rag_gt.observability.cost_tracker import TrackedLLM

        compose_llm = TrackedLLM(
            llm, cost_tracker, stage="arm_compose", doc_id=doc_id
        )
    from rag_gt.observability.cost_tracker import LiveCallBudgetExceeded

    try:
        gold_answer = compose_llm.generate(
            compose_prompt, temperature=0.0, max_tokens=256
        ).strip()
    except LiveCallBudgetExceeded:
        raise
    except Exception as e:
        logger.warning(f"[ARM] compose LLM error: {e}; joining step claims")
        gold_answer = " ".join(s.claim for s in steps)

    fallback_answer = _compose_fallback_answer(
        steps,
        relation_claims,
        chain_facts,
        threshold=threshold,
    )
    if not gold_answer:
        gold_answer = fallback_answer
    elif not _final_answer_covers_steps(gold_answer, steps, threshold=threshold):
        logger.debug("[ARM] composed answer dropped at least one step; joining claims")
        gold_answer = fallback_answer
    elif not _answer_entailed_by_sources(gold_answer, chain_facts, threshold=threshold):
        logger.debug("[ARM] composed answer added unsupported content; joining claims")
        gold_answer = fallback_answer

    return ARMResult(
        gold_answer=gold_answer,
        reasoning_trace=[s.to_dict() for s in steps],
        all_steps_pass=all_pass,
        retries=0,
        construction_method="SFU+ARM",
    )
