"""AT/CT twin generator — v16 P4.

For each accepted strict QuestionGT row, produces:
  AT (abstention twin) — removes the most load-bearing chain fact; gold answer
      becomes the abstention text.
  CT (counterfactual twin) — perturbs one numeric/NE span in a chain fact via
      LLM mask-infill; gold answer is regenerated from the perturbed chain.

Twins carry twin_of / twin_type / twin_metadata and difficulty_adversarial so
the difficulty matrix can place them in the (hops, sem_dist, adversarial) cell.
"""

from __future__ import annotations

import copy
import hashlib
import re
from typing import TYPE_CHECKING, List, Optional

from loguru import logger

from rag_gt.core.types import MSFS, Fact, QuestionGT
from rag_gt.core.llm import APIError
from rag_gt.generation.answers import ABSTENTION_TEXT, generate_answer
from rag_gt.generation.counterfactual_infill import infill_counterfactual

if TYPE_CHECKING:
    from rag_gt.core.llm import LLM


# ---------- Load-bearing fact detection ----------

def _load_bearing_fact_id(q: QuestionGT) -> str:
    """Return the fact_id that is most load-bearing for answering the question.

    Strategy (in priority order):
    1. If question_assumptions is populated, return the fact that grounds the
       REL assumption with the LOWEST NLI score (the weakest link — removing it
       is most likely to make the question unanswerable).
    2. Otherwise, return the LAST fact in required_fact_ids (closest to answer).
    """
    if q.question_assumptions:
        rel_verdicts = [
            a for a in q.question_assumptions
            if str(a.get("type", "")).upper() == "REL"
        ]
        if rel_verdicts:
            weakest = min(rel_verdicts, key=lambda a: float(a.get("nli_score", 1.0)))
            sfu_idx = int(weakest.get("entails_from_sfu", -1))
            if 0 <= sfu_idx < len(q.required_fact_ids):
                return q.required_fact_ids[sfu_idx]

    return q.required_fact_ids[-1] if q.required_fact_ids else ""


# ---------- AT twin ----------

def _make_abstention_twin(q: QuestionGT) -> Optional[QuestionGT]:
    """Derive one AT twin by removing the load-bearing fact."""
    if len(q.required_fact_ids) < 2:
        return None  # single-fact chain has no internal facts to remove

    removed_id = _load_bearing_fact_id(q)
    if not removed_id:
        return None

    remaining_ids = [fid for fid in q.required_fact_ids if fid != removed_id]
    remaining_facts = [f for f in q.required_facts if f.fact_id != removed_id]

    if not remaining_ids:
        return None

    twin_q_id = f"{q.q_id}_at"
    msfs_id = f"{twin_q_id}_msfs1"

    return QuestionGT(
        q_id=twin_q_id,
        question=q.question,
        gold_answer=ABSTENTION_TEXT,
        msfs_list=[MSFS(msfs_id=msfs_id, fact_ids=remaining_ids)],
        doc_ids=list(q.doc_ids),
        required_fact_ids=remaining_ids,
        difficulty_reasoning_depth=q.difficulty_reasoning_depth,
        difficulty_semantic_distance=q.difficulty_semantic_distance,
        required_facts=remaining_facts,
        required_fact_groups=[[fid] for fid in remaining_ids],
        hop_type=q.hop_type,
        minimum_required_fact_count=len(remaining_ids),
        relation_type=q.relation_type,
        # v16 fields
        tf_sfg_edges=list(q.tf_sfg_edges),
        question_assumptions=list(q.question_assumptions),
        distractor_spans=list(q.distractor_spans),
        twin_of=q.q_id,
        twin_type="abstention",
        twin_metadata={"removed_fact_id": removed_id},
        difficulty_adversarial="abstention",
        intent=q.intent,
    )


# ---------- CT twin ----------

def _make_counterfactual_twin(
    q: QuestionGT,
    llm: "LLM",
    *,
    ct_threshold: float = 0.40,
    ct_max_retries: int = 3,
    require_answer_change: bool = True,
    validate_answer_entailment: bool = False,
    cost_tracker: object = None,
    doc_id: str = "",
) -> Optional[QuestionGT]:
    """Derive one CT twin by perturbing a span in a chain fact.

    Selects the fact most likely to have a perturb-able span: prefers facts
    with numeric spans, falls back to facts with named-entity spans.
    """
    if not q.required_facts:
        return None

    # Try each required fact in order until one produces a valid infill.
    ct_result = None
    perturbed_fact_orig: Optional[Fact] = None
    infill_llm = llm
    answer_llm = llm
    if cost_tracker is not None:
        from rag_gt.observability.cost_tracker import TrackedLLM

        infill_llm = TrackedLLM(
            llm, cost_tracker, stage="counterfactual_infill", doc_id=doc_id
        )
        answer_llm = TrackedLLM(
            llm, cost_tracker, stage="answer_gen", doc_id=doc_id
        )
    for fact in q.required_facts:
        result = infill_counterfactual(
            fact,
            infill_llm,
            contradiction_threshold=ct_threshold,
            max_retries=ct_max_retries,
        )
        if result is not None:
            ct_result = result
            perturbed_fact_orig = fact
            break

    if ct_result is None or perturbed_fact_orig is None:
        return None

    # Build perturbed fact (shallow copy with text replaced).
    perturbed_fact = copy.copy(perturbed_fact_orig)
    perturbed_fact.text = ct_result.perturbed_fact_text
    perturbed_fact.canonical_form = ct_result.perturbed_fact_text
    perturbed_fact.raw_text = perturbed_fact_orig.raw_text or perturbed_fact_orig.text

    perturbed_facts = [
        perturbed_fact if f.fact_id == perturbed_fact_orig.fact_id else f
        for f in q.required_facts
    ]

    # Regenerate gold answer from perturbed chain.
    try:
        new_answer = generate_answer(q.question, perturbed_facts, answer_llm)
    except APIError:
        raise
    except Exception as e:
        logger.warning(f"[CT] answer regen failed for {q.q_id}: {e}")
        return None

    if not new_answer or new_answer == ABSTENTION_TEXT:
        return None
    if require_answer_change and _norm_answer(new_answer) == _norm_answer(q.gold_answer):
        return None
    if validate_answer_entailment and not _ct_answer_is_valid(
        strict_answer=q.gold_answer,
        new_answer=new_answer,
        perturbed_facts=perturbed_facts,
    ):
        return None

    twin_q_id = f"{q.q_id}_ct"
    msfs_id = f"{twin_q_id}_msfs1"

    return QuestionGT(
        q_id=twin_q_id,
        question=q.question,
        gold_answer=new_answer,
        msfs_list=[MSFS(msfs_id=msfs_id, fact_ids=list(q.required_fact_ids))],
        doc_ids=list(q.doc_ids),
        required_fact_ids=list(q.required_fact_ids),
        difficulty_reasoning_depth=q.difficulty_reasoning_depth,
        difficulty_semantic_distance=q.difficulty_semantic_distance,
        required_facts=perturbed_facts,
        required_fact_groups=[[fid] for fid in q.required_fact_ids],
        hop_type=q.hop_type,
        minimum_required_fact_count=len(q.required_fact_ids),
        relation_type=q.relation_type,
        # v16 fields
        tf_sfg_edges=list(q.tf_sfg_edges),
        question_assumptions=list(q.question_assumptions),
        distractor_spans=list(q.distractor_spans),
        twin_of=q.q_id,
        twin_type="counterfactual",
        twin_metadata={
            "perturbed_fact_id": perturbed_fact_orig.fact_id,
            "original_span": ct_result.original_span,
            "perturbed_span": ct_result.perturbed_span,
            "original_value": ct_result.original_span,
            "perturbed_value": ct_result.perturbed_span,
            "contradiction_score": round(ct_result.contradiction_score, 4),
        },
        difficulty_adversarial="counterfactual",
        intent=q.intent,
    )


# ---------- Orchestrator ----------

def derive_twins(
    strict_rows: List[QuestionGT],
    llm: "LLM",
    *,
    twins_enabled: bool = True,
    ct_threshold: float = 0.40,
    ct_max_retries: int = 3,
    ct_sample_rate: float = 1.0,
    require_answer_change: bool = True,
    validate_answer_entailment: bool = False,
    cost_tracker: object = None,
    doc_id: str = "",
) -> List[QuestionGT]:
    """Derive AT + CT twins for each accepted strict row.

    Returns the twin rows only (not the original strict rows). Callers should
    concatenate `strict_rows + derive_twins(...)` for the final GT file.
    """
    if not twins_enabled:
        return []

    twins: List[QuestionGT] = []
    at_ok = at_skip = ct_ok = ct_skip = 0

    for q in strict_rows:
        at = _make_abstention_twin(q)
        if at is not None:
            twins.append(at)
            at_ok += 1
        else:
            at_skip += 1

        ct = None
        if _include_ct(q.q_id, ct_sample_rate):
            ct = _make_counterfactual_twin(
                q,
                llm,
                ct_threshold=ct_threshold,
                ct_max_retries=ct_max_retries,
                require_answer_change=require_answer_change,
                validate_answer_entailment=validate_answer_entailment,
                cost_tracker=cost_tracker,
                doc_id=doc_id,
            )
        if ct is not None:
            twins.append(ct)
            ct_ok += 1
        else:
            ct_skip += 1

    logger.info(
        f"[Twins] AT: {at_ok} ok / {at_skip} skipped; "
        f"CT: {ct_ok} ok / {ct_skip} skipped"
    )
    return twins


def _norm_answer(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _include_ct(q_id: str, sample_rate: float) -> bool:
    if sample_rate >= 1.0:
        return True
    if sample_rate <= 0:
        return False
    digest = hashlib.sha256(q_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return bucket <= sample_rate


def _ct_answer_is_valid(
    *,
    strict_answer: str,
    new_answer: str,
    perturbed_facts: List[Fact],
) -> bool:
    try:
        from rag_gt.validation.nli_check import GT_CONSISTENCY_THRESHOLD, nli_entailment

        context = " ".join(f.text for f in perturbed_facts)
        if nli_entailment(context, new_answer) < GT_CONSISTENCY_THRESHOLD:
            return False
        if strict_answer and nli_entailment(context, strict_answer) >= GT_CONSISTENCY_THRESHOLD:
            return False
    except Exception as e:
        logger.debug(f"[CT] answer validation failed: {e}")
        return False
    return True
