"""Constructive Gold Answer (CGA) builder.

Phase 2 of the SFU+CGA+SAE plan (plan v6). Builds a gold answer for a
multi-hop question from a chain of Semantic Fact Units (SFUs) under
clause-level NLI grounding. The property this produces:

  For every atomic clause C in the gold answer, there exists at least one
  SFU S in the chain such that S NLI-entails C above the threshold.

This is the guarantee that a perfect RAG can reconstruct the gold answer
from the retrieved SFUs alone — no world-knowledge leak.

Construction pipeline:

  1. Compose: concatenate each SFU's canonical_form in chain order.
  2. Paraphrase-only LLM pass: rewrite for fluency at temperature 0 with a
     strict no-new-content prompt.
  3. Decompose: split the result into atomic clauses
     (`validation.atomic_decomp.decompose_into_atomic_clauses`).
  4. Guard: NLI-check each clause against the union of SFU canonical_forms
     (`validation.nli_check.atomic_clause_entailment`).
  5. If all clauses pass: return the answer. Else retry up to N times,
     then either return the best candidate or surface a structured failure.

Distinct from `generation.answers.generate_answer` which is the *system
under test* (runtime answer generator). CGA is only ever used at GT
construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from loguru import logger

from rag_gt.core.types import Fact
from rag_gt.validation.atomic_decomp import decompose_into_atomic_clauses
from rag_gt.validation.nli_check import (
    GT_CONSISTENCY_THRESHOLD,
    atomic_clause_entailment,
)

if TYPE_CHECKING:
    from rag_gt.core.llm import LLM


MAX_CGA_RETRIES = 3
CGA_NLI_THRESHOLD = GT_CONSISTENCY_THRESHOLD  # 0.55 by default; configurable


_PARAPHRASE_PROMPT = """Rewrite the passage below for fluency, keeping the
meaning IDENTICAL.

Strict rules:
- Do NOT add ANY new factual content.
- Do NOT remove ANY factual content.
- Do NOT import outside knowledge.
- Combine the parts into ONE coherent paragraph.
- Preserve all numerical values, named entities, and symbols verbatim.
- Output ONLY the rewritten paragraph — no preamble, no headers, no quotes.

Passage:
<<PASSAGE>>
{passage}
<</PASSAGE>>

Rewrite:"""


@dataclass
class AtomicClauseGrounding:
    clause: str
    passes: bool
    best_score: float
    grounded_sfu_index: int  # index into the original chain_facts list


@dataclass
class CGAResult:
    gold_answer: str
    atomic_clauses: List[str]
    clause_grounding: List[AtomicClauseGrounding]
    all_clauses_pass: bool
    retries: int
    construction_method: str = "SFU+CGA"
    # v16 P5: populated when ARM is used in place of the paraphrase step.
    reasoning_trace: List[dict] = field(default_factory=list)

    @property
    def grounded_clause_count(self) -> int:
        return sum(1 for g in self.clause_grounding if g.passes)

    @property
    def total_clause_count(self) -> int:
        return len(self.clause_grounding)


def _support_texts_for_chain(chain_facts: List[Fact]) -> List[str]:
    """Return the SFU support texts in chain order.

    Prefer ``canonical_form`` when populated (the post-SFU-upgrade case).
    Otherwise fall back to ``text``. This lets CGA work on partially-upgraded
    GTs as well as legacy V11 GTs.
    """
    out: List[str] = []
    for f in chain_facts:
        choice = f.canonical_form.strip() or f.text.strip()
        out.append(choice)
    return out


def build_constructive_gold_answer(
    question: str,
    chain_facts: List[Fact],
    paraphrase_llm: "LLM",
    *,
    decomposer_llm: "LLM | None" = None,
    threshold: float = CGA_NLI_THRESHOLD,
    max_retries: int = MAX_CGA_RETRIES,
    max_np_overlap: float | None = None,
    use_arm: bool = False,
    tf_sfg_edges: Optional[List[dict]] = None,
    cost_tracker: object = None,
    doc_id: str = "",
) -> CGAResult:
    """Compose + paraphrase + decompose + NLI-guard a gold answer.

    `chain_facts` are the SFUs in chain order. `paraphrase_llm` runs the
    fluency rewrite; `decomposer_llm` is optional — when None, the heuristic
    splitter in `atomic_decomp` is used (cheaper, lower-quality but
    deterministic).

    When `use_arm=True` (v16 P5), delegates to ARM's decompose-then-synthesize
    pipeline instead of paraphrase-only. `tf_sfg_edges` is forwarded to ARM so
    it can annotate reasoning steps with edge relation types.

    The returned `CGAResult.all_clauses_pass=True` is the contract: a perfect
    RAG can reproduce this gold answer from the SFUs alone. Callers (e.g.
    `pipeline.gt_pipeline`) should reject the question when False — or
    accept and flag it as a "partially grounded" GT.
    """
    if use_arm:
        from rag_gt.generation.arm import ARM_MAX_NP_OVERLAP, build_abstractive_answer

        arm_result = build_abstractive_answer(
            question,
            chain_facts,
            paraphrase_llm,
            tf_sfg_edges=tf_sfg_edges or [],
            threshold=threshold,
            max_np_overlap=max_np_overlap if max_np_overlap is not None else ARM_MAX_NP_OVERLAP,
            max_step_retries=max_retries,
            cost_tracker=cost_tracker,
            doc_id=doc_id,
        )
        # Wrap ARM result as CGAResult so callers need no changes.
        pseudo_grounding = [
            AtomicClauseGrounding(
                clause=step["claim"],
                passes=step["passes"],
                best_score=step["nli_score"],
                grounded_sfu_index=i,
            )
            for i, step in enumerate(arm_result.reasoning_trace)
        ]
        return CGAResult(
            gold_answer=arm_result.gold_answer,
            atomic_clauses=[s["claim"] for s in arm_result.reasoning_trace],
            clause_grounding=pseudo_grounding,
            all_clauses_pass=arm_result.all_steps_pass,
            retries=arm_result.retries,
            construction_method="SFU+ARM",
            reasoning_trace=arm_result.reasoning_trace,
        )
    if not chain_facts:
        raise ValueError("CGA requires at least one SFU in chain_facts")

    support_texts = _support_texts_for_chain(chain_facts)

    # Step 1: compose canonical_forms in chain order.
    composed = "\n\n".join(s.strip() for s in support_texts if s.strip())
    if not composed:
        raise ValueError("CGA: composed support text is empty after stripping")

    from rag_gt.core.llm import APIError

    best_result: Optional[CGAResult] = None

    for attempt in range(max_retries):
        # Step 2: paraphrase-only LLM pass.
        prompt = _PARAPHRASE_PROMPT.format(passage=composed)
        try:
            paraphrased = paraphrase_llm.generate(
                prompt, temperature=0.0, max_tokens=512
            ).strip()
        except APIError:
            raise
        except Exception as e:
            logger.warning(
                f"[CGA] paraphrase LLM error attempt {attempt + 1}: "
                f"{type(e).__name__}: {e}; falling back to verbatim composition"
            )
            paraphrased = composed

        if not paraphrased:
            paraphrased = composed

        # Step 3: decompose into atomic clauses.
        clauses = decompose_into_atomic_clauses(
            paraphrased,
            llm=decomposer_llm,
            use_llm=decomposer_llm is not None,
        )
        if not clauses:
            logger.warning(
                f"[CGA] attempt {attempt + 1}: no atomic clauses produced; retrying"
            )
            continue

        # Step 4: NLI-guard each clause against SFU canonical_forms.
        grounding = atomic_clause_entailment(
            clauses, support_texts, threshold=threshold
        )
        agc = [
            AtomicClauseGrounding(
                clause=clause,
                passes=passes,
                best_score=score,
                grounded_sfu_index=best_idx,
            )
            for clause, (passes, score, best_idx) in zip(clauses, grounding)
        ]
        all_pass = all(g.passes for g in agc)

        candidate = CGAResult(
            gold_answer=paraphrased,
            atomic_clauses=clauses,
            clause_grounding=agc,
            all_clauses_pass=all_pass,
            retries=attempt,
        )

        if all_pass:
            return candidate

        # Track the candidate that passed the most clauses, in case all
        # retries fail.
        if best_result is None or candidate.grounded_clause_count > best_result.grounded_clause_count:
            best_result = candidate

        logger.debug(
            f"[CGA] attempt {attempt + 1}: "
            f"{candidate.grounded_clause_count}/{candidate.total_clause_count} clauses grounded; retrying"
        )

    # All retries exhausted: return the best candidate, marked as failing.
    if best_result is not None:
        logger.info(
            f"[CGA] retries exhausted: best candidate "
            f"{best_result.grounded_clause_count}/{best_result.total_clause_count} clauses grounded"
        )
        return best_result

    # Hard fallback: verbatim composition with no decomposition.
    return CGAResult(
        gold_answer=composed,
        atomic_clauses=[composed],
        clause_grounding=[
            AtomicClauseGrounding(
                clause=composed,
                passes=True,
                best_score=1.0,
                grounded_sfu_index=0,
            )
        ],
        all_clauses_pass=True,
        retries=max_retries,
        construction_method="SFU+CGA-fallback-verbatim",
    )
