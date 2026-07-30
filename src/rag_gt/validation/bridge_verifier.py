"""Stage D deterministic-first verifier for bridge and single-control QA."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from loguru import logger

from rag_gt.core.config import load_config
from rag_gt.core.llm import get_llm
from rag_gt.generation.answer_first import bridge_is_hidden
from rag_gt.validation.nli_check import nli_batch


def _fact_id(record: Mapping[str, object]) -> str:
    return str(record.get("fact_id") or record.get("id") or "").strip()


def _fact_text(record: Mapping[str, object]) -> str:
    return str(record.get("canonical_form") or record.get("text") or "").strip()


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _bridge_present(text: str, bridge: str) -> bool:
    text_norm = f" {_normalize(text)} "
    bridge_norm = _normalize(bridge)
    if not bridge_norm:
        return False
    if f" {bridge_norm} " in text_norm:
        return True
    compact = bridge_norm.replace(" ", "")
    return len(compact) >= 6 and compact in text_norm.replace(" ", "")


def _token_jaccard(left: str, right: str) -> float:
    a = set(_normalize(left).split())
    b = set(_normalize(right).split())
    return len(a & b) / len(a | b) if a and b else 0.0


# Advisory-only floor for single-fact QA. A question whose content words barely
# overlap the source fact is the QUESTION_NOT_FROM_FACTS audit failure. We do NOT
# reject on this (the fix is the improved generation prompt); we surface the score
# and a boolean so the failure mode is measurable inside the pipeline instead of
# only in a post-hoc audit (BUG-F).
_SINGLE_QUESTION_OVERLAP_FLOOR = 0.30
_QF_STOPWORDS = frozenset(
    "a an the is are was were be been being do does did what which who how when "
    "where why of in on at to for with by from and or but not this that these "
    "those it its as also than then there their they both of s".split()
)


def _question_fact_overlap(question: str, fact_text: str) -> float:
    """Fraction of the question's content words that appear in the fact text."""
    q = {w for w in _normalize(question).split() if w not in _QF_STOPWORDS and len(w) > 2}
    f = {w for w in _normalize(fact_text).split() if w not in _QF_STOPWORDS and len(w) > 2}
    if not q:
        return 0.0
    return len(q & f) / len(q)


def _settings() -> dict:
    cfg = load_config()
    out = dict(cfg["bridge_verifier"])
    answer_first = cfg["answer_first"]
    out["clause_entailment_min"] = float(answer_first["clause_entailment_min"])
    out["single_fact_answer_max"] = float(answer_first["single_fact_answer_max"])
    out["joint_answer_min"] = float(answer_first["joint_answer_min"])
    return out


def _judge_prompt(qa: Mapping[str, object], facts: Sequence[Mapping[str, object]]) -> str:
    fact_block = "\n".join(
        f"SOURCE {index + 1}: {_fact_text(fact)}" for index, fact in enumerate(facts)
    )
    return f"""You are the borderline judge in a grounded QA validation cascade.
{fact_block}

QUESTION: {qa.get('question', '')}
ANSWER: {qa.get('answer', '')}

Decide whether the answer is fully supported by BOTH sources and whether removing
either source makes the complete answer unavailable. Do not add outside knowledge.
Return only JSON: {{"verdict":"PASS|FIX|REJECT","reason":"brief reason"}}
PASS means faithful and genuinely needs both sources. FIX means only question wording
needs repair. REJECT means unsupported, redundant, or effectively single-source."""


def _borderline_judge(
    qa: Mapping[str, object],
    facts: Sequence[Mapping[str, object]],
    llm: object,
    settings: Mapping[str, object],
) -> tuple[str, str]:
    try:
        result = llm.generate_json(
            _judge_prompt(qa, facts),
            temperature=float(settings["judge_temperature"]),
            max_tokens=int(settings["judge_max_tokens"]),
        )
    except Exception as exc:
        return "REJECT", f"borderline_judge_error:{type(exc).__name__}"
    verdict = str(result.get("verdict") or "").strip().upper()
    if verdict not in {"PASS", "FIX", "REJECT"}:
        return "REJECT", "borderline_judge_invalid"
    return verdict, str(result.get("reason") or "borderline_judge").strip()


def verify_qa_pairs(
    qa_pairs: Sequence[Mapping[str, object]],
    facts: Sequence[Mapping[str, object]],
    *,
    llm: object | None = None,
    nli_fn: Callable[[list[tuple[str, str]]], list[float]] = nli_batch,
    settings: Mapping[str, object] | None = None,
) -> dict:
    """Apply the Stage D cascade and return every QA with a final verdict."""
    cfg = dict(settings or _settings())
    facts_by_id = {_fact_id(fact): fact for fact in facts if _fact_id(fact)}
    pending: list[tuple[int, Mapping[str, object], list[Mapping[str, object]]]] = []
    nli_inputs: list[tuple[str, str]] = []
    output = [dict(qa) for qa in qa_pairs]
    reasons: Counter = Counter()

    for index, qa in enumerate(qa_pairs):
        if str(qa.get("hop_type")) == "single":
            fact_ids = list(qa.get("gold_fact_ids") or [])
            fact = facts_by_id.get(str(fact_ids[0])) if fact_ids else None
            faithful = bool(fact and str(qa.get("answer") or "").strip() == _fact_text(fact))
            # Advisory question-grounding signal (BUG-F): the answer==fact check is
            # true by construction and says nothing about whether the QUESTION was
            # drawn from the fact. Measure it so QUESTION_NOT_FROM_FACTS is visible
            # in-pipeline. This does NOT gate the verdict — quality is fixed at the
            # generation prompt, not by rejecting rows here.
            q_overlap = (
                _question_fact_overlap(str(qa.get("question") or ""), _fact_text(fact))
                if fact
                else 0.0
            )
            question_grounded = q_overlap >= _SINGLE_QUESTION_OVERLAP_FLOOR
            verdict = "PASS" if faithful else "REJECT"
            reason = "single_control" if faithful else "single_fact_mismatch"
            output[index]["verify"] = {
                "bridge_in_both": None,
                "necessity_passed": None,
                "bridge_hidden": None,
                "duplicate": False,
                "faithful": faithful,
                "question_fact_overlap": round(q_overlap, 3),
                "question_grounded": question_grounded,
                "borderline": False,
                "judge_used": False,
                "verdict": verdict,
                "reason": reason,
            }
            reasons[reason] += 1
            if faithful and not question_grounded:
                reasons["single_question_low_overlap_advisory"] += 1
            continue

        fact_ids = [str(value) for value in qa.get("gold_fact_ids") or []]
        support = [facts_by_id[value] for value in fact_ids if value in facts_by_id]
        bridge = str(qa.get("bridge_entity") or "")
        # Whether two facts genuinely share the bridge is a property of the SOURCE,
        # not of the self-containment rewrite. The canonical form may legitimately
        # reword a common-noun bridge phrase; if the verbatim source span still
        # carries the entity, the bridge is real. Check canonical OR raw source.
        bridge_in_both = len(support) == 2 and all(
            _bridge_present(_fact_text(fact), bridge)
            or _bridge_present(str(fact.get("raw_text") or ""), bridge)
            for fact in support
        )
        necessity_passed = bool((qa.get("necessity") or {}).get("passed"))
        hidden = bridge_is_hidden(str(qa.get("question") or ""), bridge)
        verify = {
            "bridge_in_both": bridge_in_both,
            "necessity_passed": necessity_passed,
            "bridge_hidden": hidden,
            "duplicate": None,
            "faithful": None,
            "borderline": False,
            "judge_used": False,
            "verdict": "REJECT",
            "reason": "",
        }
        if not bridge_in_both:
            verify["reason"] = "bridge_not_in_both"
        elif not necessity_passed:
            verify["reason"] = "necessity_failed"
        elif not hidden:
            verify["reason"] = "bridge_leaked"
        elif len(qa.get("answer_clauses") or []) != 2:
            verify["reason"] = "invalid_answer_clauses"
        else:
            output[index]["verify"] = verify
            clauses = [str(c.get("text") or "") for c in qa["answer_clauses"]]
            answer = str(qa.get("answer") or "")
            pending.append((index, qa, support))
            nli_inputs.extend(
                [
                    (_fact_text(support[0]), clauses[0]),
                    (_fact_text(support[1]), clauses[1]),
                    (f"{_fact_text(support[0])} {_fact_text(support[1])}", answer),
                    (clauses[0], clauses[1]),
                    (clauses[1], clauses[0]),
                ]
            )
            continue
        output[index]["verify"] = verify
        reasons[str(verify["reason"])] += 1

    scores = nli_fn(nli_inputs) if nli_inputs else []
    if len(scores) != len(nli_inputs):
        raise RuntimeError(f"NLI returned {len(scores)} scores for {len(nli_inputs)} inputs")

    for score_index, (index, qa, support) in enumerate(pending):
        own_a, own_b, joint, a_to_b, b_to_a = [
            float(value) for value in scores[score_index * 5 : score_index * 5 + 5]
        ]
        clauses = [str(c.get("text") or "") for c in qa["answer_clauses"]]
        jaccard = _token_jaccard(clauses[0], clauses[1])
        mutual = (
            a_to_b >= float(cfg["mutual_entailment_min"])
            and b_to_a >= float(cfg["mutual_entailment_min"])
        )
        duplicate = jaccard >= float(cfg["duplicate_jaccard_max"]) or mutual
        faithful = (
            own_a >= float(cfg["clause_entailment_min"])
            and own_b >= float(cfg["clause_entailment_min"])
            and joint >= float(cfg["joint_answer_min"])
        )
        verify = output[index]["verify"]
        verify.update(
            {
                "duplicate": duplicate,
                "faithful": faithful,
                "scores": {
                    "clause_a": round(own_a, 3),
                    "clause_b": round(own_b, 3),
                    "joint": round(joint, 3),
                    "a_to_b": round(a_to_b, 3),
                    "b_to_a": round(b_to_a, 3),
                    "clause_jaccard": round(jaccard, 3),
                },
            }
        )
        if duplicate:
            verify["reason"] = "duplicate_clauses"
            reasons["duplicate_clauses"] += 1
            continue
        if not faithful:
            verify["reason"] = "not_faithful"
            reasons["not_faithful"] += 1
            continue

        margin = float(cfg["borderline_margin"])
        necessity = qa.get("necessity") or {}
        borderline = any(
            [
                abs(own_a - float(cfg["clause_entailment_min"])) <= margin,
                abs(own_b - float(cfg["clause_entailment_min"])) <= margin,
                abs(joint - float(cfg["joint_answer_min"])) <= margin,
                abs(jaccard - float(cfg["duplicate_jaccard_max"])) <= margin,
                abs(float(necessity.get("nli_a", 0.0)) - float(cfg["single_fact_answer_max"])) <= margin,
                abs(float(necessity.get("nli_b", 0.0)) - float(cfg["single_fact_answer_max"])) <= margin,
            ]
        )
        verify["borderline"] = borderline
        if borderline and bool(cfg["use_llm_borderline"]):
            if llm is None:
                llm = get_llm("gt")
            verdict, reason = _borderline_judge(qa, support, llm, cfg)
            verify["judge_used"] = True
            verify["verdict"] = verdict
            verify["reason"] = reason
            reasons[f"judge_{verdict.lower()}"] += 1
        else:
            verify["verdict"] = "PASS"
            verify["reason"] = "deterministic_pass"
            reasons["deterministic_pass"] += 1

    verdicts = Counter(str(qa["verify"]["verdict"]) for qa in output)
    bridge_rows = [qa for qa in output if qa.get("hop_type") == "bridge"]
    single_rows = [qa for qa in output if qa.get("hop_type") == "single"]
    deterministic_bridge = sum(
        1 for qa in bridge_rows if not qa["verify"].get("judge_used")
    )
    n_judge_used = sum(1 for qa in output if qa["verify"].get("judge_used"))
    single_grounded = sum(
        1 for qa in single_rows if qa["verify"].get("question_grounded")
    )
    try:
        from rag_gt.validation.nli_check import truncation_stats
        nli_trunc = truncation_stats()
    except Exception:
        nli_trunc = {}
    return {
        "pairs": output,
        "stats": {
            "n_input": len(qa_pairs),
            "n_bridge": len(bridge_rows),
            "verdicts": dict(sorted(verdicts.items())),
            "reasons": dict(sorted(reasons.items())),
            "deterministic_bridge_rate": round(
                deterministic_bridge / len(bridge_rows), 4
            )
            if bridge_rows
            else 1.0,
            # N2 — cascade economics: how much of verification the deterministic
            # path resolved on its own vs. how many pairs needed the LLM judge.
            "cascade": {
                "n_bridge": len(bridge_rows),
                "n_deterministic": deterministic_bridge,
                "n_llm_judge_used": n_judge_used,
                "llm_judge_fraction": round(n_judge_used / len(output), 4)
                if output
                else 0.0,
            },
            # BUG-F advisory rollup: fraction of single QAs whose question is
            # actually grounded in its source fact.
            "single_question_grounded_rate": round(
                single_grounded / len(single_rows), 4
            )
            if single_rows
            else 1.0,
            # BUG-J: NLI-input truncation events observed this process.
            "nli_truncation": nli_trunc,
        },
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage D bridge QA verification")
    parser.add_argument("facts", type=Path)
    parser.add_argument("qa_pairs", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    facts = json.loads(args.facts.read_text(encoding="utf-8"))
    payload = json.loads(args.qa_pairs.read_text(encoding="utf-8"))
    result = verify_qa_pairs(payload["pairs"], facts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Stage D wrote {args.output}: {result['stats']}")
    return 0 if result["stats"]["verdicts"].get("PASS", 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
