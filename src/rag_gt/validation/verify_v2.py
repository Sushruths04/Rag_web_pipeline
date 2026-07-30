"""Stage D verifier for v2 evidence-dense QA (neighbor-pair singles + 2+2
clusters).

``bridge_verifier.verify_qa_pairs`` is a v1-schema instrument: its
``hop_type=="single"`` branch requires ``answer == fact_text`` verbatim (a
single fact's own text), and its bridge branch hard-codes
``len(answer_clauses) != 2``. Neither assumption holds for v2:
``neighbor_pair_v2`` composes its answer from 2 clauses via ``_compose_multi``
(never equal to one fact's text), and ``cluster_2plus2`` carries 4 clauses.
Running v2 records through ``verify_qa_pairs`` unmodified would reject every
one of them by construction, not because they are wrong.

This module generalizes the SAME deterministic-first cascade (per-clause
entailment, joint entailment, pairwise duplicate-clause check, borderline
-margin LLM judge escalation) to n in {2, 4} evidence units, reusing
``bridge_verifier``'s judge primitives rather than duplicating them.
``bridge_pair_v1`` records (the demoted 2-fact cross-page bridges) already
match ``verify_qa_pairs``' bridge-branch assumptions exactly and are routed
there unchanged — no reason to reinvent what already fits.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Callable, Mapping, Sequence

from rag_gt.core.config import load_config
from rag_gt.core.llm import get_llm
from rag_gt.generation.answer_first import bridge_is_hidden
from rag_gt.validation.bridge_verifier import (
    _borderline_judge,
    _fact_id,
    _fact_text,
    _token_jaccard,
    verify_qa_pairs,
)
from rag_gt.validation.nli_check import nli_batch

_N_BY_STRATEGY = {"neighbor_pair_v2": 2, "cluster_2plus2": 4}


def _settings() -> dict:
    cfg = load_config()
    out = dict(cfg["bridge_verifier"])
    answer_first = cfg["answer_first"]
    out["clause_entailment_min"] = float(answer_first["clause_entailment_min"])
    out["single_fact_answer_max"] = float(answer_first["single_fact_answer_max"])
    out["joint_answer_min"] = float(answer_first["joint_answer_min"])
    return out


def _rejected(reasons: Counter, reason: str) -> dict:
    reasons[reason] += 1
    return {
        "bridge_hidden": None, "duplicate": None, "faithful": None,
        "borderline": False, "judge_used": False, "verdict": "REJECT", "reason": reason,
    }


def verify_v2_pairs(
    qa_pairs: Sequence[Mapping[str, object]],
    facts: Sequence[Mapping[str, object]],
    *,
    llm: object | None = None,
    nli_fn: Callable[[list[tuple[str, str]]], list[float]] = nli_batch,
    settings: Mapping[str, object] | None = None,
) -> dict:
    """Apply the Stage D cascade to a mixed batch of v2 evidence strategies.

    Returns every input QA with a final ``verify`` field, in original order,
    plus aggregate stats (verdict histogram, routing counts, NLI truncation).
    """
    cfg = dict(settings or _settings())
    facts_by_id = {_fact_id(f): f for f in facts if _fact_id(f)}
    output: list[dict] = [dict(qa) for qa in qa_pairs]
    reasons: Counter = Counter()

    v1_indices = [i for i, qa in enumerate(qa_pairs)
                  if str(qa.get("evidence_strategy")) == "bridge_pair_v1"]
    if v1_indices:
        v1_result = verify_qa_pairs(
            [qa_pairs[i] for i in v1_indices], facts, llm=llm, nli_fn=nli_fn, settings=cfg,
        )
        for local_i, orig_i in enumerate(v1_indices):
            output[orig_i] = v1_result["pairs"][local_i]
        # Merge the sub-verifier's own BUCKETED reason histogram (e.g.
        # "judge_pass"/"bridge_not_in_both"), not each pair's raw verify
        # ["reason"] field — for a judge-escalated row that field holds the
        # LLM's free-text justification, which would otherwise pollute this
        # histogram with one-off sentences instead of a countable bucket.
        reasons.update(v1_result["stats"]["reasons"])

    v1_set = set(v1_indices)
    pending: list[tuple] = []
    nli_inputs: list[tuple[str, str]] = []

    for i, qa in enumerate(qa_pairs):
        if i in v1_set:
            continue
        strategy = str(qa.get("evidence_strategy") or "")
        n = _N_BY_STRATEGY.get(strategy)
        if n is None:
            output[i]["verify"] = _rejected(reasons, "unknown_evidence_strategy")
            continue
        fact_ids = [str(v) for v in qa.get("gold_fact_ids") or []]
        clauses = [str(c.get("text") or "") for c in qa.get("answer_clauses") or []]
        if len(fact_ids) != n or len(clauses) != n:
            output[i]["verify"] = _rejected(reasons, "invalid_answer_clauses")
            continue
        support = [facts_by_id.get(fid) for fid in fact_ids]
        if any(f is None for f in support):
            output[i]["verify"] = _rejected(reasons, "missing_fact")
            continue
        bridge = str(qa.get("bridge_entity") or "")
        hidden: bool | None = None
        if strategy == "cluster_2plus2":
            hidden = bridge_is_hidden(str(qa.get("question") or ""), bridge)
            if not hidden:
                output[i]["verify"] = _rejected(reasons, "bridge_leaked")
                continue

        fact_texts = [_fact_text(f) for f in support]  # type: ignore[arg-type]
        answer = str(qa.get("answer") or "")
        output[i]["verify"] = {
            "bridge_hidden": hidden, "duplicate": None, "faithful": None,
            "borderline": False, "judge_used": False, "verdict": "REJECT", "reason": "",
        }

        start = len(nli_inputs)
        for fact_text, clause in zip(fact_texts, clauses):
            nli_inputs.append((fact_text, clause))
        nli_inputs.append((" ".join(fact_texts), answer))
        pair_slots = list(combinations(range(n), 2))
        for a, b in pair_slots:
            nli_inputs.append((clauses[a], clauses[b]))
            nli_inputs.append((clauses[b], clauses[a]))
        pending.append((i, qa, support, clauses, start, n, pair_slots))

    scores = nli_fn(nli_inputs) if nli_inputs else []
    if len(scores) != len(nli_inputs):
        raise RuntimeError(f"NLI returned {len(scores)} scores for {len(nli_inputs)} inputs")

    for (i, qa, support, clauses, start, n, pair_slots) in pending:
        cursor = start
        clause_scores = [float(v) for v in scores[cursor:cursor + n]]
        cursor += n
        joint = float(scores[cursor])
        cursor += 1
        max_jaccard = 0.0
        mutual = False
        for a, b in pair_slots:
            a_to_b = float(scores[cursor])
            b_to_a = float(scores[cursor + 1])
            cursor += 2
            max_jaccard = max(max_jaccard, _token_jaccard(clauses[a], clauses[b]))
            if a_to_b >= float(cfg["mutual_entailment_min"]) and b_to_a >= float(cfg["mutual_entailment_min"]):
                mutual = True

        duplicate = max_jaccard >= float(cfg["duplicate_jaccard_max"]) or mutual
        faithful = (
            all(s >= float(cfg["clause_entailment_min"]) for s in clause_scores)
            and joint >= float(cfg["joint_answer_min"])
        )
        verify = output[i]["verify"]
        verify.update({
            "duplicate": duplicate,
            "faithful": faithful,
            "scores": {
                "clauses": [round(s, 3) for s in clause_scores],
                "joint": round(joint, 3),
                "max_pairwise_jaccard": round(max_jaccard, 3),
            },
        })
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
        singles = [float(v) for v in (necessity.get("nli_singles") or [])]
        borderline = (
            any(abs(s - float(cfg["clause_entailment_min"])) <= margin for s in clause_scores)
            or abs(joint - float(cfg["joint_answer_min"])) <= margin
            or abs(max_jaccard - float(cfg["duplicate_jaccard_max"])) <= margin
            or any(abs(s - float(cfg["single_fact_answer_max"])) <= margin for s in singles)
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

    verdicts = Counter(str(row["verify"]["verdict"]) for row in output)
    try:
        from rag_gt.validation.nli_check import truncation_stats
        nli_trunc = truncation_stats()
    except Exception:
        nli_trunc = {}

    return {
        "pairs": output,
        "stats": {
            "n_input": len(qa_pairs),
            "n_v1_bridge_routed": len(v1_indices),
            "n_v2_native": len(qa_pairs) - len(v1_indices),
            "verdicts": dict(sorted(verdicts.items())),
            "reasons": dict(sorted(reasons.items())),
            "nli_truncation": nli_trunc,
        },
    }
