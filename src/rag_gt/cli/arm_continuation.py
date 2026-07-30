"""Targeted paid continuation — ARM threshold 0.38 test for ranks 3 and 5.

Reads the two arm_grounding_failed drop events from the Sutton p50 trace,
reconstructs Fact objects, re-runs ARM with max_np_overlap=0.38, then runs
quality / answer-NLI / minimality gates.  Budget: ≤20 live API calls.

Usage:
    python -m rag_gt.cli.arm_continuation
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import List

import yaml
from loguru import logger


TRACE_PATH = "data/eval_runs/sutton_nebius_q3_p50_dualanswerqg_v1611_cap150.trace.jsonl"
CONFIG_PATH = "configs/v16.yaml"
TARGET_CHAINS = {"chain_eea1df1fb246", "chain_1e2500c29571"}
MAX_LIVE_CALLS = 20


# ---------- Fact reconstruction ----------


def _reconstruct_facts(chain_dict: dict) -> List:
    from rag_gt.core.types import Fact, Span

    facts_raw = chain_dict.get("facts", [])
    facts: List[Fact] = []
    for fd in facts_raw:
        spans_raw = fd.get("supporting_spans_full", "[]")
        if isinstance(spans_raw, str):
            try:
                spans_list = ast.literal_eval(spans_raw)
            except Exception:
                spans_list = []
        else:
            spans_list = spans_raw

        supporting_spans = [Span.from_any(s) for s in spans_list]

        weight = float(fd.get("weight", 1.0) or 1.0)
        weight = max(0.01, min(weight, 2.0))

        facts.append(
            Fact(
                fact_id=str(fd["fact_id"]),
                text=str(fd["text"]),
                role=str(fd.get("role", "definition")),
                supporting_spans=supporting_spans,
                weight=weight,
                canonical_form="",
            )
        )
    return facts


# ---------- Gate helpers ----------


def _run_gates(question: str, answer: str, facts, reasoning_trace: list, min_quality: float) -> dict:
    from rag_gt.validation.structure import check_structure
    from rag_gt.validation.gt_quality import score_generated_pair
    from rag_gt.validation.nli_check import batch_answer_entailment
    from rag_gt.validation.minimality import support_minimality_check

    ok, struct_reason = check_structure(question, answer)
    if not ok:
        return {"passed": False, "stage": "structure", "reason": struct_reason}

    quality = score_generated_pair(question, answer, facts)
    if quality["quality"] < min_quality:
        flagged = [k for k, v in quality["flags"].items() if v]
        return {
            "passed": False,
            "stage": "quality",
            "score": quality["quality"],
            "flagged": flagged,
        }

    nli_pass = batch_answer_entailment([(answer, facts)])
    if not nli_pass[0]:
        return {"passed": False, "stage": "answer_nli"}

    if len(facts) > 1:
        support_verdict = support_minimality_check(
            question,
            facts,
            answer,
            question_assumptions=[],
            reasoning_trace=reasoning_trace,
        )
        if not support_verdict.passed:
            return {
                "passed": False,
                "stage": "minimality",
                "reason": support_verdict.to_dict().get("reason", ""),
            }

    return {"passed": True, "quality": quality["quality"]}


# ---------- Main ----------


def main() -> None:
    trace_path = Path(TRACE_PATH)
    if not trace_path.exists():
        raise SystemExit(f"Trace not found: {trace_path}")

    with open(CONFIG_PATH, encoding="utf-8") as fh:
        v16_cfg = yaml.safe_load(fh)
    arm_cfg = v16_cfg.get("arm", {})
    max_np_overlap = float(arm_cfg.get("max_np_overlap", 0.38))
    step_nli_threshold = float(arm_cfg.get("step_nli_threshold", 0.55))
    max_retries = int(arm_cfg.get("max_step_retries", 3))
    min_gt_quality = float(v16_cfg.get("validation", {}).get("min_gt_quality", 0.7))

    logger.info(f"ARM max_np_overlap={max_np_overlap}  nli_threshold={step_nli_threshold}")
    logger.info(f"Budget: {MAX_LIVE_CALLS} live calls")

    # Read target chains from trace
    chains: dict = {}
    with trace_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") != "drop" or ev.get("reason") != "arm_grounding_failed":
                continue
            data = ev.get("data", {})
            chain = data.get("chain", {})
            cid = chain.get("chain_id", "")
            if cid in TARGET_CHAINS:
                chains[cid] = {"question": data.get("question", ""), "chain": chain}
            if len(chains) == len(TARGET_CHAINS):
                break

    if len(chains) < len(TARGET_CHAINS):
        raise SystemExit(f"Found only {len(chains)}/{len(TARGET_CHAINS)} target chains in trace")

    # Set up LLM with budget
    from rag_gt.core.llm import get_llm
    from rag_gt.observability.cost_tracker import CostTracker, TrackedLLM

    base_llm = get_llm("answer")
    cost_tracker = CostTracker(max_live_api_calls=MAX_LIVE_CALLS)
    answer_llm = TrackedLLM(base_llm, cost_tracker, stage="arm_continuation", doc_id="sutton")

    from rag_gt.generation.cga import build_constructive_gold_answer

    print(f"\nRunning ARM continuation at max_np_overlap={max_np_overlap}\n")
    print(f"{'chain_id':<26} {'fact_ids':<22} {'ARM':<6} {'quality':<9} {'gate_result'}")
    print("-" * 95)

    results = {}
    for cid in sorted(chains):
        entry = chains[cid]
        question = entry["question"]
        chain_dict = entry["chain"]
        fact_ids = chain_dict.get("fact_ids", [])
        short_ids = "->".join(fid.split("_")[-1] for fid in fact_ids)

        facts = _reconstruct_facts(chain_dict)
        edges = chain_dict.get("edges", [])

        try:
            cga = build_constructive_gold_answer(
                question,
                facts,
                answer_llm,
                threshold=step_nli_threshold,
                max_retries=max_retries,
                max_np_overlap=max_np_overlap,
                use_arm=True,
                tf_sfg_edges=edges,
                cost_tracker=cost_tracker,
                doc_id="sutton",
            )
        except Exception as e:
            print(f"{cid:<26} {short_ids:<22} {'ERR':<6} {'N/A':<9} {type(e).__name__}: {e}")
            results[cid] = {"arm": "error", "error": str(e)}
            continue

        arm_status = "PASS" if cga.all_clauses_pass else "FAIL"

        if not cga.all_clauses_pass:
            blocking = [
                f"s{i+1}:np={s['np_overlap']:.4f}" for i, s in enumerate(cga.reasoning_trace)
                if not s.get("passes")
            ]
            print(
                f"{cid:<26} {short_ids:<22} {arm_status:<6} {'N/A':<9} "
                f"blocked={blocking}"
            )
            results[cid] = {"arm": "fail", "blocking_steps": blocking}
            continue

        gate = _run_gates(question, cga.gold_answer, facts, cga.reasoning_trace, min_gt_quality)
        gate_label = f"PASS (q={gate.get('quality', 0):.3f})" if gate["passed"] else f"FAIL@{gate['stage']}"
        print(
            f"{cid:<26} {short_ids:<22} {arm_status:<6} "
            f"{gate.get('quality', 0):.3f}     {gate_label}"
        )
        results[cid] = {"arm": "pass", "gate": gate, "gold_answer": cga.gold_answer}

    live = cost_tracker._reserved_live_calls if hasattr(cost_tracker, "_reserved_live_calls") else "N/A"
    print(f"\nLive API calls used: {live}/{MAX_LIVE_CALLS}")

    # Summary
    print("\n=== SUMMARY ===")
    arm_passes = sum(1 for r in results.values() if r.get("arm") == "pass")
    gate_passes = sum(1 for r in results.values() if r.get("arm") == "pass" and r.get("gate", {}).get("passed"))
    print(f"ARM passed: {arm_passes}/{len(results)}")
    print(f"All gates passed: {gate_passes}/{len(results)}")

    for cid, r in sorted(results.items()):
        fact_ids = chains[cid]["chain"].get("fact_ids", [])
        short_ids = "->".join(fid.split("_")[-1] for fid in fact_ids)
        if r.get("arm") == "pass" and r.get("gate", {}).get("passed"):
            print(f"\n[NEW ROW] {short_ids}")
            print(f"  Q: {chains[cid]['question'][:100]}")
            print(f"  A: {r['gold_answer'][:150]}")


if __name__ == "__main__":
    main()
