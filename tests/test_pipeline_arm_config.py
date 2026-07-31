"""Verify that max_np_overlap from v16.yaml reaches build_constructive_gold_answer.

Tests 1-2: config extraction checks (YAML value + dict-access path).
Test 3:    calls _generate_and_answer_v16 with ARM enabled and captures the
           keyword argument actually passed to build_constructive_gold_answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _load_arm_cfg() -> dict:
    config_path = Path("configs/v16.yaml")
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return raw.get("arm", {})


def test_v16_yaml_declares_max_np_overlap():
    arm_cfg = _load_arm_cfg()
    assert "max_np_overlap" in arm_cfg
    assert float(arm_cfg["max_np_overlap"]) == pytest.approx(0.35)


def test_pipeline_arm_cfg_extraction_builds_correct_kwargs():
    """Replicates the dict-access extraction in _generate_and_answer_v16."""
    arm_cfg = _load_arm_cfg()

    kwargs = {
        "threshold": float(arm_cfg.get("step_nli_threshold", 0.55)),
        "max_retries": int(arm_cfg.get("max_step_retries", 3)),
        "max_np_overlap": float(arm_cfg.get("max_np_overlap", 0.35)),
        "use_arm": True,
    }

    assert kwargs["max_np_overlap"] == pytest.approx(0.35)
    assert kwargs["threshold"] == pytest.approx(0.55)
    assert kwargs["max_retries"] == 5


def test_generate_and_answer_v16_passes_max_np_overlap_to_cga(monkeypatch):
    """_generate_and_answer_v16 with arm.enabled=True passes max_np_overlap to CGA."""
    from rag_gt.core.types import Fact
    from rag_gt.generation.cga import AtomicClauseGrounding, CGAResult
    from rag_gt.pipeline.gt_pipeline import _generate_and_answer_v16
    from rag_gt.validation.question_assumption_gate import QANLIGateVerdict

    captured: dict = {}

    def _mock_cga(*args, **kwargs):
        captured.update(kwargs)
        return CGAResult(
            gold_answer="mock answer",
            atomic_clauses=["mock clause"],
            clause_grounding=[
                AtomicClauseGrounding(
                    clause="mock clause", passes=True, best_score=0.9, grounded_sfu_index=0
                )
            ],
            all_clauses_pass=True,
            retries=0,
            reasoning_trace=[{"step_id": "s1", "claim": "mock", "fact_ids": ["F001"],
                               "nli_score": 0.9, "np_overlap": 0.1, "passes": True,
                               "relation_type": "definition", "bbox_refs": []}],
        )

    passing_verdict = QANLIGateVerdict(
        accepted=True,
        assumption_verdicts=[],
        fail_rate=0.0,
        rel_fail_rate=0.0,
        reason="pass",
        guidance="",
        missing_fact_indices=[],
    )

    monkeypatch.setattr(
        "rag_gt.pipeline.gt_pipeline.generate_question",
        lambda *a, **kw: "How does the learning rate affect optimization convergence?",
    )
    monkeypatch.setattr(
        "rag_gt.validation.question_assumption_gate.check_question_assumptions",
        lambda *a, **kw: passing_verdict,
    )
    monkeypatch.setattr(
        "rag_gt.generation.cga.build_constructive_gold_answer",
        _mock_cga,
    )

    class _StubLLM:
        model = "stub"
        def generate(self, *a, **kw): return ""

    facts = [
        Fact(fact_id="F001", text="Learning rate controls the step size in gradient descent optimization.", role="definition"),
        Fact(fact_id="F002", text="Larger step sizes cause divergence in gradient-based optimization.", role="definition"),
    ]

    v16_cfg = {
        "arm": {
            "enabled": True,
            "max_np_overlap": 0.38,
            "step_nli_threshold": 0.55,
            "max_step_retries": 3,
        },
        "qa_nli": {},
        "question_generation": {},
    }

    _generate_and_answer_v16(facts, [], _StubLLM(), _StubLLM(), v16_cfg)

    assert "max_np_overlap" in captured, "build_constructive_gold_answer was not called"
    assert captured["max_np_overlap"] == pytest.approx(0.38)
