"""Tests for Phase 2 — Atomic decomposition + Constructive Gold Answer (CGA).

Plan v6. NLI calls are stubbed out so these tests are deterministic and run
without API access or model weights.
"""

from __future__ import annotations

from rag_gt.core.types import Fact, Span
from rag_gt.generation import cga as cga_mod
from rag_gt.validation import atomic_decomp


def _fake_llm(responses):
    queue = list(responses)

    class _LLM:
        model = "fake-test-model"

        def generate(self, prompt, temperature=0.0, max_tokens=256):
            if not queue:
                return ""
            return queue.pop(0)

    return _LLM()


def _make_sfu(fact_id: str, canonical: str, text: str | None = None) -> Fact:
    return Fact(
        fact_id=fact_id,
        text=text or canonical,
        role="definition",
        supporting_spans=[Span(doc_id="doc", chunk_id="doc_c000000", start_token=0, end_token=1)],
        canonical_form=canonical,
        raw_text=text or canonical,
    )


# ---- atomic_decomp tests ----


def test_heuristic_decompose_splits_on_period():
    out = atomic_decomp._heuristic_decompose("A is X. B is Y. C is Z.")
    assert out == ["A is X.", "B is Y.", "C is Z."]


def test_heuristic_decompose_splits_on_conjunctions():
    out = atomic_decomp._heuristic_decompose(
        "Sarsa learns quickly and switches policies."
    )
    assert len(out) == 2
    assert "Sarsa learns quickly" in out[0]
    assert "switches policies" in out[1]


def test_heuristic_decompose_drops_short_fragments():
    out = atomic_decomp._heuristic_decompose("ok.")
    assert out == []


def test_decompose_into_atomic_clauses_falls_back_when_no_llm():
    out = atomic_decomp.decompose_into_atomic_clauses(
        "First claim is here. Second claim is here.", llm=None
    )
    assert len(out) == 2


def test_decompose_into_atomic_clauses_uses_llm_when_provided():
    llm = _fake_llm(['["Claim one is solid.", "Claim two is also solid."]'])
    out = atomic_decomp.decompose_into_atomic_clauses(
        "A composite paragraph.", llm=llm, use_llm=True
    )
    assert out == ["Claim one is solid.", "Claim two is also solid."]


def test_decompose_into_atomic_clauses_falls_back_on_bad_json():
    llm = _fake_llm(["not json at all"])
    out = atomic_decomp.decompose_into_atomic_clauses(
        "First. Second.", llm=llm, use_llm=True
    )
    # Falls back to heuristic split.
    assert len(out) == 2


# ---- CGA tests ----


def test_cga_passes_when_all_clauses_nli_entailed(monkeypatch):
    sfus = [
        _make_sfu("F1", "Monte Carlo estimates q*."),
        _make_sfu("F2", "Monte Carlo control approximates optimal policies."),
    ]
    paraphrase_llm = _fake_llm([
        "Monte Carlo estimates q* and approximates optimal policies."
    ])
    # Stub atomic_clause_entailment: all clauses pass against SFU 0.
    monkeypatch.setattr(
        cga_mod,
        "atomic_clause_entailment",
        lambda clauses, support, threshold=None: [(True, 0.9, 0) for _ in clauses],
    )

    result = cga_mod.build_constructive_gold_answer(
        "What does Monte Carlo do?",
        sfus,
        paraphrase_llm,
    )

    assert result.all_clauses_pass is True
    assert result.retries == 0
    assert "Monte Carlo estimates" in result.gold_answer
    assert all(g.passes for g in result.clause_grounding)


def test_cga_retries_when_some_clauses_unsupported(monkeypatch):
    sfus = [_make_sfu("F1", "Monte Carlo estimates q*.")]
    paraphrase_llm = _fake_llm([
        "Monte Carlo estimates q*. It averages returns over episodes.",  # attempt 1 — 2 clauses, one invented
        "Monte Carlo estimates q*. It is a key goal.",                    # attempt 2 — 2 clauses, both clean
    ])

    call_count = {"n": 0}

    def fake_entailment(clauses, support, threshold=None):
        call_count["n"] += 1
        assert len(clauses) >= 2, f"expected ≥2 clauses, got {len(clauses)}"
        if call_count["n"] == 1:
            # First attempt: one clause unsupported.
            return [(True, 0.9, 0)] + [(False, 0.2, -1)] * (len(clauses) - 1)
        # Subsequent attempts: all pass.
        return [(True, 0.9, 0) for _ in clauses]

    monkeypatch.setattr(cga_mod, "atomic_clause_entailment", fake_entailment)

    result = cga_mod.build_constructive_gold_answer(
        "What does Monte Carlo do?",
        sfus,
        paraphrase_llm,
    )

    assert result.retries == 1
    assert result.all_clauses_pass is True


def test_cga_returns_best_candidate_after_retries_exhausted(monkeypatch):
    sfus = [_make_sfu("F1", "Monte Carlo estimates q*.")]
    paraphrase_llm = _fake_llm([
        "Monte Carlo estimates q*. Invented detail one is here.",
        "Monte Carlo estimates q*. Invented detail two is here.",
        "Monte Carlo estimates q*. Invented detail three is here.",
    ])
    # Always fail one of two clauses.
    monkeypatch.setattr(
        cga_mod,
        "atomic_clause_entailment",
        lambda clauses, support, threshold=None: [(True, 0.9, 0), (False, 0.2, -1)],
    )

    result = cga_mod.build_constructive_gold_answer(
        "Q?",
        sfus,
        paraphrase_llm,
        max_retries=3,
    )

    assert result.all_clauses_pass is False
    # The candidate's `retries` is the attempt index when it was captured
    # (best-so-far accumulator), not the total retry budget. The contract is
    # that we return SOME candidate after exhaustion.
    assert result.grounded_clause_count == 1
    assert result.total_clause_count == 2


def test_cga_uses_canonical_form_when_present(monkeypatch):
    """When SFU.canonical_form is populated, CGA uses it, NOT SFU.text."""
    sfu = Fact(
        fact_id="F1",
        text="Original sentence-fragment from PDF.",
        role="rule",
        supporting_spans=[Span(doc_id="doc", chunk_id="doc_c000000", start_token=0, end_token=1)],
        canonical_form="Self-contained canonical form of the claim.",
        raw_text="Original sentence-fragment from PDF.",
    )
    # Capture the support_texts passed to entailment.
    captured = {}

    def capture(clauses, support, threshold=None):
        captured["support"] = support
        return [(True, 0.9, 0) for _ in clauses]

    monkeypatch.setattr(cga_mod, "atomic_clause_entailment", capture)
    paraphrase_llm = _fake_llm(["A clean rewrite of the claim."])

    cga_mod.build_constructive_gold_answer("Q?", [sfu], paraphrase_llm)

    assert captured["support"] == ["Self-contained canonical form of the claim."]


def test_cga_falls_back_to_text_when_no_canonical_form(monkeypatch):
    """Legacy V11 facts (no canonical_form) still work — CGA uses .text."""
    legacy = Fact(
        fact_id="F1",
        text="Some sentence ending in a period.",
        role="rule",
        supporting_spans=[Span(doc_id="doc", chunk_id="doc_c000000", start_token=0, end_token=1)],
    )
    captured = {}

    def capture(clauses, support, threshold=None):
        captured["support"] = support
        return [(True, 0.9, 0) for _ in clauses]

    monkeypatch.setattr(cga_mod, "atomic_clause_entailment", capture)
    paraphrase_llm = _fake_llm(["A clean rewrite."])

    cga_mod.build_constructive_gold_answer("Q?", [legacy], paraphrase_llm)

    assert captured["support"] == ["Some sentence ending in a period."]


def test_cga_raises_on_empty_chain():
    import pytest

    with pytest.raises(ValueError, match="at least one SFU"):
        cga_mod.build_constructive_gold_answer("Q?", [], _fake_llm([]))
