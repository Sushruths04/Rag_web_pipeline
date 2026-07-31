from rag_gt.generation.answers import ABSTENTION_TEXT, _normalize_answer_output


def test_normalize_answer_output_rejects_meta_reasoning_without_final_answer():
    raw = (
        "We need to answer this question using only the provided facts. "
        "The provided facts do not contain enough information."
    )
    assert _normalize_answer_output(raw) == ABSTENTION_TEXT


def test_normalize_answer_output_extracts_final_answer_after_meta_prefix():
    raw = (
        "We need to answer this carefully.\n\n"
        "Answer: Every state-action pair must be sampled infinitely often."
    )
    assert _normalize_answer_output(raw) == "Every state-action pair must be sampled infinitely often."


def test_normalize_answer_output_maps_context_refusal_to_abstention():
    raw = (
        "The provided facts discuss another topic and do not contain any information "
        "about the requested concept. Consequently, the question cannot be answered."
    )
    assert _normalize_answer_output(raw) == ABSTENTION_TEXT


# ---- Phase 3.3 hardening (plan v4): new CoT-stripping regression tests ----


def test_normalize_strips_leading_quoted_question_restatement():
    """gpt-oss-120b style: quote the question, then deliberate."""
    raw = (
        '"What primary objective of Monte Carlo methods makes them suitable for control?"\n'
        "Need to use only provided facts.\n\n"
        "Answer: The chief aim of Monte Carlo methods is to estimate q*."
    )
    assert _normalize_answer_output(raw) == "The chief aim of Monte Carlo methods is to estimate q*."


def test_normalize_strips_quoted_question_when_no_explicit_marker():
    """Quoted-question restatement without explicit Answer: marker — still strip."""
    raw = (
        '"How does Sarsa avoid the optimal-policy difficulty?"\n'
        "Step-by-step methods such as Sarsa quickly learn that poor policies are bad and switch."
    )
    out = _normalize_answer_output(raw)
    assert "Step-by-step methods" in out
    assert "How does Sarsa" not in out


def test_normalize_jumps_past_LAST_answer_marker():
    """Reasoning models sometimes emit Answer: ... wait, actually Final answer: ..."""
    raw = (
        "Answer: This is a placeholder.\n"
        "Wait, let me reconsider the facts.\n"
        "Final answer: Monte Carlo control follows generalized policy iteration."
    )
    out = _normalize_answer_output(raw)
    assert out == "Monte Carlo control follows generalized policy iteration."


def test_normalize_keeps_partial_answer_with_unrelated_negation_phrase():
    """A partial answer that mentions 'do not contain' in passing must NOT abstain."""
    raw = (
        "Monte Carlo methods do not contain a model of the environment, "
        "and therefore must estimate action values directly."
    )
    out = _normalize_answer_output(raw)
    assert out.startswith("Monte Carlo methods do not contain")
    assert out != ABSTENTION_TEXT


def test_normalize_handles_look_at_facts_cot_prefix():
    """The 'Look at facts:' deliberation pattern from the v2 answer log."""
    raw = (
        "Look at facts: F1 about MC methods, F2 about action values. "
        "Need to combine."
    )
    assert _normalize_answer_output(raw) == ABSTENTION_TEXT


def test_normalize_idempotent_on_clean_answer():
    """A clean answer must pass through unchanged."""
    clean = "The chief aim of Monte Carlo methods is to estimate q*."
    assert _normalize_answer_output(clean) == clean


def test_normalize_handles_smart_quotes_in_question_restatement():
    """Some models emit smart curly quotes around the restated question."""
    raw = (
        "“What is the role of ε-greedy policies?”\n"
        "Answer: ε-greedy policies are the ε-soft policies closest to greedy."
    )
    out = _normalize_answer_output(raw)
    assert out == "ε-greedy policies are the ε-soft policies closest to greedy."
