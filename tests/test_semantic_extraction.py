"""Tests for Phase 1 SFU extraction (plan v6)."""

from __future__ import annotations

from types import SimpleNamespace

from rag_gt.core.types import Fact, Span
from rag_gt.facts import semantic_extraction as sx


def _fake_llm(responses):
    """LLM stub: returns the next item in `responses` on each .generate() call."""
    queue = list(responses)

    class _LLM:
        model = "fake-test-model"

        def generate(self, prompt, temperature=0.0, max_tokens=256):
            if not queue:
                return ""
            return queue.pop(0)

    return _LLM()


def test_parse_float_score_clamps_to_unit_interval():
    assert sx._parse_float_score("0.85") == 0.85
    assert sx._parse_float_score("Score: 0.7") == 0.7
    assert sx._parse_float_score("1.7") == 1.0
    assert sx._parse_float_score("-0.3") == 0.0
    assert sx._parse_float_score("garbage") is None
    assert sx._parse_float_score("") is None


def test_self_containment_score_parses_llm_output(monkeypatch):
    llm = _fake_llm(["0.85"])
    score = sx.self_containment_score("Some claim.", llm)
    assert score == 0.85


def test_self_containment_score_falls_back_on_unparseable_output(monkeypatch):
    llm = _fake_llm(["explanation only, no number"])
    score = sx.self_containment_score("Some claim.", llm, default_on_error=0.4)
    assert score == 0.4


def test_canonical_form_rewrite_skips_nli_when_model_absent():
    llm = _fake_llm(["The rewritten self-contained version of the claim."])
    result = sx.canonical_form_rewrite(
        passage="Thus, q* is the goal.",
        context="In Monte Carlo control we seek q*.",
        llm=llm,
        nli_model=None,
    )
    assert result.passed is True
    assert result.canonical_form == "The rewritten self-contained version of the claim."
    assert result.retries == 0


def test_canonical_form_rewrite_passes_with_strong_nli(monkeypatch):
    llm = _fake_llm(["Estimating q* is a primary goal of Monte Carlo methods."])
    # Stub nli_batch to return high entailment both ways.
    monkeypatch.setattr(
        sx, "_forward_nli_guard", lambda a, b, m, t: (True, 0.92, 0.88)
    )
    result = sx.canonical_form_rewrite(
        passage="Thus, one primary goal of Monte Carlo methods is to estimate q*.",
        context="In Monte Carlo control we seek q*.",
        llm=llm,
        nli_model=object(),
    )
    assert result.passed is True
    assert result.nli_forward == 0.92
    assert result.nli_backward == 0.88


def test_canonical_form_rewrite_retries_then_falls_back(monkeypatch):
    llm = _fake_llm([
        "Rewrite v1 — adds invented content.",
        "Rewrite v2 — still invents.",
        "Rewrite v3 — still bad.",
    ])
    # Always fail NLI guard.
    monkeypatch.setattr(
        sx, "_forward_nli_guard", lambda a, b, m, t: (False, 0.3, 0.2)
    )
    result = sx.canonical_form_rewrite(
        passage="Original passage.",
        context="ctx",
        llm=llm,
        nli_model=object(),
        max_retries=3,
    )
    assert result.passed is False
    assert result.retries == 3
    # Falls back to the VERBATIM ORIGINAL passage, not a meaning-changing rewrite
    # that failed NLI entailment (audit fix: never trust an unverified rewrite).
    assert result.canonical_form == "Original passage."


def test_upgrade_fact_to_sfu_populates_all_three_fields(monkeypatch):
    """End-to-end: canonical_form + raw_text + self_containment_score all set."""
    llm = _fake_llm([
        "Canonical rewrite goes here without new content.",  # rewrite
        "0.9",                                                 # self-containment
    ])
    monkeypatch.setattr(
        sx, "_forward_nli_guard", lambda a, b, m, t: (True, 0.85, 0.83)
    )

    fact = Fact(
        fact_id="F1",
        text="Thus, q* is the goal of Monte Carlo methods.",
        role="consequence",
        supporting_spans=[Span(doc_id="doc", chunk_id="doc_c000000", start_token=0, end_token=1)],
    )

    sx.upgrade_fact_to_sfu(fact, context_text="surrounding chunk text", llm=llm, nli_model=object())

    assert fact.raw_text == "Thus, q* is the goal of Monte Carlo methods."
    assert fact.canonical_form == "Canonical rewrite goes here without new content."
    assert fact.self_containment_score == 0.9
    # Original text field unchanged so legacy downstream code is unaffected.
    assert fact.text == "Thus, q* is the goal of Monte Carlo methods."


def test_upgrade_fact_to_sfu_keeps_raw_text_if_already_set(monkeypatch):
    """Re-running upgrade must not overwrite raw_text with the (possibly already
    canonicalised) text field."""
    llm = _fake_llm([
        "Re-rewritten canonical form.",
        "0.7",
    ])
    monkeypatch.setattr(
        sx, "_forward_nli_guard", lambda a, b, m, t: (True, 0.8, 0.8)
    )

    fact = Fact(
        fact_id="F1",
        text="Some sentence with sufficient length.",
        role="rule",
        supporting_spans=[Span(doc_id="doc", chunk_id="doc_c000000", start_token=0, end_token=1)],
        raw_text="Pristine original text from the source PDF.",
    )

    sx.upgrade_fact_to_sfu(fact, context_text="", llm=llm, nli_model=object())

    # raw_text must be preserved across runs.
    assert fact.raw_text == "Pristine original text from the source PDF."


def test_segment_chunk_into_sfus_parses_json_segments():
    """The chunk segmenter returns a list of SFUSegment objects from JSON."""
    chunk_text = (
        "Sentence one is about A. Sentence two extends sentence one. "
        "Sentence three is independent."
    )
    # JSON output: two segments — first merges sentences 1+2, second is sentence 3.
    json_out = (
        '[{"start_char": 0, "end_char": 56, "role_hint": "definition", '
        '"claim_summary": "Combined claim from sentence 1 and 2."}, '
        '{"start_char": 57, "end_char": ' + str(len(chunk_text)) + ', '
        '"role_hint": "rule", "claim_summary": "Independent claim."}]'
    )
    llm = _fake_llm([json_out])
    segs = sx.segment_chunk_into_sfus(chunk_text, llm)
    assert len(segs) == 2
    assert segs[0].role_hint == "definition"
    assert segs[1].role_hint == "rule"
    # Verbatim slice from chunk_text — no LLM-supplied text.
    assert segs[0].text == chunk_text[0:56].strip()


def test_segment_chunk_into_sfus_returns_empty_on_bad_json():
    llm = _fake_llm(["this is not JSON at all"])
    segs = sx.segment_chunk_into_sfus("some chunk text here", llm)
    assert segs == []


def test_segment_chunk_into_sfus_drops_out_of_range_offsets():
    chunk_text = "Short chunk."  # length 12
    bad_json = '[{"start_char": 0, "end_char": 99999, "role_hint": "rule", "claim_summary": "x"}]'
    llm = _fake_llm([bad_json])
    segs = sx.segment_chunk_into_sfus(chunk_text, llm)
    assert segs == []
