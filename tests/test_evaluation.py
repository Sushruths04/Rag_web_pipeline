"""Tests for production evaluation metrics."""

from types import SimpleNamespace

import pytest

from rag_gt.cli import evaluate as eval_mod
from rag_gt.core import models as core_models
from rag_gt.core.types import AnswerLog, Fact, MSFS, QuestionGT, RetrievalLog, Span


def _make_question() -> QuestionGT:
    facts = [
        Fact(
            fact_id="F1",
            text="The temperature must not exceed 1500C.",
            role="rule",
            supporting_spans=[
                Span(doc_id="doc", chunk_id="doc_c000000", start_token=0, end_token=6)
            ],
        ),
        Fact(
            fact_id="F2",
            text="Exceeding the limit causes deformation of the material.",
            role="consequence",
            supporting_spans=[
                Span(doc_id="doc", chunk_id="doc_c000001", start_token=6, end_token=14)
            ],
        ),
    ]
    return QuestionGT(
        q_id="doc_q001",
        question="Why must the temperature stay below 1500C?",
        gold_answer=(
            "The temperature must stay below 1500C because exceeding that "
            "limit causes deformation of the material."
        ),
        msfs_list=[MSFS(msfs_id="doc_q001_msfs1", fact_ids=["F1", "F2"])],
        doc_ids=["doc"],
        required_fact_ids=["F1", "F2"],
        difficulty_reasoning_depth=2,
        difficulty_semantic_distance="intra_doc",
        required_facts=facts,
    )


def test_evaluate_answer_uses_real_persisted_facts(monkeypatch):
    q_gt = _make_question()
    retrieval = RetrievalLog(
        q_id="doc_q001",
        retrieved_chunk_ids=["doc_c000000", "doc_c000001"],
    )
    answer = AnswerLog(
        q_id="doc_q001",
        predicted_answer=(
            "The temperature must stay below 1500C because exceeding that "
            "limit causes deformation of the material."
        ),
    )

    # Single seam: patch the cross-encoder. The contradiction index is
    # resolved from this fake's id2label.
    fake_model = SimpleNamespace(
        predict=lambda pairs, apply_softmax=True: [[0.05, 0.9, 0.05] for _ in pairs],
        model=SimpleNamespace(
            config=SimpleNamespace(
                id2label={0: "contradiction", 1: "entailment", 2: "neutral"}
            )
        ),
    )
    monkeypatch.setattr(core_models.MM, "get_nli", lambda: fake_model)
    monkeypatch.setattr(core_models.MM, "load_nli", lambda: None)
    monkeypatch.setattr(eval_mod, "nli_entailment", lambda p, h: 0.9)
    monkeypatch.setattr(eval_mod, "nli_batch", lambda pairs: [0.9 for _ in pairs])
    # Make sure the contradiction-index lookup finds slot 0.
    core_models.NLI_LABEL_INDEX.update(
        {"contradiction": 0, "entailment": 1, "neutral": 2}
    )

    result = eval_mod.evaluate_answer(q_gt, retrieval, answer)
    assert result["fact_span_recall"] == 1.0
    assert result["fact_span_precision"] == 1.0
    assert result["fact_precision"] == 0.9
    assert result["faithfulness"] == 0.9
    assert result["contradiction_rate"] == 0.05


def test_evaluation_fails_without_required_facts():
    q_gt = QuestionGT(
        q_id="doc_q001",
        question="Why must the temperature stay below 1500C?",
        gold_answer="Because exceeding the limit deforms the material.",
        msfs_list=[MSFS(msfs_id="doc_q001_msfs1", fact_ids=["F1"])],
        doc_ids=["doc"],
        required_fact_ids=["F1"],
        difficulty_reasoning_depth=1,
        difficulty_semantic_distance="local",
        required_facts=[],
    )
    retrieval = RetrievalLog(q_id="doc_q001", retrieved_chunk_ids=["doc_c000000"])
    answer = AnswerLog(
        q_id="doc_q001",
        predicted_answer="Because exceeding the limit deforms the material.",
    )

    with pytest.raises(ValueError, match="required_facts"):
        eval_mod.evaluate_answer(q_gt, retrieval, answer)


# ---- Phase 4 (plan v5): ChunkResolver projection in fact-span metrics ----


def _resolver_with_one_active_chunk(
    *,
    doc_id: str,
    active_chunk_id: str,
    active_char_start: int,
    active_char_end: int,
):
    """Build a tiny ChunkResolver whose only record overlaps the given span."""
    from rag_gt.comparison.chunk_resolver import ChunkRecord, ChunkResolver

    rec = ChunkRecord(
        chunk_id=active_chunk_id,
        doc_id=doc_id,
        text="x" * (active_char_end - active_char_start),
        char_start=active_char_start,
        char_end=active_char_end,
        sha1="deadbeef",
        page_start=42,
        page_end=42,
    )
    return ChunkResolver({rec.chunk_id: rec})


def test_fact_span_recall_legacy_path_unchanged_when_no_resolver():
    """No resolver passed → behaviour identical to chunk-id equality."""
    facts = [
        Fact(
            fact_id="F1",
            text="This is a placeholder fact text.",
            role="rule",
            supporting_spans=[
                Span(
                    doc_id="doc",
                    chunk_id="doc_c000000",
                    start_token=0,
                    end_token=6,
                    char_start=100,
                    char_end=200,
                )
            ],
        ),
    ]
    # Retrieved a different chunk profile id — without resolver this must be 0.
    fsr = eval_mod._fact_span_recall(["doc_v11_c0042"], facts)
    assert fsr == 0.0


def test_fact_span_recall_projects_via_resolver_to_active_profile():
    """
    GT span sits at char [100, 200] in chunk doc_c000000.
    Active V11 chunk profile has chunk doc_v11_c0042 covering chars [50, 250].
    When the V11 retrieval log returns doc_v11_c0042, recall should be 1.0
    (the span overlaps that active chunk), NOT 0.0.
    """
    facts = [
        Fact(
            fact_id="F1",
            text="This is a placeholder fact text.",
            role="rule",
            supporting_spans=[
                Span(
                    doc_id="doc",
                    chunk_id="doc_c000000",
                    start_token=0,
                    end_token=6,
                    char_start=100,
                    char_end=200,
                )
            ],
        ),
    ]
    resolver = _resolver_with_one_active_chunk(
        doc_id="doc",
        active_chunk_id="doc_v11_c0042",
        active_char_start=50,
        active_char_end=250,
    )
    fsr = eval_mod._fact_span_recall(["doc_v11_c0042"], facts, resolver=resolver)
    assert fsr == 1.0


def test_fact_span_precision_projects_via_resolver():
    """The required-chunk set must include resolver-projected IDs."""
    facts = [
        Fact(
            fact_id="F1",
            text="This is a placeholder fact text.",
            role="rule",
            supporting_spans=[
                Span(
                    doc_id="doc",
                    chunk_id="doc_c000000",
                    start_token=0,
                    end_token=6,
                    char_start=100,
                    char_end=200,
                )
            ],
        ),
    ]
    resolver = _resolver_with_one_active_chunk(
        doc_id="doc",
        active_chunk_id="doc_v11_c0042",
        active_char_start=50,
        active_char_end=250,
    )
    # Retrieved the active-profile chunk; precision must be 1/1 = 1.0.
    fsp = eval_mod._fact_span_precision(
        ["doc_v11_c0042"], facts, resolver=resolver
    )
    assert fsp == 1.0


def test_fact_span_recall_legacy_chunk_id_still_counts_when_resolver_present():
    """The resolver path is additive: an exact chunk_id match still counts
    even when the resolver projects to other IDs."""
    facts = [
        Fact(
            fact_id="F1",
            text="This is a placeholder fact text.",
            role="rule",
            supporting_spans=[
                Span(
                    doc_id="doc",
                    chunk_id="doc_c000000",
                    start_token=0,
                    end_token=6,
                    char_start=100,
                    char_end=200,
                )
            ],
        ),
    ]
    resolver = _resolver_with_one_active_chunk(
        doc_id="doc",
        active_chunk_id="doc_v11_c0042",
        active_char_start=50,
        active_char_end=250,
    )
    # Retrieval log uses the ORIGINAL chunk_id (legacy run on V10 cache);
    # must still count as hit.
    fsr = eval_mod._fact_span_recall(
        ["doc_c000000"], facts, resolver=resolver
    )
    assert fsr == 1.0


# ---- Attempt 5: abstention-aware faithfulness / contradiction_rate ----


def _patch_nli_with_constant(monkeypatch, *, faith=0.9, contra=0.05):
    """Common stub used by Attempt 5 tests."""
    fake_model = SimpleNamespace(
        predict=lambda pairs, apply_softmax=True: [
            [contra, faith, 1.0 - faith - contra] for _ in pairs
        ],
        model=SimpleNamespace(
            config=SimpleNamespace(
                id2label={0: "contradiction", 1: "entailment", 2: "neutral"}
            )
        ),
    )
    monkeypatch.setattr(core_models.MM, "get_nli", lambda: fake_model)
    monkeypatch.setattr(core_models.MM, "load_nli", lambda: None)
    monkeypatch.setattr(eval_mod, "nli_entailment", lambda p, h: faith)
    monkeypatch.setattr(eval_mod, "nli_batch", lambda pairs: [faith for _ in pairs])
    core_models.NLI_LABEL_INDEX.update(
        {"contradiction": 0, "entailment": 1, "neutral": 2}
    )


def test_abstained_answer_emits_nan_for_faithfulness_and_contradiction(monkeypatch):
    """Per-question path: an abstained answer has no propositional content;
    faithfulness and contradiction_rate must be NaN, not 0/high contradiction."""
    q_gt = _make_question()
    retrieval = RetrievalLog(
        q_id="doc_q001",
        retrieved_chunk_ids=["doc_c000000", "doc_c000001"],
    )
    answer = AnswerLog(
        q_id="doc_q001",
        predicted_answer="Insufficient information",
        abstained=True,
    )
    _patch_nli_with_constant(monkeypatch, faith=0.9, contra=0.05)

    result = eval_mod.evaluate_answer(q_gt, retrieval, answer)
    assert result["faithfulness"] != result["faithfulness"]  # NaN
    assert result["contradiction_rate"] != result["contradiction_rate"]  # NaN
    # Hallucination must be 0 because the model abstained — no claim made.
    assert result["hallucination_flag"] == 0.0
    # fact_precision (vs context) is still computed; abstention doesn't void it.
    assert result["fact_precision"] == 0.9


def test_non_abstained_answer_keeps_faithfulness_and_contradiction(monkeypatch):
    """Regression: when the model actually answers, faith/contra are unchanged
    from the pre-Attempt-5 behaviour."""
    q_gt = _make_question()
    retrieval = RetrievalLog(
        q_id="doc_q001",
        retrieved_chunk_ids=["doc_c000000", "doc_c000001"],
    )
    answer = AnswerLog(
        q_id="doc_q001",
        predicted_answer=(
            "The temperature must stay below 1500C because exceeding that "
            "limit causes deformation of the material."
        ),
        abstained=False,
    )
    _patch_nli_with_constant(monkeypatch, faith=0.9, contra=0.05)

    result = eval_mod.evaluate_answer(q_gt, retrieval, answer)
    assert result["faithfulness"] == 0.9
    assert result["contradiction_rate"] == 0.05
    assert result["hallucination_flag"] == 0.0


def test_batched_eval_skips_abstained_rows_for_faith_and_contra(monkeypatch):
    """Corpus batched path: with one abstained + one non-abstained row, the
    abstained row's faith/contra are NaN; the non-abstained row keeps the
    real NLI scores; corpus mean (via _safe_mean) skips NaN."""
    q1 = _make_question()
    q2_question = _make_question()
    q2_question.q_id = "doc_q002"
    q_map = {q1.q_id: q1, q2_question.q_id: q2_question}
    ret = {
        q1.q_id: RetrievalLog(
            q_id=q1.q_id, retrieved_chunk_ids=["doc_c000000", "doc_c000001"]
        ),
        q2_question.q_id: RetrievalLog(
            q_id=q2_question.q_id,
            retrieved_chunk_ids=["doc_c000000", "doc_c000001"],
        ),
    }
    ans = {
        q1.q_id: AnswerLog(
            q_id=q1.q_id,
            predicted_answer=(
                "The temperature must stay below 1500C because exceeding that "
                "limit causes deformation of the material."
            ),
            abstained=False,
        ),
        q2_question.q_id: AnswerLog(
            q_id=q2_question.q_id,
            predicted_answer="Insufficient information",
            abstained=True,
        ),
    }
    _patch_nli_with_constant(monkeypatch, faith=0.9, contra=0.05)

    rows = eval_mod._evaluate_corpus_batched(q_map, ret, ans)
    by_id = {r_in: r_out for r_in, r_out in zip(q_map, rows)}
    # Non-abstained row: real numbers preserved.
    assert by_id[q1.q_id]["faithfulness"] == 0.9
    assert by_id[q1.q_id]["contradiction_rate"] == 0.05
    # Abstained row: NaN.
    assert by_id[q2_question.q_id]["faithfulness"] != by_id[q2_question.q_id]["faithfulness"]
    assert by_id[q2_question.q_id]["contradiction_rate"] != by_id[q2_question.q_id]["contradiction_rate"]

    # Corpus mean over both rows skips the NaN abstained row.
    faith_mean = eval_mod._safe_mean([r["faithfulness"] for r in rows])
    contra_mean = eval_mod._safe_mean([r["contradiction_rate"] for r in rows])
    assert faith_mean == 0.9
    assert contra_mean == 0.05


def test_fact_span_recall_no_projection_when_span_lacks_offsets():
    """If a span has no char offsets, resolver is useless and we fall back
    to chunk-id equality only."""
    facts = [
        Fact(
            fact_id="F1",
            text="This is a placeholder fact text.",
            role="rule",
            supporting_spans=[
                Span(
                    doc_id="doc",
                    chunk_id="doc_c000000",
                    start_token=0,
                    end_token=6,
                    # NO char_start/char_end
                )
            ],
        ),
    ]
    resolver = _resolver_with_one_active_chunk(
        doc_id="doc",
        active_chunk_id="doc_v11_c0042",
        active_char_start=50,
        active_char_end=250,
    )
    # Retrieved active-profile id, but no char offsets to project from →
    # falls back to chunk-id equality → 0.
    fsr = eval_mod._fact_span_recall(
        ["doc_v11_c0042"], facts, resolver=resolver
    )
    assert fsr == 0.0
