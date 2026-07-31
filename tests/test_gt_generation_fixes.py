"""Regression tests for the GT-generation audit fixes (GT_GENERATION_AUDIT_PLAN.md).

All LLM and NLI behavior is stubbed ($0). One test per bug: C, B, D, G, E, A,
plus the Track-B prompt smoke.
"""

import uuid

import pytest

from rag_gt.generation import answer_first as af
from rag_gt.generation.answer_first import (
    _bridge_cache_key,
    _compose_answer,
    _doc_name,
    _lower_first,
    _single_prompt,
    bridge_is_hidden,
    build_answer_first_pairs,
)
from rag_gt.graph.bridge_linker import _contains_phrase, build_bridge_pairs


class _FakeLLM:
    def generate_json(self, prompt, temperature=0.0, max_tokens=0):
        # Distinct questions per fact so neither single is dropped as a duplicate.
        if "diamond pyramid indenter" in prompt:
            return {"question": "What indenter shape does the Vickers hardness test use?"}
        return {"question": "What is the test force range for the Vickers test?"}


def _facts(with_chunk_id: bool):
    base = [
        {"id": "F1", "doc": "din_iso_6507_vickers_full",
         "text": "The Vickers hardness test uses a diamond pyramid indenter.", "page": 3},
        {"id": "F2", "doc": "din_iso_6507_vickers_full",
         "text": "The test force ranges from 1 to 100 kilograms force.", "page": 5},
    ]
    if with_chunk_id:
        base[0]["chunk_id"] = "din_iso_6507_vickers_full_c000003"
        base[1]["chunk_id"] = "din_iso_6507_vickers_full_c000005"
    return base


# ── BUG-C: grounding fail-safe ────────────────────────────────────────────────

def test_bugc_ungrounded_singles_flagged_not_silently_faked():
    result = build_answer_first_pairs(
        _facts(with_chunk_id=False), [], _FakeLLM(),
        include_singles=True, workers=1, nli_fn=lambda _i: [],
    )
    # Pairs still emitted (backward compatible) but honestly marked incomplete.
    assert result["stats"]["n_pairs_ungrounded"] == result["stats"]["n_single_pass"]
    assert all(qa["grounding_complete"] is False for qa in result["pairs"])
    assert all(qa["gold_chunk_ids"][0].startswith("fact:") for qa in result["pairs"])


def test_bugc_strict_mode_rejects_ungrounded():
    result = build_answer_first_pairs(
        _facts(with_chunk_id=False), [], _FakeLLM(),
        include_singles=True, workers=1, require_chunk_ids=True, nli_fn=lambda _i: [],
    )
    assert result["stats"]["n_single_pass"] == 0
    assert result["stats"]["single_rejection_reasons"].get("ungrounded_chunk_ids") == 2


def test_bugc_real_chunk_ids_are_grounded():
    result = build_answer_first_pairs(
        _facts(with_chunk_id=True), [], _FakeLLM(),
        include_singles=True, workers=1, require_chunk_ids=True, nli_fn=lambda _i: [],
    )
    assert result["stats"]["n_single_pass"] == 2
    assert all(qa["grounding_complete"] is True for qa in result["pairs"])
    assert all(not qa["gold_chunk_ids"][0].startswith("fact:") for qa in result["pairs"])


# ── BUG-B: content-derived draft-cache key ────────────────────────────────────

def test_bugb_cache_key_is_content_not_positional():
    pair_early = {"pair_id": "BP0001", "fact_a": "F1", "fact_b": "F2", "bridge_norm": "indenter"}
    pair_late = {"pair_id": "BP0997", "fact_a": "F1", "fact_b": "F2", "bridge_norm": "indenter"}
    # Same facts + bridge → same key regardless of positional pair_id.
    assert _bridge_cache_key(pair_early) == _bridge_cache_key(pair_late)
    # Different facts → different key.
    other = {"pair_id": "BP0001", "fact_a": "F1", "fact_b": "F9", "bridge_norm": "indenter"}
    assert _bridge_cache_key(pair_early) != _bridge_cache_key(other)


def test_bugb_stale_v1_cache_entries_are_ignored(tmp_path):
    import json
    cache = tmp_path / "drafts.jsonl"
    # A v1-style line with no cache_version must be skipped, not misread.
    cache.write_text(
        json.dumps({"kind": "bridge", "key": "BP0001", "question": "stale?"}) + "\n",
        encoding="utf-8",
    )
    loaded = af._load_draft_cache(cache)
    assert loaded == {}


# ── BUG-D: acronym preservation in composed answers ───────────────────────────

def test_bugd_compose_answer_preserves_acronyms():
    assert _lower_first("ISO 3834 requires records") == "ISO 3834 requires records"
    assert _lower_first("The indenter is a diamond") == "the indenter is a diamond"
    composed = _compose_answer("Clause one", "ISO 3834 mandates traceability")
    assert "ISO 3834" in composed and "iSO" not in composed


# ── BUG-G: short-bridge plural/possessive leak ─────────────────────────────────

def test_bugg_short_bridge_plural_leak_detected():
    assert not bridge_is_hidden("What do the WPSs require for preheating?", "WPS")
    assert not bridge_is_hidden("What does the WPS specify?", "WPS")
    assert bridge_is_hidden("What does the procedure specification require?", "WPS")


# ── BUG-E: anti-fabrication word-boundary ─────────────────────────────────────

def test_buge_phrase_match_respects_word_boundaries():
    assert not _contains_phrase("the greatest method for testing", "test method")
    assert _contains_phrase("this test method applies to steel", "test method")


def test_buge_bridge_pair_rejected_when_bridge_only_a_substring():
    facts = [
        {"id": "A", "text": "The greatest method for hardness is documented.", "page": 1},
        {"id": "B", "text": "A different test method is defined for steel.", "page": 2},
    ]
    # Bridge group asserts both facts share "test method", but fact A only contains
    # it as a substring of "greatest method" — the boundary fix must drop the pair.
    bridge_index = {
        "bridge_groups": [{
            "doc": "d", "norm": "test method", "type": "TERM",
            "surface": "test method", "fact_ids": ["A", "B"], "pages": [1, 2],
        }]
    }
    res = build_bridge_pairs(facts, bridge_index)
    assert res["stats"]["verified_pairs"] == 0
    assert res["stats"]["dropped_bridge_missing"] == 1


# ── BUG-A: parallel extraction is deterministic vs sequential ─────────────────

def test_buga_parallel_extraction_matches_sequential(monkeypatch):
    from types import SimpleNamespace

    from rag_gt.allpdf import extract as ex
    from rag_gt.core.types import Fact

    def _stub_llm_facts(doc_id, chunk, chunk_idx, llm, nli_model):
        # Position-derived id, exactly like the real extractor.
        return [Fact(
            fact_id=f"{doc_id}_F{chunk_idx * 1000 + 1:06d}",
            text=f"fact for chunk {chunk_idx}",
            raw_text=f"fact for chunk {chunk_idx}",
            canonical_form=f"fact for chunk {chunk_idx}",
            self_containment_score=1.0,
            self_containment_known=False,
            role="descriptive",
            weight=0.5,
            supporting_spans=[],
        )]

    monkeypatch.setattr(ex, "_llm_facts_for_chunk", _stub_llm_facts)

    profile = SimpleNamespace(doc_id="testdoc")
    chunk_result = SimpleNamespace(
        chunks=[{"text": f"chunk {i} body", "char_start": 0} for i in range(12)]
    )

    seq = ex.extract_sfu_facts(profile, chunk_result, llm=object(), workers=1)
    par = ex.extract_sfu_facts(profile, chunk_result, llm=object(), workers=6)

    seq_ids = [f.fact_id for f in seq.facts]
    par_ids = [f.fact_id for f in par.facts]
    assert seq_ids == par_ids
    assert [f.text for f in seq.facts] == [f.text for f in par.facts]
    assert len(seq_ids) == 12


# ── BUG-I / BUG-J: NLI dedupe + truncation counters ───────────────────────────

def test_bugi_nli_batch_dedupes_identical_inputs(monkeypatch):
    import rag_gt.validation.nli_check as nc

    predicted = []

    class _FakeModel:
        def predict(self, batch, apply_softmax=True):
            predicted.append(len(batch))
            return [[0.0, 1.0] for _ in batch]

    monkeypatch.setattr(nc.MM, "get_nli", lambda: _FakeModel())
    monkeypatch.setattr(nc, "_entailment_index", lambda: 1)

    # Unique strings so the shared SQLite cache cannot short-circuit the test.
    p, h = f"prem {uuid.uuid4()}", f"hyp {uuid.uuid4()}"
    scores = nc.nli_batch([(p, h), (p, h), (p, h)])
    assert scores == [1.0, 1.0, 1.0]
    assert sum(predicted) == 1  # the unique pair was predicted exactly once


def test_bugj_truncation_is_counted():
    import rag_gt.validation.nli_check as nc

    nc.reset_truncation_stats()
    tp, th = nc._truncate("x" * (nc.PREMISE_MAX + 50), "y" * (nc.HYPOTHESIS_MAX + 50))
    assert len(tp) == nc.PREMISE_MAX
    assert len(th) == nc.HYPOTHESIS_MAX
    stats = nc.truncation_stats()
    assert stats["premise_truncated"] >= 1
    assert stats["hypothesis_truncated"] >= 1


# ── Track B: doc display name is injected, forbidden phrases still forbidden ───

def test_trackb_single_prompt_names_document():
    prompt = _single_prompt("The Vickers test uses a diamond indenter.",
                            "ISO 6507 (Vickers hardness test)")
    assert "ISO 6507 (Vickers hardness test)" in prompt
    # The forbidden-phrase rule is still present.
    assert "this document" in prompt  # appears inside the prohibition text
    assert "DOCUMENT: ISO 6507" in prompt


def test_trackb_doc_name_falls_back_for_unknown_id():
    settings = {"doc_display_names": {"known_full": "the Known book"}}
    assert _doc_name("known_full", settings) == "the Known book"
    assert _doc_name("mystery_doc", settings) == "the source document"
