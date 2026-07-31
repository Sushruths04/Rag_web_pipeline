"""Tests for V16.2 adaptive budget."""

from __future__ import annotations

import math

import pytest

from rag_gt.budget.adaptive_budget import (
    AdaptiveBudgetConfig,
    DocBudget,
    DocTypeBudget,
    compute_doc_budget,
)


def _full_cfg() -> AdaptiveBudgetConfig:
    return AdaptiveBudgetConfig.from_dict(
        {
            "enabled": True,
            "attempt_multiplier": 4,
            "global_strict_per_doc_cap": 60,
            "base_budget": {
                "TEXTBOOK": {
                    "tf_sfg_k": 9.0,
                    "fb_k": 2.0,
                    "target_k": 1.5,
                    "tf_sfg_pairs_hi": 800,
                },
                "ISO_STANDARD": {
                    "tf_sfg_k": 6.0,
                    "fb_k": 1.0,
                    "target_k": 1.0,
                    "tf_sfg_pairs_hi": 400,
                },
                "KT_DOC": {
                    "tf_sfg_k": 5.0,
                    "fb_k": 1.5,
                    "target_k": 1.0,
                    "tf_sfg_pairs_hi": 400,
                },
                "REPORT": {
                    "tf_sfg_k": 4.0,
                    "fb_k": 1.5,
                    "target_k": 0.8,
                    "tf_sfg_pairs_hi": 400,
                },
                "NARRATIVE": {
                    "tf_sfg_k": 3.0,
                    "fb_k": 2.0,
                    "target_k": 0.6,
                    "tf_sfg_pairs_hi": 400,
                },
            },
        }
    )


def test_dataclass_parse_from_dict() -> None:
    cfg = _full_cfg()
    assert cfg.enabled is True
    assert cfg.attempt_multiplier == 4
    assert cfg.global_strict_per_doc_cap == 60
    assert "TEXTBOOK" in cfg.base_budget
    assert cfg.base_budget["TEXTBOOK"].tf_sfg_pairs_hi == 800
    assert cfg.base_budget["REPORT"].tf_sfg_pairs_hi == 400


def test_returns_docbudget_with_expected_fields() -> None:
    cfg = _full_cfg()
    b = compute_doc_budget(fact_count=200, page_count=50, doc_type="REPORT", cfg=cfg)
    assert isinstance(b, DocBudget)
    assert b.fact_count == 200
    assert b.page_count == 50
    assert b.doc_type == "REPORT"
    assert b.size_signal == pytest.approx(math.log2(201) * math.log2(51))
    assert b.tf_sfg_pairs >= 24
    assert b.attempt_cap == b.strict_target * 4
    # default yield-controller fractions threaded through
    assert b.mh_floor_fraction == pytest.approx(0.60)
    assert b.untyped_mh_cap_fraction == pytest.approx(0.10)
    assert b.fb_cap_fraction == pytest.approx(0.40)


def test_monotonic_in_fact_count() -> None:
    cfg = _full_cfg()
    smaller = compute_doc_budget(
        fact_count=50, page_count=30, doc_type="TEXTBOOK", cfg=cfg
    )
    larger = compute_doc_budget(
        fact_count=2000, page_count=30, doc_type="TEXTBOOK", cfg=cfg
    )
    assert larger.tf_sfg_pairs >= smaller.tf_sfg_pairs
    assert larger.strict_target >= smaller.strict_target


def test_monotonic_in_page_count() -> None:
    cfg = _full_cfg()
    smaller = compute_doc_budget(
        fact_count=500, page_count=10, doc_type="TEXTBOOK", cfg=cfg
    )
    larger = compute_doc_budget(
        fact_count=500, page_count=400, doc_type="TEXTBOOK", cfg=cfg
    )
    assert larger.tf_sfg_pairs >= smaller.tf_sfg_pairs


def test_textbook_clamp_higher_than_report() -> None:
    """TEXTBOOK gets a higher tf_sfg_pairs ceiling so big books don't saturate
    at the same level as a small report."""
    cfg = _full_cfg()
    # Extreme size signal forces both into clamp territory.
    tb = compute_doc_budget(
        fact_count=10_000, page_count=800, doc_type="TEXTBOOK", cfg=cfg
    )
    rp = compute_doc_budget(
        fact_count=10_000, page_count=800, doc_type="REPORT", cfg=cfg
    )
    assert tb.tf_sfg_pairs == 800  # TEXTBOOK upper clamp
    assert rp.tf_sfg_pairs == 400  # non-TEXTBOOK upper clamp


def test_lower_clamps_floor_small_docs() -> None:
    cfg = _full_cfg()
    b = compute_doc_budget(
        fact_count=2, page_count=2, doc_type="NARRATIVE", cfg=cfg
    )
    assert b.tf_sfg_pairs >= 24
    assert b.fallback_singlehop >= 8
    assert b.strict_target >= 4


def test_global_strict_cap_respected() -> None:
    cfg = _full_cfg()
    b = compute_doc_budget(
        fact_count=10_000, page_count=800, doc_type="TEXTBOOK", cfg=cfg
    )
    assert b.strict_target <= cfg.global_strict_per_doc_cap


def test_unknown_doc_type_falls_back_to_report() -> None:
    cfg = _full_cfg()
    unknown = compute_doc_budget(
        fact_count=200, page_count=50, doc_type="UNKNOWN", cfg=cfg
    )
    report = compute_doc_budget(
        fact_count=200, page_count=50, doc_type="REPORT", cfg=cfg
    )
    # Both should produce the same budget (UNKNOWN falls through to REPORT).
    assert unknown.tf_sfg_pairs == report.tf_sfg_pairs
    assert unknown.strict_target == report.strict_target


def test_missing_base_budget_uses_safe_default() -> None:
    """Empty base_budget shouldn't crash — should use a moderate default."""
    cfg = AdaptiveBudgetConfig.from_dict(
        {"enabled": True, "attempt_multiplier": 4, "global_strict_per_doc_cap": 60}
    )
    b = compute_doc_budget(
        fact_count=200, page_count=50, doc_type="REPORT", cfg=cfg
    )
    assert b.tf_sfg_pairs >= 24
    assert b.strict_target >= 4


def test_zero_fact_count_handled() -> None:
    cfg = _full_cfg()
    b = compute_doc_budget(fact_count=0, page_count=10, doc_type="REPORT", cfg=cfg)
    # size_signal = log2(1) * log2(11) = 0 → all clamps fire at lo
    assert b.size_signal == pytest.approx(0.0)
    assert b.tf_sfg_pairs == 24
    assert b.strict_target == 4


def test_zero_page_count_handled() -> None:
    cfg = _full_cfg()
    b = compute_doc_budget(fact_count=100, page_count=0, doc_type="REPORT", cfg=cfg)
    assert b.size_signal == pytest.approx(0.0)
    assert b.tf_sfg_pairs == 24


def test_negative_counts_treated_as_zero() -> None:
    cfg = _full_cfg()
    b = compute_doc_budget(fact_count=-5, page_count=-2, doc_type="REPORT", cfg=cfg)
    assert b.size_signal == pytest.approx(0.0)
    assert b.tf_sfg_pairs == 24


def test_size_signal_formula_log2_log2() -> None:
    """log2(fact+1) * log2(page+1) — guards the paper claim about formula."""
    cfg = _full_cfg()
    b = compute_doc_budget(fact_count=15, page_count=7, doc_type="REPORT", cfg=cfg)
    expected = math.log2(16) * math.log2(8)
    assert b.size_signal == pytest.approx(expected)


def test_attempt_cap_scales_with_target() -> None:
    cfg = _full_cfg()
    b = compute_doc_budget(
        fact_count=500, page_count=100, doc_type="TEXTBOOK", cfg=cfg
    )
    assert b.attempt_cap == b.strict_target * cfg.attempt_multiplier


def test_doctype_ordering_textbook_gets_most_aggressive_budget() -> None:
    """For the same doc shape, TEXTBOOK should produce >= budget than REPORT/NARRATIVE."""
    cfg = _full_cfg()
    tb = compute_doc_budget(
        fact_count=300, page_count=100, doc_type="TEXTBOOK", cfg=cfg
    )
    rp = compute_doc_budget(
        fact_count=300, page_count=100, doc_type="REPORT", cfg=cfg
    )
    nr = compute_doc_budget(
        fact_count=300, page_count=100, doc_type="NARRATIVE", cfg=cfg
    )
    assert tb.tf_sfg_pairs >= rp.tf_sfg_pairs
    assert tb.tf_sfg_pairs >= nr.tf_sfg_pairs


def test_yield_fractions_overrideable() -> None:
    cfg = _full_cfg()
    b = compute_doc_budget(
        fact_count=200,
        page_count=50,
        doc_type="REPORT",
        cfg=cfg,
        mh_floor_fraction=0.75,
        untyped_mh_cap_fraction=0.05,
        fb_cap_fraction=0.25,
    )
    assert b.mh_floor_fraction == pytest.approx(0.75)
    assert b.untyped_mh_cap_fraction == pytest.approx(0.05)
    assert b.fb_cap_fraction == pytest.approx(0.25)
