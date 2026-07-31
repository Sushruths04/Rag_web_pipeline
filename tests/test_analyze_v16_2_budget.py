"""Tests for V16.2 budget comparison analyzer."""

from __future__ import annotations

from rag_gt.cli.analyze_v16_2_budget import build_markdown


def test_build_markdown_contains_per_doc_and_aggregate() -> None:
    v1 = {
        "cost_tracker": {
            "docA": {
                "live_api_calls": 100,
                "cache_hit_calls": 0,
                "total_logical_calls": 100,
            }
        },
        "cascade_stats": {
            "docA": {"yield": {"strict_total": 10}},
        },
    }
    v2 = {
        "cost_tracker": {
            "docA": {
                "live_api_calls": 60,
                "cache_hit_calls": 10,
                "total_logical_calls": 70,
            }
        },
        "cascade_stats": {
            "docA": {
                "yield": {
                    "strict_total": 10,
                    "typed_mh": {"accepted": 7},
                    "typed_mh_share": 0.7,
                }
            },
        },
    }
    md = build_markdown(v1, v2)
    assert "`docA`" in md
    assert "`aggregate`" in md
    assert "40.0%" in md
    assert "70.0%" in md
