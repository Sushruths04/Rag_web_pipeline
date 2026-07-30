"""Document-intrinsic adaptive budget for V16.2.

Per-doc API spend is a function of (fact_count, page_count, doc_type). The size
signal is log2 * log2 (intentionally sub-linear; log2 * sqrt saturated the
clamp on big textbooks). Doc-type-specific upper clamps let TEXTBOOK reach a
higher tf_sfg_pairs ceiling than REPORT/ISO documents.

See docs/V16_2_FILTER_PLAN_20260519.md §4.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


# Doc-type values come from src/rag_gt/core/types.py::DocType.
# Unknown / unrecognised doc_types fall back to REPORT (mid-range profile).
_FALLBACK_DOC_TYPE = "REPORT"


@dataclass(frozen=True)
class DocTypeBudget:
    """Per-doc-type scaling constants and upper clamp.

    `tf_sfg_pairs_hi` is set higher for TEXTBOOK so big books (Sutton,
    Bonaventure) don't saturate at the same ceiling as a short report.
    """

    tf_sfg_k: float
    fb_k: float
    target_k: float
    tf_sfg_pairs_hi: int


@dataclass(frozen=True)
class AdaptiveBudgetConfig:
    """Top-level adaptive-budget config (parsed from v16_2.adaptive_budget)."""

    enabled: bool = True
    attempt_multiplier: int = 4
    global_strict_per_doc_cap: int = 60
    base_budget: Mapping[str, DocTypeBudget] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "AdaptiveBudgetConfig":
        raw = raw or {}
        base_raw = raw.get("base_budget", {}) or {}
        base = {
            doc_type: DocTypeBudget(
                tf_sfg_k=float(spec["tf_sfg_k"]),
                fb_k=float(spec["fb_k"]),
                target_k=float(spec["target_k"]),
                tf_sfg_pairs_hi=int(spec["tf_sfg_pairs_hi"]),
            )
            for doc_type, spec in base_raw.items()
        }
        return cls(
            enabled=bool(raw.get("enabled", True)),
            attempt_multiplier=int(raw.get("attempt_multiplier", 4)),
            global_strict_per_doc_cap=int(raw.get("global_strict_per_doc_cap", 60)),
            base_budget=base,
        )


@dataclass(frozen=True)
class DocBudget:
    """Computed per-document budget.

    `mh_floor_fraction` / `untyped_mh_cap_fraction` / `fb_cap_fraction` come
    from the yield_controller config and are duplicated here so downstream
    callers receive a single self-contained budget object.
    """

    fact_count: int
    page_count: int
    doc_type: str
    size_signal: float

    tf_sfg_pairs: int
    fallback_singlehop: int
    strict_target: int
    attempt_cap: int

    mh_floor_fraction: float
    untyped_mh_cap_fraction: float
    fb_cap_fraction: float


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _size_signal(fact_count: int, page_count: int) -> float:
    """log2(fact_count + 1) * log2(page_count + 1).

    Sub-linear so Sutton-scale docs don't saturate the clamp. Both factors are
    log2-bounded; the +1 protects against fact_count=0 or page_count=0.
    """
    fact_count = max(0, fact_count)
    page_count = max(0, page_count)
    return math.log2(fact_count + 1) * math.log2(page_count + 1)


def compute_doc_budget(
    fact_count: int,
    page_count: int,
    doc_type: str,
    cfg: AdaptiveBudgetConfig,
    *,
    mh_floor_fraction: float = 0.60,
    untyped_mh_cap_fraction: float = 0.10,
    fb_cap_fraction: float = 0.40,
) -> DocBudget:
    """Compute the adaptive per-document budget.

    Yield-controller fractions are passed in (they live in a different config
    block) so the budget object is self-contained for downstream callers.
    """
    base = cfg.base_budget.get(doc_type) or cfg.base_budget.get(_FALLBACK_DOC_TYPE)
    if base is None:
        # Defensive default: no config entry at all. Use a moderate profile so
        # the pipeline still runs in test/dev with a minimal config.
        base = DocTypeBudget(tf_sfg_k=4.0, fb_k=1.5, target_k=0.8, tf_sfg_pairs_hi=400)

    signal = _size_signal(fact_count, page_count)
    tf_sfg_pairs = _clamp(round(base.tf_sfg_k * signal), lo=24, hi=base.tf_sfg_pairs_hi)
    fallback_singlehop = _clamp(round(base.fb_k * signal), lo=8, hi=128)
    strict_target = _clamp(
        round(base.target_k * signal), lo=4, hi=cfg.global_strict_per_doc_cap
    )
    attempt_cap = round(strict_target * cfg.attempt_multiplier)

    return DocBudget(
        fact_count=fact_count,
        page_count=page_count,
        doc_type=doc_type,
        size_signal=signal,
        tf_sfg_pairs=tf_sfg_pairs,
        fallback_singlehop=fallback_singlehop,
        strict_target=strict_target,
        attempt_cap=attempt_cap,
        mh_floor_fraction=mh_floor_fraction,
        untyped_mh_cap_fraction=untyped_mh_cap_fraction,
        fb_cap_fraction=fb_cap_fraction,
    )


# ---------------------------------------------------------------------------
# Future scope: warm-up yield-feedback rescaling (V16.3).
#
# Intentionally scaffolded but disabled in V16.2 so the adaptive-budget claim
# in the paper is strictly intrinsic (function of doc structure only) and
# does not overlap with optimal-stopping prior work (BEACON 2025).
#
# def rescale_after_warmup(
#     budget: DocBudget,
#     warmup_attempts: int,
#     warmup_accepted: int,
#     target_acceptance: float,
# ) -> DocBudget:
#     """Rescale remaining budget based on observed acceptance rate.
#
#     This is the V16.3 entry point — leave commented out until that work begins.
#     """
#     ...
# ---------------------------------------------------------------------------
