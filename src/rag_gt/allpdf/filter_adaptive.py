"""Stage 4 — Adaptive domain filter.

Wraps the existing ``filter_fact_domain`` with doc-type-aware rule relaxation.
The existing filter was tuned for prose textbooks (Sutton). Applied unchanged to
ISO standards, financial reports, or OCR scanned docs it incorrectly drops:
- ISO clause facts (start with section numbers like "4.3.1 The system shall ...")
- Financial table rows (match CAPTION_OR_HEADING_RE on numbered headings)
- OCR blocks that trigger noise heuristics

Strategy: route each doc_type to a filter tier:
- strict   : TEXTBOOK / NARRATIVE — apply full filter (existing behaviour)
- relaxed  : ISO_STANDARD / REPORT — skip caption/section-number filter; still
             reject bibliography, front-matter artifacts, exercises, duplicates
- minimal  : UNKNOWN (scanned OCR) — only reject empty text + bibliography
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Tuple

from loguru import logger

from rag_gt.allpdf.preflight import DocProfile
from rag_gt.core.types import Fact
from rag_gt.facts.domain_filter import (
    BIBLIOGRAPHY_RE,
    CAPTION_OR_HEADING_RE,
    COMPACT_HEADER_RE,
    COT_META_RE,
    FRONT_MATTER_ARTIFACT_RE,
    ISO_BOILERPLATE_RE,
    NUMBERED_REFERENCE_RE,
    REFERENCE_HEADING_RE,
    SOURCE_TASK_HEADING_RE,
    SOURCE_TASK_PROMPT_RE,
    filter_fact_domain,
    fact_domain_reject_reason,
    has_dangling_anaphora,
    is_bare_numbered_heading,
    is_printout_metadata,
    is_reference_list_dump,
    is_table_artifact,
    strip_running_artifacts,
)


_ISO_CLAUSE_RE = re.compile(r"^\s*\d+(?:\.\d+)+\s+\S", re.I)

# Relaxed-tier self-containment floor. The Stage-3 default (0.5) is low enough
# that genuine boilerplate (e.g. a CEN country list scored 0.7) clears it.
_RELAXED_MIN_SELF_CONTAINMENT = 0.75
# Facts shorter than this (after stripping) are list-residue / heading fragments.
_RELAXED_MIN_WORDS = 8


def _fact_page(fact: Fact):
    """First supporting-span page for a fact, or None."""
    spans = getattr(fact, "supporting_spans", None) or []
    if spans:
        return getattr(spans[0], "page_start", None)
    return None


_ORPHAN_LIST_MARKER_RE = re.compile(r"\b[a-z]\)\s")
_BULLET_START_RE = re.compile(r"^\s*(?:[—–•*]|[-]{1,2})\s+")


def _is_fragment(text: str) -> bool:
    """Mid-sentence / list-residue fragment heuristic.

    Tuned to avoid two real false-positives seen on din_iso_6507:
    - a valid short sentence ("If possible, avoid testing etched surfaces.") —
      so the word-floor is 5, not 8, and only applies with a lower-case start;
    - an OCR-mangled opener ("lf such faults are detected ...", "If"->"lf") —
      so a lower-case start alone does not condemn a long sentence.
    A real ISO clause (starts with a section number) is never a fragment.
    """
    if _ISO_CLAUSE_RE.match(text):
        return False
    stripped = text.strip()
    if not stripped:
        return True
    # BUG-1 backstop: a fact that still opens with a list/bullet marker is an
    # unresolved orphan list item ("— Wire speed feed range ...").
    if _BULLET_START_RE.match(stripped):
        return True
    words = stripped.split()
    # Too short to stand alone as a fact.
    if len(words) < 5:
        return True
    first = words[0]
    # A lower-case OPENER signals a mid-sentence split — but only when the first
    # token is a plain lower-case word. Scientific tokens that legitimately start
    # lower-case but carry an internal capital (pH, pKa, mV, dB, kPa, mmHg) are
    # NOT fragment openers, so a short "pH values below 7.0 indicate acidity." is
    # kept.
    starts_lower = first[:1].islower() and first.isalpha() and first.islower()
    # Mid-sentence list residue: a lower-case opener trailing item markers
    # ("correction is applicable; f) ... g) ..."). The lower-case gate keeps a
    # legitimate enumerated definition ("From this definition ... : a) ... b) ...")
    # — which starts upper-case — out of the fragment bucket.
    if starts_lower and _ORPHAN_LIST_MARKER_RE.search(stripped):
        return True
    # Mid-sentence start: lower-case opener AND still short (dangling clause).
    if starts_lower and len(words) < _RELAXED_MIN_WORDS:
        return True
    return False


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _relaxed_reject(fact: Fact, front_matter_pages: frozenset[int] = frozenset()) -> str | None:
    """Reject reason for ISO_STANDARD / REPORT.

    Drops academic *and* ISO/standards document-structure artefacts at the
    domain-filter stage (before the graph), so no downstream consumer — normal
    multi-hop pairs or Phase-3 clusters — ever sees them. ``front_matter_pages``
    comes from the Stage-0 profile and lets us drop front-matter facts by
    provenance instead of guessing from text.
    """
    text = " ".join((fact.text or "").split())
    if not text:
        return "empty_text"
    ta = _ascii(text)

    # --- raw-text structural guard (BEFORE the fluent-rewrite checks) ---
    # The canonical rewrite turns an Annex form/table region into fluent prose
    # ("Oscillation: amplitude... :" -> "The following parameters shall be
    # specified: ..."), erasing the colon/dotted-leader signature so the checks
    # below on `text` can no longer see it. We therefore test the PRE-REWRITE
    # `raw_text` against the same structural signatures: a fact whose source span
    # is a form/table/reference-list region is rejected regardless of how fluent
    # the rewrite reads. This is the root-cause guard for informative-form facts
    # (F065001/F067001) being laundered into normative requirements.
    raw = " ".join((fact.raw_text or "").split())
    if raw and raw != text:
        if is_table_artifact(raw):
            return "table_artifact_raw"
        if is_reference_list_dump(raw):
            return "reference_list_dump_raw"

    # --- academic artefacts (original behaviour) ---
    if REFERENCE_HEADING_RE.match(ta):
        return "reference_heading"
    if SOURCE_TASK_HEADING_RE.match(ta):
        return "source_task_heading"
    if SOURCE_TASK_PROMPT_RE.match(ta):
        return "source_task_prompt"
    if FRONT_MATTER_ARTIFACT_RE.search(ta):
        return "front_matter_artifact"
    if BIBLIOGRAPHY_RE.search(ta):
        return "bibliography"

    # --- ISO / standards document-structure artefacts (RC-1, RC-2) ---
    # Note: the controlled-copy watermark (RC-4) is STRIPPED, not rejected, in
    # filter_facts_adaptive before this runs — the remainder is real content.
    # The CEN adoption country-list facts are caught here (administrative
    # context required), not by a blunt standalone country-list regex.
    if ISO_BOILERPLATE_RE.search(text):
        return "iso_boilerplate"
    if NUMBERED_REFERENCE_RE.match(text):
        return "numbered_reference"
    if is_reference_list_dump(text):
        return "reference_list_dump"
    if is_table_artifact(text):
        return "table_artifact"
    if is_bare_numbered_heading(text):
        return "bare_numbered_heading"
    if has_dangling_anaphora(text):
        return "dangling_anaphora"
    if is_printout_metadata(text):
        return "printout_metadata"
    if COT_META_RE.search(text):
        return "cot_meta_leak"

    # --- mid-sentence / list-residue fragments (RC-5) ---
    # Checked before the self-containment floor so a fragment is reported as
    # "fragment", not mis-attributed to "weak_self_containment".
    if _is_fragment(text):
        return "fragment"

    # --- self-containment floor (RC-6) ---
    sc = getattr(fact, "self_containment_score", None)
    if getattr(fact, "self_containment_known", False) and sc is not None \
            and sc < _RELAXED_MIN_SELF_CONTAINMENT:
        return "weak_self_containment"

    # --- front-matter by page provenance (RC-1 fill-the-gap) ---
    # Front-matter pages (title/TOC/foreword/adoption notices) carry no technical
    # body. Drop facts originating there unless the text is a real ISO clause
    # (protects a Scope/normative clause mis-paged into front matter).
    page = _fact_page(fact)
    if page is not None and page in front_matter_pages and not _ISO_CLAUSE_RE.match(text):
        return "front_matter_page"

    # --- caption/heading, only if NOT an ISO clause ---
    if (CAPTION_OR_HEADING_RE.match(ta) or COMPACT_HEADER_RE.match(ta)):
        if not _ISO_CLAUSE_RE.match(text):
            return "caption_or_heading"
    return None


def _minimal_reject(fact: Fact) -> str | None:
    """Reject reason for UNKNOWN (scanned OCR) — keep almost everything."""
    text = " ".join((fact.text or "").split())
    if not text:
        return "empty_text"
    ta = _ascii(text)
    if REFERENCE_HEADING_RE.match(ta):
        return "reference_heading"
    if BIBLIOGRAPHY_RE.search(ta):
        return "bibliography"
    if len(text) < 20:
        return "too_short"
    return None


def filter_facts_adaptive(
    facts: Iterable[Fact],
    profile: DocProfile,
) -> Tuple[List[Fact], dict]:
    """Filter facts with doc-type-aware rule relaxation.

    Returns (kept_facts, drop_counts_by_reason).
    """
    doc_type = profile.doc_type_guess
    front_pages = frozenset(getattr(profile, "front_matter_pages", []) or [])

    if doc_type in ("TEXTBOOK", "NARRATIVE"):
        tier = "strict"
        reject_fn = fact_domain_reject_reason
    elif doc_type in ("ISO_STANDARD", "REPORT"):
        tier = "relaxed"
        reject_fn = lambda f: _relaxed_reject(f, front_pages)  # noqa: E731
    else:  # UNKNOWN / scanned OCR
        tier = "minimal"
        reject_fn = _minimal_reject

    kept: List[Fact] = []
    dropped: Counter = Counter()
    n_stripped = 0
    for fact in facts:
        # RC-4: strip controlled-copy watermarks glued into the fact text BEFORE
        # judging it, so the technical remainder is preserved (strip, not drop).
        if tier in ("relaxed", "minimal"):
            cleaned = strip_running_artifacts(fact.text or "")
            if cleaned != (fact.text or ""):
                fact.text = cleaned
                if fact.canonical_form:
                    fact.canonical_form = strip_running_artifacts(fact.canonical_form)
                n_stripped += 1
        reason = reject_fn(fact)
        if reason:
            dropped[reason] += 1
        else:
            kept.append(fact)
    if n_stripped:
        logger.info(f"[filter_adaptive] {profile.doc_id}: stripped watermark from {n_stripped} facts")

    n_in = len(kept) + sum(dropped.values())
    logger.info(
        f"[filter_adaptive] {profile.doc_id}: tier={tier} doc_type={doc_type} "
        f"in={n_in} kept={len(kept)} dropped={sum(dropped.values())} "
        f"reasons={dict(dropped)}"
    )
    return kept, dict(dropped)
