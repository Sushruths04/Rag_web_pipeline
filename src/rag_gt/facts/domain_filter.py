"""Pre-graph source-section and fact-domain filtering.

This gate runs after span resolution and before vector indexing. Its job is to
keep non-content artifacts out of the fact graph, where they otherwise become
required evidence for generated questions.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Iterable, Optional

from rag_gt.core.types import Fact
from rag_gt.core.config import load_config


REFERENCE_HEADING_RE = re.compile(
    r"^\s*(references|bibliography|works cited|further reading|index)\s*$",
    re.I,
)
SOURCE_TASK_HEADING_RE = re.compile(
    r"^\s*(exercise|exercises|problem|problems|questions|review questions)"
    r"(?:\s+\d+(?:\.\d+)*)?\s*$",
    re.I,
)
SOURCE_TASK_PROMPT_RE = re.compile(
    r"^\s*(exercise|problem|question)\s+\d+(?:\.\d+)*\b|"
    r"^\s*(show|prove|derive|calculate|compute)\b",
    re.I,
)
SECTION_NUMBER_PREFIX_RE = re.compile(
    r"^\s*[*∗]?\s*\d+(?:\.\d+)+\s+[A-Z]",
    re.I,
)
CAPTION_OR_HEADING_RE = re.compile(
    r"^\s*(figure|fig\.|table|chapter|section|appendix)\s+\d+(?:\.\d+)*"
    r"[\s:.-]+",
    re.I,
)
EMBEDDED_CAPTION_RE = re.compile(
    r"\b(?:figure|fig\.|table)\s+\d+(?:\.\d+)*\s*:",
    re.I,
)
COMPACT_HEADER_RE = re.compile(
    r"^\s*\d{1,4}\s*(chapter|section|appendix)\s+\d+(?:\.\d+)*",
    re.I,
)
BIBLIOGRAPHY_RE = re.compile(
    r"\b("
    r"proceedings|workshop|conference|journal|transactions|"
    r"university press|morgan kaufmann|springer|elsevier|wiley|"
    r"mit press|acm|ieee|cambridge university press|"
    r"technical report|tech\.?\s+report|dissertation|ph\.?\s*d\.?|"
    r"\bthesis\b|isbn|doi|arxiv|vol\.|no\."
    r")\b",
    re.I,
)
# --- ISO / standards document-structure artefacts -------------------------
# ISO and CEN standards carry administrative boilerplate (patent-rights
# disclaimers, CEN-CENELEC internal-regulation notices, national-body adoption
# lists) that BIBLIOGRAPHY_RE / FRONT_MATTER_ARTIFACT_RE (tuned for academic
# textbooks) never match. These are document structure, not technical content,
# and must be dropped at the domain-filter stage before they enter the graph.
ISO_BOILERPLATE_RE = re.compile(
    r"\b("
    r"attention is drawn to the possibility|"
    r"subject of patent rights|"
    r"shall not be held responsible for identifying|"
    r"cen[-‑– ]?cenelec internal regulations|"
    r"national standards (?:bodies|organi[sz]ations) of|"
    r"cen members are the national standards|"
    r"the secretariat of which is held by|"
    r"this (?:european|international) standard shall be given the status"
    r")\b",
    re.I,
)
# Numbered bibliography reference: a fact that begins with "[N]" is, in ISO
# documents, a reference-list entry regardless of the publisher named.
NUMBERED_REFERENCE_RE = re.compile(r"^\s*\[\d{1,3}\]\s+\S")
# NOTE: a standalone "country list" regex (>=5 comma-separated ProperCase tokens)
# was deliberately NOT added — it over-rejects legitimate technical enumerations
# (languages, chemical elements, author/site lists) on REPORT-type docs. The two
# real CEN adoption-list facts are already caught by ISO_BOILERPLATE_RE via the
# "national standards bodies/organizations of" / "CEN members are the national
# standards" anchors, which require the administrative context to be present.
# Running-header / controlled-copy watermark glued onto a fact by PDF text
# extraction (the header line is pulled in-flow and concatenated to the body).
# This is a STRIP target, not a reject target: the technical remainder after the
# watermark is real content and must be preserved.
PRINTED_COPY_WATERMARK_RE = re.compile(
    r"(?:user name\s*:\s*\S+\s+)?"
    r"(?:printed copies are uncontrolled|uncontrolled when printed)",
    re.I,
)


# BUG-2: form/table pages (e.g. the WPS template) are flattened by PDF text
# extraction into a single run of colon-separated field labels with footnote
# superscripts glued to words ("Voltage2", "Wire feed speed2", "Arc energy1, 2").
# These are fill-in templates, not factual claims — reject them, but only on a
# strong signature so genuine prose with a colon or a number is never touched.
_GLUED_SUPERSCRIPT_RE = re.compile(r"[A-Za-z]{3,}\d")


def is_table_artifact(text: str) -> bool:
    """True when text is flattened form/table residue, not a prose fact."""
    t = text or ""
    # >=4 colon-separated field labels = a form template, not a sentence.
    if t.count(":") >= 4:
        return True
    # >=3 words with a footnote digit glued on = a flattened table row.
    if len(_GLUED_SUPERSCRIPT_RE.findall(t)) >= 3:
        return True
    # Dotted leaders (TOC / fill-in form lines): 4+ consecutive dots or an
    # ellipsis run. Three-dot ellipsis in prose ("...") is intentionally excluded.
    if _DOTTED_LEADER_RE.search(t):
        return True
    return False


# Dotted leaders appear in tables of contents and blank form fields
# ("Heating and cooling rates: ......."). 4+ dots (or a unicode ellipsis run).
_DOTTED_LEADER_RE = re.compile(r"\.{4,}|…")

# A normative-references list flattens into one fact as several
# "ISO 6848, Arc welding ..." entries. A genuine prose sentence almost never
# writes a standard number immediately followed by a comma three+ times.
_STANDARD_REF_ENTRY_RE = re.compile(
    r"\b(?:ISO|EN|DIN|IEC|ASTM|BS|AWS)(?:/TR|/TS)?\s*\d{2,5}(?:[-‑]\d+)?\s*,",
    re.I,
)


def is_reference_list_dump(text: str) -> bool:
    """True when the fact is a flattened standards reference list (>=3 entries)."""
    return len(_STANDARD_REF_ENTRY_RE.findall(text or "")) >= 3


# A bare top-level section header ("4 Technical content of ... (WPS)"): a single
# integer, then a Title, no finite verb, no terminal sentence punctuation. ISO
# clauses ("4.2 The process shall ...") carry a dotted number and are exempt.
_BARE_NUMBERED_HEADING_RE = re.compile(r"^\s*\d{1,2}\s+[A-Z][A-Za-z]")
_TERMINAL_PUNCT_RE = re.compile(r"[.?!]\s*$")


def is_bare_numbered_heading(text: str) -> bool:
    t = " ".join((text or "").split())
    if not _BARE_NUMBERED_HEADING_RE.match(t):
        return False
    # A dotted clause number ("4.2 ...") is a real clause, not a bare header.
    if re.match(r"^\s*\d+\.\d", t):
        return False
    # Headers are short and do not end in sentence punctuation.
    if _TERMINAL_PUNCT_RE.search(t):
        return False
    return len(t.split()) <= 10


# A fact whose meaning hinges on an unresolved demonstrative used as a bare
# object: "...not assigned in those," / "...applies to these." The antecedent
# lives in a different fact, so any edge built on it re-binds the pronoun and
# fabricates a relationship. A demonstrative FOLLOWED BY A NOUN ("in those
# material groups") is resolved and is left untouched.
_DANGLING_ANAPHORA_RE = re.compile(
    r"\b(?:in|of|on|to|from|for|with|by|than|as|into|within)\s+"
    r"(?:those|these|that|them)\s*(?:[,.;:)]|$)",
    re.I,
)


def has_dangling_anaphora(text: str) -> bool:
    return bool(_DANGLING_ANAPHORA_RE.search(" ".join((text or "").split())))


# Per-page print/export artifacts: a "printout" date-time stamp or a fact that
# is essentially "The date and time of the printout are 2025-04-23, 12:00:03."
# These are NLI-faithful, grammatically complete sentences (so they pass the
# self-containment + NLI guards) but carry zero technical content.
_PRINTOUT_METADATA_RE = re.compile(
    r"\bprintout\b|"
    r"\bdate and time\b[^.]*\d{1,2}:\d{2}|"
    r"\b\d{4}-\d{2}-\d{2}\s*,?\s*\d{1,2}:\d{2}:\d{2}\b",
    re.I,
)


def is_printout_metadata(text: str) -> bool:
    return bool(_PRINTOUT_METADATA_RE.search(text or ""))


def strip_running_artifacts(text: str) -> str:
    """Remove controlled-copy / running-header watermarks glued into a fact.

    Deterministic text normalisation applied at the chunk->fact gateway so the
    SFU segmenter never folds a page header into a fact (root cause for RC-4).
    Preserves the technical remainder; collapses whitespace.
    """
    if not text:
        return text
    # Only the full watermark (optionally including the "User name: X" prefix) is
    # stripped. A bare "user name:" is NOT stripped on its own — it is a
    # legitimate phrase in IT/security standards (ISO 27001 etc.) and stripping
    # it would corrupt real content.
    cleaned = PRINTED_COPY_WATERMARK_RE.sub(" ", text)
    return " ".join(cleaned.split())
# Reasoning-model chain-of-thought captured as a fact (defence-in-depth net for
# the Stage-3 canonical_form_rewrite guard). Restricted to model-meta phrasings
# that do not occur in real document prose — generic phrases like "resolve
# pronouns" or "understood in isolation" are NLP-paper vocabulary and were
# deliberately excluded to avoid over-rejection.
COT_META_RE = re.compile(
    r"\b(we need to rewrite the passage|"
    r"rewrite the passage so it can be understood|"
    r"the passage (?:is|reads)\s*:\s*[\"'])",
    re.I,
)

PAGE_RANGE_RE = re.compile(
    r"\b(pp\.|pages?)\s*\d+\s*(?:--|-|\u2013|\u2014|to)\s*\d+\b|"
    r"\b\d+\s*(?:--|-|\u2013|\u2014)\s*\d+\b|"
    r"\b\d+\s*:\s*\d+\s*(?:--|-|\u2013|\u2014)\s*\d+\b",
    re.I,
)
CITATION_PREFIX_RE = re.compile(
    r"^\s*(?:\[\d+\]|\(\d{4}[a-z]?\)|[A-Z][A-Za-z-]+,\s+[A-Z]\.)"
)
NAMED_CITATION_CONTEXT_RE = re.compile(
    r"^\s*[A-Z][A-Za-z-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z-]+)?"
    r"\s*\(\d{4}[a-z]?\)\s+"
    r"(?:discussed|considered|introduced|proposed|studied|showed|reported)\b",
    re.I,
)
SOURCE_WORDING_RE = re.compile(
    r"\b(the|this|that)\s+"
    r"(above|following|preceding|given|provided)\s+"
    r"(statement|text|passage|paragraph|description|example)\b",
    re.I,
)
DANGLING_SECTION_REF_RE = re.compile(
    r"\b(?:in|section)\s+\d+(?:\.\d+)+\.?$",
    re.I,
)
FRONT_MATTER_ARTIFACT_RE = re.compile(
    r"\b("
    r"copyright|creative commons|cc[- ]?by|license(?:d)?|"
    r"free\s+pdf|low\s+cost\s+print|printed\s+version|"
    r"textbook equity|saylor\.org|open textbook challenge|"
    r"about the authors?|foreword|with contributions from|translated from|"
    r"translator|published by|publisher|isbn|"
    r"table of contents|contents|preface|dear reader|"
    r"study guide|examinations?--study guides?|"
    r"library of congress|catalog(?:ue|ing)?|"
    r"all rights reserved|springer(?:\s+vieweg)?"
    r")\b",
    re.I,
)
URL_OR_EMAIL_RE = re.compile(
    r"\b(?:https?://|www\.|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
    re.I,
)
AUTHOR_BIO_RE = re.compile(
    r"\b("
    r"holds a full professorship|received his ph\.?d|received her ph\.?d|"
    r"\bco-?author of\b|peer-reviewed publications|"
    r"served as program and general chair|"
    r"chief consultant|founder and executive partner|"
    r"you can find (?:more|further) information"
    r")\b",
    re.I,
)
RELEASE_HEADER_RE = re.compile(
    r"\b(Computer Networking\s*:\s*Principles,\s*Protocols and Practice,\s*Release|"
    r"Release\s+\d+(?:\.\d+)?)\b",
    re.I,
)
HEADING_WITH_PAGE_RE = re.compile(
    r"^[A-Z][A-Za-z0-9 ,:;'/()&+-]{4,80}\s+\d{1,4}\s+"
    r"[A-Z][A-Za-z0-9 ,:;'/()&+-]{4,}",
    re.I,
)
SOURCE_FIGURE_ATTRIBUTION_RE = re.compile(
    r"\breprinted\s+from\b|"
    r"\b(?:results?|data)\s+in\s+(?:figure|table)\s+\d+(?:\.\d+)*\s+"
    r"(?:are\s+)?(?:due\s+to|from)\b",
    re.I,
)
UNRESOLVED_SOURCE_REFERENCE_RE = re.compile(
    r"\b(?:algorithm|method|example|case)\s+"
    r"(?:discussed|described|shown|introduced)\s+in\s+(?:the\s+)?"
    r"(?:previous|preceding|following)\s+(?:section|figure|diagram)\b|"
    r"^\s*(?:this|that)\s+(?:idea|example|diagram|relationship|case)\b|"
    r"^\s*another\s+way\s+of\s+(?:saying|seeing)\s+this\b|"
    r"^\s*(?:in\s+other\s+words|we\s+have\s+just\s+shown\s+that|this\s+means\s+that)\b|"
    r"^\s*if\s+we\s+look\s+closely\s+at\s+(?:this|that|the)\s+example\b|"
    r"\bas\s+in\s+the\s+(?:left|right)\s+(?:diagram|figure)\b",
    re.I,
)
UNRESOLVED_DEICTIC_OPENER_RE = re.compile(
    r"^\s*(?:this|it|these|those|that|they)\s+"
    r"(?:can|is|are|was|were|will|would|may|might|"
    r"makes?|means?|allows?|enables?|provides?|"
    r"leads?|causes?|ensures?|results?|shows?|"
    r"suggests?|implies?|indicates?|represents?|demonstrates?|"
    r"has|have|had|does|do|did)\b",
    re.I,
)

VISUAL_ONLY_FACT_RE = re.compile(
    r"^\s*(?:for\s+example,\s+)?(?:the\s+)?backup\s+diagram\s+"
    r"(?:for|corresponding\s+to|showing)\b",
    re.I,
)
ALGORITHM_LISTING_MARKERS = (
    re.compile(r"\brepeat\s*\(for each", re.I),
    re.compile(r"\binitialize\s+[A-Z]", re.I),
    re.compile(r"\btake action\b", re.I),
    re.compile(r"\bchoose\s+[A-Z].*\busing policy\b", re.I),
    re.compile(r"\bargmax\b|(?:←|<-)", re.I),
)


@lru_cache(maxsize=1)
def _doc_page_excludes() -> dict[str, list[tuple[int, int]]]:
    try:
        raw = load_config().get("fact_domain_filter", {}).get("doc_page_excludes", {})
    except Exception:
        return {}
    out: dict[str, list[tuple[int, int]]] = {}
    if not isinstance(raw, dict):
        return out
    for doc_id, ranges in raw.items():
        parsed: list[tuple[int, int]] = []
        if not isinstance(ranges, list):
            continue
        for item in ranges:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                start, end = int(item[0]), int(item[1])
            except (TypeError, ValueError):
                continue
            parsed.append((start, end))
        if parsed:
            out[str(doc_id)] = parsed
    return out


def _in_excluded_source_section(fact: Fact) -> bool:
    excludes = _doc_page_excludes()
    if not excludes:
        return False
    for span in fact.supporting_spans:
        doc_id = str(span.doc_id or "")
        ranges = excludes.get(doc_id)
        if not ranges or span.page_start is None:
            continue
        page = int(span.page_start)
        if any(start <= page <= end for start, end in ranges):
            return True
    return False


def _has_all_caps_heading_prefix(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", text)
    if len(words) < 5:
        return False
    prefix = words[: min(6, len(words))]
    upper_run = 0
    for word in prefix:
        if len(word) > 1 and word.upper() == word:
            upper_run += 1
            continue
        break
    if upper_run >= 3 and upper_run < len(words):
        return True
    return (
        upper_run >= 2
        and upper_run < len(words)
        and all(len(w) > 3 for w in words[:upper_run])
    )


def _looks_like_algorithm_listing(text: str) -> bool:
    return sum(bool(pattern.search(text)) for pattern in ALGORITHM_LISTING_MARKERS) >= 2


def fact_domain_reject_reason(fact: Fact) -> Optional[str]:
    text = " ".join((fact.text or "").split())
    text_ascii = unicodedata.normalize("NFKD", text).encode(
        "ascii", "ignore"
    ).decode("ascii")
    if not text:
        return "empty_text"

    if _in_excluded_source_section(fact):
        return "excluded_source_section"
    if REFERENCE_HEADING_RE.match(text_ascii):
        return "reference_heading"
    if SOURCE_TASK_HEADING_RE.match(text_ascii):
        return "source_task_heading"
    if SOURCE_TASK_PROMPT_RE.match(text_ascii):
        return "source_task_prompt"
    if CAPTION_OR_HEADING_RE.match(text_ascii) or COMPACT_HEADER_RE.match(text_ascii):
        return "caption_or_heading"
    if EMBEDDED_CAPTION_RE.search(text_ascii):
        return "embedded_caption_fragment"
    if _looks_like_algorithm_listing(text):
        return "algorithm_listing_fragment"
    if SECTION_NUMBER_PREFIX_RE.match(text_ascii):
        return "section_prefix_fragment"
    if _has_all_caps_heading_prefix(text_ascii):
        return "heading_prefix_fragment"
    if SOURCE_WORDING_RE.search(text_ascii):
        return "source_wording"
    if SOURCE_FIGURE_ATTRIBUTION_RE.search(text_ascii):
        return "source_figure_attribution"
    if UNRESOLVED_SOURCE_REFERENCE_RE.search(text_ascii):
        return "unresolved_source_reference"
    if VISUAL_ONLY_FACT_RE.search(text_ascii):
        return "visual_only_fact"
    if URL_OR_EMAIL_RE.search(text_ascii):
        return "front_matter_artifact"
    if FRONT_MATTER_ARTIFACT_RE.search(text_ascii):
        return "front_matter_artifact"
    if AUTHOR_BIO_RE.search(text_ascii):
        return "author_bio"
    if RELEASE_HEADER_RE.search(text_ascii):
        return "source_release_header"
    if HEADING_WITH_PAGE_RE.match(text_ascii):
        return "caption_or_heading"
    if DANGLING_SECTION_REF_RE.search(text_ascii):
        return "dangling_section_reference"
    if NAMED_CITATION_CONTEXT_RE.match(text_ascii):
        return "named_citation_context"
    if CITATION_PREFIX_RE.match(text_ascii) and BIBLIOGRAPHY_RE.search(text_ascii):
        return "bibliography_or_citation"
    if BIBLIOGRAPHY_RE.search(text_ascii) and PAGE_RANGE_RE.search(text_ascii):
        return "bibliography_or_citation"
    if PAGE_RANGE_RE.search(text_ascii) and len(text_ascii.split()) <= 16:
        return "page_range_or_locator"

    return None


def fact_has_unresolved_deictic(fact: Fact) -> bool:
    """True when a fact text opens with a context-dependent pronoun and hasn't been SFU-resolved.

    Facts like "This can make X..." or "It is because..." hard-fail Q7_bad_fact_fragment
    in quality scoring.  If canonical_form is populated the SFU upgrader has resolved the
    reference and the fact is treated as self-contained.
    """
    if fact.canonical_form:
        return False
    text = " ".join((fact.text or "").split())
    return bool(UNRESOLVED_DEICTIC_OPENER_RE.match(text))


def filter_fact_domain(facts: Iterable[Fact]) -> tuple[list[Fact], dict[str, int]]:
    kept: list[Fact] = []
    dropped: Counter[str] = Counter()
    for fact in facts:
        reason = fact_domain_reject_reason(fact)
        if reason:
            dropped[reason] += 1
        else:
            kept.append(fact)
    return kept, dict(dropped)
