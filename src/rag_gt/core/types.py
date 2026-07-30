"""Core dataclasses shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

Role = Literal[
    "definition",
    "rule",
    "constraint",
    "condition",
    "exception",
    "consequence",
    "example",
]
Difficulty = Literal["easy", "medium", "hard"]
DocType = Literal[
    "ISO_STANDARD", "KT_DOC", "REPORT", "NARRATIVE", "TEXTBOOK", "UNKNOWN"
]
SemDistance = Literal["local", "intra_doc", "cross_doc"]

# v16 PASS-GT extension types
AdversarialAxis = Literal["clean", "distractor", "abstention", "counterfactual"]
TwinType = Literal["abstention", "counterfactual"]
QuestionIntent = Literal[
    "factoid",
    "inferential",
    "comparative",
    "numerical",
    "procedural",
    "counterfactual",
    "list",
    "unanswerable",
]


@dataclass
class Document:
    doc_id: str
    text: str
    version: str = ""
    doc_type: DocType = "UNKNOWN"
    source_path: str = ""
    source_sha1: str = ""
    source_backend: str = ""
    source_units: List["SourceUnit"] = field(default_factory=list)


@dataclass
class SourceBBox:
    page_no: int
    l: float
    t: float
    r: float
    b: float
    coord_origin: str = "BOTTOMLEFT"

    @classmethod
    def from_any(cls, value: Any) -> "SourceBBox":
        if isinstance(value, cls):
            return value
        return cls(
            page_no=int(value.get("page_no", 0) or 0),
            l=float(value.get("l", 0.0) or 0.0),
            t=float(value.get("t", 0.0) or 0.0),
            r=float(value.get("r", 0.0) or 0.0),
            b=float(value.get("b", 0.0) or 0.0),
            coord_origin=str(value.get("coord_origin", "BOTTOMLEFT") or "BOTTOMLEFT"),
        )

    def to_dict(self) -> dict:
        return {
            "page_no": self.page_no,
            "l": self.l,
            "t": self.t,
            "r": self.r,
            "b": self.b,
            "coord_origin": self.coord_origin,
        }


@dataclass
class SourceUnit:
    """A canonical text range mapped back to one source PDF layout item."""

    doc_id: str
    char_start: int
    char_end: int
    text: str = ""
    page_no: Optional[int] = None
    block_id: str = ""
    paragraph_id: str = ""
    bboxes: List[SourceBBox] = field(default_factory=list)
    source_path: str = ""
    source_sha1: str = ""
    extractor: str = ""

    @classmethod
    def from_any(cls, value: Any) -> "SourceUnit":
        if isinstance(value, cls):
            return value
        bboxes = [SourceBBox.from_any(b) for b in value.get("bboxes", [])]
        page_no = value.get("page_no")
        return cls(
            doc_id=str(value.get("doc_id", "") or ""),
            char_start=int(value.get("char_start", 0) or 0),
            char_end=int(value.get("char_end", 0) or 0),
            text=str(value.get("text", "") or ""),
            page_no=int(page_no) if page_no not in (None, "") else None,
            block_id=str(value.get("block_id", "") or ""),
            paragraph_id=str(value.get("paragraph_id", "") or ""),
            bboxes=bboxes,
            source_path=str(value.get("source_path", "") or ""),
            source_sha1=str(value.get("source_sha1", "") or ""),
            extractor=str(value.get("extractor", "") or ""),
        )

    def to_dict(self, include_text: bool = False) -> dict:
        out = {
            "doc_id": self.doc_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "page_no": self.page_no,
            "block_id": self.block_id,
            "paragraph_id": self.paragraph_id,
            "bboxes": [b.to_dict() for b in self.bboxes],
            "source_path": self.source_path,
            "source_sha1": self.source_sha1,
            "extractor": self.extractor,
        }
        if include_text:
            out["text"] = self.text
        return out


@dataclass
class Span:
    doc_id: str
    chunk_id: str
    start_token: int
    end_token: int
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    bboxes: List[SourceBBox] = field(default_factory=list)
    block_ids: List[str] = field(default_factory=list)
    paragraph_ids: List[str] = field(default_factory=list)
    source_text_sha1: str = ""
    source_path: str = ""
    source_sha1: str = ""
    extractor: str = ""

    @classmethod
    def from_any(cls, value: Any) -> "Span":
        if isinstance(value, cls):
            return value
        return cls(
            doc_id=str(value["doc_id"]),
            chunk_id=str(value.get("chunk_id", "") or ""),
            start_token=int(value.get("start_token", 0) or 0),
            end_token=int(value.get("end_token", 0) or 0),
            char_start=_optional_int(value.get("char_start")),
            char_end=_optional_int(value.get("char_end")),
            page_start=_optional_int(value.get("page_start")),
            page_end=_optional_int(value.get("page_end")),
            bboxes=[SourceBBox.from_any(b) for b in value.get("bboxes", [])],
            block_ids=[str(x) for x in value.get("block_ids", []) if str(x)],
            paragraph_ids=[
                str(x) for x in value.get("paragraph_ids", []) if str(x)
            ],
            source_text_sha1=str(value.get("source_text_sha1", "") or ""),
            source_path=str(value.get("source_path", "") or ""),
            source_sha1=str(value.get("source_sha1", "") or ""),
            extractor=str(value.get("extractor", "") or ""),
        )

    def to_dict(self) -> dict:
        out = {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "start_token": self.start_token,
            "end_token": self.end_token,
        }
        optional = {
            "char_start": self.char_start,
            "char_end": self.char_end,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "bboxes": [b.to_dict() for b in self.bboxes],
            "block_ids": self.block_ids,
            "paragraph_ids": self.paragraph_ids,
            "source_text_sha1": self.source_text_sha1,
            "source_path": self.source_path,
            "source_sha1": self.source_sha1,
            "extractor": self.extractor,
        }
        for key, value in optional.items():
            if value not in (None, "", []):
                out[key] = value
        return out


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


@dataclass
class Fact:
    """A single fact unit extracted from source documents.

    Phase 1 (plan v6) introduces three optional fields to support Semantic
    Fact Units (SFU). They default to empty / 1.0 so V10 and V11 facts that
    don't carry them load unchanged:

    - ``canonical_form``: LLM-rewritten version of ``text`` where pronouns,
      discourse markers, and named-but-undefined terms are resolved so the
      claim is readable in isolation. Must be NLI-entailed by ``text``.
    - ``self_containment_score``: 0..1, set by a small LLM-as-judge that
      checks whether the claim can be understood without prior context.
    - ``raw_text``: original verbatim span from the source PDF, preserved
      separately from ``text`` so audit/visualisation tools can show the
      unmodified source even when ``text`` is set to the canonical form.

    The semantic of ``text`` does NOT change: it remains the canonical field
    used by all matching, embedding, prompt-construction, and NLI code. SFU
    extractors that want canonical forms to flow downstream should write the
    rewrite into ``text`` AND preserve the original in ``raw_text``.
    """

    fact_id: str
    text: str
    role: Role
    supporting_spans: List[Span] = field(default_factory=list)
    weight: float = 1.0
    difficulty: Difficulty = "medium"
    canonical_form: str = ""
    self_containment_score: float = 1.0
    raw_text: str = ""
    # True only when upgrade_fact_to_sfu() has scored this fact; False for all
    # facts produced by extract_candidate_facts() so C3 gate is never applied
    # to unscored facts.
    self_containment_known: bool = False

    def __post_init__(self) -> None:
        if not self.fact_id or not self.fact_id.strip():
            raise ValueError("fact_id cannot be empty or whitespace")
        if len(self.text) < 10:
            raise ValueError(f"Fact text too short: '{self.text[:40]}'")
        if not (0.0 < self.weight <= 2.0):
            raise ValueError(f"weight out of range (0, 2]: {self.weight}")
        if not (0.0 <= self.self_containment_score <= 1.0):
            raise ValueError(
                f"self_containment_score must be in [0, 1]: {self.self_containment_score}"
            )


@dataclass
class MSFS:
    msfs_id: str
    fact_ids: List[str]

    def __post_init__(self) -> None:
        if not self.msfs_id or not self.msfs_id.strip():
            raise ValueError("msfs_id cannot be empty or whitespace")
        if not self.fact_ids:
            raise ValueError("MSFS must have >= 1 fact_id")


@dataclass
class FactChain:
    """A semantically-coherent chain of facts produced by the multi-hop sampler."""

    fact_ids: List[str]
    depth: int = 0
    anchor_id: str = ""
    mean_cosine: float = 0.0
    role_path: List[Role] = field(default_factory=list)
    # v16: TF-SFG typed edges that link consecutive facts in this chain.
    chain_edges: List[dict] = field(default_factory=list)
    # v16: QA-NLI assumption verdicts for the question generated from this chain.
    question_assumptions: List[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.fact_ids:
            raise ValueError("FactChain must have >= 1 fact_id")
        # Compute depth from fact_ids; reject explicit mismatches if caller passed one.
        if self.depth and self.depth != len(self.fact_ids):
            raise ValueError(
                f"depth={self.depth} disagrees with len(fact_ids)={len(self.fact_ids)}"
            )
        self.depth = len(self.fact_ids)
        if not self.anchor_id:
            self.anchor_id = self.fact_ids[0]


# Minimum gold-answer length. Short factoids ("Yes", "42", "ISO 9001") are
# legitimate, so this is intentionally permissive. Override at call sites that
# require longer answers.
MIN_GOLD_ANSWER_LEN = 1


@dataclass
class QuestionGT:
    q_id: str
    question: str
    gold_answer: str
    msfs_list: List[MSFS]
    doc_ids: List[str]
    required_fact_ids: List[str]
    difficulty_reasoning_depth: int
    difficulty_semantic_distance: SemDistance
    required_facts: List[Fact] = field(default_factory=list)
    required_fact_groups: List[List[str]] = field(default_factory=list)
    hop_type: str = ""
    minimum_required_fact_count: int = 0
    chunk_profile_id: str = ""
    fact_to_chunk_mappings: List[dict] = field(default_factory=list)
    # QRSG verdict fields (v13). Empty/None when QRSG was not run.
    relation_type: str = ""
    risky_frame_hit: Optional[str] = None
    qrsg_evidence_map: dict = field(default_factory=dict)
    # v16 PASS-GT fields. All default to empty/None so v11–v15 GT files load unchanged.
    # P1: TF-SFG typed edges between chain facts.
    tf_sfg_edges: List[dict] = field(default_factory=list)
    # P2: QA-NLI decomposed question assumptions.
    question_assumptions: List[dict] = field(default_factory=list)
    # P3: provenance-anchored distractor spans.
    distractor_spans: List[dict] = field(default_factory=list)
    # P4: twin provenance (null for strict rows).
    twin_of: Optional[str] = None
    twin_type: Optional[str] = None   # "abstention" | "counterfactual" | None
    twin_metadata: Optional[dict] = None
    # P5: ARM abstractive reasoning trace.
    reasoning_trace: List[dict] = field(default_factory=list)
    # 3rd difficulty axis (clean / distractor / abstention / counterfactual).
    difficulty_adversarial: str = "clean"
    # DataMorgana-style question intent.
    intent: str = ""

    def __post_init__(self) -> None:
        if not self.msfs_list:
            raise ValueError("QuestionGT must have >= 1 MSFS")
        if len(self.gold_answer.strip()) < MIN_GOLD_ANSWER_LEN:
            raise ValueError("gold_answer is empty")
        if not self.minimum_required_fact_count:
            self.minimum_required_fact_count = len(self.required_fact_ids)
        if not self.required_fact_groups and self.required_fact_ids:
            self.required_fact_groups = [list(self.required_fact_ids)]


@dataclass
class RetrievalLog:
    q_id: str
    retrieved_chunk_ids: List[str]

    def __post_init__(self) -> None:
        if not self.q_id or not self.q_id.strip():
            raise ValueError("q_id cannot be empty")


@dataclass
class AnswerLog:
    q_id: str
    predicted_answer: str
    abstained: bool = False

    def __post_init__(self) -> None:
        if not self.q_id or not self.q_id.strip():
            raise ValueError("q_id cannot be empty")
