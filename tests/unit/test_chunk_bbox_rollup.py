"""Every chunking strategy must surface chunk-level bboxes, not just table_aware.

Found 2026-08-01 by inspecting a live 5-document run. Stage 3's _make_span()
builds a fact's provenance from ``chunk["bboxes"]``. The table_aware /
ocr_block path goes through _pack_units(), which rolls its source units'
bboxes up into that key. Every other strategy (clause / heading / recursive)
goes through chunk_document(), which emits no "bboxes" key at all -- so
_make_span() reads [] and every fact from those documents lands with empty
bbox provenance.

Measured on the rerun: din_iso_13919_1 (table_aware) had bboxes on 7/7
facts; din_iso_3452_1, din_iso_3834_1, din_iso_4136 and din_iso_6507_1 (all
clause) had 0 on 533 facts between them. The data was never lost -- all 1231
source_units carried bboxes -- it just never reached chunk level. bbox
observability is a hard requirement for this project, so this is a real
provenance gap, not cosmetics.
"""

from __future__ import annotations

from rag_gt.allpdf.chunk import agentic_chunk
from rag_gt.allpdf.ingest import IngestResult
from rag_gt.allpdf.preflight import BACKEND_LEGACY, DocProfile
from rag_gt.core.types import SourceBBox, SourceUnit


def _profile(doc_type: str, table_density: float = 0.0) -> DocProfile:
    return DocProfile(
        doc_id="d",
        path="d.pdf",
        source_sha256="abc",
        page_count=3,
        sampled_pages=3,
        digital_text_ratio=1.0,
        scanned=False,
        mixed=False,
        avg_chars_per_page=1500.0,
        figure_density=0.0,
        table_density=table_density,
        multi_column=False,
        reading_order_risk="low",
        doc_type_guess=doc_type,
        recommended_backend=BACKEND_LEGACY,
    )


def _units(n: int = 6) -> list[SourceUnit]:
    out, cursor = [], 0
    for i in range(n):
        text = (
            f"{i + 1}.1 Clause heading number {i}. "
            "This clause states a requirement that is long enough to survive "
            "sentence splitting and chunk packing without being discarded. "
        )
        out.append(
            SourceUnit(
                doc_id="d",
                char_start=cursor,
                char_end=cursor + len(text),
                text=text,
                page_no=(i // 2) + 1,
                block_id=f"b{i}",
                paragraph_id=f"p{i}",
                bboxes=[
                    SourceBBox(
                        page_no=(i // 2) + 1,
                        l=10.0 + i, t=20.0 + i, r=300.0, b=40.0 + i,
                        coord_origin="TOPLEFT",
                    )
                ],
                extractor="pymupdf",
            )
        )
        cursor += len(text)
    return out


def _ingest(units) -> IngestResult:
    text = "".join(u.text for u in units)
    return IngestResult(
        doc_id="d",
        backend_used=BACKEND_LEGACY,
        text=text,
        units=units,
        char_count=len(text),
        n_units=len(units),
        pages_covered=len({u.page_no for u in units}),
        front_matter_units=0,
        back_matter_units=0,
    )


def _chunks_with_bboxes(chunks) -> int:
    return sum(1 for c in chunks if c.get("bboxes"))


class TestBboxesReachChunkLevel:
    def test_table_aware_rolls_bboxes_up(self):
        """The strategy that already worked -- guards against regressing it."""
        units = _units()
        res = agentic_chunk(_profile("REPORT", table_density=0.9), _ingest(units))
        assert res.strategy == "table_aware"
        assert _chunks_with_bboxes(res.chunks) == len(res.chunks)

    def test_clause_strategy_also_rolls_bboxes_up(self):
        """ISO standards chunk via `clause`; facts from them had no bboxes."""
        units = _units()
        res = agentic_chunk(_profile("ISO_STANDARD"), _ingest(units))
        assert res.strategy == "clause"
        assert res.chunks, "expected at least one chunk"
        assert _chunks_with_bboxes(res.chunks) == len(res.chunks), (
            "clause chunks carry no chunk-level 'bboxes', so Stage 3's "
            "_make_span() reads [] and every fact loses bbox provenance"
        )

    def test_rolled_up_bboxes_are_well_formed(self):
        res = agentic_chunk(_profile("ISO_STANDARD"), _ingest(_units()))
        for c in res.chunks:
            for bb in c["bboxes"]:
                assert set(("page_no", "l", "t", "r", "b")) <= set(bb)
                assert isinstance(bb["page_no"], int)

    def test_bbox_pages_stay_within_the_chunk_page_span(self):
        res = agentic_chunk(_profile("ISO_STANDARD"), _ingest(_units()))
        for c in res.chunks:
            pages = {bb["page_no"] for bb in c["bboxes"]}
            if c.get("page_start") is not None and pages:
                assert min(pages) >= c["page_start"]
                assert max(pages) <= c["page_end"]
