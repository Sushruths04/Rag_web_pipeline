"""Docling tables must be serialized, not silently dropped.

Found 2026-08-01 while tracing why DIN EN ISO 13919-1 produced no facts from
its imperfection tables -- the tables that ARE the standard.

_docling_units_to_text() reads only ``item.text``:

    raw = getattr(item, "text", None)
    cleaned = clean_text(str(raw or ""))
    if not cleaned:
        continue

For a Docling ``TableItem``, ``.text`` is ``None``. So every table hit the
``continue`` and vanished -- on a backend (``docling_table``) whose entire
reason for existing is table structure, with ``do_table_structure=True``
already paying for the extraction. Confirmed against the real PDF: pages
19-20 yielded ``{'SectionHeaderItem': 4, 'PictureItem': 2, 'TableItem': 2}``
and zero TextItems, so those pages emitted no units at all.

That also explains why the page-repair added earlier fired on those pages:
Docling produced nothing, so PyMuPDF refilled them as a flat stream of cells
with all row/column association destroyed ("D C B" headers in one chunk, the
values in the next). Fixing the drop at source means the repair no longer has
to paper over it, and Stage 3 sees a real table.

Tables are now serialized with ``export_to_markdown()``, which preserves the
header row and cell alignment that make a row templatable into a
self-contained fact.
"""

from __future__ import annotations

from rag_gt.ingestion.docling_pdf import _docling_units_to_text

MARKDOWN_TABLE = (
    "|   No. | ISO 6520-1 reference | Imperfection designation |\n"
    "|-------|----------------------|--------------------------|\n"
    "|   3.1 |                  507 | Linear misalignment      |"
)


class _BBox:
    l, t, r, b = 1.0, 2.0, 3.0, 4.0
    coord_origin = type("O", (), {"value": "BOTTOMLEFT"})()


class _Prov:
    def __init__(self, page_no=19):
        self.page_no = page_no
        self.bbox = _BBox()


class _TableItem:
    """Mirrors Docling: .text is None, structure only via export_to_markdown."""

    text = None

    def __init__(self, md=MARKDOWN_TABLE, page_no=19):
        self._md = md
        self.prov = [_Prov(page_no)]
        self.self_ref = "#/tables/0"

    def export_to_markdown(self, doc=None):
        return self._md


class _TextItem:
    def __init__(self, text, page_no=1):
        self.text = text
        self.prov = [_Prov(page_no)]
        self.self_ref = "#/texts/0"


class _Doc:
    def __init__(self, items):
        self._items = items

    def iterate_items(self):
        for it in self._items:
            yield it, 0


def _run(items):
    return _docling_units_to_text(
        document=_Doc(items), doc_id="d", source_path="d.pdf",
        source_sha1="sha", cursor=0,
    )


class TestTableItemsAreSerialized:
    def test_table_item_is_not_dropped(self):
        text, units = _run([_TableItem()])
        assert units, "TableItem was silently dropped (.text is None)"
        assert "ISO 6520-1 reference" in text
        assert "Linear misalignment" in text

    def test_table_keeps_markdown_row_structure(self):
        """Row/column association is what makes a row templatable."""
        text, _ = _run([_TableItem()])
        assert "|" in text, "pipes lost; row/column association destroyed"
        header_line = [ln for ln in text.splitlines() if "ISO 6520-1" in ln][0]
        assert "No." in header_line and "Imperfection designation" in header_line, (
            "header cells must stay on one line so a row can be read against them"
        )

    def test_table_unit_keeps_page_and_bbox_provenance(self):
        _, units = _run([_TableItem(page_no=19)])
        u = units[0]
        assert u.page_no == 19
        assert u.bboxes and u.bboxes[0].page_no == 19
        assert u.extractor == "docling"

    def test_offsets_stay_consistent_with_mixed_text_and_tables(self):
        text, units = _run([
            _TextItem("Clause 3 introduces the limits."),
            _TableItem(),
            _TextItem("The limits relate to deviations.", page_no=20),
        ])
        assert len(units) == 3
        for u in units:
            assert text[u.char_start:u.char_end] == u.text

    def test_a_table_that_cannot_be_serialized_is_skipped_not_fatal(self):
        class _Broken(_TableItem):
            def export_to_markdown(self, doc=None):
                raise RuntimeError("docling internal")

        text, units = _run([_TextItem("Kept."), _Broken()])
        assert len(units) == 1, "a broken table must not take the page down"
        assert "Kept." in text

    def test_plain_text_items_are_unaffected(self):
        text, units = _run([_TextItem("Ordinary prose sentence.")])
        assert len(units) == 1
        assert text == "Ordinary prose sentence."
