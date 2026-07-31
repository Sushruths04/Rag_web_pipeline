import json

from rag_gt.integration.user_chunks import make_user_chunk_row, write_user_chunks_jsonl


def test_make_user_chunk_row_normalizes_external_chunk():
    row = make_user_chunk_row(
        {
            "id": "external-1",
            "text": "Alpha beta gamma.",
            "source_char_start": 10,
            "source_char_end": 27,
            "page_start": 2,
            "chunking_strategy": "semantic",
        },
        doc_id="doc",
        source_path="source.pdf",
        source_sha1="abc",
    )

    assert row["chunk_id"] == "external-1"
    assert row["doc_id"] == "doc"
    assert row["char_start"] == 10
    assert row["char_end"] == 27
    assert row["page_start"] == 2
    assert row["page_end"] == 2


def test_write_user_chunks_jsonl(tmp_path):
    out = tmp_path / "user_chunks.jsonl"

    count = write_user_chunks_jsonl(
        [{"text": "Alpha beta gamma.", "char_start": 0, "char_end": 17}],
        doc_id="doc",
        output_path=out,
    )

    assert count == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["chunk_id"] == "doc_user_c000000"
    assert rows[0]["doc_id"] == "doc"
