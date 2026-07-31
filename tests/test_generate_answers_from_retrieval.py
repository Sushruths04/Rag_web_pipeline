from rag_gt.cli.generate_answers_from_retrieval import _facts_from_contexts
from rag_gt.comparison.chunk_resolver import ChunkResolver


def test_facts_from_contexts_uses_cached_chunk_text(tmp_path):
    cache = tmp_path / "chunks.jsonl"
    cache.write_text(
        '{"chunk_id":"doc_c000001","doc_id":"doc","text":"This is a sufficiently long chunk of context text.","char_start":0,"char_end":50}\n'
        '{"chunk_id":"doc_c000002","doc_id":"doc","text":"short","char_start":51,"char_end":56}\n',
        encoding="utf-8",
    )
    resolver = ChunkResolver.from_cache(cache)

    facts = _facts_from_contexts(
        "doc_q001",
        ["doc_c000001", "doc_c000002", "missing_chunk"],
        resolver,
        max_context_chunks=5,
        max_chars_per_chunk=20,
    )

    assert len(facts) == 1
    assert facts[0].fact_id == "doc_q001_ctx001"
    assert facts[0].role == "example"
    assert facts[0].text == "This is a sufficient"
