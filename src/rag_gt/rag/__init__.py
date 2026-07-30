"""No-API RAG evaluation pipeline.

Consumes the GT dataset (allpdf_phase2_pairs.jsonl) and evaluates retrieval
strategies by checking whether retrieved chunks contain the GT facts using
>=60% token overlap.  Zero LLM API calls.
"""
