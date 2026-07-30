"""Fact-level vector store: embeddings, FAISS index, clustering."""

from rag_gt.vectorstore.clustering import cluster_facts
from rag_gt.vectorstore.embedding import embed_facts, embed_query, embed_texts
from rag_gt.vectorstore.faiss_index import FactIndex

__all__ = [
    "FactIndex",
    "cluster_facts",
    "embed_facts",
    "embed_query",
    "embed_texts",
]
