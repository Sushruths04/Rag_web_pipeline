import numpy as np

from tests.helpers import cleanup_workspace_tmp, make_workspace_tmp
from rag_gt.vectorstore.clustering import cluster_facts
from rag_gt.vectorstore.faiss_index import FactIndex


def _norm(v):
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def test_roundtrip_search():
    embs = _norm(np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]], dtype=np.float32))
    idx = FactIndex(dim=3, backend="sklearn")
    idx.add(embs, ["a", "b", "c"])
    results = idx.search(embs[0], k=2)
    assert results[0][0] == "a"
    assert results[1][0] == "b"


def test_save_load():
    embs = _norm(np.eye(4, dtype=np.float32))
    idx = FactIndex(dim=4, backend="sklearn")
    idx.add(embs, ["a", "b", "c", "d"])
    tmpdir = make_workspace_tmp("vectorstore")
    try:
        p = tmpdir / "idx"
        idx.save(p)
        loaded = FactIndex.load(p)
        assert loaded.fact_ids == ["a", "b", "c", "d"]
        res = loaded.search(embs[2], k=1)
        assert res[0][0] == "c"
    finally:
        cleanup_workspace_tmp(tmpdir)


def test_cluster_collapses_duplicates():
    embs = _norm(
        np.array(
            [
                [1.0, 0, 0],
                [1.0, 0, 0],
                [0, 1.0, 0],
            ],
            dtype=np.float32,
        )
    )
    clusters = cluster_facts(embs, ["f1", "f2", "f3"], cosine_threshold=0.95)
    groups = {frozenset(v) for v in clusters.values()}
    assert frozenset({"f1", "f2"}) in groups
    assert frozenset({"f3"}) in groups


def test_cluster_single_fact():
    embs = np.array([[1.0, 0, 0]], dtype=np.float32)
    clusters = cluster_facts(embs, ["only"], cosine_threshold=0.8)
    assert clusters == {0: ["only"]}


def test_cluster_empty():
    clusters = cluster_facts(np.zeros((0, 3), dtype=np.float32), [])
    assert clusters == {}
