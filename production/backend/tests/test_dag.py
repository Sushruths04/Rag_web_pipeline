import pytest

from app.orchestrator.dag import PipelineDAG, StageDef
from app.stages.topology import GRAFT_TOPOLOGY


def _dag(*defs):
    return PipelineDAG([StageDef(name=n, label=n, deps=d) for n, d in defs])


def test_topo_order_respects_deps():
    dag = _dag(("a", ()), ("b", ("a",)), ("c", ("a",)), ("d", ("b", "c")))
    order = dag.topo_order()
    assert order.index("a") < order.index("b") < order.index("d")
    assert order.index("a") < order.index("c") < order.index("d")


def test_downstream_transitive():
    dag = _dag(("a", ()), ("b", ("a",)), ("c", ("b",)), ("x", ()))
    assert dag.downstream("a") == {"b", "c"}
    assert dag.downstream("x") == set()


def test_cycle_rejected():
    with pytest.raises(ValueError, match="cycle"):
        _dag(("a", ("b",)), ("b", ("a",)))


def test_unknown_dep_rejected():
    with pytest.raises(ValueError, match="unknown"):
        _dag(("a", ("ghost",)))


def test_duplicate_name_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        _dag(("a", ()), ("a", ()))


def test_graft_topology_is_valid_and_complete():
    dag = PipelineDAG(GRAFT_TOPOLOGY)
    order = dag.topo_order()
    assert order[0] == "profile"
    assert set(order) == {
        "ingest", "profile", "chunk", "extract", "clean", "graph", "sample",
        "qagen", "verify", "gt_dataset", "bm25_index", "vector_index",
        "fusion", "rerank_warmup", "sweep", "leaderboard", "select_config",
        "final_eval", "report",
    }
    # eval track needs BOTH the GT and the RAG indexes
    assert set(dag.stage("sweep").deps) == {"gt_dataset", "fusion", "rerank_warmup"}
    assert dag.stage("report").deps == ("final_eval",)
    for s in GRAFT_TOPOLOGY:
        assert s.track in ("gt", "rag", "eval")
