"""Tests for the block SDK adapters wrapping the M4-refactored draft_*/gate_*
functions (05_BLOCK_CATALOG.md secs. 3 "Generation" #15-17, sec. 4). All PAID
-- every test injects a FakeLLM via params["llm"], exactly like the existing
answer_first_v2 test suite; the real get_llm() default path is never hit.
"""
import json

from rag_gt.blocks import qa_gen_bridges, qa_gen_clusters, qa_gen_pairs
from tests.test_answer_first_v2 import (
    BRIDGE_PAIRS,
    CLUSTER,
    FACTS,
    FakeLLM,
    _accepting_nli,
    _passing_necessity,
)


def _write_json(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _artifact(type_, ref):
    return {"type": type_, "ref": str(ref)}


def test_qa_gen_pairs_produces_qa_artifact_matching_gate_neighbor_pairs_shape(tmp_path):
    facts_ref = _write_json(tmp_path, "facts.json", FACTS)
    pair = {"fact_a": "A", "fact_b": "A2"}
    candidates_ref = _write_json(tmp_path, "candidates.json", [pair])

    out = qa_gen_pairs.run(
        inputs={"facts": _artifact("facts", facts_ref), "candidates": _artifact("candidates", candidates_ref)},
        params={
            "doc": "din_iso_15609_welding_procedure_full",
            "llm": FakeLLM(),
            "nli_fn": _accepting_nli,
            "necessity_fn": _passing_necessity,
            "workers": 1,
        },
        artifacts_dir=tmp_path,
    )
    assert out["qa"]["type"] == "qa"
    qa_records = json.loads(open(out["qa"]["ref"], encoding="utf-8").read())
    assert len(qa_records) == 1
    assert qa_records[0]["evidence_strategy"] == "neighbor_pair_v2"
    assert qa_records[0]["gold_fact_ids"] == ["A", "A2"]
    assert out["qa"]["meta"]["n_qa"] == 1


def test_qa_gen_clusters_produces_qa_artifact_matching_gate_clusters_shape(tmp_path):
    facts_ref = _write_json(tmp_path, "facts.json", FACTS)
    candidates_ref = _write_json(tmp_path, "candidates.json", [CLUSTER])

    out = qa_gen_clusters.run(
        inputs={"facts": _artifact("facts", facts_ref), "candidates": _artifact("candidates", candidates_ref)},
        params={
            "doc": "din_iso_15609_welding_procedure_full",
            "llm": FakeLLM(),
            "nli_fn": _accepting_nli,
            "necessity_fn": _passing_necessity,
            "workers": 1,
        },
        artifacts_dir=tmp_path,
    )
    assert out["qa"]["type"] == "qa"
    qa_records = json.loads(open(out["qa"]["ref"], encoding="utf-8").read())
    assert len(qa_records) == 1
    assert qa_records[0]["evidence_strategy"] == "cluster_2plus2"
    assert qa_records[0]["gold_fact_ids"] == ["A", "A2", "B", "B2"]
    assert out["qa"]["meta"]["n_qa"] == 1


def test_qa_gen_bridges_produces_qa_artifact_via_build_answer_first_pairs(tmp_path):
    facts_ref = _write_json(tmp_path, "facts.json", FACTS)
    bridges_ref = _write_json(tmp_path, "bridges.json", BRIDGE_PAIRS)

    out = qa_gen_bridges.run(
        inputs={"facts": _artifact("facts", facts_ref), "bridges": _artifact("bridges", bridges_ref)},
        params={
            "llm": FakeLLM(),
            "nli_fn": _accepting_nli,
            "necessity_fn": _passing_necessity,
            "workers": 1,
        },
        artifacts_dir=tmp_path,
    )
    assert out["qa"]["type"] == "qa"
    qa_records = json.loads(open(out["qa"]["ref"], encoding="utf-8").read())
    assert len(qa_records) == 1
    assert qa_records[0]["gold_fact_ids"] == ["A", "B"]


def test_qa_gen_pairs_never_calls_real_get_llm_when_llm_param_given(tmp_path, monkeypatch):
    """PAID blocks must not silently reach for a real LLM in a test context;
    proves the adapter honors an injected llm rather than always calling
    get_llm()."""
    from rag_gt.blocks import qa_gen_pairs as mod

    def _boom(role="gt"):
        raise AssertionError("get_llm() must not be called when params['llm'] is provided")

    monkeypatch.setattr(mod, "get_llm", _boom)

    facts_ref = _write_json(tmp_path, "facts.json", FACTS)
    pair = {"fact_a": "A", "fact_b": "A2"}
    candidates_ref = _write_json(tmp_path, "candidates.json", [pair])
    qa_gen_pairs.run(
        inputs={"facts": _artifact("facts", facts_ref), "candidates": _artifact("candidates", candidates_ref)},
        params={
            "doc": "din_iso_15609_welding_procedure_full",
            "llm": FakeLLM(),
            "nli_fn": _accepting_nli,
            "necessity_fn": _passing_necessity,
            "workers": 1,
        },
        artifacts_dir=tmp_path,
    )


def test_qa_gen_pairs_default_artifacts_dir_is_still_callable_standalone(tmp_path):
    """Matches the FREE-spine blocks' contract (see rag_gt.blocks.facts_import):
    omitting artifacts_dir must not error -- it falls back to the process
    -wide default temp directory, so a block stays callable on its own."""
    facts_ref = _write_json(tmp_path, "facts.json", FACTS)
    pair = {"fact_a": "A", "fact_b": "A2"}
    candidates_ref = _write_json(tmp_path, "candidates.json", [pair])

    out = qa_gen_pairs.run(
        inputs={"facts": _artifact("facts", facts_ref), "candidates": _artifact("candidates", candidates_ref)},
        params={
            "doc": "din_iso_15609_welding_procedure_full",
            "llm": FakeLLM(),
            "nli_fn": _accepting_nli,
            "necessity_fn": _passing_necessity,
            "workers": 1,
        },
    )
    assert out["qa"]["type"] == "qa"
    assert json.loads(open(out["qa"]["ref"], encoding="utf-8").read())
