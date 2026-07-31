"""Stage A acceptance — cross-page bridges exist and are cleanly extracted ($0)."""
import json
from pathlib import Path

from rag_gt.graph.bridge_index import build_bridge_index, extract_bridge_keys

V11_FACTS = Path(
    "D:/Mini Thesis/RAG_GT/data/test_corpus_allpdf/pipeline_run_v11/checkpoints/s4_facts.json"
)


def _mini_facts():
    # facts always carry a doc id (real schemas have `doc` or a doc-prefixed id);
    # bridges are formed WITHIN a doc.
    return [
        {"doc": "d", "fact_id": "A", "canonical_form": "The shielding gas designation is given in accordance with ISO 14175.", "page_start": 13},
        {"doc": "d", "fact_id": "B", "canonical_form": "The shielding gas flow rate and nozzle diameter shall be specified.", "page_start": 14},
        {"doc": "d", "fact_id": "C", "canonical_form": "The shielding gas composition must be recorded.", "page_start": 13},  # same page as A
        {"doc": "d", "fact_id": "D", "canonical_form": "The weld zone temperature shall be maintained.", "page_start": 13},
    ]


def test_cross_page_bridge_detected():
    idx = build_bridge_index(_mini_facts())
    norms = {g["norm"] for g in idx["bridge_groups"]}
    assert "shielding ga" in norms or "shielding gas" in norms, idx["bridge_groups"]
    # the shielding-gas group must span pages 13 and 14 (A/B), not just same-page A/C
    sg = next(g for g in idx["bridge_groups"] if g["norm"].startswith("shielding ga"))
    assert set(sg["pages"]) >= {13, 14}


def test_same_page_only_is_not_a_group():
    # 'weld zone' appears once -> no group; no spurious single-page bridge
    idx = build_bridge_index(_mini_facts())
    assert all(len(g["pages"]) >= 2 for g in idx["bridge_groups"])


def test_bridges_never_span_documents():
    # same phrase, two DIFFERENT docs -> must NOT form a bridge
    facts = [
        {"doc": "doc1", "id": "doc1_F1", "text": "The shielding gas designation per ISO 14175.", "page": 13},
        {"doc": "doc2", "id": "doc2_F1", "text": "The shielding gas flow rate is specified.", "page": 14},
    ]
    idx = build_bridge_index(facts)
    assert idx["stats"]["n_cross_page_groups"] == 0, idx["bridge_groups"]


def test_doc_inferred_from_prefixed_fact_id():
    # s4_facts shape: no `doc` field, doc inferred from the fact_id prefix
    facts = [
        {"fact_id": "din_iso_15609_welding_procedure_F029001",
         "canonical_form": "The welding procedure specification defines the content.", "page_start": 10},
        {"fact_id": "din_iso_15609_welding_procedure_F036001",
         "canonical_form": "The welding procedure specification provides weld information.", "page_start": 11},
    ]
    idx = build_bridge_index(facts)
    assert idx["stats"]["n_cross_page_groups"] >= 1


def test_pure_number_fragments_rejected():
    keys = extract_bridge_keys("DIN EN ISO 15609-1:2019-12 EN ISO 15609-1:2019 (E)")
    terms = [k["norm"] for k in keys if k["type"] == "TERM"]
    assert not any(t.replace(" ", "").isdigit() for t in terms), terms
    # the standard ref itself is still captured
    assert any(k["type"] == "STANDARD_REF" for k in keys)


def test_v11_corpus_has_cross_page_bridges():
    if not V11_FACTS.exists():
        return  # skip if checkpoint not present
    facts = json.loads(V11_FACTS.read_text(encoding="utf-8"))
    idx = build_bridge_index(facts)
    assert idx["stats"]["n_cross_page_groups"] >= 1
    surfaces = {g["surface"].lower() for g in idx["bridge_groups"]}
    assert any("welding procedure" in s for s in surfaces)
    assert any("shielding gas" in s for s in surfaces)
