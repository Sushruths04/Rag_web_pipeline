"""TDD: rag_gt.blocks.facts_import and rag_gt.blocks.bridges_import wrap
rag_gt.generation.answer_first._load_records verbatim (05_BLOCK_CATALOG.md
§3.3/§3.4). Uses the checked-in grounded facts file for din_iso_6507_vickers
(118 real facts) and the checked-in cleaned bridge-pairs file, filtered to
the same document (172 real pairs) -- exactly the counts used as the wire
badge examples in the catalog.
"""
from __future__ import annotations

import json
from pathlib import Path

from rag_gt.generation.answer_first import _load_records

REPO_ROOT = Path(__file__).resolve().parents[1]
FACTS_PATH = REPO_ROOT / "data" / "eval_results" / "facts_v1_grounded" / "facts_din_iso_6507_vickers_full.json"
BRIDGE_PAIRS_PATH = REPO_ROOT / "data" / "eval_results" / "allpdf_bridge_pairs_clean.json"


def _din_iso_6507_bridge_pairs_file(tmp_path) -> Path:
    payload = json.loads(BRIDGE_PAIRS_PATH.read_text(encoding="utf-8"))
    sub = [p for p in payload["pairs"] if p.get("doc") == "din_iso_6507_vickers_full"]
    out = tmp_path / "bridge_pairs_din_iso_6507.json"
    out.write_text(json.dumps({"pairs": sub}), encoding="utf-8")
    return out


# --- facts_import -----------------------------------------------------------


def test_facts_import_returns_facts_artifact(tmp_path):
    from rag_gt.blocks.facts_import import run

    out = run({}, {"path": str(FACTS_PATH)}, artifacts_dir=tmp_path)

    assert set(out.keys()) == {"facts"}
    artifact = out["facts"]
    assert artifact["type"] == "facts"
    assert artifact["meta"]["count"] == 118
    assert artifact["meta"]["grounded"] is True


def test_facts_import_writes_data_matching_load_records(tmp_path):
    from rag_gt.blocks.facts_import import run

    expected = _load_records(FACTS_PATH)
    out = run({}, {"path": str(FACTS_PATH)}, artifacts_dir=tmp_path)

    written = json.loads(open(out["facts"]["ref"], encoding="utf-8").read())
    assert written == expected


# --- bridges_import ----------------------------------------------------------


def test_bridges_import_returns_bridges_artifact(tmp_path):
    from rag_gt.blocks.bridges_import import run

    bridges_path = _din_iso_6507_bridge_pairs_file(tmp_path)
    out = run({}, {"path": str(bridges_path)}, artifacts_dir=tmp_path)

    assert set(out.keys()) == {"bridges"}
    artifact = out["bridges"]
    assert artifact["type"] == "bridges"
    assert artifact["meta"]["count"] == 172


def test_bridges_import_writes_data_matching_load_records(tmp_path):
    from rag_gt.blocks.bridges_import import run

    bridges_path = _din_iso_6507_bridge_pairs_file(tmp_path)
    expected = _load_records(bridges_path, "pairs")
    out = run({}, {"path": str(bridges_path)}, artifacts_dir=tmp_path)

    written = json.loads(open(out["bridges"]["ref"], encoding="utf-8").read())
    assert written == expected
