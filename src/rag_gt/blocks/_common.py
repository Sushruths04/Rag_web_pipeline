"""Shared helpers for ``rag_gt.blocks.*`` adapters: artifact I/O + the
``{"type", "ref", "meta"}`` convention from ``studio/backend/stubs.py``.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

_DEFAULT_ARTIFACTS_DIR = Path(tempfile.gettempdir()) / "rag_gt_blocks_artifacts"


def default_artifacts_dir() -> Path:
    """Process-wide fallback output directory, used when a block is called
    without an explicit ``artifacts_dir`` (e.g. directly from a script)."""
    _DEFAULT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_ARTIFACTS_DIR


def resolve_artifacts_dir(artifacts_dir: Path | str | None) -> Path:
    out_dir = Path(artifacts_dir) if artifacts_dir else default_artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def artifact(type_id: str, ref: str, meta: dict) -> dict:
    return {"type": type_id, "ref": ref, "meta": meta}


def write_json_artifact(
    artifacts_dir: Path | str | None,
    block_type: str,
    payload: Any,
) -> Path:
    """Write ``payload`` as a content-addressed JSON file under
    ``artifacts_dir`` and return its path. Content-addressing (sha256 of the
    serialized payload) means re-running a block with unchanged output
    reuses the same artifact file."""
    out_dir = resolve_artifacts_dir(artifacts_dir)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    path = out_dir / f"{block_type}_{digest}.json"
    if not path.exists():
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
        )
    return path


def read_json_artifact(ref: str) -> Any:
    return json.loads(Path(ref).read_text(encoding="utf-8"))


def write_text_artifact(
    artifacts_dir: Path | str | None,
    block_type: str,
    text: str,
    extension: str,
) -> Path:
    """Write a non-JSON (md/html) artifact, content-addressed like
    write_json_artifact, and return its path."""
    out_dir = resolve_artifacts_dir(artifacts_dir)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    path = out_dir / f"{block_type}_{digest}.{extension}"
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    return path


def read_list_input(artifact_dict: dict | None) -> list:
    """Read the JSON list payload referenced by an input artifact's ``ref``.
    Returns ``[]`` for a missing/optional input."""
    if not artifact_dict:
        return []
    payload = read_json_artifact(artifact_dict["ref"])
    if isinstance(payload, list):
        return payload
    return payload.get("items", []) if isinstance(payload, dict) else []
