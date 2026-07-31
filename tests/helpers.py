"""Test helpers that keep temporary files inside the writable workspace."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path


def make_workspace_tmp(name: str) -> Path:
    root = Path.cwd() / "datacache" / "pytest_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_workspace_tmp(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
