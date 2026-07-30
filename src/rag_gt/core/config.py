"""Configuration loader. Resolves `configs/config.yaml` relative to repo root."""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


def load_env() -> None:
    load_dotenv()


def _find_config() -> Path:
    """Search plausible locations, deterministic order.

    `configs/config.yaml` is the canonical location. The repo-root `config.yaml`
    is treated as a legacy mirror; if both exist we still prefer `configs/`.
    """
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "configs" / "config.yaml",
        here.parents[3] / "configs" / "config.yaml",
        Path.cwd() / "config.yaml",
        here.parents[3] / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "config.yaml not found in ./configs/, ./, or repo root"
    )


@lru_cache(maxsize=4)
def _load_config_cached(path_str: str, mtime_ns: int) -> dict:
    with open(path_str, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Optional[str] = None) -> dict:
    """Return a fresh deep copy of the parsed config.

    Cache is keyed by (resolved path, file mtime), so editing the YAML in
    a long-running process is picked up on the next call. The returned
    dict is a deep copy, so caller mutations cannot leak across callers.
    """
    path = Path(config_path) if config_path else _find_config()
    mtime_ns = path.stat().st_mtime_ns
    cached = _load_config_cached(str(path), mtime_ns)
    cfg = copy.deepcopy(cached)
    _apply_env_overrides(cfg)
    return cfg


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _apply_env_overrides(cfg: dict) -> None:
    """Apply narrow runtime overrides used for A/B ingestion experiments."""
    ingestion = cfg.setdefault("ingestion", {})
    qrsg = cfg.setdefault("v13_qrsg", {})

    if os.getenv("RAG_GT_PDF_BACKEND"):
        ingestion["pdf_backend"] = os.getenv("RAG_GT_PDF_BACKEND")
    if os.getenv("RAG_GT_DOCLING_EXPORT_FORMAT"):
        ingestion["docling_export_format"] = os.getenv(
            "RAG_GT_DOCLING_EXPORT_FORMAT"
        )
    if os.getenv("RAG_GT_DOCLING_DO_OCR"):
        ingestion["docling_do_ocr"] = _env_bool("RAG_GT_DOCLING_DO_OCR")
    if os.getenv("RAG_GT_DOCLING_DO_TABLE_STRUCTURE"):
        ingestion["docling_do_table_structure"] = _env_bool(
            "RAG_GT_DOCLING_DO_TABLE_STRUCTURE"
        )
    if os.getenv("RAG_GT_DOCLING_BATCH_SIZE"):
        ingestion["docling_batch_size"] = int(
            os.getenv("RAG_GT_DOCLING_BATCH_SIZE", "1")
        )
    if os.getenv("RAG_GT_DOCLING_PAGE_RANGE_SIZE"):
        ingestion["docling_page_range_size"] = int(
            os.getenv("RAG_GT_DOCLING_PAGE_RANGE_SIZE", "20")
        )
    if os.getenv("RAG_GT_DOCLING_MIN_TEXT_CHARS"):
        ingestion["docling_min_text_chars"] = int(
            os.getenv("RAG_GT_DOCLING_MIN_TEXT_CHARS", "1000")
        )
    if os.getenv("RAG_GT_DOCLING_MIN_TEXT_FILE_RATIO"):
        ingestion["docling_min_text_file_ratio"] = float(
            os.getenv("RAG_GT_DOCLING_MIN_TEXT_FILE_RATIO", "0.05")
        )
    if os.getenv("RAG_GT_QRSG_ENABLED") is not None:
        qrsg["enabled"] = _env_bool("RAG_GT_QRSG_ENABLED")
    if os.getenv("RAG_GT_QRSG_GATE_VERSION"):
        qrsg["gate_version"] = os.getenv("RAG_GT_QRSG_GATE_VERSION")
    if os.getenv("RAG_GT_QRSG_LLM_ROLE"):
        qrsg["llm_role"] = os.getenv("RAG_GT_QRSG_LLM_ROLE")
    if os.getenv("RAG_GT_QRSG_MAX_TOKENS"):
        qrsg["max_tokens"] = int(os.getenv("RAG_GT_QRSG_MAX_TOKENS", "2048"))
    if os.getenv("RAG_GT_QRSG_CACHE_PATH"):
        qrsg["cache_path"] = os.getenv("RAG_GT_QRSG_CACHE_PATH")


def repo_root() -> Path:
    """Return the repo root (parent of src/)."""
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """Return the data directory path (does not create it)."""
    return Path(os.getenv("RAG_GT_DATA_DIR", repo_root() / "data"))


def ensure_data_dir() -> Path:
    """Return the data directory and create it if missing."""
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root
