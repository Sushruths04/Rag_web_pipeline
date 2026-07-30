"""Lifecycle manager for local PyTorch models (NLI + embedding).

LLMs run out-of-process (remote API) and are not managed here.

This module exposes the resolved NLI label indices via:

    NLI_LABEL_INDEX: dict[str, int]   # 'entailment' | 'neutral' | 'contradiction' -> int

Always look up indices by name (`NLI_LABEL_INDEX["entailment"]`) — never hard-code
slot 0/1/2. Different cross-encoder NLI checkpoints use different orderings.
"""

from __future__ import annotations

import gc
import os
import threading
import time
from typing import Dict, Optional

import requests
import torch
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

NLI_MODEL_NAME = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# If ACTIVE_EMBEDDING is set, embeddings are served by the API endpoint (so a
# large server-side embedder like Qwen3-Embedding-8B never loads locally).
ACTIVE_EMBEDDING = os.getenv("ACTIVE_EMBEDDING", "").strip()

# Resolved at NLI load time. Always look up via NLI_LABEL_INDEX[name].
NLI_LABEL_INDEX: Dict[str, int] = {}


class _APIEmbedder:
    """OpenAI-compatible ``/embeddings`` adapter — a drop-in for SentenceTransformer.

    Exposes only the subset the pipeline uses (``encode`` +
    ``get_sentence_embedding_dimension``) so a large server-side embedder
    (e.g. Qwen3-Embedding-8B) runs via the API with no local model load.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.base_url = os.getenv("API_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("API_KEY", "").strip().strip('"').strip()
        if not self.base_url or not self.api_key:
            raise ValueError(
                "ACTIVE_EMBEDDING is set but API_BASE_URL / API_KEY are missing in .env"
            )
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._session = requests.Session()
        self._dim: Optional[int] = None

    def get_sentence_embedding_dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self._embed(["probe"])[0])
        return self._dim

    def _embed(self, texts):
        out = []
        batch = 64
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            last = None
            for attempt in range(5):
                try:
                    r = self._session.post(
                        f"{self.base_url}/embeddings",
                        headers=self._headers,
                        json={"model": self.model, "input": chunk},
                        timeout=120,
                    )
                    if r.status_code in (401, 403):
                        raise RuntimeError(
                            f"Auth failed ({r.status_code}) on /embeddings — check API_KEY"
                        )
                    if r.status_code == 429 or 500 <= r.status_code < 600:
                        last = f"http {r.status_code}"
                        time.sleep(min(2 ** attempt, 30))
                        continue
                    r.raise_for_status()
                    data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
                    out.extend(d["embedding"] for d in data)
                    break
                except Exception as e:  # noqa: BLE001
                    last = str(e)
                    if attempt == 4:
                        raise RuntimeError(f"/embeddings failed: {last}")
                    time.sleep(min(2 ** attempt, 30))
        return out

    def encode(self, texts, batch_size: int = 32, show_progress_bar: bool = False,
               convert_to_numpy: bool = True, normalize_embeddings: bool = True, **_):
        import numpy as np

        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.zeros((0, self.get_sentence_embedding_dimension()), dtype=np.float32)
        arr = np.asarray(self._embed(items), dtype=np.float32)
        if normalize_embeddings and arr.size:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = (arr / norms).astype(np.float32)
        return arr[0] if single else arr


class ModelManager:
    def __init__(self) -> None:
        self._nli_model = None
        self._embed_model = None
        self._embed_dim: Optional[int] = None
        self._lock = threading.Lock()

    # ---------- NLI ----------

    def load_nli(self) -> None:
        with self._lock:
            if self._nli_model is not None:
                return
            from sentence_transformers import CrossEncoder

            self._nli_model = CrossEncoder(
                NLI_MODEL_NAME,
                device="cuda" if torch.cuda.is_available() else "cpu",
                max_length=512,
            )
            self._resolve_label_indices()
            self._verify_nli_labels()
            logger.info(
                f"[MM] NLI loaded ({NLI_MODEL_NAME}) | labels={NLI_LABEL_INDEX} | {self.vram_report()}"
            )

    def get_nli(self):
        self.load_nli()
        return self._nli_model

    def unload_nli(self) -> None:
        with self._lock:
            if self._nli_model is not None:
                del self._nli_model
                self._nli_model = None
                NLI_LABEL_INDEX.clear()
                self._flush()
                logger.info("[MM] NLI unloaded")

    def _resolve_label_indices(self) -> None:
        """Read id2label from the underlying HF model and populate NLI_LABEL_INDEX.

        Handles label name variants ("ENTAILMENT", "entailment", "ent"). Falls back
        to the default DeBERTa-v3 ordering only if id2label is unavailable.
        """
        global NLI_LABEL_INDEX
        NLI_LABEL_INDEX.clear()
        id2label = None
        # sentence-transformers CrossEncoder wraps an HF model
        inner = getattr(self._nli_model, "model", None)
        cfg = getattr(inner, "config", None) if inner is not None else None
        if cfg is not None:
            id2label = getattr(cfg, "id2label", None)

        if id2label:
            for idx, label in id2label.items():
                norm = str(label).strip().lower()
                if "entail" in norm:
                    NLI_LABEL_INDEX["entailment"] = int(idx)
                elif "contradict" in norm:
                    NLI_LABEL_INDEX["contradiction"] = int(idx)
                elif "neutral" in norm:
                    NLI_LABEL_INDEX["neutral"] = int(idx)

        # Fallback for models that don't expose id2label cleanly.
        if not NLI_LABEL_INDEX:
            logger.warning(
                "[MM] Could not resolve NLI id2label; falling back to "
                "[contradiction=0, entailment=1, neutral=2] which matches "
                "deberta-v3-base. Verify if you swap models."
            )
            NLI_LABEL_INDEX.update(
                {"contradiction": 0, "entailment": 1, "neutral": 2}
            )

    def _verify_nli_labels(self) -> None:
        """Probe both an entailment example and a contradiction example.

        Confirms our resolved indices put the right slot at the top of softmax for
        each known case. If a model swap silently inverts the label order, this
        assertion fires before any downstream metric is computed.
        """
        ent_idx = NLI_LABEL_INDEX.get("entailment")
        contra_idx = NLI_LABEL_INDEX.get("contradiction")
        if ent_idx is None or contra_idx is None:
            raise RuntimeError(
                f"NLI label indices incomplete: {NLI_LABEL_INDEX}"
            )
        scores = self._nli_model.predict(
            [
                ("The cat is on the mat.", "There is a cat on the mat."),
                ("Cats are mammals.", "Cats are not mammals."),
            ],
            apply_softmax=True,
        )
        ent_score = float(scores[0][ent_idx])
        contra_score = float(scores[1][contra_idx])
        if ent_score < 0.5:
            raise RuntimeError(
                f"NLI entailment probe failed: scores[0]={scores[0]} ent_idx={ent_idx}"
            )
        if contra_score < 0.5:
            raise RuntimeError(
                f"NLI contradiction probe failed: scores[1]={scores[1]} contra_idx={contra_idx}"
            )

    # ---------- Embeddings ----------

    def load_embedding(self) -> None:
        with self._lock:
            if self._embed_model is not None:
                return
            if ACTIVE_EMBEDDING:
                self._embed_model = _APIEmbedder(ACTIVE_EMBEDDING)
                try:
                    self._embed_dim = int(self._embed_model.get_sentence_embedding_dimension())
                except Exception:
                    self._embed_dim = None
                logger.info(
                    f"[MM] Embedding via API: {ACTIVE_EMBEDDING} dim={self._embed_dim}"
                )
                return
            from sentence_transformers import SentenceTransformer

            device = os.getenv("EMBED_DEVICE")
            if not device:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
            try:
                self._embed_dim = int(self._embed_model.get_sentence_embedding_dimension())
            except Exception:
                self._embed_dim = None
            logger.info(
                f"[MM] Embedding loaded: {EMBED_MODEL_NAME} ({device}) dim={self._embed_dim}"
            )

    def get_embedding(self):
        self.load_embedding()
        return self._embed_model

    def embedding_dim(self) -> int:
        """Return the embedding model's output dimension (loads model if needed)."""
        self.load_embedding()
        if self._embed_dim is None:
            # Encode a probe to recover dim
            vec = self._embed_model.encode(["probe"], normalize_embeddings=True)
            self._embed_dim = int(vec.shape[1])
        return self._embed_dim

    def unload_embedding(self) -> None:
        with self._lock:
            if self._embed_model is not None:
                del self._embed_model
                self._embed_model = None
                self._embed_dim = None
                logger.info("[MM] Embedding unloaded")

    def embed_model_id(self) -> str:
        return ACTIVE_EMBEDDING or EMBED_MODEL_NAME

    # ---------- Housekeeping ----------

    def unload_all(self) -> None:
        self.unload_nli()
        self.unload_embedding()
        self._flush()

    def _flush(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(0.3)

    def vram_report(self) -> str:
        if not torch.cuda.is_available():
            return "CPU mode"
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"VRAM {used:.2f}/{total:.2f} GB"


MM = ModelManager()
