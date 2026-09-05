"""Embedding layer (SVG I, J, J1, K; PDF §5).

J  - ``intfloat/multilingual-e5-small`` (384-d, multilingual, retrieval-focused,
     light enough for the Pi when exported to ONNX INT8).
J1 - E5 requires prefixes: ``passage:`` for stored chunks and ``query:`` for
     questions — without them retrieval quality drops (PDF §5).
K  - outputs are L2-normalized float32 vectors, ready for LanceDB.

The heavy backends (sentence-transformers on the PC, ONNX on the Pi) are
imported lazily and are swappable, so the prefix/normalize front end is
testable without any model installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
EMBED_DIM = 384
MAX_LENGTH = 512

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


# ---------------------------------------------------------------------------
# Front end: prefixes (J1) + L2 normalization (K)
# ---------------------------------------------------------------------------


def apply_prefix(text: str, kind: str) -> str:
    """Prefix text for E5 (J1): ``passage:`` for chunks, ``query:`` for questions.

    Idempotent: already-prefixed text is left untouched.
    """
    if kind == "passage":
        prefix = PASSAGE_PREFIX
    elif kind == "query":
        prefix = QUERY_PREFIX
    else:
        raise ValueError(f"kind must be 'passage' or 'query', got {kind!r}")
    t = (text or "").strip()
    if t.lower().startswith(prefix.strip().lower()):
        return t
    return prefix + t


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row; all-zero rows stay zero. Returns float32 (K)."""
    vecs = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean-pool token embeddings over non-padding tokens (E5 pooling)."""
    mask = np.asarray(attention_mask, dtype=np.float32)[..., None]
    summed = (np.asarray(last_hidden, dtype=np.float32) * mask).sum(axis=1)
    counts = mask.sum(axis=1)
    counts[counts == 0] = 1.0
    return (summed / counts).astype(np.float32)


def _embed(texts: Sequence[str], kind: str, backend: "EmbeddingBackend") -> np.ndarray:
    prefixed = [apply_prefix(t, kind) for t in texts]
    raw = np.asarray(backend.encode(prefixed), dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"backend.encode must return 2-D vectors, got shape {raw.shape}")
    return normalize(raw)


def embed_passages(texts: Sequence[str], backend: "EmbeddingBackend") -> np.ndarray:
    """Embed stored chunks: ``passage:`` prefix -> model -> L2 normalize."""
    return _embed(texts, "passage", backend)


def embed_question(text: str, backend: "EmbeddingBackend") -> np.ndarray:
    """Embed a runtime question: ``query:`` prefix -> model -> L2 normalize."""
    return _embed([text], "query", backend)[0]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class EmbeddingBackend:
    """Encodes prefixed texts into raw float32 vectors (n, dim)."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerBackend(EmbeddingBackend):
    """Build-time backend: e5-small via sentence-transformers (PC)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = MAX_LENGTH,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = self.max_length
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        return np.asarray(
            model.encode(list(texts), normalize_embeddings=True), dtype=np.float32
        )


class OnnxBackend(EmbeddingBackend):
    """Runtime backend: ONNX (INT8) e5-small via onnxruntime (Pi, CPU)."""

    def __init__(
        self,
        model_dir: str | Path,
        max_length: int = MAX_LENGTH,
        provider: Optional[str] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.provider = provider
        self._session = None
        self._tokenizer = None

    def _load(self):
        if self._session is None:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            model_path = self.model_dir / "model.onnx"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found: {model_path} (run scripts/export_onnx_model.py first)"
                )
            providers = [self.provider] if self.provider else ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        return self._session, self._tokenizer

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        session, tokenizer = self._load()
        inputs = tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.max_length, return_tensors="np",
        )
        input_names = {i.name for i in session.get_inputs()}
        feed = {name: inputs[name] for name in input_names if name in inputs}
        last_hidden = session.run(["last_hidden_state"], feed)[0]
        return mean_pool(last_hidden, feed["attention_mask"])


def load_embedder(
    model_name: Optional[str] = None,
    onnx_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
    max_length: int = MAX_LENGTH,
) -> EmbeddingBackend:
    """Pick a backend: ONNX (Pi runtime) when ``onnx_dir`` is given, else
    sentence-transformers (PC build)."""
    if onnx_dir:
        return OnnxBackend(onnx_dir, max_length=max_length)
    return SentenceTransformerBackend(model_name or DEFAULT_MODEL, max_length=max_length, device=device)
