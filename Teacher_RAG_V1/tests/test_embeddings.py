"""Unit tests for src/embeddings.py (Phase 5: SVG I, J, J1, K; PDF §5)."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from src.embeddings import (
    apply_prefix,
    embed_passages,
    embed_question,
    mean_pool,
    normalize,
)

try:
    import sentence_transformers  # noqa: F401

    HAS_ST = True
except ImportError:
    HAS_ST = False


class _FakeBackend:
    """Records the texts it receives and returns deterministic vectors."""

    def __init__(self, dim: int = 384, seed: int = 0) -> None:
        self.seen: list[str] = []
        rng = np.random.default_rng(seed)
        self.vectors = rng.standard_normal((64, dim)).astype(np.float32)

    def encode(self, texts):
        self.seen.extend(texts)
        return np.stack([self.vectors[len(self.seen) - i - 1] for i in range(len(texts))])


# --- J1: prefixes ----------------------------------------------------------


def test_apply_prefix_passage_and_query():
    assert apply_prefix("hello world", "passage") == "passage: hello world"
    assert apply_prefix("hello world", "query") == "query: hello world"


def test_apply_prefix_is_idempotent():
    assert apply_prefix("passage: hello", "passage") == "passage: hello"
    assert apply_prefix("query: hello", "query") == "query: hello"
    assert apply_prefix("Passage: Hello", "passage") == "Passage: Hello"


def test_apply_prefix_rejects_unknown_kind():
    with pytest.raises(ValueError):
        apply_prefix("x", "bogus")


# --- K: L2 normalization ---------------------------------------------------


def test_normalize_rows_and_preserves_zero():
    v = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    out = normalize(v)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out[:1], axis=1), 1.0)  # non-zero rows -> unit norm
    assert np.allclose(np.linalg.norm(out[2:], axis=1), 1.0)
    assert np.allclose(out[1], 0.0)  # zero row stays zero, not NaN


def test_normalize_zero_row_does_not_nan():
    out = normalize(np.zeros((2, 4), dtype=np.float32))
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)


# --- pooling ---------------------------------------------------------------


def test_mean_pool_over_non_padding_tokens():
    hidden = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]])
    out = mean_pool(hidden, mask)
    assert np.allclose(out, [[2.0, 3.0]])  # mean of tokens 0 and 1 only


# --- embed functions with a fake backend -----------------------------------


def test_embed_passages_prefixes_and_normalizes():
    backend = _FakeBackend()
    out = embed_passages(["plants make food", "sunlight helps"], backend)
    assert backend.seen == ["passage: plants make food", "passage: sunlight helps"]
    assert out.shape == (2, 384)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_embed_question_returns_single_vector():
    backend = _FakeBackend()
    out = embed_question("How do plants make food?", backend)
    assert backend.seen == ["query: How do plants make food?"]
    assert out.shape == (384,)


def test_embed_does_not_double_prefix():
    backend = _FakeBackend()
    embed_passages(["passage: already prefixed"], backend)
    assert backend.seen == ["passage: already prefixed"]


def test_embed_rejects_non_matrix_output():
    class _BadBackend(_FakeBackend):
        def encode(self, texts):
            return np.zeros(4, dtype=np.float32)

    with pytest.raises(ValueError, match="2-D"):
        embed_passages(["x"], _BadBackend())


def test_heading_enriched_chunk_text_embeds_cleanly():
    # embed_text from chunking is already heading-enriched; prefix is still applied.
    backend = _FakeBackend()
    chunk_text = "Unit: Unit 3 · Topic: Oxygen — Plants release oxygen into the air."
    out = embed_passages([chunk_text], backend)
    assert out.shape == (1, 384)
    assert backend.seen[0].startswith("passage: Unit: Unit 3")


# --- real-model sanity (runs only when sentence-transformers is installed) --


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@pytest.mark.skipif(not HAS_ST, reason="sentence-transformers not installed")
def test_real_model_cosine_sanity():
    from src.embeddings import load_embedder

    backend = load_embedder()
    same1 = embed_passages(["Plants make their own food."], backend)
    same2 = embed_passages(["Plants produce their food."], backend)
    different = embed_passages(["The capital of France is Paris."], backend)
    assert _cos(same1, same2) > _cos(same1, different)


# --- export script ---------------------------------------------------------


def test_export_script_help_runs_without_heavy_deps():
    result = subprocess.run(
        [sys.executable, "scripts/export_onnx_model.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--int8" in result.stdout
