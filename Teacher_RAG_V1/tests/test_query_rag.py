"""Tests for query_rag.py (Phase 11: wired Pi/laptop runtime)."""
from __future__ import annotations

import numpy as np
import pytest

from src.chunking import Chunk, make_lexical_text
from src.config import Config
from src.lancedb_store import LanceStore
from src.rag_chain import LLMConnectionError

from query_rag import (
    ask,
    format_blocks,
    format_sources,
    load_backend,
    resolve_backend_source,
    resolve_db_path,
)

DIM = 8


def _unit(values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


V_PLANTS = _unit([1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])


def _chunk(chunk_id, parent_id, text, *, page=1, kind="child"):
    return Chunk(
        chunk_id=chunk_id, parent_id=parent_id, kind=kind,
        text_clean=text, text_lexical=make_lexical_text(text), embed_text=text,
        page_start=page, page_end=page, book="Test Book", board="SNC",
        grade="5", subject="Science", unit="Unit 3", chapter="Chapter 2",
        topic="", language="english", ocr_confidence=0.9,
        pipeline_version="test-v1",
    )


class _FixedBackend:
    def encode(self, texts):
        return np.repeat(V_PLANTS[np.newaxis, :], len(texts), axis=0)


@pytest.fixture
def store(tmp_path):
    s = LanceStore.create(tmp_path / "lancedb")
    s.add_parents([
        _chunk("p_plants", "p_plants", "Plants make their own food using sunlight.", kind="parent"),
    ])
    s.add_children(
        [
            _chunk("c_plants", "p_plants", "Plants make their own food using sunlight.", page=30),
            _chunk("c_photo", "p_plants", "Photosynthesis needs sunlight, water and carbon dioxide.", page=31),
        ],
        np.stack([V_PLANTS, V_PLANTS * 0.9]),
    )
    s.build_indexes()
    return s


CFG = {"vector_top_k": 20, "bm25_top_k": 20, "rrf_k": 60,
       "final_context_blocks": 5, "context_tokens": 1200}


def _write_config(tmp_path, pi_db_exists=False, pi_onnx_exists=False, pc_db_exists=False):
    """Write a minimal config.yaml for resolve_* tests, optionally creating
    the directories the config points at so the "exists on this machine"
    fallback logic has something real to check."""
    pi_db = tmp_path / "pi_db"
    pi_onnx = tmp_path / "pi_onnx"
    pc_db = tmp_path / "pc_db"
    if pi_db_exists:
        pi_db.mkdir()
    if pi_onnx_exists:
        pi_onnx.mkdir()
    if pc_db_exists:
        pc_db.mkdir()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
pc:
  db_path: {pc_db}
  embed_model: intfloat/multilingual-e5-small
pi:
  db_path: {pi_db}
  onnx_model: {pi_onnx}
  vector_top_k: 20
  bm25_top_k: 20
  relevance_threshold: 0.55
  llm:
    provider: ollama
    model: qwen2.5:1.5b
""",
        encoding="utf-8",
    )
    return Config.load(config_path), pi_db, pi_onnx, pc_db


# --- db path resolution (local-testing fallback) ----------------------------


def test_resolve_db_path_prefers_override(tmp_path):
    cfg, pi_db, _onnx, _pc = _write_config(tmp_path, pi_db_exists=True)
    override = tmp_path / "somewhere_else"
    from query_rag import resolve_db_path
    assert resolve_db_path(cfg, str(override)) == override


def test_resolve_db_path_uses_pi_path_when_it_exists(tmp_path):
    cfg, pi_db, _onnx, _pc = _write_config(tmp_path, pi_db_exists=True, pc_db_exists=True)
    assert resolve_db_path(cfg) == pi_db


def test_resolve_db_path_falls_back_to_pc_path_for_local_testing(tmp_path):
    cfg, pi_db, _onnx, pc_db = _write_config(tmp_path, pi_db_exists=False, pc_db_exists=True)
    assert resolve_db_path(cfg) == pc_db


def test_resolve_db_path_returns_pi_path_when_neither_exists(tmp_path):
    """Neither exists — return the pi path anyway so LanceStore.open raises a
    clear, actionable FileNotFoundError instead of resolve_db_path guessing."""
    cfg, pi_db, _onnx, _pc = _write_config(tmp_path)
    assert resolve_db_path(cfg) == pi_db


# --- embedder backend resolution (local-testing fallback) -------------------


def test_resolve_backend_prefers_override():
    class _Cfg:
        pi = {"onnx_model": "/does/not/matter"}
    kind, path = resolve_backend_source(_Cfg(), onnx_dir="/my/onnx/dir")
    assert (kind, path) == ("onnx", "/my/onnx/dir")


def test_resolve_backend_uses_pi_onnx_when_it_exists(tmp_path):
    cfg, _db, onnx_dir, _pc = _write_config(tmp_path, pi_onnx_exists=True)
    kind, path = resolve_backend_source(cfg)
    assert kind == "onnx"
    assert path == str(onnx_dir)


def test_resolve_backend_falls_back_to_sentence_transformers(tmp_path):
    """ONNX model not exported yet / not present on this machine (Phase 5
    output missing) -> fall back to the PC backend instead of failing."""
    cfg, _db, _onnx, _pc = _write_config(tmp_path, pi_onnx_exists=False)
    kind, path = resolve_backend_source(cfg)
    assert kind == "sentence-transformers"
    assert path is None


def test_load_backend_sentence_transformers_fallback_uses_pc_embed_model(tmp_path):
    cfg, _db, _onnx, _pc = _write_config(tmp_path, pi_onnx_exists=False)
    backend = load_backend(cfg)
    assert backend.model_name == "intfloat/multilingual-e5-small"


# --- console formatting ------------------------------------------------------


def test_format_sources_renders_page_range_and_location():
    sources = [{"page_start": 30, "page_end": 31, "unit": "Unit 3", "chapter": "Chapter 2", "topic": ""}]
    out = format_sources(sources)
    assert "page 30-31" in out
    assert "Unit 3 / Chapter 2" in out


def test_format_sources_single_page_no_location():
    sources = [{"page_start": 30, "page_end": 30, "unit": "", "chapter": "", "topic": ""}]
    out = format_sources(sources)
    assert "page 30" in out
    assert "(" not in out  # no location parenthetical when unit/chapter/topic are empty


def test_format_blocks_empty():
    assert "nothing retrieved" in format_blocks([])


def test_format_blocks_truncates_long_snippets():
    from src.retriever import ContextBlock
    long_text = "word " * 100
    block = ContextBlock(
        chunk_id="c1", parent_id="p1", score=0.1234,
        child_text=long_text, metadata={"page_start": 5, "page_end": 5},
    )
    out = format_blocks([block])
    assert "..." in out
    assert "score=0.1234" in out
    assert "page 5" in out


# --- ask() orchestration ------------------------------------------------------


def test_ask_no_llm_shows_retrieved_passages(store):
    out = ask("How do plants make food?", store, _FixedBackend(), CFG, no_llm=True)
    assert "Retrieved passages" in out
    assert "page 30" in out
    assert "Plants make their own food" in out


def test_ask_no_llm_declines_when_nothing_retrieved(store):
    cfg = dict(CFG, relevance_threshold=0.99)  # everything below the gate
    out = ask("anything", store, _FixedBackend(), cfg, no_llm=True)
    assert "not covered" in out


def test_ask_with_llm_includes_sources(store, monkeypatch):
    import src.rag_chain as rag_chain

    def fake_call_llm(messages, cfg=None):
        return "Plants make food through photosynthesis (page 30)."

    monkeypatch.setattr(rag_chain, "call_llm", fake_call_llm)
    out = ask("How do plants make food?", store, _FixedBackend(), CFG, no_llm=False)
    assert "photosynthesis" in out
    assert "Sources:" in out
    assert "page 30" in out or "page 31" in out


def test_ask_surfaces_llm_connection_error_without_crashing(store, monkeypatch):
    import src.rag_chain as rag_chain

    def failing_call_llm(messages, cfg=None):
        raise rag_chain.LLMConnectionError("Ollama call failed: connection refused")

    monkeypatch.setattr(rag_chain, "call_llm", failing_call_llm)
    out = ask("How do plants make food?", store, _FixedBackend(), CFG, no_llm=False)
    assert "trouble reaching" in out
    assert "connection/engine error" in out
    # Sources should still be shown even though the LLM call failed, since
    # retrieval itself succeeded.
    assert "Sources:" in out


def test_ask_declines_with_llm_mode_when_context_too_weak(store):
    cfg = dict(CFG, relevance_threshold=0.99)
    out = ask("anything", store, _FixedBackend(), cfg, no_llm=False)
    assert out == "This is not covered in the provided book."
