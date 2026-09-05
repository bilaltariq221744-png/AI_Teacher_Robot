"""Tests for src/rag_chain.py (Phase 8: SVG W, X; PDF §8)."""
from __future__ import annotations

import numpy as np
import pytest

from src.chunking import Chunk, make_lexical_text
from src.lancedb_store import LanceStore
from src.rag_chain import (
    DECLINE_MESSAGE,
    LLM_ERROR_MESSAGE,
    Answer,
    LLMConnectionError,
    answer,
    build_messages,
    build_prompt,
    call_llm,
    decline,
    format_context,
)
from src.retriever import ContextBlock

DIM = 8


def _unit(values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


V_PLANTS = _unit([1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
V_BIRDS = _unit([-1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])


def _chunk(chunk_id, parent_id, text, *, unit="Unit 3", grade="5", page=1, kind="child"):
    return Chunk(
        chunk_id=chunk_id, parent_id=parent_id, kind=kind,
        text_clean=text, text_lexical=make_lexical_text(text), embed_text=text,
        page_start=page, page_end=page, book="Test Book", board="SNC",
        grade=grade, subject="English", unit=unit, chapter="Chapter 2",
        topic="", language="english", ocr_confidence=0.9,
        pipeline_version="test-v1",
    )


class _FixedBackend:
    def __init__(self, vector):
        self.vector = vector

    def encode(self, texts):
        return np.repeat(self.vector[np.newaxis, :], len(texts), axis=0)


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


# --- prompt construction (W) -------------------------------------------------


def test_format_context_includes_pages():
    block = ContextBlock(
        chunk_id="c1", parent_id="p1", score=0.5,
        child_text="Plants make food.",
        parent_text="Longer parent passage.",
        metadata={"page_start": 30, "page_end": 31},
    )
    rendered = format_context([block])
    assert "(pages 30-31)" in rendered
    assert "Plants make food." in rendered
    assert "Longer parent passage." in rendered


def test_format_context_single_page():
    block = ContextBlock("c1", "p1", 0.5, "x", metadata={"page_start": 30})
    assert "(page 30)" in format_context([block])


def test_build_prompt_contains_rules_context_question():
    block = ContextBlock("c1", "p1", 0.5, "Plants make food.", metadata={"page_start": 30})
    prompt = build_prompt([block], "How do plants make food?")
    assert "teacher robot" in prompt
    assert DECLINE_MESSAGE in prompt
    assert "Plants make food." in prompt
    assert "How do plants make food?" in prompt


def test_build_messages_are_langchain_messages():
    block = ContextBlock("c1", "p1", 0.5, "text", metadata={})
    messages = build_messages([block], "question?")
    assert len(messages) == 2
    assert messages[0].type == "system"
    assert messages[1].type == "human"


# --- decline ----------------------------------------------------------------


def test_decline_message():
    assert decline() == "This is not covered in the provided book."


# --- answer orchestration ---------------------------------------------------


def test_answer_returns_llm_text_and_sources(store):
    calls = []

    def fake_llm(messages, cfg):
        calls.append(messages)
        return "Plants make food through photosynthesis (page 30)."

    result = answer("How do plants make food?", store, _FixedBackend(V_PLANTS), CFG, llm=fake_llm)
    assert isinstance(result, Answer)
    assert not result.declined
    assert result.text == "Plants make food through photosynthesis (page 30)."
    assert result.sources, "sources should carry page citations"
    assert result.sources[0]["page_start"] in (30, 31)
    assert "p_plants" in {s["parent_id"] for s in result.sources}
    assert len(calls) == 1
    assert calls[0][-1].type == "human"  # question is the human message


def test_answer_declines_when_no_strong_context(store):
    calls = []

    def fake_llm(messages, cfg):
        calls.append(messages)
        return "should not be called"

    cfg = dict(CFG, relevance_threshold=0.9)  # everything declines
    result = answer("anything", store, _FixedBackend(V_PLANTS), cfg, llm=fake_llm)
    assert result.declined
    assert result.text == DECLINE_MESSAGE
    assert result.sources == []
    assert calls == []  # LLM never called when retrieval is weak


# --- LLM connection failure: graceful degradation, not a crash -------------


def test_answer_returns_friendly_message_on_llm_connection_error(store):
    def failing_llm(messages, cfg):
        raise LLMConnectionError("Ollama call failed: connection refused")

    result = answer("How do plants make food?", store, _FixedBackend(V_PLANTS), CFG, llm=failing_llm)
    assert result.error is True
    assert result.declined is False  # a backend failure is not a content decline
    assert result.text == LLM_ERROR_MESSAGE
    # Retrieval succeeded even though the LLM call failed — sources should
    # still be available (useful for debugging / showing what was found).
    assert result.sources


def test_answer_success_has_error_false_by_default(store):
    result = answer(
        "How do plants make food?", store, _FixedBackend(V_PLANTS), CFG,
        llm=lambda messages, cfg: "Plants make food via photosynthesis (page 30).",
    )
    assert result.error is False


def test_call_llm_wraps_backend_failures_as_llm_connection_error(monkeypatch):
    import src.rag_chain as rag_chain_module

    class _BoomChatOllama:
        def __init__(self, *a, **kw):
            raise ConnectionError("no route to host")

    monkeypatch.setattr(
        "langchain_ollama.ChatOllama", _BoomChatOllama, raising=False
    )
    with pytest.raises(LLMConnectionError):
        call_llm([], {"llm": {"model": "qwen2.5:1.5b"}})
