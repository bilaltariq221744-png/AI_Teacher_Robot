"""Tests for src/retriever.py (Phase 7: SVG Q-V; PDF §8)."""
from __future__ import annotations

import numpy as np
import pytest

from src.chunking import Chunk, make_lexical_text
from src.lancedb_store import LanceStore
from src.retriever import (
    DECLINE_MESSAGE,
    ContextBlock,
    dedupe_and_expand,
    detect_language,
    estimate_tokens,
    gate_context,
    infer_filters,
    retrieve,
    rrf_fuse,
)

DIM = 8


def _unit(values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


V_PLANTS = _unit([1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
V_BIRDS = _unit([-1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
V_PHOTO = _unit([0.9, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])


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
    """Returns the same vector for every input; records what it saw."""

    def __init__(self, vector):
        self.vector = vector
        self.seen: list[str] = []

    def encode(self, texts):
        self.seen.extend(texts)
        return np.repeat(self.vector[np.newaxis, :], len(texts), axis=0)


@pytest.fixture
def store(tmp_path):
    s = LanceStore.create(tmp_path / "lancedb")
    s.add_parents([
        _chunk("p_plants", "p_plants", "Plants make their own food using sunlight. This process is called photosynthesis.", kind="parent"),
        _chunk("p_birds", "p_birds", "Birds build nests in trees and feed their young.", unit="Unit 4", grade="6", kind="parent"),
    ])
    s.add_children(
        [
            _chunk("c_plants", "p_plants", "Plants make their own food using sunlight."),
            _chunk("c_photo", "p_plants", "Photosynthesis needs sunlight, water and carbon dioxide."),
            _chunk("c_birds", "p_birds", "Birds build nests in trees and feed their young.", unit="Unit 4", grade="6"),
        ],
        np.stack([V_PLANTS, V_PHOTO, V_BIRDS]),
    )
    s.build_indexes()
    return s


# --- Q: language detection --------------------------------------------------


def test_detect_language_urdu_script():
    assert detect_language("پودے کھانا کیسے بناتے ہیں؟") == "urdu"


def test_detect_language_roman_urdu():
    assert detect_language("poday kya khana banate hain") == "roman_urdu"
    assert detect_language("yeh kahan hota hai") == "roman_urdu"


def test_detect_language_english():
    assert detect_language("How do plants make their food?") == "english"


# --- Q: curriculum filters --------------------------------------------------


def test_infer_filters_grade_and_section():
    assert infer_filters("Explain grade 5 unit 3 topic about plants") == {
        "grade": "5", "unit": "Unit 3",
    }
    assert infer_filters("chapter 2 animals") == {"chapter": "Chapter 2"}


def test_infer_filters_subject_and_book():
    q = "In English, how do plants make food?"
    assert infer_filters(q).get("subject") == "English"
    curriculum = {"books": [{"book": "New Oxford Modern English"}]}
    assert infer_filters("From New Oxford Modern English, what is photosynthesis?",
                         curriculum)["book"] == "New Oxford Modern English"


def test_infer_filters_empty():
    assert infer_filters("hello world") == {}


# --- T: RRF fusion ----------------------------------------------------------


def _hit(cid, parent, text, score):
    from src.lancedb_store import Hit

    return Hit(chunk_id=cid, parent_id=parent, text_clean=text, score=score, row={})


def test_rrf_fuses_and_ranks():
    vec = [_hit("a", "p", "text a", 0.1), _hit("b", "p", "text b", 0.2)]
    bm25 = [_hit("b", "p", "text b", 5.0), _hit("c", "p", "text c", 4.0)]
    fused = rrf_fuse(vec, bm25, k=60)
    assert [f.chunk_id for f in fused] == ["b", "a", "c"]
    # b appears at rank 2 in vector and rank 1 in BM25.
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-9


def test_rrf_deduplicates():
    vec = [_hit("a", "p", "t", 0.1), _hit("a", "p", "t", 0.1)]
    fused = rrf_fuse(vec, [], k=60)
    assert len(fused) == 1


# --- U: dedupe + expansion ---------------------------------------------------


def test_dedupe_and_expand_attaches_parent_and_siblings(store):
    from src.retriever import FusedHit

    fused = [FusedHit("c_plants", "p_plants", "plants text", 0.5, {})]
    blocks = dedupe_and_expand(fused, store)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.parent_text and "photosynthesis" in block.parent_text
    assert block.siblings == ["Photosynthesis needs sunlight, water and carbon dioxide."]
    assert "plants text" in block.text
    assert block.parent_text in block.text


# --- V: gate -----------------------------------------------------------------


def test_gate_context_budget_and_max():
    blocks = [
        ContextBlock("a", "p", 0.9, "alpha " * 100),
        ContextBlock("b", "p", 0.8, "beta " * 100),
        ContextBlock("c", "p", 0.7, "gamma " * 100),
    ]
    kept = gate_context(blocks, budget_tokens=50, max_blocks=5, min_score=None)
    assert len(kept) == 1  # first block always kept, second exceeds the 50-token budget
    assert kept[0].chunk_id == "a"


def test_gate_context_min_score_drops_weak():
    blocks = [
        ContextBlock("a", "p", 0.05, "weak"),
        ContextBlock("b", "p", 0.5, "strong"),
    ]
    kept = gate_context(blocks, min_score=0.1)
    assert [b.chunk_id for b in kept] == ["b"]
    assert gate_context(blocks, min_score=0.9) == []  # nothing passes -> decline


def test_gate_context_empty():
    assert gate_context([]) == []


def test_estimate_tokens_positive():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 40) == 10


# --- full retrieve() ---------------------------------------------------------


def test_retrieve_end_to_end(store):
    backend = _FixedBackend(V_PLANTS)
    cfg = {"vector_top_k": 20, "bm25_top_k": 20, "rrf_k": 60,
           "final_context_blocks": 5, "context_tokens": 1200}
    blocks = retrieve("How do plants make their own food?", store, backend, cfg)
    assert blocks
    assert blocks[0].parent_id == "p_plants"
    assert "photosynthesis" in blocks[0].parent_text
    assert "plants" in blocks[0].child_text.lower()
    # query embedding used the query: prefix
    assert backend.seen and backend.seen[0].startswith("query: ")
    # siblings of the matched parent are included somewhere
    all_text = "\n".join(b.text for b in blocks)
    assert "carbon dioxide" in all_text


def test_retrieve_applies_unit_filter(store):
    backend = _FixedBackend(V_BIRDS)
    cfg = {"vector_top_k": 20, "bm25_top_k": 20, "rrf_k": 60,
           "final_context_blocks": 5, "context_tokens": 1200}
    blocks = retrieve("unit 4: how do birds make nests?", store, backend, cfg)
    assert blocks
    assert all(b.metadata.get("unit") == "Unit 4" or b.parent_id == "p_birds" for b in blocks)


def test_retrieve_declines_when_threshold_blocks(store):
    backend = _FixedBackend(V_PLANTS)
    cfg = {"vector_top_k": 20, "bm25_top_k": 20, "rrf_k": 60,
           "final_context_blocks": 5, "context_tokens": 1200,
           "relevance_threshold": 0.9}  # RRF scores are ~0.03; everything declines
    assert retrieve("How do plants make food?", store, backend, cfg) == []
    assert DECLINE_MESSAGE == "This is not covered in the provided book."


def test_retrieve_respects_context_budget(store):
    backend = _FixedBackend(V_PLANTS)
    cfg = {"vector_top_k": 20, "bm25_top_k": 20, "rrf_k": 60,
           "final_context_blocks": 1, "context_tokens": 1200}
    blocks = retrieve("How do plants make food?", store, backend, cfg)
    assert len(blocks) == 1


# --- filter-fallback: a bad grade/subject/book guess must not cause a
# false decline (retrieve() retries once, unfiltered, on a filtered miss) --


class _FakeStore:
    """Records the filters each search was called with, so the fallback
    retry logic in retrieve() can be tested precisely without depending on
    real LanceDB filtering semantics."""

    def __init__(self, filtered_hits, unfiltered_hits):
        self.filtered_hits = filtered_hits
        self.unfiltered_hits = unfiltered_hits
        self.vector_calls: list = []
        self.fts_calls: list = []

    def search_children_vector(self, query_vector, filters=None, top_k=20):
        self.vector_calls.append(filters)
        return self.filtered_hits if filters else self.unfiltered_hits

    def search_children_fts(self, query_text, filters=None, top_k=20):
        self.fts_calls.append(filters)
        return self.filtered_hits if filters else self.unfiltered_hits

    def get_parent(self, parent_id):
        return None

    def get_children(self, parent_id, limit=50):
        return []


def _hit(chunk_id, parent_id="p1", text="Plants make their own food.", score=1.0):
    from src.lancedb_store import Hit

    return Hit(
        chunk_id=chunk_id, parent_id=parent_id, text_clean=text, score=score,
        row={"page_start": 1, "page_end": 1, "unit": "Unit 3"},
    )


def test_retrieve_falls_back_to_unfiltered_when_filtered_search_is_empty():
    backend = _FixedBackend(V_PLANTS)
    fake = _FakeStore(filtered_hits=[], unfiltered_hits=[_hit("c1")])
    cfg = {"vector_top_k": 20, "bm25_top_k": 20, "rrf_k": 60,
           "final_context_blocks": 5, "context_tokens": 1200}
    # "grade 9" won't match any stored chunk (the store only has grades 5/6
    # in the other tests' fixtures) — the inferred filter would zero out
    # both searches. The fallback must still surface the real content once
    # filters are dropped.
    blocks = retrieve("grade 9: how do plants make their own food?", fake, backend, cfg)
    assert blocks and blocks[0].chunk_id == "c1"
    assert any(f for f in fake.vector_calls)      # first attempt: filtered
    assert any(f is None for f in fake.vector_calls)  # retry: unfiltered
    assert any(f for f in fake.fts_calls)
    assert any(f is None for f in fake.fts_calls)


def test_retrieve_no_fallback_retry_when_filtered_results_are_found():
    backend = _FixedBackend(V_PLANTS)
    fake = _FakeStore(filtered_hits=[_hit("c1")], unfiltered_hits=[])
    cfg = {"vector_top_k": 20, "bm25_top_k": 20, "rrf_k": 60,
           "final_context_blocks": 5, "context_tokens": 1200}
    blocks = retrieve("grade 5 english: how do plants make food?", fake, backend, cfg)
    assert blocks and blocks[0].chunk_id == "c1"
    # Found on the first (filtered) attempt — no unnecessary retry.
    assert len(fake.vector_calls) == 1
    assert len(fake.fts_calls) == 1


def test_retrieve_no_fallback_retry_when_no_filters_were_inferred():
    backend = _FixedBackend(V_PLANTS)
    fake = _FakeStore(filtered_hits=[], unfiltered_hits=[])
    cfg = {"vector_top_k": 20, "bm25_top_k": 20, "rrf_k": 60,
           "final_context_blocks": 5, "context_tokens": 1200}
    # Nothing in this question maps to a grade/subject/book/unit filter, so
    # there's nothing to retry — a genuinely empty result stays empty.
    blocks = retrieve("xyzzy completely unrelated gibberish", fake, backend, cfg)
    assert blocks == []
    assert len(fake.vector_calls) == 1
    assert len(fake.fts_calls) == 1


# --- gate_context: strict behavior when nothing clears the threshold -------


def test_gate_context_min_score_drops_even_the_top_block():
    """Regression guard for the fixed docstring: an earlier version claimed
    the top block was always kept regardless of score. It never actually
    was, and the strict behavior (decline when nothing clears the bar) is
    the one we keep — a confident answer from a below-threshold match is a
    worse failure than declining for a teacher robot."""
    blocks = [ContextBlock("only", "p", 0.01, "the single best match, but still weak")]
    assert gate_context(blocks, min_score=0.5) == []
