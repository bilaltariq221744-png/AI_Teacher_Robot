"""Round-trip tests for src/lancedb_store.py (Phase 6: SVG L, M, M1-M3)."""
from __future__ import annotations

import numpy as np
import pytest

from src.chunking import Chunk, make_lexical_text
from src.lancedb_store import CHILD_SCALAR_COLUMNS, LanceStore, _where_clause

DIM = 8


def _unit(values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


# Deterministic unit vectors: plants/photo point +x, birds point -x (far away).
V_PLANTS = _unit([1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
V_BIRDS = _unit([-1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
V_PHOTO = _unit([0.9, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])


def _chunk(chunk_id, parent_id, text, *, unit="Unit 3", grade="5", page=1, kind="child", book="Test Book"):
    return Chunk(
        chunk_id=chunk_id, parent_id=parent_id, kind=kind,
        text_clean=text, text_lexical=make_lexical_text(text),
        embed_text=text, page_start=page, page_end=page,
        book=book, board="SNC", grade=grade, subject="English",
        unit=unit, chapter="Chapter 2", topic="", language="english",
        ocr_confidence=0.9, pipeline_version="test-v1",
    )


@pytest.fixture
def store(tmp_path):
    s = LanceStore.create(tmp_path / "lancedb")
    children = [
        _chunk("c_plants", "p_plants", "Plants make their own food using sunlight and water."),
        _chunk("c_birds", "p_birds", "Birds build nests in trees and feed their young.", unit="Unit 4", grade="6"),
        _chunk("c_photo", "p_plants", "Photosynthesis needs sunlight, water and carbon dioxide."),
    ]
    parents = [
        _chunk("p_plants", "p_plants", "Plants make their own food using sunlight and water. Photosynthesis needs sunlight.", kind="parent"),
        _chunk("p_birds", "p_birds", "Birds build nests in trees and feed their young.", unit="Unit 4", grade="6", kind="parent"),
    ]
    s.add_parents(parents)
    s.add_children(children, np.stack([V_PLANTS, V_BIRDS, V_PHOTO]))
    s.build_indexes()
    return s


# --- where-clause builder --------------------------------------------------


def test_where_clause_equality_and_in():
    assert _where_clause({"grade": "5"}) == "grade = '5'"
    assert _where_clause({"grade": ["5", "6"]}) == "grade IN ('5', '6')"
    assert _where_clause(None) is None
    assert _where_clause({}) is None


def test_where_clause_escapes_quotes():
    assert _where_clause({"unit": "Unit 3's"}).endswith("'Unit 3''s'")


# --- round-trips ------------------------------------------------------------


def test_vector_search_roundtrip(store):
    hits = store.search_children_vector(V_PLANTS, top_k=3)
    assert len(hits) == 3
    assert hits[0].chunk_id == "c_plants"
    assert hits[0].score < hits[1].score  # L2 distance: lower is better
    assert hits[0].row["grade"] == "5"
    assert "vector" not in hits[0].row
    assert hits[0].text_clean.startswith("Plants make")


def test_fts_finds_keyword_vector_search_misses(store):
    vec_ids = [h.chunk_id for h in store.search_children_vector(V_PLANTS, top_k=2)]
    assert vec_ids == ["c_plants", "c_photo"]  # birds is far in vector space
    fts_hits = store.search_children_fts("nests", top_k=3)
    assert fts_hits[0].chunk_id == "c_birds"  # BM25 catches the exact keyword


def test_metadata_filter_restricts_results(store):
    hits = store.search_children_vector(V_PLANTS, filters={"grade": "5"}, top_k=10)
    assert {h.row["grade"] for h in hits} == {"5"}
    assert {h.chunk_id for h in hits} == {"c_plants", "c_photo"}

    hits = store.search_children_vector(V_PLANTS, filters={"unit": "Unit 4"}, top_k=10)
    assert [h.chunk_id for h in hits] == ["c_birds"]

    hits = store.search_children_vector(V_PLANTS, filters={"grade": ["5", "6"]}, top_k=10)
    assert len(hits) == 3


def test_hybrid_search_ranks_and_filters(store):
    hits = store.hybrid_search("nests", V_PLANTS, top_k=3)
    assert hits[0].chunk_id == "c_birds"  # FTS term dominates
    filtered = store.hybrid_search("nests", V_PLANTS, filters={"grade": "5"}, top_k=3)
    assert "c_birds" not in [h.chunk_id for h in filtered]
    assert filtered


def test_parent_roundtrip(store):
    row = store.get_parent("p_plants")
    assert row is not None
    assert row["chunk_id"] == "p_plants"
    assert "Plants make their own food" in row["text_clean"]
    assert store.get_parent("nope") is None


def test_stats(store):
    assert store.stats() == {"parents": 2, "children": 3}


# --- lifecycle --------------------------------------------------------------


def test_open_existing_store(tmp_path):
    db_path = tmp_path / "lancedb"
    s1 = LanceStore.create(db_path)
    s1.add_parents([_chunk("p1", "p1", "parent text", kind="parent")])
    s1.add_children([_chunk("c1", "p1", "hello world")], V_PLANTS.reshape(1, DIM))
    s2 = LanceStore.open(db_path)
    hits = s2.search_children_vector(V_PLANTS, top_k=1)
    assert hits[0].chunk_id == "c1"
    assert s2.stats() == {"parents": 1, "children": 1}


def test_open_incomplete_store_raises(tmp_path):
    db_path = tmp_path / "lancedb"
    s = LanceStore.create(db_path)
    s.add_children([_chunk("c1", "p1", "hello")], V_PLANTS.reshape(1, DIM))
    with pytest.raises(FileNotFoundError):
        LanceStore.open(db_path)


def test_search_without_build_raises(tmp_path):
    s = LanceStore.create(tmp_path / "lancedb")
    with pytest.raises(RuntimeError, match="build the store first"):
        s.search_children_vector(V_PLANTS)


def test_add_children_validates_vector_count(tmp_path):
    s = LanceStore.create(tmp_path / "lancedb")
    with pytest.raises(ValueError, match="length mismatch"):
        s.add_children([_chunk("c1", "p1", "text")], np.stack([V_PLANTS, V_PHOTO]))
    with pytest.raises(ValueError, match="2-D"):
        s.add_children([_chunk("c1", "p1", "text")], np.zeros(DIM, dtype=np.float32))


def test_build_indexes_idempotent(store):
    store.build_indexes()  # replace=True -> safe to rebuild
    store.build_indexes()
    assert store.search_children_fts("photosynthesis", top_k=1)[0].chunk_id == "c_photo"


def test_scalar_index_columns_defined():
    assert "grade" in CHILD_SCALAR_COLUMNS
    assert "unit" in CHILD_SCALAR_COLUMNS
    assert "language" in CHILD_SCALAR_COLUMNS


# --- seq: true sibling reading order (replaces the old alphabetic sort) ----


def test_get_children_orders_by_seq_not_alphabetically(tmp_path):
    """The old behavior sorted siblings by raw text, which is an alphabetic
    accident, not reading order. A child whose text alphabetically sorts
    first but was written LAST in the section must still come last."""
    s = LanceStore.create(tmp_path / "lancedb")
    parent = _chunk("p1", "p1", "Full parent text.", kind="parent")
    children = [
        Chunk(
            chunk_id="c_second", parent_id="p1", kind="child",
            text_clean="Aardvark comes alphabetically first but is written second.",
            text_lexical="x", embed_text="x", page_start=1, page_end=1,
            book="Test Book", unit="Unit 3", seq=1,
        ),
        Chunk(
            chunk_id="c_first", parent_id="p1", kind="child",
            text_clean="Zebra comes alphabetically last but is written first.",
            text_lexical="x", embed_text="x", page_start=1, page_end=1,
            book="Test Book", unit="Unit 3", seq=0,
        ),
    ]
    s.add_parents([parent])
    s.add_children(children, np.stack([V_PLANTS, V_PLANTS]))
    s.build_indexes()
    ordered = s.get_children("p1")
    assert [c["chunk_id"] for c in ordered] == ["c_first", "c_second"]


def test_child_row_stores_seq(store):
    rows = store.get_children("p_plants")
    assert all("seq" in r for r in rows)


# --- _table_names(): defensive handling across lancedb API shapes ----------


def test_table_names_handles_dot_tables_attribute(tmp_path, monkeypatch):
    s = LanceStore.create(tmp_path / "lancedb")

    class _Result:
        tables = ["parents", "children"]

    monkeypatch.setattr(s._db, "list_tables", lambda: _Result())
    assert s._table_names() == {"parents", "children"}


def test_table_names_handles_plain_list_return(tmp_path, monkeypatch):
    s = LanceStore.create(tmp_path / "lancedb")
    monkeypatch.setattr(s._db, "list_tables", lambda: ["parents", "children"])
    assert s._table_names() == {"parents", "children"}
