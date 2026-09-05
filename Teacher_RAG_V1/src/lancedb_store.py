"""LanceDB storage layer (SVG L, M, M1-M3; PDF §6-7).

Two embedded tables in one portable folder (copied PC -> Pi):

* ``parents``  - parent chunks (180-300 words), no vector, retrieved by ID.
* ``children`` - child chunks (60-120 words) with the full PDF §7 schema:
  ``chunk_id, parent_id, board, grade, subject, book, edition, unit, chapter,
  topic, page_start, page_end, language, text_clean, text_lexical, vector,
  ocr_confidence, pipeline_version, seq``. ``seq`` is the child's order within
  its parent's section (true reading order for sibling expansion, U).

Indexes (built with ``build_indexes``):
  M1 - vector index on ``vector`` (LanceDB default L2; our vectors are
       L2-normalized so L2 ranking == cosine ranking).
  M2 - BM25/full-text index on ``text_lexical``.
  M3 - scalar (BTree) indexes on the curriculum metadata columns.

Notes on the API: lancedb 0.37.x deprecates ``create_fts_index`` /
``create_scalar_index`` in favor of ``create_index(col, config=...)``, but the
new config path currently has a serialization bug (``FtsParams.with_position``),
so this module uses the working legacy calls with the deprecation warning
suppressed. The vector metric is the table default (L2); ordering is identical
to cosine for unit-norm vectors.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .chunking import Chunk

PARENT_TABLE = "parents"
CHILD_TABLE = "children"

# Curriculum metadata columns that get a scalar (BTree) index (M3).
CHILD_SCALAR_COLUMNS = [
    "board", "grade", "subject", "book", "edition",
    "unit", "chapter", "topic", "language",
    "page_start", "page_end",
]


@dataclass
class Hit:
    """One retrieved child chunk.

    ``score`` semantics differ per search: ``_distance`` (vector, lower is
    better), ``_score`` (FTS, higher is better), ``_relevance_score`` (hybrid).
    Retrieval fusion (Phase 7, RRF) only uses rank order.
    """

    chunk_id: str
    parent_id: str
    text_clean: str
    score: float
    row: dict


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _where_clause(filters: Optional[dict[str, Any]]) -> Optional[str]:
    """Build a LanceDB SQL where-clause from equality / IN filters."""
    if not filters:
        return None
    clauses = []
    for key, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            vals = ", ".join(_quote(str(v)) for v in value)
            clauses.append(f"{key} IN ({vals})")
        else:
            clauses.append(f"{key} = {_quote(str(value))}")
    return " AND ".join(clauses)


def _apply_filters(query, filters: Optional[dict[str, Any]]):
    where = _where_clause(filters)
    return query.where(where) if where else query


def _rows_to_hits(rows: Sequence[dict], score_key: str) -> list[Hit]:
    hits = []
    for row in rows:
        hits.append(
            Hit(
                chunk_id=row["chunk_id"],
                parent_id=row.get("parent_id", ""),
                text_clean=row.get("text_clean", ""),
                score=float(row.get(score_key, 0.0)),
                row={k: v for k, v in row.items() if k != "vector"},
            )
        )
    return hits


def _parent_row(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "board": chunk.board,
        "grade": chunk.grade,
        "subject": chunk.subject,
        "book": chunk.book,
        "edition": chunk.edition,
        "unit": chunk.unit,
        "chapter": chunk.chapter,
        "topic": chunk.topic,
        "language": chunk.language,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "text_clean": chunk.text_clean,
        "pipeline_version": chunk.pipeline_version,
    }


def _child_row(chunk: Chunk, vector: np.ndarray) -> dict:
    row = _parent_row(chunk)
    row.update(
        {
            "text_lexical": chunk.text_lexical,
            "vector": np.asarray(vector, dtype=np.float32),
            "ocr_confidence": float(chunk.ocr_confidence),
            "seq": int(getattr(chunk, "seq", 0) or 0),
        }
    )
    return row


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


class LanceStore:
    """Embedded LanceDB store: a ``parents`` and a ``children`` table."""

    def __init__(self, db, db_path: Path) -> None:
        self._db = db
        self.path = Path(db_path)

    # -- creation / opening -------------------------------------------------

    @classmethod
    def create(cls, db_path: str | Path, mode: str = "overwrite") -> "LanceStore":
        """Create (or overwrite) a store at ``db_path``.

        Tables are created lazily on the first ``add_*`` call so the schema is
        inferred from the rows. ``mode="overwrite"`` drops existing tables.
        """
        import lancedb

        db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(db_path))
        store = cls(db, db_path)
        if mode == "overwrite":
            store.reset()
        return store

    @classmethod
    def open(cls, db_path: str | Path) -> "LanceStore":
        """Open an existing store; raises FileNotFoundError if it is incomplete."""
        import lancedb

        db_path = Path(db_path)
        db = lancedb.connect(str(db_path))
        store = cls(db, db_path)
        existing = store._table_names()
        missing = [n for n in (PARENT_TABLE, CHILD_TABLE) if n not in existing]
        if missing:
            raise FileNotFoundError(
                f"incomplete LanceDB store at {db_path}: missing table(s) {missing}"
            )
        return store

    def reset(self) -> None:
        for name in (PARENT_TABLE, CHILD_TABLE):
            if name in self._table_names():
                self._db.drop_table(name)

    # -- tables -------------------------------------------------------------

    def _table_names(self) -> set[str]:
        """Names of existing tables.

        lancedb has changed this return shape across versions: some releases
        return a plain list of names from ``list_tables()``, others return a
        response object with a ``.tables`` attribute. Handle both so a
        version bump doesn't silently break every store operation.
        """
        result = self._db.list_tables()
        names = getattr(result, "tables", result)
        return set(names)

    def _table(self, name: str):
        if name not in self._table_names():
            return None
        return self._db.open_table(name)

    def _require_table(self, name: str):
        table = self._table(name)
        if table is None:
            raise RuntimeError(
                f"table '{name}' missing at {self.path} — build the store first "
                f"(LanceStore.create + add_parents/add_children)"
            )
        return table

    def _ensure_table(self, name: str, rows: list[dict]):
        table = self._table(name)
        if table is None:
            return self._db.create_table(name, rows)
        if rows:
            table.add(rows)
        return table

    # -- writes -------------------------------------------------------------

    def add_parents(self, parents: Sequence[Chunk]) -> None:
        """Store parent chunks (no vector)."""
        if parents:
            self._ensure_table(PARENT_TABLE, [_parent_row(c) for c in parents])

    def add_children(self, children: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Store child chunks with their embedding vectors.

        ``vectors[i]`` is the 384-d embedding of ``children[i].embed_text``
        (produce with ``embeddings.embed_passages``).
        """
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 2:
            raise ValueError(f"vectors must be 2-D (n, dim), got shape {vecs.shape}")
        if len(children) != vecs.shape[0]:
            raise ValueError(
                f"children ({len(children)}) and vectors ({vecs.shape[0]}) length mismatch"
            )
        if children:
            self._ensure_table(CHILD_TABLE, [_child_row(c, vecs[i]) for i, c in enumerate(children)])

    # -- indexes (M1, M2, M3) ------------------------------------------------

    def build_indexes(self) -> None:
        """Create/refresh vector, FTS and scalar metadata indexes."""
        child = self._require_table(CHILD_TABLE)
        if child.count_rows() == 0:
            raise RuntimeError("cannot index an empty children table — add chunks first")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # M2: BM25/full-text index over the normalized lexical text.
            child.create_fts_index("text_lexical", replace=True)
            # M3: scalar metadata indexes for curriculum prefiltering.
            columns = child.schema.names
            for col in CHILD_SCALAR_COLUMNS:
                if col in columns:
                    child.create_scalar_index(col, replace=True)
        # M1: LanceDB keeps a default vector index on the "vector" column; the
        # search path below uses the table default (L2). No explicit index
        # creation needed for correctness — LanceDB builds/uses it on demand.

    # -- searches (M1 + M2 + filters) ----------------------------------------

    def search_children_vector(
        self,
        query_vector: np.ndarray,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 20,
    ) -> list[Hit]:
        """Vector search (M1) with optional metadata prefilter."""
        table = self._require_table(CHILD_TABLE)
        query = table.search(
            np.asarray(query_vector, dtype=np.float32), query_type="vector"
        ).limit(top_k)
        return _rows_to_hits(_apply_filters(query, filters).to_list(), "_distance")

    def search_children_fts(
        self,
        query_text: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 20,
    ) -> list[Hit]:
        """BM25 full-text search (M2) with optional metadata prefilter."""
        table = self._require_table(CHILD_TABLE)
        query = table.search(query_text, query_type="fts").limit(top_k)
        return _rows_to_hits(_apply_filters(query, filters).to_list(), "_score")

    def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 20,
    ) -> list[Hit]:
        """LanceDB native hybrid search (M1 + M2) with metadata prefilter."""
        table = self._require_table(CHILD_TABLE)
        query = (
            table.search(query_type="hybrid")
            .vector(np.asarray(query_vector, dtype=np.float32))
            .text(query_text)
            .limit(top_k)
        )
        return _rows_to_hits(_apply_filters(query, filters).to_list(), "_relevance_score")

    # -- parent retrieval ----------------------------------------------------

    def get_parent(self, parent_id: str) -> Optional[dict]:
        """Fetch a parent chunk row by ``parent_id`` (expansion step, U)."""
        table = self._require_table(PARENT_TABLE)
        rows = table.search().where(f"parent_id = {_quote(parent_id)}").limit(1).to_list()
        return rows[0] if rows else None

    def get_children(self, parent_id: str, limit: int = 50) -> list[dict]:
        """All child chunks of a parent (neighbor expansion, U). Ordered by
        printed page then ``seq`` (true reading order within the section) —
        sorting by raw text (the old behavior) was an alphabetic tie-break
        that didn't reflect actual document order."""
        table = self._require_table(CHILD_TABLE)
        rows = table.search().where(f"parent_id = {_quote(parent_id)}").limit(limit).to_list()
        rows.sort(key=lambda r: (r.get("page_start") or 0, r.get("seq") or 0))
        return [{k: v for k, v in r.items() if k != "vector"} for r in rows]

    # -- stats ---------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        result = {}
        for name in (PARENT_TABLE, CHILD_TABLE):
            table = self._table(name)
            result[name] = table.count_rows() if table is not None else 0
        return result
