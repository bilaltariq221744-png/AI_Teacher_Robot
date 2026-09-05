"""Hybrid retrieval runtime (SVG Q-V; PDF §8).

Implements the PDF §8 query flow verbatim:

1. ``detect_language`` / ``resolve_language_curriculum`` (Q): question
   language (Urdu script / Roman Urdu cues / English) and curriculum metadata
   filters inferred from the question.
2. ``embed_query`` (R): ``query:``-prefixed question embedding.
3. Vector search top-20 (S, M1) and BM25 top-20 (S, M2), both metadata-prefiltered.
   If the inferred filters return nothing from either search, they are
   retried once unfiltered before giving up (see ``retrieve``) — a bad
   grade/subject/book guess should never cause a false decline on its own.
4. ``rrf_fuse`` (T): reciprocal-rank fusion + deduplication.
5. ``dedupe_and_expand`` (U): parent chunk + sibling (neighbor) expansion.
6. ``gate_context`` (V): context budget (800-1200 tokens) + relevance threshold;
   an empty result means the caller should decline (see DECLINE_MESSAGE).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .embeddings import embed_question

# Deterministic decline answer (PDF §8, rule 10).
DECLINE_MESSAGE = "This is not covered in the provided book."

_URDU_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")
_ROMAN_URDU_CUES = {
    "kya", "hai", "hain", "ka", "ki", "ke", "kahan", "kaise", "kyun", "kis",
    "kyaa", "mein", "tha", "thi", "the", "nahi", "na", "batao", "bataye",
    "samjhao", "karna", "hota", "hoti", "karne", "karo",
}
_STRONG_ROMAN_URDU = {"kya", "hai", "hain", "kahan", "kaise", "kyun"}

_GRADE_RE = re.compile(r"\bgrade\s*([0-9]+)\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"\b(unit|chapter|lesson|section|module|topic)\s*([0-9]+)\b", re.IGNORECASE)
_SUBJECTS = [
    "english", "urdu", "science", "mathematics", "maths", "math",
    "social studies", "islamiat", "general knowledge", "computer",
]


@dataclass
class FusedHit:
    """A deduplicated hit carrying its RRF score."""

    chunk_id: str
    parent_id: str
    text_clean: str
    score: float
    row: dict


@dataclass
class ContextBlock:
    """One context unit for the LLM: matched child + parent + sibling neighbors."""

    chunk_id: str
    parent_id: str
    score: float
    child_text: str
    parent_text: Optional[str] = None
    siblings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        parts = [self.child_text]
        if self.parent_text:
            parts.append(self.parent_text)
        parts.extend(self.siblings)
        return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Q - language and curriculum resolution
# ---------------------------------------------------------------------------


def detect_language(question: str) -> str:
    """Detect the question language: ``urdu`` (script), ``roman_urdu``, else
    ``english`` (covers mixed English queries)."""
    q = question or ""
    if _URDU_SCRIPT_RE.search(q):
        return "urdu"
    words = set(re.findall(r"[a-z']+", q.lower()))
    if words & _STRONG_ROMAN_URDU:
        return "roman_urdu"
    if len(words & _ROMAN_URDU_CUES) >= 2:
        return "roman_urdu"
    return "english"


def infer_filters(question: str, curriculum: Optional[dict] = None) -> dict[str, str]:
    """Infer metadata filters from the question (grade/subject/book/unit/...).

    ``curriculum`` is the loaded curriculum.yaml dict (``{books: [...]}``)
    used to recognize book titles; grade/subject/unit/chapter/topic are
    extracted by pattern. Filters match the canonical metadata labels stored
    in LanceDB (e.g. "Unit 3").
    """
    filters: dict[str, str] = {}
    q = question or ""

    m = _GRADE_RE.search(q)
    if m:
        filters["grade"] = m.group(1)

    m = _SECTION_RE.search(q)
    if m:
        label = m.group(1).lower()
        key = "chapter" if label in ("chapter", "lesson", "section", "module") else label
        filters[key] = f"{label.capitalize()} {m.group(2)}"

    ql = q.lower()
    for subject in _SUBJECTS:
        if re.search(rf"\b{re.escape(subject)}\b", ql):
            filters["subject"] = subject.title()
            break

    for book in (curriculum or {}).get("books", []) or []:
        name = str(book.get("book", ""))
        if name and name.lower() in ql:
            filters["book"] = name
            break
    return filters


def resolve_language_curriculum(
    question: str, store=None, curriculum: Optional[dict] = None
) -> tuple[str, dict[str, str]]:
    """(Q) Returns ``(language, metadata_filters)`` for a student question."""
    return detect_language(question), infer_filters(question, curriculum)


# ---------------------------------------------------------------------------
# R - question embedding
# ---------------------------------------------------------------------------


def embed_query(text: str, backend) -> np.ndarray:
    """Embed a runtime question with the ``query:`` prefix (R)."""
    return embed_question(text, backend)


# ---------------------------------------------------------------------------
# T - RRF fusion
# ---------------------------------------------------------------------------


def rrf_fuse(vector_hits: Sequence, bm25_hits: Sequence, k: int = 60) -> list[FusedHit]:
    """Reciprocal rank fusion (T): ``score = sum(1 / (k + rank))`` over the
    vector and BM25 result lists; duplicates merge, ties break by chunk id."""
    scores: dict[str, float] = {}
    first: dict[str, Any] = {}
    for source in (vector_hits, bm25_hits):
        for rank, hit in enumerate(source, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            first.setdefault(hit.chunk_id, hit)
    fused = []
    for chunk_id in sorted(scores, key=lambda c: (-scores[c], c)):
        hit = first[chunk_id]
        fused.append(
            FusedHit(
                chunk_id=chunk_id,
                parent_id=hit.parent_id,
                text_clean=hit.text_clean,
                score=scores[chunk_id],
                row=hit.row,
            )
        )
    return fused


# ---------------------------------------------------------------------------
# U - deduplication and parent/neighbor expansion
# ---------------------------------------------------------------------------


def dedupe_and_expand(fused: Sequence[FusedHit], store) -> list[ContextBlock]:
    """(U) Turn fused hits into context blocks: dedupe by chunk id, then
    attach the parent chunk and sibling (neighbor) children."""
    blocks: list[ContextBlock] = []
    seen: set[str] = set()
    for hit in fused:
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        parent = store.get_parent(hit.parent_id)
        siblings = [
            s["text_clean"]
            for s in store.get_children(hit.parent_id)
            if s["chunk_id"] != hit.chunk_id
        ]
        blocks.append(
            ContextBlock(
                chunk_id=hit.chunk_id,
                parent_id=hit.parent_id,
                score=hit.score,
                child_text=hit.text_clean,
                parent_text=parent["text_clean"] if parent else None,
                siblings=siblings,
                metadata=hit.row,
            )
        )
    return blocks


# ---------------------------------------------------------------------------
# V - context budget and relevance gate
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for budget accounting."""
    return max(1, len(text or "") // 4)


def gate_context(
    blocks: Sequence[ContextBlock],
    budget_tokens: int = 1200,
    max_blocks: int = 5,
    min_score: Optional[float] = None,
) -> list[ContextBlock]:
    """(V) Keep the top ``max_blocks`` blocks that fit ``budget_tokens``.

    When ``min_score`` is set (calibrated threshold), blocks below it are
    dropped entirely — including the top block. This is intentional: a
    teacher robot answering confidently from a weak, below-threshold match
    is a worse failure than declining, so nothing overrides the gate. (An
    earlier version of this docstring claimed the top block was always kept
    regardless of score; that was never actually implemented, and the
    stricter behavior — decline when nothing clears the bar — is the one we
    keep.) An empty result means the caller should decline.
    """
    if not blocks:
        return []
    ordered = sorted(blocks, key=lambda b: -b.score)
    if min_score is not None:
        ordered = [b for b in ordered if b.score >= min_score]
    kept: list[ContextBlock] = []
    used = 0
    for block in ordered[:max_blocks]:
        est = estimate_tokens(block.text)
        if kept and used + est > budget_tokens:
            break
        kept.append(block)
        used += est
    return kept


# ---------------------------------------------------------------------------
# Full flow (PDF §8, steps 1-8)
# ---------------------------------------------------------------------------


def _retrieve_impl(
    question: str, store, backend, cfg: Optional[dict] = None
) -> tuple[list[ContextBlock], dict]:
    """Shared implementation for ``retrieve()`` and ``retrieve_debug()``.

    Runs the exact same Q -> V pipeline once and returns both the final
    context blocks and a diagnostics dict describing every stage, so
    ``retrieve()``'s public behavior can never drift from what
    ``retrieve_debug()`` reports.
    """
    cfg = cfg or {}
    vector_top_k = int(cfg.get("vector_top_k", 20))
    bm25_top_k = int(cfg.get("bm25_top_k", 20))
    rrf_k = int(cfg.get("rrf_k", 60))
    max_blocks = int(cfg.get("final_context_blocks", 5))
    budget_tokens = int(cfg.get("context_tokens", 1200))
    threshold = cfg.get("relevance_threshold")

    lang, filters = resolve_language_curriculum(question, store, cfg.get("curriculum"))
    query_vec = embed_query(question, backend)

    vector_hits = store.search_children_vector(query_vec, filters=filters, top_k=vector_top_k)
    bm25_hits = store.search_children_fts(question, filters=filters, top_k=bm25_top_k)

    # Filter-fallback: inferred metadata filters (grade/subject/book/unit...)
    # are a best-effort guess from the question text. If they don't exactly
    # match the stored curriculum labels, they can zero out both result
    # lists and cause a false decline even though the book covers the
    # question. Retry once, unfiltered, before accepting an empty result.
    fallback_used = False
    if filters and not vector_hits and not bm25_hits:
        fallback_used = True
        vector_hits = store.search_children_vector(query_vec, filters=None, top_k=vector_top_k)
        bm25_hits = store.search_children_fts(question, filters=None, top_k=bm25_top_k)

    fused = rrf_fuse(vector_hits, bm25_hits, k=rrf_k)
    blocks = dedupe_and_expand(fused, store)
    gated = gate_context(
        blocks, budget_tokens=budget_tokens, max_blocks=max_blocks, min_score=threshold
    )

    diagnostics = {
        "language": lang,
        "filters": filters,
        "filter_fallback_used": fallback_used,
        "vector_hit_count": len(vector_hits),
        "vector_top_score": round(vector_hits[0].score, 5) if vector_hits else None,
        "bm25_hit_count": len(bm25_hits),
        "bm25_top_score": round(bm25_hits[0].score, 5) if bm25_hits else None,
        "fused_count": len(fused),
        "fused_top_score": round(fused[0].score, 5) if fused else None,
        "expanded_count": len(blocks),
        "expanded_scores": [round(b.score, 5) for b in blocks],
        "relevance_threshold": threshold,
        "gated_count": len(gated),
    }
    return gated, diagnostics


def retrieve(question: str, store, backend, cfg: Optional[dict] = None) -> list[ContextBlock]:
    """Run the full retrieval pipeline (Q -> V) for a student question.

    Returns context blocks (possibly empty — the caller answers with
    ``DECLINE_MESSAGE`` when empty). ``cfg`` uses the ``pi`` section keys:
    ``vector_top_k``, ``bm25_top_k``, ``rrf_k``, ``final_context_blocks``,
    ``context_tokens``, ``relevance_threshold`` (None until calibrated).

    See ``retrieve_debug`` for the same pipeline with per-stage diagnostics
    (useful when retrieval unexpectedly returns nothing).
    """
    blocks, _diagnostics = _retrieve_impl(question, store, backend, cfg)
    return blocks


def retrieve_debug(
    question: str, store, backend, cfg: Optional[dict] = None
) -> tuple[list[ContextBlock], dict]:
    """Same pipeline as ``retrieve()``, plus a diagnostics dict covering
    every stage (Q -> V): detected language and inferred metadata filters,
    whether the filter-fallback retry fired, vector/BM25 hit counts and top
    scores, the fused RRF count/top score, how many blocks survived parent/
    sibling expansion and their scores, the relevance threshold actually
    used, and how many blocks cleared the gate.

    This exists because "why did it decline?" has several very different
    possible causes (no matches at all, a bad metadata filter, or — the
    most common one — an uncalibrated ``relevance_threshold`` sitting above
    where real RRF scores ever land) and the plain ``retrieve()`` return
    value can't distinguish between them. ``query_rag.py --debug`` uses
    this directly.
    """
    return _retrieve_impl(question, store, backend, cfg)