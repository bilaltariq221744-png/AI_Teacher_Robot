"""Layer-by-layer pipeline inspector.

Verifies each stage of the RAG pipeline independently, against artifacts
that already exist on disk from a previous ``build_db.py`` run — no
rebuild/re-OCR needed. Every subcommand talks directly to the lowest-level
API available (raw files, the LanceDB store's own search methods) rather
than going through retriever.py/rag_chain.py's orchestration, so a "this
layer looks fine" verdict here can't be an artifact of a bug in a higher
layer wrapping it.

Usage:
    python inspect_pipeline.py raw   <book_stem> [--page N] [--work work]
    python inspect_pipeline.py clean <book_stem> [--page N] [--work work]
    python inspect_pipeline.py store [--db work/lancedb]
    python inspect_pipeline.py chunks [--db work/lancedb] [--n 5]
    python inspect_pipeline.py search <query> [--db work/lancedb] [--top-k 10] [--onnx-dir DIR]
    python inspect_pipeline.py compare <relevant_query> <irrelevant_query> [--db work/lancedb]

Layers, in order:
    raw     - Layer 1 (OCR/ingestion): what ingest_pdf actually extracted per page.
    clean   - Layer 2 (cleaning): what survived watermark/header-footer removal.
    store   - Layer 3+4+5 (chunking + embedding + LanceDB): table stats, schema,
              and whether stored vectors are sane (non-zero, unit-norm).
    chunks  - Sample stored chunks' text, so you can visually confirm chunking
              produced coherent, on-topic passages.
    search  - Layer 6 (retrieval), talking directly to LanceStore's own vector
              and BM25 search methods, bypassing retriever.py entirely.
    compare - Runs a clearly-relevant and a clearly-irrelevant question through
              raw retrieval side by side, to prove (or disprove) that the
              scoring is internally discriminative BEFORE touching any
              threshold constant.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import click
import numpy as np

from src.config import DEFAULT_CONFIG_PATH, Config
from src.embeddings import embed_question, load_embedder
from src.lancedb_store import LanceStore
from src.retriever import infer_filters, rrf_fuse


def _print_header(title: str) -> None:
    click.echo(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


@click.group()
def cli():
    """Layer-by-layer RAG pipeline inspector."""


@cli.command()
@click.argument("book_stem")
@click.option("--page", type=int, default=1, show_default=True, help="1-based page number to show")
@click.option("--work", default="work", show_default=True, help="work directory from build_db.py")
def raw(book_stem: str, page: int, work: str) -> None:
    """Layer 1 (OCR/ingestion): print the raw extracted text for one page."""
    p = Path(work) / "raw" / book_stem / f"page_{page:04d}.txt"
    _print_header(f"LAYER 1 - RAW OCR/embedded text | {p}")
    if not p.exists():
        click.echo(f"NOT FOUND: {p}")
        click.echo("Available pages:")
        parent = p.parent
        if parent.exists():
            for f in sorted(parent.glob("page_*.txt"))[:10]:
                click.echo(f"  {f.name}")
        return
    text = p.read_text(encoding="utf-8")
    click.echo(f"length: {len(text)} chars")
    click.echo("---")
    click.echo(text)


@cli.command()
@click.argument("book_stem")
@click.option("--page", type=int, default=1, show_default=True, help="1-based page number to show")
@click.option("--work", default="work", show_default=True, help="work directory from build_db.py")
def clean(book_stem: str, page: int, work: str) -> None:
    """Layer 2 (cleaning): print the cleaned text for one page (post
    watermark/header-footer removal) — compare against `raw` for the same
    page to see exactly what was stripped."""
    p = Path(work) / "clean" / book_stem / f"page_{page:04d}.txt"
    _print_header(f"LAYER 2 - CLEANED text | {p}")
    if not p.exists():
        click.echo(f"NOT FOUND: {p}")
        return
    text = p.read_text(encoding="utf-8")
    click.echo(f"length: {len(text)} chars")
    click.echo("---")
    click.echo(text)


@cli.command()
@click.option("--db", default=None, help="store path (default: resolved from config.yaml)")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH))
def store(db: str | None, config_path: str) -> None:
    """Layer 3+4+5 (chunking + embedding + storage): table stats, schema,
    and a sanity check that stored vectors are real (non-zero, unit-norm) —
    a silent embedding failure would show up here as all-zero vectors."""
    cfg = Config.load(config_path)
    db_path = db or cfg.pi.get("db_path") or cfg.pc.get("db_path", "work/lancedb")
    if not Path(db_path).exists():
        db_path = cfg.pc.get("db_path", "work/lancedb")
    _print_header(f"LAYER 3+4+5 - STORE | {db_path}")
    st = LanceStore.open(db_path)
    click.echo(f"stats: {st.stats()}")

    child = st._require_table("children")
    click.echo(f"children columns: {child.schema.names}")

    rows = child.search().limit(5).to_list()
    click.echo(f"\nsample of {len(rows)} child row(s):")
    for r in rows:
        vec = np.asarray(r.get("vector", []), dtype=np.float32)
        norm = float(np.linalg.norm(vec)) if vec.size else 0.0
        click.echo(
            f"  chunk_id={r.get('chunk_id')} unit={r.get('unit')!r} "
            f"chapter={r.get('chapter')!r} page={r.get('page_start')}-{r.get('page_end')} "
            f"vector_dim={vec.size} vector_norm={norm:.4f} (should be ~1.0, NOT 0.0)"
        )
        if norm < 0.5:
            click.echo("    *** WARNING: near-zero vector norm — embedding likely failed for this row ***")


@cli.command()
@click.option("--db", default=None, help="store path (default: resolved from config.yaml)")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH))
@click.option("--n", default=5, show_default=True, help="how many sample chunks to print")
def chunks(db: str | None, config_path: str, n: int) -> None:
    """Print full text of N sample stored chunks, so you can visually
    confirm chunking produced coherent, on-topic passages (not scrambled
    OCR fragments or garbled headings)."""
    cfg = Config.load(config_path)
    db_path = db or cfg.pi.get("db_path") or cfg.pc.get("db_path", "work/lancedb")
    if not Path(db_path).exists():
        db_path = cfg.pc.get("db_path", "work/lancedb")
    _print_header(f"SAMPLE STORED CHUNKS | {db_path}")
    st = LanceStore.open(db_path)
    child = st._require_table("children")
    rows = child.search().limit(n).to_list()
    for i, r in enumerate(rows, 1):
        click.echo(f"\n[{i}] chunk_id={r.get('chunk_id')}")
        click.echo(f"    unit={r.get('unit')!r} chapter={r.get('chapter')!r} "
                   f"page={r.get('page_start')}-{r.get('page_end')}")
        click.echo(f"    text_clean: {r.get('text_clean')!r}")


@cli.command()
@click.argument("query")
@click.option("--db", default=None, help="store path (default: resolved from config.yaml)")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH))
@click.option("--top-k", default=10, show_default=True)
@click.option("--onnx-dir", default=None, help="ONNX embedder dir (default: sentence-transformers)")
def search(query: str, db: str | None, config_path: str, top_k: int, onnx_dir: str | None) -> None:
    """Layer 6 (retrieval), talking DIRECTLY to LanceStore's vector and BM25
    search — bypasses retriever.py's RRF fusion/gating entirely, so you can
    see the two raw signals before anything combines or filters them."""
    cfg = Config.load(config_path)
    db_path = db or cfg.pi.get("db_path") or cfg.pc.get("db_path", "work/lancedb")
    if not Path(db_path).exists():
        db_path = cfg.pc.get("db_path", "work/lancedb")
    st = LanceStore.open(db_path)
    backend = load_embedder(onnx_dir=onnx_dir, model_name=cfg.pc.get("embed_model"))

    filters = infer_filters(query, None)
    _print_header(f"LAYER 6 - RAW RETRIEVAL for: {query!r}")
    click.echo(f"inferred metadata filters: {filters or '(none)'}")

    query_vec = embed_question(query, backend)
    vec_hits = st.search_children_vector(query_vec, filters=filters or None, top_k=top_k)
    click.echo(f"\n--- vector search: {len(vec_hits)} hit(s) (L2 distance, LOWER = more similar) ---")
    for h in vec_hits:
        snippet = h.text_clean[:100].replace("\n", " ")
        click.echo(f"  distance={h.score:.5f} chunk={h.chunk_id[:12]}... {snippet!r}")

    bm25_hits = st.search_children_fts(query, filters=filters or None, top_k=top_k)
    click.echo(f"\n--- BM25 search: {len(bm25_hits)} hit(s) (higher = more similar) ---")
    for h in bm25_hits:
        snippet = h.text_clean[:100].replace("\n", " ")
        click.echo(f"  score={h.score:.5f} chunk={h.chunk_id[:12]}... {snippet!r}")

    fused = rrf_fuse(vec_hits, bm25_hits, k=int(cfg.pi.get("rrf_k", 60)))
    click.echo(f"\n--- RRF fused: {len(fused)} unique hit(s) (this is what the threshold gate compares against) ---")
    for f in fused[:top_k]:
        snippet = f.text_clean[:100].replace("\n", " ")
        click.echo(f"  rrf_score={f.score:.5f} chunk={f.chunk_id[:12]}... {snippet!r}")


@cli.command()
@click.argument("relevant_query")
@click.argument("irrelevant_query")
@click.option("--db", default=None, help="store path (default: resolved from config.yaml)")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH))
@click.option("--onnx-dir", default=None, help="ONNX embedder dir (default: sentence-transformers)")
def compare(relevant_query: str, irrelevant_query: str, db: str | None, config_path: str, onnx_dir: str | None) -> None:
    """Run a clearly-relevant and a clearly-irrelevant question through raw
    retrieval side by side. The point: prove whether the RRF scoring is
    internally discriminative (relevant scores meaningfully higher than
    irrelevant) BEFORE deciding what the threshold constant should be. If
    relevant and irrelevant score about the same, that's a real retrieval
    problem, not a threshold problem — this command tells you which one
    you actually have.
    """
    cfg = Config.load(config_path)
    db_path = db or cfg.pi.get("db_path") or cfg.pc.get("db_path", "work/lancedb")
    if not Path(db_path).exists():
        db_path = cfg.pc.get("db_path", "work/lancedb")
    st = LanceStore.open(db_path)
    backend = load_embedder(onnx_dir=onnx_dir, model_name=cfg.pc.get("embed_model"))
    rrf_k = int(cfg.pi.get("rrf_k", 60))

    def top_score(q: str) -> tuple[float, str]:
        filters = infer_filters(q, None) or None
        qv = embed_question(q, backend)
        vh = st.search_children_vector(qv, filters=filters, top_k=20)
        bh = st.search_children_fts(q, filters=filters, top_k=20)
        fused = rrf_fuse(vh, bh, k=rrf_k)
        if not fused:
            return 0.0, "(no hits at all)"
        return fused[0].score, fused[0].text_clean[:100].replace("\n", " ")

    rel_score, rel_text = top_score(relevant_query)
    irr_score, irr_text = top_score(irrelevant_query)

    _print_header("COMPARE: is retrieval actually discriminative?")
    click.echo(f"RELEVANT   {relevant_query!r}")
    click.echo(f"  top RRF score: {rel_score:.5f}")
    click.echo(f"  best match: {rel_text!r}")
    click.echo(f"\nIRRELEVANT {irrelevant_query!r}")
    click.echo(f"  top RRF score: {irr_score:.5f}")
    click.echo(f"  best match: {irr_text!r}")

    click.echo(f"\n{'-' * 70}")
    if rel_score <= irr_score:
        click.echo(
            "VERDICT: relevant did NOT score higher than irrelevant. This is a "
            "real retrieval-quality problem (bad embeddings, bad chunking, or "
            "a genuine content gap) — not something a threshold change fixes."
        )
    elif rel_score < 0.1:
        click.echo(
            f"VERDICT: relevant DOES score meaningfully higher ({rel_score:.5f} vs "
            f"{irr_score:.5f}) — the retrieval math is discriminating correctly. "
            "Both scores are small because that is how RRF fusion scores work "
            "(bounded well under 1.0 by construction), not because of a bug. "
            f"The fix here is calibrating relevance_threshold to sit between "
            f"{irr_score:.5f} and {rel_score:.5f} for this book — "
            "run: python src/eval.py --db <store> --apply"
        )
    else:
        click.echo("VERDICT: relevant scores higher and is not unusually small — looks healthy.")


if __name__ == "__main__":
    cli()