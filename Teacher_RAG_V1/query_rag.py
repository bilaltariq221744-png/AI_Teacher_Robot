"""Raspberry Pi 5 runtime (Phase 11: wired retriever + rag_chain).

Also runnable on the PC for local testing. Phase 10 (the deployment bundle /
scripts/deploy_to_pi.py) hasn't landed yet, so this includes a laptop-testing
fallback: if the configured Pi paths (``pi.db_path``, ``pi.onnx_model``)
don't exist on this machine, it falls back to the PC build output
(``pc.db_path``) and the sentence-transformers embedder instead of failing
outright. The deployed Pi runtime should still use the ONNX INT8 model and
the bundled store — this fallback exists purely so the chain is testable
before a bundle exists.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import click

from src.config import DEFAULT_CONFIG_PATH, Config
from src.embeddings import load_embedder
from src.lancedb_store import LanceStore
from src.rag_chain import answer
from src.retriever import ContextBlock, retrieve, retrieve_debug


# ---------------------------------------------------------------------------
# Resolution helpers (pure functions, unit-testable without a live store/CLI)
# ---------------------------------------------------------------------------


def _is_valid_store(path: Path) -> bool:
    """True only if ``path`` contains a real, complete LanceDB store (both
    the parents and children tables) — not just an empty directory.

    ``lancedb.connect()`` silently creates the target directory as a side
    effect of connecting, even when no tables exist yet. A bare
    ``Path.exists()`` check therefore false-positives on a stale empty
    directory left behind by an earlier failed or exploratory connection
    attempt (e.g. someone ran the tool once against the placeholder
    ``pi.db_path`` before a real store was ever built there) — confirmed as
    a real bug via testing. Actually attempting to open the store is the
    only reliable check.
    """
    if not path.exists():
        return False
    try:
        LanceStore.open(path)
        return True
    except Exception:
        return False


def resolve_db_path(cfg: Config, db_override: Optional[str] = None) -> Path:
    """Pick the LanceDB store path.

    Prefers an explicit ``--db`` override, then ``pi.db_path`` (the deployed
    Pi location). Falls back to ``pc.db_path`` (where ``build_db.py`` writes
    locally) when the Pi path isn't a real, complete store on this machine —
    the common case before a Phase 10 deployment bundle exists.
    """
    if db_override:
        return Path(db_override)
    pi_path = Path(cfg.pi.get("db_path", ""))
    if _is_valid_store(pi_path):
        return pi_path
    pc_path = Path(cfg.pc.get("db_path", "work/lancedb"))
    if _is_valid_store(pc_path):
        return pc_path
    return pi_path  # doesn't exist either; LanceStore.open will raise a clear error


def resolve_backend_source(
    cfg: Config, onnx_dir: Optional[str] = None
) -> tuple[str, Optional[str]]:
    """Decide which embedding backend to use.

    Returns ``(kind, path_or_none)`` where ``kind`` is ``"onnx"`` or
    ``"sentence-transformers"``. Same local-testing fallback as
    ``resolve_db_path``: prefers an explicit override, then the configured
    Pi ONNX model if it exists on this machine, else falls back to the PC
    sentence-transformers backend.
    """
    if onnx_dir:
        return "onnx", onnx_dir
    pi_onnx = cfg.pi.get("onnx_model")
    if pi_onnx and Path(pi_onnx).exists():
        return "onnx", pi_onnx
    return "sentence-transformers", None


def load_backend(cfg: Config, onnx_dir: Optional[str] = None, device: Optional[str] = None):
    """Load the resolved embedding backend (see ``resolve_backend_source``)."""
    kind, path = resolve_backend_source(cfg, onnx_dir)
    if kind == "onnx":
        return load_embedder(onnx_dir=path)
    return load_embedder(model_name=cfg.pc.get("embed_model"), device=device)


# ---------------------------------------------------------------------------
# Console formatting
# ---------------------------------------------------------------------------


def format_sources(sources: list[dict]) -> str:
    """Render an Answer's sources as short citation lines for the console."""
    lines = []
    for s in sources:
        start, end = s.get("page_start"), s.get("page_end")
        if start is not None and end is not None and end != start:
            pages = f"page {start}-{end}"
        elif start is not None:
            pages = f"page {start}"
        else:
            pages = "page ?"
        where = " / ".join(
            p for p in (s.get("unit", ""), s.get("chapter", ""), s.get("topic", "")) if p
        )
        lines.append(f"  - {pages}" + (f" ({where})" if where else ""))
    return "\n".join(lines)


def format_blocks(blocks: list[ContextBlock]) -> str:
    """Render raw retrieved context blocks (``--no-llm`` inspection mode)."""
    if not blocks:
        return "  (nothing retrieved)"
    lines = []
    for i, b in enumerate(blocks, 1):
        meta = b.metadata
        start, end = meta.get("page_start"), meta.get("page_end")
        if start is not None and end is not None and end != start:
            pages = f"page {start}-{end}"
        elif start is not None:
            pages = f"page {start}"
        else:
            pages = "page ?"
        snippet = b.child_text.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        lines.append(f"  [{i}] score={b.score:.4f} {pages}\n      {snippet}")
    return "\n".join(lines)


def format_diagnostics(diag: dict) -> str:
    """Render retrieve_debug()'s diagnostics dict as a readable pipeline
    trace: what each stage (Q -> V) actually saw, so it's obvious which
    stage is responsible when nothing comes back.
    """
    lines = ["--- retrieval debug trace ---"]
    lines.append(f"  language detected: {diag['language']}")
    lines.append(f"  metadata filters inferred: {diag['filters'] or '(none)'}")
    if diag["filter_fallback_used"]:
        lines.append("  filter-fallback: YES (filtered search found nothing, retried unfiltered)")
    lines.append(
        f"  vector search: {diag['vector_hit_count']} hit(s), "
        f"top score {diag['vector_top_score']} (L2 distance, LOWER is better)"
    )
    lines.append(
        f"  BM25 search:   {diag['bm25_hit_count']} hit(s), "
        f"top score {diag['bm25_top_score']} (higher is better)"
    )
    lines.append(f"  RRF fused:     {diag['fused_count']} unique hit(s), top score {diag['fused_top_score']}")
    lines.append(
        f"  after parent/sibling expansion: {diag['expanded_count']} block(s), "
        f"scores {diag['expanded_scores']}"
    )
    threshold = diag["relevance_threshold"]
    lines.append(f"  relevance_threshold in effect: {threshold}")
    lines.append(f"  blocks that cleared the gate: {diag['gated_count']}")

    if diag["expanded_count"] == 0:
        lines.append(
            "  => VERDICT: nothing matched at all (no vector or BM25 hits). "
            "Check the store actually has content for this topic, or that "
            "OCR/chunking produced real text for the relevant pages."
        )
    elif diag["gated_count"] == 0 and threshold is not None:
        best = max(diag["expanded_scores"]) if diag["expanded_scores"] else 0.0
        lines.append(
            f"  => VERDICT: {diag['expanded_count']} candidate(s) were found (best score "
            f"{best}), but relevance_threshold={threshold} is higher than any of them, so "
            "everything was gated out. RRF scores are typically small (roughly 0.01-0.04) "
            "— a threshold like the shipped placeholder 0.55 will decline almost everything. "
            "Try --threshold 0 to see raw matches, or calibrate a real threshold for this "
            "book with: python src/eval.py --db <store> --apply"
        )
    elif diag["gated_count"] > 0:
        lines.append("  => VERDICT: retrieval succeeded, answer should be grounded in real content.")
    return "\n".join(lines)


def ask(question: str, store, backend, cfg: dict, no_llm: bool = False, debug: bool = False) -> str:
    """Answer one question; returns the text to print to the console.

    ``no_llm`` skips ``rag_chain`` entirely and shows raw retrieved passages
    — useful for inspecting retrieval quality without a running Ollama.
    ``debug`` prints a full pipeline trace (see ``format_diagnostics``)
    before the answer/passages, so it's clear which stage produced the
    result — most commonly useful when something unexpectedly declines.
    """
    out_parts = []
    if debug:
        _debug_blocks, diag = retrieve_debug(question, store, backend, cfg)
        out_parts.append(format_diagnostics(diag))
        out_parts.append("")  # blank line separator

    if no_llm:
        blocks = retrieve(question, store, backend, cfg)
        if not blocks:
            out_parts.append("This is not covered in the provided book. (no context retrieved)")
        else:
            out_parts.append("Retrieved passages:\n" + format_blocks(blocks))
        return "\n".join(out_parts)

    result = answer(question, store, backend, cfg)
    if result.declined:
        out_parts.append(result.text)
        return "\n".join(out_parts)
    out = result.text
    if result.sources:
        out += "\n\nSources:\n" + format_sources(result.sources)
    if result.error:
        out += "\n\n(note: this was a connection/engine error, not a content decline)"
    out_parts.append(out)
    return "\n".join(out_parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.argument("question", required=False)
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), show_default=True)
@click.option(
    "--db", "db_override", default=None,
    help="Override the LanceDB store path (defaults to pi.db_path, falling back to "
    "pc.db_path for local testing before a Pi deployment bundle exists)",
)
@click.option(
    "--onnx-dir", default=None,
    help="Override the ONNX embedder directory (defaults to pi.onnx_model, falling "
    "back to the PC sentence-transformers backend for local testing)",
)
@click.option("--device", default=None, help="torch device for the sentence-transformers fallback backend")
@click.option(
    "--threshold", "threshold_override", type=float, default=None,
    help="Override pi.relevance_threshold (e.g. 0 to see all retrieval results). "
    "Useful for local testing before the threshold has been calibrated with src/eval.py.",
)
@click.option(
    "--no-llm", is_flag=True, default=False,
    help="Skip the LLM call; just show retrieved passages (no Ollama needed)",
)
@click.option(
    "--debug", is_flag=True, default=False,
    help="Print a full retrieval pipeline trace before the answer/passages — language "
    "detected, vector/BM25 hit counts and scores, RRF fusion, and exactly why the "
    "relevance gate did or didn't pass. Use this whenever a question unexpectedly declines.",
)
def main(
    question: str | None,
    config_path: str,
    db_override: str | None,
    onnx_dir: str | None,
    device: str | None,
    threshold_override: float | None,
    no_llm: bool,
    debug: bool,
) -> None:
    """Answer a student question using the built LanceDB store + Ollama/Qwen.

    Run with no QUESTION for an interactive loop; type 'quit' or 'exit' (or
    press Ctrl+D) to leave.
    """
    cfg = Config.load(config_path)
    pi = dict(cfg.pi)
    if threshold_override is not None:
        pi["relevance_threshold"] = threshold_override

    db_path = resolve_db_path(cfg, db_override)
    try:
        store = LanceStore.open(db_path)
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"{exc}\n"
            "Build a store first: python build_db.py <pdf-or-dir>\n"
            "(then pass --db work/lancedb here, or point config.yaml's pc.db_path/pi.db_path at it)."
        )

    backend_kind, backend_path = resolve_backend_source(cfg, onnx_dir)
    try:
        backend = load_backend(cfg, onnx_dir, device)
    except ImportError as exc:
        raise click.ClickException(
            "embedding backend unavailable — install sentence-transformers "
            f"(or pass --onnx-dir to a built ONNX model) ({exc})"
        )

    llm_cfg = pi.get("llm", {})
    click.echo(f"teacher_rag runtime | db={db_path}")
    if backend_kind == "onnx":
        click.echo(f"  embedder: onnx ({backend_path})")
    else:
        click.echo(
            "  embedder: sentence-transformers (no ONNX model found — this is the "
            "local-testing fallback; the deployed Pi runtime uses ONNX INT8)"
        )
    if not no_llm:
        click.echo(f"  llm: {llm_cfg.get('provider')} / {llm_cfg.get('model')}")
    click.echo(
        f"  hybrid search: vector_top_k={pi.get('vector_top_k')} "
        f"bm25_top_k={pi.get('bm25_top_k')} threshold={pi.get('relevance_threshold')}"
    )

    if question:
        click.echo()
        click.echo(ask(question, store, backend, pi, no_llm=no_llm, debug=debug))
        return

    click.echo("\nInteractive mode — type a question, or 'quit'/'exit' to leave.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo()
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        click.echo(ask(q, store, backend, pi, no_llm=no_llm, debug=debug))
        click.echo()


if __name__ == "__main__":
    main()