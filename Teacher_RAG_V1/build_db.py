"""PC pipeline orchestrator (Phases 0-8).

Usage:
    python build_db.py <pdf-file-or-directory> [--config config.yaml] [--no-ocr] [--no-store]

Runs the full build: ingest -> clean -> structure parse -> parent-child chunk
-> embed children -> LanceDB store (parents + children + indexes).

The embedding step needs sentence-transformers + the model (installed via
requirements.txt). Use --no-store to run the text pipeline only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import click

from src.chunking import (
    build_parent_child_chunks,
    flatten_chunks,
    load_curriculum,
    match_book,
    parse_curriculum,
)
from src.cleaning import clean_book
from src.config import DEFAULT_CONFIG_PATH, Config
from src.embeddings import embed_passages, load_embedder
from src.lancedb_store import LanceStore
from src.ocr import PageRecord, ingest_pdf


def _collect_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise click.BadParameter(f"not a PDF: {input_path}")
        return [input_path]
    return sorted(p for p in input_path.rglob("*.pdf") if p.is_file())


def _write_page_record(rec: PageRecord, work_dir: Path) -> None:
    book = Path(rec.pdf_path).stem
    for kind, text in (("raw", rec.raw_text), ("clean", rec.clean_text)):
        out_dir = work_dir / kind / book
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"page_{rec.page_number:04d}.txt").write_text(text, encoding="utf-8")


def _build_store(pc: dict, parents: list, children: list) -> None:
    """Embed children and write the LanceDB store with indexes (M1-M3)."""
    try:
        backend = load_embedder(model_name=pc.get("embed_model"))
    except ImportError as exc:
        raise click.ClickException(
            "sentence-transformers is not installed — run `pip install -r requirements.txt` "
            f"to build the store ({exc})"
        )

    db_path = Path(pc.get("db_path", "work/lancedb"))
    click.echo(f"embedding {len(children)} child chunk(s) with {pc.get('embed_model')} ...")
    try:
        vectors = embed_passages([c.embed_text for c in children], backend)
    except ImportError as exc:
        raise click.ClickException(
            "sentence-transformers is not installed — run `pip install -r requirements.txt` "
            f"to build the store ({exc})"
        )
    except (RuntimeError, OSError) as exc:
        raise click.ClickException(f"embedding failed: {exc}")

    store = LanceStore.create(db_path, mode="overwrite")
    store.add_parents(parents)
    store.add_children(children, vectors)
    store.build_indexes()
    click.echo(f"LanceDB store written to {db_path}: {store.stats()}")


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--config", "config_path", default=str(DEFAULT_CONFIG_PATH), show_default=True,
    help="Shared config.yaml",
)
@click.option("--no-ocr", is_flag=True, default=False, help="Embedded-text-only mode (skips OCR)")
@click.option("--no-store", is_flag=True, default=False, help="Skip the LanceDB store build")
def main(input: str, config_path: str, no_ocr: bool, no_store: bool) -> None:
    """Build the database. INPUT is a PDF file or a directory of PDFs."""
    cfg = Config.load(config_path)
    pc = cfg.pc
    if no_ocr:
        pc.setdefault("ocr", {})["enabled"] = False

    pdfs = _collect_pdfs(Path(input))
    if not pdfs:
        raise click.ClickException("No PDF files found.")

    work_dir = Path(pc.get("work_dir", "work"))
    ocr_enabled = pc.get("ocr", {}).get("enabled", True)

    curriculum_path = Path(pc.get("curriculum", "config/curriculum.yaml"))
    books = load_curriculum(curriculum_path).get("books", []) if curriculum_path.exists() else []

    click.echo(
        f"teacher_rag builder | dpi={pc.get('dpi')} langs={pc.get('ocr_langs')} "
        f"ocr_enabled={ocr_enabled}"
    )
    watermark_keywords = list((pc.get("watermark") or {}).get("keywords", []) or [])
    ignore_heading_keywords = list((pc.get("chunk") or {}).get("ignore_heading_keywords", []) or [])

    methods: dict[str, int] = {}
    total_watermarks_removed = 0
    all_parents: list = []
    all_children: list = []
    for pdf in pdfs:
        click.echo(f"  ingesting + cleaning {pdf.name} ...", nl=False)
        records = clean_book(ingest_pdf(pdf, pc), pc)
        for rec in records:
            _write_page_record(rec, work_dir)
            methods[rec.extraction_method] = methods.get(rec.extraction_method, 0) + 1
        mean_conf = sum(r.ocr_confidence for r in records) / max(1, len(records))
        page_methods = sorted({r.extraction_method for r in records})
        book_watermarks = sum(getattr(r, "watermark_removed_count", 0) for r in records)
        total_watermarks_removed += book_watermarks
        click.echo(f" {len(records)} page(s), methods {page_methods}, mean OCR conf {mean_conf:.2f}")
        click.echo(f"    watermark/artifact lines removed: {book_watermarks}")

        book_meta = match_book(pdf.stem, books) if books else None
        if book_meta is None:
            click.echo(f"  (no curriculum entry for '{pdf.stem}' — structure/chunking skipped)")
            continue
        structured = parse_curriculum(
            records, book_meta,
            watermark_keywords=watermark_keywords,
            ignore_heading_keywords=ignore_heading_keywords,
        )
        units = build_parent_child_chunks(
            structured.sections, records, cfg=pc, page_map=structured.page_map
        )
        parents, children = flatten_chunks(units)
        click.echo(
            f"  structure: {len(structured.sections)} section(s) -> "
            f"{len(parents)} parent(s), {len(children)} child chunk(s)"
        )
        all_parents.extend(parents)
        all_children.extend(children)

    click.echo(f"done: {sum(methods.values())} page record(s) -> {work_dir}")
    click.echo(f"extraction methods: {methods}")
    click.echo(f"total watermark/artifact lines removed across all books: {total_watermarks_removed}")

    if not all_children:
        click.echo("no chunks produced — nothing to store")
    elif no_store:
        click.echo(f"--no-store: {len(all_parents)} parent(s), {len(all_children)} child chunk(s) ready")
    else:
        _build_store(pc, all_parents, all_children)


if __name__ == "__main__":
    main()
