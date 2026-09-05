"""Curriculum structure parser and parent-child chunking (SVG G, G1-G3, H, H1, H2; PDF §4).

G1 - board/grade/subject/book metadata comes from config/curriculum.yaml (per-book),
     not from OCR text: robust and auditable.
G2 - ``detect_heading``: unit/chapter/topic/exercise/generic headings from cleaned
     lines, using numbered patterns and title-case short lines.
G3 - ``build_page_map``: PDF page -> printed page using the page numbers detected
     during cleaning, interpolating unnumbered pages (offset scans).
H  - ``build_parent_child_chunks``: child chunks of 60-120 words with a 1-sentence
     overlap; parent chunks of 180-300 words; never crossing section (unit/chapter)
     boundaries. ``chunk_id = sha256(book + page + text)`` (deterministic) and
     ``parent_id`` links every child to its parent (PDF §7).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

_UNIT_RE = re.compile(r"^(unit|part)\s*([0-9ivxlcdmIVXLCDM]+)\s*[:.\-]?\s*(.*)$", re.IGNORECASE)
_CHAPTER_RE = re.compile(r"^(chapter|lesson|section|module)\s*([0-9ivxlcdmIVXLCDM]+)\s*[:.\-]?\s*(.*)$", re.IGNORECASE)
_TOPIC_RE = re.compile(r"^topic\s*([0-9ivxlcdmIVXLCDM]+)?\s*[:.\-]?\s*(.*)$", re.IGNORECASE)
_EXERCISE_RE = re.compile(r"^exercise\b", re.IGNORECASE)
_PURE_NUMBER_RE = re.compile(r"^\d+$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?۔])\s+")
_PREFIX = {"unit": "Unit", "chapter": "Chapter", "topic": "Topic"}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Heading:
    """A heading detected in cleaned text (G2). ``level`` is one of
    unit | chapter | topic | exercise | heading."""

    level: str
    title: str
    number: str = ""


@dataclass
class BookMeta:
    book: str
    edition: str = ""
    board: str = ""
    grade: str = ""
    subject: str = ""
    language: str = "english"


@dataclass
class Section:
    """A contiguous block of text under one heading, with inherited metadata."""

    level: str
    title: str
    number: str = ""
    unit: str = ""
    unit_no: str = ""
    chapter: str = ""
    chapter_no: str = ""
    topic: str = ""
    topic_no: str = ""
    board: str = ""
    grade: str = ""
    subject: str = ""
    book: str = ""
    edition: str = ""
    language: str = ""
    pdf_page_start: Optional[int] = None
    pdf_page_end: Optional[int] = None
    page_start: Optional[int] = None  # printed page
    page_end: Optional[int] = None  # printed page
    text_lines: list[str] = field(default_factory=list)
    line_pages: list[int] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.text_lines)


@dataclass
class StructuredBook:
    meta: BookMeta
    sections: list[Section]
    page_map: dict[int, int]  # pdf page -> printed page


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    kind: str  # "parent" | "child"
    text_clean: str
    text_lexical: str
    embed_text: str = ""  # heading-chain enriched text (children only)
    page_start: Optional[int] = None  # printed page
    page_end: Optional[int] = None
    pdf_page_start: Optional[int] = None
    pdf_page_end: Optional[int] = None
    board: str = ""
    grade: str = ""
    subject: str = ""
    book: str = ""
    edition: str = ""
    unit: str = ""
    chapter: str = ""
    topic: str = ""
    language: str = ""
    ocr_confidence: float = 0.0
    pipeline_version: str = ""
    section_level: str = ""
    section_title: str = ""
    seq: int = 0  # order within the parent's children (true reading order for siblings)


@dataclass
class ParentChildUnit:
    parent: Chunk
    children: list[Chunk]


# ---------------------------------------------------------------------------
# G2 - heading detection
# ---------------------------------------------------------------------------


def _make_heading(level: str, number: str, title: str, original: str) -> Heading:
    title = (title or "").strip(" :.-")
    if not title:
        title = original.strip()
    return Heading(level=level, title=title, number=number or "")


def detect_heading(line: str) -> Optional[Heading]:
    """Detect a heading in a cleaned line (G2), or None for prose/noise."""
    s = line.strip()
    if not s:
        return None
    m = _UNIT_RE.match(s)
    if m:
        return _make_heading("unit", m.group(2), m.group(3), s)
    m = _CHAPTER_RE.match(s)
    if m:
        return _make_heading("chapter", m.group(2), m.group(3), s)
    m = _TOPIC_RE.match(s)
    if m:
        return _make_heading("topic", m.group(1) or "", m.group(2), s)
    if _EXERCISE_RE.match(s):
        return _make_heading("exercise", "", s, s)
    if _is_generic_heading(s):
        return _make_heading("heading", "", s, s)
    return None


def _is_generic_heading(s: str) -> bool:
    if not 2 <= len(s) <= 60:
        return False
    if s.endswith((".", "!", "?", "۔")):
        return False
    if _PURE_NUMBER_RE.match(s):
        return False
    words = s.split()
    if not 1 <= len(words) <= 8:
        return False
    title_case = sum(1 for w in words if w[:1].isupper()) / len(words)
    return title_case >= 0.7


# ---------------------------------------------------------------------------
# G3 - PDF page -> printed page mapping
# ---------------------------------------------------------------------------


def build_page_map(pages: Sequence["PageRecord"]) -> dict[int, int]:
    """Map PDF page -> printed page (G3).

    Anchors are pages where cleaning detected a printed page number. All other
    pages are interpolated/extrapolated sequentially, so offset scans (blank
    or unnumbered front pages) still get stable citations.
    """
    anchors = sorted((r.page_number, r.printed_page) for r in pages if r.printed_page is not None)
    if not anchors:
        return {}
    result: dict[int, int] = {}
    first_pdf, first_print = anchors[0]
    for pdf in range(1, first_pdf):
        result[pdf] = first_print - (first_pdf - pdf)
    result[first_pdf] = first_print
    for (p1, q1), (p2, q2) in zip(anchors, anchors[1:]):
        for pdf in range(p1 + 1, p2):
            result[pdf] = q1 + (pdf - p1)
        result[p2] = q2
    last_pdf, last_print = anchors[-1]
    max_pdf = max(r.page_number for r in pages)
    for pdf in range(last_pdf + 1, max_pdf + 1):
        result[pdf] = last_print + (pdf - last_pdf)
    return result


# ---------------------------------------------------------------------------
# Curriculum parser (G1 + G2 + G3)
# ---------------------------------------------------------------------------


def _canonical(level: str, number: str, title: str) -> str:
    """Canonical metadata label for a context level, e.g. "Unit 3" or a title."""
    if number:
        prefix = _PREFIX.get(level, level)
        try:
            return f"{prefix} {int(number)}"
        except ValueError:  # roman numerals
            return f"{prefix} {number}"
    return title


def _matches_watermark_keyword(line: str, keywords: Sequence[str]) -> bool:
    """Defense-in-depth check (mirrors cleaning.py's F1c): if a watermark
    keyword still shows up in ``clean_text`` (e.g. keywords were added after
    a page was already cleaned, or the same run's config changed), never let
    it become a Section heading/title or get chunked as body text.

    cleaning.py is the primary defense; this is a cheap second check so a
    single leaked line can't corrupt structure metadata shown to students.
    """
    if not keywords:
        return False
    low = line.lower()
    return any(k and k.lower() in low for k in keywords)


def _new_section(heading: Heading, context: dict, meta: BookMeta, rec, printed: Optional[int]) -> Section:
    return Section(
        level=heading.level,
        title=heading.title,
        number=heading.number,
        unit=context["unit"],
        unit_no=context["unit_no"],
        chapter=context["chapter"],
        chapter_no=context["chapter_no"],
        topic=context["topic"],
        topic_no=context["topic_no"],
        board=meta.board,
        grade=meta.grade,
        subject=meta.subject,
        book=meta.book,
        edition=meta.edition,
        language=meta.language,
        pdf_page_start=rec.page_number,
        page_start=printed,
    )


def parse_curriculum(
    pages: Sequence["PageRecord"],
    curriculum: dict,
    watermark_keywords: Sequence[str] = (),
    ignore_heading_keywords: Sequence[str] = (),
) -> StructuredBook:
    """Parse a cleaned book into ordered sections (G1-G3).

    ``curriculum`` is the per-book metadata dict from config/curriculum.yaml:
    ``{book, edition, board, grade, subject, language}``.

    ``watermark_keywords`` (from ``pc.watermark.keywords`` in config.yaml) is
    a defense-in-depth line filter — see ``_matches_watermark_keyword``.

    ``ignore_heading_keywords`` (from ``pc.chunk.ignore_heading_keywords``)
    is a curated list of known false-positive heading titles: OCR
    occasionally misreads a diagram caption or illustration label as a
    short, title-case-looking line, which ``_is_generic_heading`` then
    promotes to a section — fragmenting real chapter content under a
    garbled title (confirmed on a real book: a plant-diagram caption
    misread as "Part of Plant Monocot Plant Oo Dicot Plant" silently
    swallowed 12 pages of a chapter under the wrong heading). Rather than
    tightening the general heading heuristic — which risks dropping
    legitimate short headings elsewhere — a matching candidate heading is
    demoted back to ordinary body text, the same curated-safety-net pattern
    already used for watermark keywords.
    """
    meta = BookMeta(
        book=str(curriculum.get("book", "")),
        edition=str(curriculum.get("edition", "")),
        board=str(curriculum.get("board", "")),
        grade=str(curriculum.get("grade", "")),
        subject=str(curriculum.get("subject", "")),
        language=str(curriculum.get("language", "english")),
    )
    page_map = build_page_map(pages)
    context = {
        "unit": "", "unit_no": "",
        "chapter": "", "chapter_no": "",
        "topic": "", "topic_no": "",
    }
    sections: list[Section] = []
    current: Optional[Section] = None

    for rec in sorted(pages, key=lambda r: r.page_number):
        printed = page_map.get(rec.page_number)
        for line in rec.clean_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _matches_watermark_keyword(stripped, watermark_keywords):
                continue  # never becomes a heading or body text
            heading = detect_heading(stripped)
            if heading is not None and _matches_watermark_keyword(heading.title, ignore_heading_keywords):
                heading = None  # known OCR-garbled false-positive heading; keep as plain body text
            if heading is not None:
                if heading.level == "unit":
                    canonical = _canonical("unit", heading.number, heading.title)
                    if canonical and canonical == context["unit"]:
                        # Running-head echo: this exact unit title repeats as
                        # a page header throughout the unit (e.g. printed at
                        # the top of every page). A book-global frequency
                        # check (cleaning.py's F1) can't catch this — a
                        # per-unit running head only covers that unit's own
                        # page range, typically far under the 70% book-wide
                        # threshold. Since we're already inside this exact
                        # unit, a repeat is unambiguously boilerplate, never
                        # a genuine second unit start — drop the line
                        # entirely rather than fragmenting the unit into one
                        # tiny section per page.
                        continue
                    context.update(unit=canonical, unit_no=heading.number,
                                   chapter="", chapter_no="", topic="", topic_no="")
                elif heading.level == "chapter":
                    canonical = _canonical("chapter", heading.number, heading.title)
                    if canonical and canonical == context["chapter"]:
                        continue  # running-head echo of the current chapter (see unit case above)
                    context.update(chapter=canonical, chapter_no=heading.number,
                                   topic="", topic_no="")
                elif heading.level == "topic":
                    canonical = _canonical("topic", heading.number, heading.title)
                    if canonical and canonical == context["topic"]:
                        continue  # running-head echo of the current topic (see unit case above)
                    context.update(topic=canonical, topic_no=heading.number)
                current = _new_section(heading, context, meta, rec, printed)
                sections.append(current)
            else:
                if current is None:
                    current = _new_section(Heading("heading", "Front matter"), context, meta, rec, printed)
                    sections.append(current)
                current.text_lines.append(stripped)
                current.line_pages.append(rec.page_number)
                current.pdf_page_end = rec.page_number
                if printed is not None:
                    current.page_end = printed
    return StructuredBook(meta=meta, sections=sections, page_map=page_map)


# ---------------------------------------------------------------------------
# Text helpers for chunking
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split cleaned text into sentences (periods, !, ?, Urdu full stop)."""
    return [p.strip() for p in _SENT_SPLIT_RE.split(text or "") if p.strip()]


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text or ""))


def make_lexical_text(text: str) -> str:
    """Normalized searchable text for BM25: lowercase, accents stripped,
    punctuation -> space, whitespace collapsed. Keeps non-ASCII script
    characters (e.g. Urdu)."""
    s = unicodedata.normalize("NFKD", text or "").lower()
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def make_chunk_id(book: str, page: Optional[int], text: str) -> str:
    """Deterministic chunk id: sha256(book + page + text) (PDF §7)."""
    key = f"{book}|{page}|{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _slug(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\u0600-\u06FF]+", "_", s)
    return s.strip("_") or "x"


def make_parent_id(
    unit: str, unit_no: str, chapter: str, chapter_no: str,
    topic: str, topic_no: str, index: int,
) -> str:
    """Deterministic parent id, e.g. ``unit_03_topic_02_parent_001``."""
    components: list[str] = []
    for label, canonical, number in (
        ("unit", unit, unit_no),
        ("chapter", chapter, chapter_no),
        ("topic", topic, topic_no),
    ):
        if not canonical:
            continue
        if number:
            try:
                components.append(f"{label}_{int(number):02d}")
            except ValueError:
                components.append(f"{label}_{_slug(number)}")
        else:
            components.append(f"{label}_{_slug(canonical)}")
    base = "_".join(components) or "front"
    return f"{base}_parent_{index:03d}"


def _embed_text(section: Section, child_text: str) -> str:
    """Heading-chain enriched embedding text (SVG I): the enclosing unit /
    chapter / topic chain prefixes the child text."""
    parts = []
    for label, value in (("Unit", section.unit), ("Chapter", section.chapter), ("Topic", section.topic)):
        if value:
            parts.append(f"{label}: {value}")
    chain = " · ".join(parts)
    return f"{chain} — {child_text}" if chain else child_text


# ---------------------------------------------------------------------------
# Parent-child chunking (H, H1, H2)
# ---------------------------------------------------------------------------


def _cumulative_word_counts(sentences: Sequence[str]) -> list[int]:
    cum = [0]
    for s in sentences:
        cum.append(cum[-1] + word_count(s))
    return cum


def _word_pages(section: Section) -> list[int]:
    pages: list[int] = []
    for line, page in zip(section.text_lines, section.line_pages):
        pages.extend([page] * word_count(line))
    return pages


def _child_chunks(
    sentences: Sequence[str], child_min: int, child_max: int, overlap: int
) -> list[tuple[str, int, int]]:
    """Split sentences into child chunks of child_min..child_max words with a
    one-sentence overlap; returns (text, start_sentence, end_sentence)."""
    n = len(sentences)
    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < n:
        end = start
        words = 0
        while end < n:
            w = word_count(sentences[end])
            if words > 0 and words + w > child_max:
                break
            words += w
            end += 1
        if words < child_min and end < n:
            words += word_count(sentences[end])
            end += 1
        chunks.append((" ".join(sentences[start:end]), start, end))
        if end >= n:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks


def _parent_groups(children: Sequence[Chunk], parent_min: int, parent_max: int) -> list[list[Chunk]]:
    """Greedily merge children into parents of parent_min..parent_max words."""
    groups: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_words = 0
    for child in children:
        w = word_count(child.text_clean)
        if current and current_words + w > parent_max and current_words >= parent_min:
            groups.append(current)
            current, current_words = [], 0
        current.append(child)
        current_words += w
    if current:
        groups.append(current)
    return groups


def _chunk_from_section(
    section: Section,
    text: str,
    si: int,
    ei: int,
    cum: Sequence[int],
    word_pages: Sequence[int],
    page_map: dict[int, int],
    conf_by_page: dict[int, float],
    pipeline_version: str,
    parent_id: str,
    kind: str,
) -> Chunk:
    pdf_start = word_pages[cum[si]]
    pdf_end = word_pages[cum[ei] - 1]
    printed_start = page_map.get(pdf_start, pdf_start)
    printed_end = page_map.get(pdf_end, pdf_end)
    conf = sum(conf_by_page.get(p, 0.0) for p in range(pdf_start, pdf_end + 1))
    conf /= max(1, pdf_end - pdf_start + 1)
    return Chunk(
        chunk_id=make_chunk_id(section.book, printed_start, text),
        parent_id=parent_id,
        kind=kind,
        text_clean=text,
        text_lexical=make_lexical_text(text),
        embed_text=_embed_text(section, text) if kind == "child" else "",
        page_start=printed_start,
        page_end=printed_end,
        pdf_page_start=pdf_start,
        pdf_page_end=pdf_end,
        board=section.board,
        grade=section.grade,
        subject=section.subject,
        book=section.book,
        edition=section.edition,
        unit=section.unit,
        chapter=section.chapter,
        topic=section.topic,
        language=section.language,
        ocr_confidence=round(conf, 3),
        pipeline_version=pipeline_version,
        section_level=section.level,
        section_title=section.title,
    )


def build_parent_child_chunks(
    sections: Sequence[Section],
    page_records: Optional[Sequence["PageRecord"]] = None,
    cfg: Optional[dict] = None,
    page_map: Optional[dict[int, int]] = None,
) -> list[ParentChildUnit]:
    """Build parent-child chunks for every non-empty section (H, H1, H2).

    Invariants: children are child_min..child_max words (a single over-long
    sentence may exceed the max), consecutive children overlap by one
    sentence, parents are parent_min..parent_max words, and chunks never cross
    section (unit/chapter) boundaries.
    """
    cfg = cfg or {}
    chunk_cfg = cfg.get("chunk", cfg) if isinstance(cfg, dict) else {}
    child_min = int(chunk_cfg.get("child_min", 60) or 60)
    child_max = int(chunk_cfg.get("child_max", 120) or 120)
    parent_min = int(chunk_cfg.get("parent_min", 180) or 180)
    parent_max = int(chunk_cfg.get("parent_max", 300) or 300)
    overlap = int(chunk_cfg.get("overlap_sentences", 1) or 1)
    pipeline_version = str(cfg.get("pipeline_version", "")) if isinstance(cfg, dict) else ""
    conf_by_page = {r.page_number: float(r.ocr_confidence or 0.0) for r in (page_records or [])}
    page_map = page_map or {}

    units: list[ParentChildUnit] = []
    parent_index = 0
    for section in sections:
        if not section.text_lines:
            continue
        sentences = split_sentences(section.text)
        if not sentences:
            continue
        cum = _cumulative_word_counts(sentences)
        word_pages = _word_pages(section)
        if len(word_pages) != cum[-1]:
            # Sentence splitting must preserve the word stream; defensive guard.
            continue
        child_specs = _child_chunks(sentences, child_min, child_max, overlap)
        templates = []
        for seq, (text, si, ei) in enumerate(child_specs):
            child = _chunk_from_section(section, text, si, ei, cum, word_pages, page_map,
                                        conf_by_page, pipeline_version, "", "child")
            child.seq = seq
            templates.append(child)
        for group in _parent_groups(templates, parent_min, parent_max):
            parent_id = make_parent_id(
                section.unit, section.unit_no,
                section.chapter, section.chapter_no,
                section.topic, section.topic_no, parent_index,
            )
            parent_index += 1
            parent_text = "\n".join(c.text_clean for c in group)
            pdf_start = min(c.pdf_page_start for c in group)
            pdf_end = max(c.pdf_page_end for c in group)
            printed_start = page_map.get(pdf_start, pdf_start)
            printed_end = page_map.get(pdf_end, pdf_end)
            parent = Chunk(
                chunk_id=make_chunk_id(section.book, printed_start, parent_text),
                parent_id=parent_id,
                kind="parent",
                text_clean=parent_text,
                text_lexical=make_lexical_text(parent_text),
                embed_text="",
                page_start=printed_start,
                page_end=printed_end,
                pdf_page_start=pdf_start,
                pdf_page_end=pdf_end,
                board=section.board,
                grade=section.grade,
                subject=section.subject,
                book=section.book,
                edition=section.edition,
                unit=section.unit,
                chapter=section.chapter,
                topic=section.topic,
                language=section.language,
                ocr_confidence=round(
                    sum(c.ocr_confidence for c in group) / max(1, len(group)), 3
                ),
                pipeline_version=pipeline_version,
                section_level=section.level,
                section_title=section.title,
            )
            for child in group:
                child.parent_id = parent_id
            units.append(ParentChildUnit(parent=parent, children=group))
    return units


def flatten_chunks(units: Sequence[ParentChildUnit]) -> tuple[list[Chunk], list[Chunk]]:
    """Split a unit list into (parents, children) for the storage layer."""
    parents = [u.parent for u in units]
    children = [c for u in units for c in u.children]
    return parents, children


# ---------------------------------------------------------------------------
# Curriculum config helpers
# ---------------------------------------------------------------------------


def load_curriculum(path: str | Path) -> dict:
    """Load config/curriculum.yaml: ``{books: [{book, edition, board, grade,
    subject, language}, ...]}``."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def match_book(stem: str, books: Sequence[dict]) -> Optional[dict]:
    """Find the curriculum entry for a PDF filename (case-insensitive)."""
    stem_l = stem.lower()
    for book in books:
        name = str(book.get("book", "")).lower()
        if name and (stem_l == name or name in stem_l or stem_l in name):
            return book
    return None
