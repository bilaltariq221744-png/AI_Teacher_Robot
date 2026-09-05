"""Cleaning and reconstruction layer (SVG nodes F, F1-F4; PDF §3).

Pipeline
--------
F1  - ``detect_repeated_headers_footers``: book-level pass. Lines repeated on
      >= ~70% of pages in the top/bottom window are running headers/footers
      and are dropped everywhere (book titles, unit running heads, "Page X
      of Y" labels). Pure page numbers are excluded here and handled
      separately.
F1b - ``detect_repeated_lines_anywhere``: book-level pass, NO position
      restriction. Catches centered/diagonal watermarks that repeat across
      most pages but never land in F1's top/bottom window.
F1c - ``flag_pattern_watermarks``: page-level pass against configured
      keywords/regex. Catches dynamic watermarks (timestamp/username baked
      in) that never repeat verbatim, so F1/F1b can't see them.
F2  - ``remove_watermarks_noise``: page-level pass. Removes F1/F1b/F1c
      artifacts, garbage lines (pure / mostly-symbol), and lines repeated
      many times on the same page (watermarks).
F3  - ``reconstruct_sentences_paragraphs``: rejoins hard-wrapped lines and
      hyphenated words; paragraphs stay separated by blank lines.
F4  - ``protect_structures`` / ``restore_structures``: numbered headings,
      tables and exercise lines are marked before F2/F3 and restored
      afterwards, so they are never re-flowed or dropped. Any line already
      flagged as an artifact (F1/F1b/F1c) is never protected — a watermark
      rendered with wide letter-spacing must not get mistaken for a table
      row and survive verbatim.

Policy note (PDF §3: "Remove repeated watermarks and page artifacts, but do
not delete real headings"): content headings appear once in a book and are
never removed. A line that repeats on >=70% of pages inside the top/bottom
window is by definition a *running head* (boilerplate), so F1 is the only
frequency-based stage allowed to drop heading-like lines; F2's noise/
watermark rules always skip heading-like lines UNLESS the line is already a
known artifact (any of F1/F1b/F1c/style-flagged from ocr.py).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

_MARKER_START = "\uF000"
_MARKER_END = "\uF001"

_PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
_ROMAN_RE = re.compile(r"^[ivxlcdm]{1,8}$", re.IGNORECASE)
_NUMBERED_HEADING_RE = re.compile(r"^(unit|chapter|lesson|section|topic|module|part)\b", re.IGNORECASE)
_EXERCISE_RE = re.compile(r"^(exercise|question|q\.?)\b", re.IGNORECASE)


@dataclass
class ProtectedSpan:
    """A heading/table/exercise extracted by F4 and re-inserted after F2/F3."""

    marker: str
    original: str
    kind: str  # "heading" | "table"


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


def _norm_line(line: str) -> str:
    """Normalized form used for artifact/watermark matching."""
    return re.sub(r"\s+", " ", line).strip().lower()


def _is_pure_number(s: str) -> bool:
    return bool(_PAGE_NUMBER_RE.match(s) or _ROMAN_RE.match(s))


def _is_heading_like(line: str) -> bool:
    """Strong heading signals: numbered unit/chapter/section lines and
    exercise/question markers.

    Title-case and ALL-CAPS short lines are deliberately NOT treated as
    headings: they are ambiguous against hard-wrapped prose line-starts and
    repeated watermarks ("SAMPLE", "DRAFT"), and the structure parser (G2)
    re-detects headings from patterns anyway.
    """
    s = line.strip()
    if not s:
        return False
    if _NUMBERED_HEADING_RE.search(s):
        return True
    if _EXERCISE_RE.search(s):
        return True
    return False


def _is_noise_line(line: str) -> bool:
    """Garbage OCR/watermark line: no alphanumerics, or mostly symbols."""
    s = line.strip()
    if not s:
        return False
    alnum = sum(1 for ch in s if ch.isalnum())
    if alnum == 0:
        return True
    return len(s) >= 4 and alnum / len(s) < 0.3


def _table_line(line: str) -> bool:
    """True when a line looks like a table row (pipe-separated or aligned columns)."""
    s = line.strip()
    if not s:
        return False
    if "|" in s:
        return True
    return len(re.split(r"\s{2,}", s)) >= 2


def _collect_table_block(lines: Sequence[str], i: int) -> Optional[tuple[int, int]]:
    """Return (start, end) of a contiguous table block starting at line i."""
    n = len(lines)
    j = i
    rows: list[str] = []
    while j < n and len(rows) < 12:
        s = lines[j].strip()
        if not s:
            if rows:
                break
            j += 1
            continue
        if not _table_line(s):
            break
        rows.append(s)
        j += 1
    if len(rows) >= 2:
        return i, j
    return None


# ---------------------------------------------------------------------------
# F1 - repeated header/footer detection (book-level)
# ---------------------------------------------------------------------------


def detect_repeated_headers_footers(
    pages: Sequence[str], min_frequency: float = 0.7, window: int = 3
) -> set[str]:
    """Find running headers/footers across all pages of a book (F1).

    A line is an artifact when it appears in the top/bottom ``window`` of at
    least ``min_frequency`` of the pages. Pure page numbers are excluded
    (they vary per page and are handled by ``find_page_number``). Lines that
    look like headings are still flagged: at this position/frequency they are
    running heads, and F1 is the only stage allowed to remove heading-like
    text.
    """
    n = len(pages)
    if n < 2:
        # Headers/footers are defined by repetition across pages; a single
        # page gives no evidence, so nothing is flagged as an artifact.
        return set()
    # A floor of 2 is required regardless of how the percentage rounds: with
    # few pages (e.g. n=2), ceil(n * min_frequency) can round down to 1,
    # which would flag content that appears on only ONE page as "repeated".
    # "Repeated" means it occurred more than once — never fewer than 2.
    threshold = max(2, math.ceil(n * min_frequency))
    counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        if not lines:
            continue
        # Dedupe: on short pages the top and bottom windows overlap and would
        # otherwise double-count the same line, inflating its frequency.
        positions = sorted(
            set(range(min(window, len(lines))))
            | set(range(max(0, len(lines) - window), len(lines)))
        )
        for pos in positions:
            ln = lines[pos]
            if ln and not _is_pure_number(ln):
                counts[_norm_line(ln)] += 1
    return {ln for ln, c in counts.items() if c >= threshold}


def detect_repeated_lines_anywhere(pages: Sequence[str], min_frequency: float = 0.5) -> set[str]:
    """Cross-page repeated-line detector with NO position restriction (F1b).

    ``detect_repeated_headers_footers`` only looks at the top/bottom
    ``window`` lines, so a centered or diagonal watermark — which rarely
    lands there — slips through. This scans every line on every page and
    flags anything repeated on at least ``min_frequency`` of pages,
    regardless of where it sits. Each page counts a repeated line at most
    once (in-page repetition is F2's job); pure page numbers are excluded.
    """
    n = len(pages)
    if n < 2:
        return set()
    # Same floor-of-2 reasoning as detect_repeated_headers_footers: at low
    # page counts a percentage threshold can round down to 1, which would
    # flag content unique to a single page as "repeated".
    threshold = max(2, math.ceil(n * min_frequency))
    counts: Counter[str] = Counter()
    for page in pages:
        seen_this_page: set[str] = set()
        for ln in page.splitlines():
            s = ln.strip()
            if not s or _is_pure_number(s):
                continue
            norm = _norm_line(s)
            if norm in seen_this_page:
                continue
            seen_this_page.add(norm)
            counts[norm] += 1
    return {ln for ln, c in counts.items() if c >= threshold}


def compile_watermark_patterns(patterns: Iterable[str]) -> list["re.Pattern"]:
    compiled = []
    for p in patterns or []:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error:
            continue  # skip a bad pattern rather than fail the whole build
    return compiled


def flag_pattern_watermarks(
    text: str, keywords: Sequence[str] = (), patterns: Sequence["re.Pattern"] = ()
) -> set[str]:
    """Lines matching a known watermark keyword or regex (F1c: dynamic watermarks).

    Frequency-based detection (F1, F1b) needs the SAME line to repeat across
    pages; a watermark that embeds a timestamp, username, or download id
    (e.g. "Downloaded by user123 on 2026-08-19") is different text on every
    page and never crosses a repetition threshold. A known keyword or regex
    match is a strong enough signal on its own — one occurrence is enough.
    """
    flagged: set[str] = set()
    kw_lower = [k.lower() for k in keywords if k]
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if any(k in low for k in kw_lower):
            flagged.add(_norm_line(s))
            continue
        if any(p.search(s) for p in patterns):
            flagged.add(_norm_line(s))
    return flagged


# ---------------------------------------------------------------------------
# F2 - watermark and OCR-noise removal (page-level)
# ---------------------------------------------------------------------------


def remove_watermarks_noise(
    text: str, artifacts: Iterable[str] = (), watermark_repeat: int = 3
) -> str:
    """Remove F1/F1b/F1c artifacts, garbage lines, and in-page repeated
    watermarks.

    Artifacts (from any upstream detector) are authoritative and are always
    dropped. All other rules (noise, in-page repetition) skip heading-like
    lines so real headings are never removed here.
    """
    artifact_set = {_norm_line(a) for a in artifacts}
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _norm_line(stripped) in artifact_set:
            continue
        if _is_heading_like(stripped):
            kept.append(stripped)
            continue
        if _is_noise_line(stripped):
            continue
        kept.append(stripped)

    counts = Counter(_norm_line(ln) for ln in kept if ln.strip())
    out: list[str] = []
    for line in kept:
        if (
            line.strip()
            and counts[_norm_line(line)] >= watermark_repeat
            and not _is_heading_like(line)
        ):
            continue
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# F3 - sentence and paragraph reconstruction
# ---------------------------------------------------------------------------


def _join_paragraph(lines: Sequence[str]) -> str:
    """Join hard-wrapped lines into one paragraph; fix hyphenated breaks."""
    out = ""
    for raw in lines:
        ln = re.sub(r"\s+", " ", raw).strip()
        if not ln:
            continue
        if out and out.endswith("-"):
            out = out[:-1] + ln
        elif out:
            out += " " + ln
        else:
            out = ln
    return out


def reconstruct_sentences_paragraphs(text: str) -> str:
    """Rejoin lines broken by OCR; split paragraphs on blank lines (F3).

    Protected markers (F4) act as paragraph boundaries so headings/tables are
    never merged into surrounding prose.
    """
    blocks: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        if not block.strip():
            continue
        sub: list[str] = []
        current: list[str] = []
        for line in block.splitlines():
            if _MARKER_START in line:
                if current:
                    sub.append(_join_paragraph(current))
                    current = []
                sub.append(line.strip())
            else:
                current.append(line)
        if current:
            sub.append(_join_paragraph(current))
        blocks.append("\n".join(part for part in sub if part is not None))
    return "\n\n".join(b for b in blocks if b)


# ---------------------------------------------------------------------------
# F4 - structure preservation (headings, tables, exercises)
# ---------------------------------------------------------------------------


def _marker(kind: str, idx: int) -> str:
    return f"{_MARKER_START}{kind}_{idx}{_MARKER_END}"


def protect_structures(
    text: str, artifacts: Iterable[str] = ()
) -> tuple[list[ProtectedSpan], str]:
    """Replace headings/tables/exercises with markers (F4, protect pass).

    ``artifacts`` (F1/F1b/F1c/style-flagged lines) are NOT protected — those
    lines stay in the body where F2 removes them. This also blocks a known
    artifact line from being mistaken for the start of a table (a watermark
    rendered with wide letter-spacing looks like a table row otherwise, and
    would survive F2 verbatim if it got protected here first). Returns
    ``(protected, body)``; call ``restore_structures`` after F2/F3 to put the
    originals back.
    """
    artifact_set = {_norm_line(a) for a in artifacts}
    lines = text.splitlines()
    protected: list[ProtectedSpan] = []
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            out.append(lines[i])
            i += 1
            continue
        # Skip table detection at a line already known to be an artifact —
        # otherwise a watermark rendered with wide letter-spacing (a common
        # watermark trick) gets misread as a table row, protected here, and
        # restored verbatim after F2 would have dropped it (F4 bypass).
        table = _collect_table_block(lines, i) if _norm_line(stripped) not in artifact_set else None
        if table is not None:
            start, end = table
            block = [ln.strip() for ln in lines[start:end] if ln.strip()]
            protected.append(
                ProtectedSpan(_marker("table", len(protected)), "\n".join(block), "table")
            )
            out.append(protected[-1].marker)
            i = end
            continue
        if _is_heading_like(stripped) and _norm_line(stripped) not in artifact_set:
            protected.append(
                ProtectedSpan(_marker("heading", len(protected)), stripped, "heading")
            )
            out.append(protected[-1].marker)
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return protected, "\n".join(out)


def restore_structures(body: str, protected: Sequence[ProtectedSpan]) -> str:
    """Put protected spans back into the cleaned body (F4, restore pass)."""
    for span in protected:
        body = body.replace(span.marker, span.original)
    return body


# ---------------------------------------------------------------------------
# Page/book orchestration
# ---------------------------------------------------------------------------


def clean_page(
    text: str,
    page_no: int,
    cfg: Optional[Mapping[str, object]] = None,
    artifacts: Sequence[str] = (),
    watermark_repeat: Optional[int] = None,
) -> str:
    """Clean a single page's raw text (F2-F4). ``page_no`` is for diagnostics.

    ``artifacts`` are the combined F1/F1b/F1c/style-flagged lines for this
    page.
    """
    if watermark_repeat is None:
        watermark_repeat = int((cfg or {}).get("watermark_repeat", 3) or 3)
    if not text or not text.strip():
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    protected, body = protect_structures(text, artifacts=artifacts)  # F4 protect
    body = remove_watermarks_noise(body, artifacts, watermark_repeat=watermark_repeat)  # F2
    body = reconstruct_sentences_paragraphs(body)  # F3
    return restore_structures(body, protected).strip()  # F4 restore


def find_page_number(text: str, window: int = 3) -> Optional[int]:
    """Best-effort printed page number from a standalone number line (G3 uses
    this later; Arabic numerals only)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    candidates = lines[:window] + lines[-window:]
    for ln in candidates:
        if _PAGE_NUMBER_RE.match(ln):
            return int(ln)
    return None


def clean_book(records, cfg: Optional[Mapping[str, object]] = None):
    """Clean every page of a book (F1-F4) in place; returns the records.

    F1/F1b run once across all pages; F1c and F2-F4 run per page. Also
    extracts the printed page number into ``record.printed_page`` when not
    already set (the structure parser refines this later) and removes that
    number line from the cleaned text.

    Watermark defense combines four independent signals into one artifact
    set per page:
      F1  - repeated in the top/bottom header/footer window (running heads)
      F1b - repeated anywhere on the page, position-agnostic (centered/
            diagonal watermarks that F1 can't see)
      F1c - matches a configured watermark keyword or regex, even once
            (dynamic watermarks that never repeat verbatim)
      C1.5 - style-flagged lines carried over from ocr.py (rotated/oversized/
             light-colored text in the embedded layer), via ``record.watermark_lines``
    """
    cfg = cfg or {}
    cleaning = cfg.get("cleaning") or {}
    watermark_cfg = cfg.get("watermark") or {}
    min_frequency = float(cleaning.get("header_footer_frequency", 0.7))
    window = int(cleaning.get("header_footer_window", 3))
    watermark_repeat = int(cleaning.get("watermark_repeat", 3))

    wm_enabled = bool(watermark_cfg.get("enabled", True))
    wm_cross_freq = float(watermark_cfg.get("cross_page_frequency", 0.5))
    wm_keywords = list(watermark_cfg.get("keywords", []) or [])
    wm_patterns = compile_watermark_patterns(watermark_cfg.get("regex_patterns", []) or [])

    pages = [rec.raw_text for rec in records]
    header_footer_artifacts = detect_repeated_headers_footers(
        pages, min_frequency=min_frequency, window=window
    )
    cross_page_artifacts = (
        detect_repeated_lines_anywhere(pages, min_frequency=wm_cross_freq) if wm_enabled else set()
    )

    for rec in records:
        if rec.printed_page is None:
            rec.printed_page = find_page_number(rec.raw_text, window=window)

        page_artifacts = set(header_footer_artifacts) | set(cross_page_artifacts)
        page_artifacts |= {_norm_line(l) for l in (getattr(rec, "watermark_lines", None) or [])}
        if wm_enabled:
            page_artifacts |= flag_pattern_watermarks(rec.raw_text, wm_keywords, wm_patterns)
        if rec.printed_page is not None:
            page_artifacts.add(str(rec.printed_page))

        before_lines = {_norm_line(l) for l in rec.raw_text.splitlines() if l.strip()}
        rec.clean_text = clean_page(
            rec.raw_text,
            rec.page_number,
            cleaning,
            artifacts=page_artifacts,
            watermark_repeat=watermark_repeat,
        )
        # Audit trail: how many distinct artifact lines actually appeared on
        # this page (build_db.py sums these into a per-book report).
        if hasattr(rec, "watermark_removed_count"):
            rec.watermark_removed_count = len(before_lines & page_artifacts)
    return records
