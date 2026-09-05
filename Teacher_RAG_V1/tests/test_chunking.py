"""Unit tests for src/chunking.py (Phase 3-4: SVG G, H; PDF §4)."""
from __future__ import annotations

import pytest

from src.chunking import (
    build_page_map,
    build_parent_child_chunks,
    detect_heading,
    flatten_chunks,
    make_chunk_id,
    make_lexical_text,
    parse_curriculum,
    split_sentences,
    word_count,
)
from src.ocr import PageRecord

BOOK_META = {
    "book": "New Oxford Modern English",
    "edition": "SNC",
    "board": "SNC",
    "grade": "5",
    "subject": "English",
    "language": "english",
}

SENTENCES = [
    "Plants make their own food.",
    "This process is called photosynthesis.",
    "It happens in green leaves.",
    "Plants need sunlight and water.",
    "They also need carbon dioxide.",
    "Chlorophyll catches the sunlight.",
    "Sugar gives the plant energy.",
    "Oxygen is released into the air.",
]


def _rec(pdf_no, clean="", printed=None, conf=0.9):
    return PageRecord(
        pdf_path="book.pdf", page_number=pdf_no, raw_text=clean, clean_text=clean,
        printed_page=printed, ocr_confidence=conf,
    )


def _book_pages():
    # Unit section spans two PDF pages (printed 30-31); topic section on page 2.
    return [
        _rec(1, clean="Unit 3: Plants and their food\n" + "\n".join(SENTENCES[:2]), printed=30),
        _rec(2, clean="\n".join(SENTENCES[2:4]) + "\nTopic: Oxygen\n" + "\n".join(SENTENCES[4:]), printed=31),
    ]


def _cfg(**kw):
    cfg = dict(
        child_min=6, child_max=12, parent_min=30, parent_max=50,
        overlap_sentences=1, pipeline_version="test-v1",
    )
    cfg.update(kw)
    return cfg


def _build(cfg=None, pages=None):
    pages = pages or _book_pages()
    book = parse_curriculum(pages, BOOK_META)
    units = build_parent_child_chunks(book.sections, pages, cfg=cfg or _cfg(), page_map=book.page_map)
    return book, units


# --- G2: heading detection --------------------------------------------------


def test_detect_unit_heading():
    h = detect_heading("Unit 3: Plants and their food")
    assert h.level == "unit" and h.number == "3" and h.title == "Plants and their food"


def test_detect_chapter_heading_without_title():
    h = detect_heading("Chapter 2")
    assert h.level == "chapter" and h.number == "2" and h.title == "Chapter 2"


def test_detect_topic_heading():
    h = detect_heading("Topic: Shipwreck")
    assert h.level == "topic" and h.title == "Shipwreck"


def test_detect_exercise_heading():
    assert detect_heading("Exercise 2: Fill in the blanks").level == "exercise"


def test_detect_generic_title_case_heading():
    h = detect_heading("Robinson Crusoe")
    assert h.level == "heading" and h.title == "Robinson Crusoe"


def test_prose_and_numbers_are_not_headings():
    assert detect_heading("Photosynthesis happens in leaves.") is None
    assert detect_heading("42") is None
    assert detect_heading("the quick brown fox jumps") is None


# --- G3: page map -----------------------------------------------------------


def test_build_page_map_offset_scan():
    records = [
        _rec(1), _rec(2),
        _rec(3, printed=1), _rec(4, printed=2), _rec(5), _rec(6, printed=4),
    ]
    assert build_page_map(records) == {1: -1, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4}


def test_build_page_map_no_anchors():
    assert build_page_map([_rec(1), _rec(2)]) == {}


# --- text helpers -----------------------------------------------------------


def test_split_sentences():
    assert split_sentences("One sentence. Two sentences! Three? Four.") == [
        "One sentence.", "Two sentences!", "Three?", "Four.",
    ]


def test_make_lexical_text_normalizes():
    assert make_lexical_text("Photosynthesis, happens (in leaves)!") == "photosynthesis happens in leaves"
    assert make_lexical_text("Café  Naïve") == "cafe naive"


def test_make_chunk_id_deterministic():
    assert make_chunk_id("Book", 5, "text") == make_chunk_id("Book", 5, "text")
    assert make_chunk_id("Book", 5, "text") != make_chunk_id("Book", 6, "text")
    assert make_chunk_id("Book", 5, "text") != make_chunk_id("Book", 5, "other")


# --- parse_curriculum (G1 + G2 + G3) ----------------------------------------


def test_parse_curriculum_sections_and_metadata():
    book = parse_curriculum(_book_pages(), BOOK_META)
    assert book.meta.board == "SNC"
    assert len(book.sections) == 2
    unit, topic = book.sections
    assert unit.level == "unit" and unit.unit == "Unit 3"
    assert unit.page_start == 30 and unit.pdf_page_start == 1
    assert unit.page_end == 31
    assert "Plants make their own food." in unit.text
    assert topic.level == "topic" and topic.topic == "Oxygen"
    assert topic.unit == "Unit 3"
    assert topic.page_start == 31
    assert topic.pdf_page_start == 2


def test_parse_front_matter_before_first_heading():
    pages = [_rec(1, clean="Preface text about the book.\nSome more preface.", printed=1)]
    book = parse_curriculum(pages, BOOK_META)
    assert len(book.sections) == 1
    assert book.sections[0].level == "heading"
    assert "Preface text about the book." in book.sections[0].text


# --- parent-child chunking invariants ----------------------------------------


def test_parent_child_invariants():
    book, units = _build()
    assert len(units) == 2
    for u in units:
        assert u.parent.kind == "parent"
        assert u.parent.unit == u.children[0].unit
        assert all(c.parent_id == u.parent.parent_id for c in u.children)
        assert all(c.kind == "child" for c in u.children)
        words = word_count(u.parent.text_clean)
        # Greedy bound: groups are cut once they would exceed parent_max and are
        # already at least parent_min; only the final group may be undersized.
        assert words <= 50 + 12 - 1  # parent_max + child_max - 1
        assert words >= 30 or u is units[-1]
        for c in u.children:
            assert 0 < word_count(c.text_clean) <= 12  # child_max (single long sentences excepted)


def test_one_sentence_overlap_between_children():
    book, units = _build()
    for u in units:
        for a, b in zip(u.children[:-1], u.children[1:]):
            a_sents = split_sentences(a.text_clean)
            b_sents = split_sentences(b.text_clean)
            assert a_sents[-1] == b_sents[0]


def test_chunks_never_cross_section_boundaries():
    pages = [
        _rec(1, clean="Unit 3: Plants\n" + "\n".join(SENTENCES[:4]), printed=1),
        _rec(2, clean="Unit 4: Animals\n" + "\n".join(SENTENCES[4:]), printed=2),
    ]
    book, units = _build(pages=pages)
    assert len(units) == 2
    for u in units:
        assert u.parent.unit in ("Unit 3", "Unit 4")
        assert all(c.unit == u.parent.unit for c in u.children)


def test_chunk_ids_deterministic():
    c1 = flatten_chunks(_build()[1])[1]
    c2 = flatten_chunks(_build()[1])[1]
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]
    first = c1[0]
    assert first.chunk_id == make_chunk_id(first.book, first.page_start, first.text_clean)


def test_chunk_metadata_flows_from_section():
    book, units = _build()
    child = units[0].children[0]
    assert child.board == "SNC" and child.grade == "5" and child.subject == "English"
    assert child.book == "New Oxford Modern English" and child.edition == "SNC"
    assert child.unit == "Unit 3" and child.language == "english"
    assert child.pipeline_version == "test-v1"
    assert child.section_level == "unit"
    assert child.text_lexical == make_lexical_text(child.text_clean)


def test_chunk_printed_pages():
    book, units = _build()
    unit_parent = units[0].parent
    assert unit_parent.page_start == 30 and unit_parent.page_end == 31
    assert any(c.page_start == 30 and c.page_end == 31 for c in units[0].children)
    topic_parent = units[1].parent
    assert topic_parent.page_start == 31 and topic_parent.page_end == 31


def test_child_embed_text_has_heading_chain():
    book, units = _build()
    unit_child = units[0].children[0]
    assert unit_child.embed_text.startswith("Unit: Unit 3")
    assert unit_child.text_clean in unit_child.embed_text
    topic_child = units[1].children[0]
    assert "Topic: Oxygen" in topic_child.embed_text


def test_parent_id_format():
    book, units = _build()
    pid = units[0].parent.parent_id
    assert pid.startswith("unit_03")
    assert pid.endswith("_parent_000")
    topic_pid = units[1].parent.parent_id
    assert topic_pid.endswith("_parent_001")
    assert "topic_oxygen" in topic_pid


def test_flatten_chunks():
    book, units = _build()
    parents, children = flatten_chunks(units)
    assert len(parents) == len(units) == 2
    # 20-word sections at ~10 words/child -> 3 children per section.
    assert len(children) == 6


def test_empty_sections_produce_no_units():
    pages = [_rec(1, clean="Unit 3: Plants and their food", printed=1)]  # heading only
    book, units = _build(pages=pages)
    assert book.sections  # section exists
    assert units == []


# --- watermark keyword defense-in-depth (mirrors cleaning.py's F1c) --------


def test_parse_curriculum_skips_watermark_keyword_lines():
    """If a watermark keyword still shows up in clean_text (e.g. cleaning
    already ran before the keyword was added to config), parse_curriculum
    must never turn it into a heading or fold it into body text."""
    pages = [
        _rec(1, clean="Chapter 1: Plants\nPlants make food.\nProperty Of ABC School\nSunlight helps them grow.", printed=1),
    ]
    book = parse_curriculum(pages, BOOK_META, watermark_keywords=["property of"])
    section = book.sections[0]
    assert "Property Of ABC School" not in section.text_lines
    assert not any("property of" in line.lower() for line in section.text_lines)
    assert section.title == "Plants"  # the real chapter heading, unaffected


def test_parse_curriculum_watermark_line_never_becomes_a_heading():
    """A watermark phrase that would otherwise pass the generic title-case
    heading heuristic must not fragment the section or become its title."""
    pages = [
        _rec(1, clean="Chapter 1: Plants\nPlants make food.\nSample Preview Copy\nMore plant facts here.", printed=1),
    ]
    book = parse_curriculum(pages, BOOK_META, watermark_keywords=["sample preview"])
    # Only one section should exist — the watermark line must not have
    # triggered a new Section via the generic-heading detector.
    assert len(book.sections) == 1
    assert book.sections[0].title == "Plants"


def test_parse_curriculum_no_keywords_configured_is_a_no_op():
    pages = [
        _rec(1, clean="Chapter 1: Plants\nPlants make food.\nSunlight helps them grow.", printed=1),
    ]
    book_with = parse_curriculum(pages, BOOK_META, watermark_keywords=[])
    book_without = parse_curriculum(pages, BOOK_META)
    assert book_with.sections[0].text_lines == book_without.sections[0].text_lines


# --- Chunk.seq: true sibling reading order -----------------------------------


def test_children_get_sequential_seq_within_their_section():
    book, units = _build()
    for unit in units:
        seqs = [c.seq for c in unit.children]
        assert seqs == sorted(seqs)  # non-decreasing
        assert seqs == list(range(len(seqs)))  # 0, 1, 2, ... within the parent group


def test_seq_resets_per_section_not_globally():
    """Each section's children are numbered from 0 — seq is section-local
    (siblings only ever come from the same section/parent group), not a
    running count across the whole book."""
    book, units = _build()
    assert len(units) >= 2
    first_seqs = [c.seq for c in units[0].children]
    second_seqs = [c.seq for c in units[1].children]
    assert first_seqs[0] == 0
    assert second_seqs[0] == 0


# --- per-chapter running-head echo: a repeated chapter heading must not
# fragment the chapter into one section per page ----------------------------
#
# Found via a real 134-page book test: a chapter running head ("Chapter 01
# Classification of Living Organisms") printed at the top of every page in
# that chapter only covers that chapter's own page range (~12% of the whole
# book) — far under cleaning.py's book-global 70% frequency threshold, so
# F1 never strips it. Left unfixed, parse_curriculum treated every repeat as
# a new chapter start, fragmenting one real 16-page chapter into 16 separate
# one-page "chapters" all sharing the same title.


def test_repeated_chapter_heading_does_not_fragment_the_chapter():
    pages = [
        _rec(1, clean="Chapter 1: Plants\n" + SENTENCES[0], printed=1),
        _rec(2, clean="Chapter 1: Plants\n" + SENTENCES[1], printed=2),
        _rec(3, clean="Chapter 1: Plants\n" + SENTENCES[2], printed=3),
    ]
    book = parse_curriculum(pages, BOOK_META)
    chapters = [s for s in book.sections if s.level == "chapter"]
    assert len(chapters) == 1
    assert chapters[0].page_start == 1 and chapters[0].page_end == 3
    assert chapters[0].text_lines == SENTENCES[:3]


def test_repeated_unit_heading_does_not_fragment_the_unit():
    pages = [
        _rec(1, clean="Unit 3: Plants\n" + SENTENCES[0], printed=1),
        _rec(2, clean="Unit 3: Plants\n" + SENTENCES[1], printed=2),
    ]
    book = parse_curriculum(pages, BOOK_META)
    units_found = [s for s in book.sections if s.level == "unit"]
    assert len(units_found) == 1
    assert units_found[0].page_start == 1 and units_found[0].page_end == 2


def test_genuinely_new_chapter_after_a_repeat_still_starts_a_new_section():
    pages = [
        _rec(1, clean="Chapter 1: Plants\n" + SENTENCES[0], printed=1),
        _rec(2, clean="Chapter 1: Plants\n" + SENTENCES[1], printed=2),  # running-head echo
        _rec(3, clean="Chapter 2: Animals\n" + SENTENCES[2], printed=3),  # genuine new chapter
    ]
    book = parse_curriculum(pages, BOOK_META)
    chapters = [s for s in book.sections if s.level == "chapter"]
    assert [c.title for c in chapters] == ["Plants", "Animals"]
    assert chapters[0].page_start == 1 and chapters[0].page_end == 2
    assert chapters[1].page_start == 3 and chapters[1].page_end == 3


def test_generic_heading_level_is_not_deduplicated_by_running_head_fix():
    """The running-head fix only applies to unit/chapter/topic — a generic
    ('heading'-level) title CAN legitimately repeat within a chapter (e.g.
    multiple "Fun Fact" callout boxes on different pages of the same
    chapter), so those must still create separate sections each time."""
    pages = [
        _rec(1, clean="Fun Fact\n" + SENTENCES[0], printed=1),
        _rec(2, clean="Fun Fact\n" + SENTENCES[1], printed=2),
    ]
    book = parse_curriculum(pages, BOOK_META)
    generic = [s for s in book.sections if s.level == "heading" and s.title == "Fun Fact"]
    assert len(generic) == 2


# --- ignore_heading_keywords: curated suppression of known bad headings ----
#
# Found via the same real-book test: an OCR-misread diagram caption ("Part
# of Plant Monocot Plant Oo Dicot Plant") passed the generic title-case
# heading heuristic and got promoted to a section, silently absorbing 12
# pages of real chapter content under the wrong title.


def test_ignore_heading_keywords_demotes_matching_heading_to_body_text():
    pages = [
        _rec(1, clean="Chapter 1: Plants\n" + SENTENCES[0]
             + "\nPart of Plant Monocot Plant Oo Dicot Plant\n" + SENTENCES[1], printed=1),
    ]
    book = parse_curriculum(pages, BOOK_META, ignore_heading_keywords=["oo dicot"])
    # Only the real chapter section exists — the garbled caption never
    # became its own section.
    assert len(book.sections) == 1
    assert book.sections[0].level == "chapter"
    # The garbled line survives as ordinary body text rather than being
    # dropped outright (unlike a watermark line, we can't be sure it's
    # worthless, so it's kept as plain content).
    assert "Part of Plant Monocot Plant Oo Dicot Plant" in book.sections[0].text_lines


def test_ignore_heading_keywords_no_config_is_a_no_op():
    pages = [
        _rec(1, clean="Chapter 1: Plants\n" + SENTENCES[0]
             + "\nPart of Plant Monocot Plant Oo Dicot Plant\n" + SENTENCES[1], printed=1),
    ]
    book = parse_curriculum(pages, BOOK_META)  # no ignore_heading_keywords
    # Without the curated entry, the garbled line IS promoted to a heading,
    # fragmenting the chapter — this confirms the test above is meaningful,
    # not vacuous.
    assert len(book.sections) == 2


def test_ignore_heading_keywords_does_not_affect_unrelated_headings():
    pages = [
        _rec(1, clean="Chapter 1: Plants\n" + SENTENCES[0]
             + "\nChapter 2: Animals\n" + SENTENCES[1], printed=1),
    ]
    book = parse_curriculum(pages, BOOK_META, ignore_heading_keywords=["oo dicot"])
    chapters = [s for s in book.sections if s.level == "chapter"]
    assert [c.title for c in chapters] == ["Plants", "Animals"]
