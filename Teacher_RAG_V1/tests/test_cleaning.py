"""Unit tests for src/cleaning.py (Phase 2: SVG F, F1-F4)."""
from __future__ import annotations

from src.cleaning import (
    clean_book,
    clean_page,
    compile_watermark_patterns,
    detect_repeated_headers_footers,
    detect_repeated_lines_anywhere,
    find_page_number,
    flag_pattern_watermarks,
    protect_structures,
    reconstruct_sentences_paragraphs,
    remove_watermarks_noise,
)
from src.ocr import PageRecord


# --- F1: repeated header/footer detection ----------------------------------


def test_detect_repeated_headers_footers_finds_common_footer():
    pages = [
        "New Oxford Modern English\n\nContent of page one about plants.\n\n1",
        "New Oxford Modern English\n\nContent of page two about animals.\n\n2",
        "New Oxford Modern English\n\nContent of page three about birds.\n\n3",
    ]
    artifacts = detect_repeated_headers_footers(pages)
    assert "new oxford modern english" in artifacts
    assert "content of page one about plants." not in artifacts


def test_detect_requires_frequency_threshold():
    pages = [f"Header on page {i}\n\nSome unique content {i}.\n\n{i}" for i in range(1, 5)]
    assert detect_repeated_headers_footers(pages, min_frequency=0.9) == set()


def test_detect_ignores_pure_page_numbers():
    pages = [f"Title\n\nContent {i}.\n\n{i}" for i in range(1, 6)]
    artifacts = detect_repeated_headers_footers(pages)
    assert "title" in artifacts
    assert not any(a.isdigit() for a in artifacts)


def test_middle_repeated_lines_not_flagged():
    pages = []
    for i in range(1, 6):
        pages.append(
            "\n".join(
                [
                    "RUNNING HEAD",
                    f"Unique first {i}",
                    f"Unique second {i}",
                    "REPEATED MIDDLE ONE",
                    "REPEATED MIDDLE TWO",
                    f"Unique third {i}",
                    f"Unique fourth {i}",
                    f"{i}",
                ]
            )
        )
    artifacts = detect_repeated_headers_footers(pages)
    assert "running head" in artifacts
    assert "repeated middle one" not in artifacts
    assert "repeated middle two" not in artifacts


def test_running_head_flagged_even_when_heading_like():
    # "Unit 3" on the top of >=70% of pages is a running head, not a content heading.
    pages = [f"Unit 3: Plants\n\nContent {i}.\n\n{i}" for i in range(1, 6)]
    artifacts = detect_repeated_headers_footers(pages)
    assert "unit 3: plants" in artifacts


# --- F2: watermark and noise removal ---------------------------------------


def test_artifacts_removed_content_kept():
    raw = "NEW OXFORD MODERN ENGLISH\n\nThe cat sat on the mat.\n\n42"
    cleaned = clean_page(raw, 1, artifacts=["new oxford modern english"])
    assert "New Oxford" not in cleaned
    assert "The cat sat on the mat." in cleaned


def test_f1_artifacts_removed_even_if_heading_like():
    raw = "Unit 3: Plants and their food\n\nSome paragraph here."
    cleaned = clean_page(raw, 1, artifacts=["unit 3: plants and their food"])
    assert "Unit 3" not in cleaned
    assert "Some paragraph here." in cleaned


def test_once_only_heading_not_removed():
    raw = "Unit 3: Plants and their food\n\nThe cat sat."
    cleaned = clean_page(raw, 1)
    assert "Unit 3: Plants and their food" in cleaned


def test_noise_lines_removed_real_words_kept():
    raw = "~~~~~~\nReal content line\n%%%%%%%\n===="
    cleaned = clean_page(raw, 1)
    assert "Real content line" in cleaned
    for noise in ("~~~~~~", "%%%%%%%", "===="):
        assert noise not in cleaned


def test_content_heading_preserved_even_when_repeated_in_page():
    raw = "\n".join(["Unit 3: Plants", "Unit 3: Plants", "Unit 3: Plants", "The cat sat on the mat."])
    cleaned = clean_page(raw, 1)
    assert cleaned.count("Unit 3: Plants") == 3  # watermark rule never drops headings
    assert "The cat sat on the mat." in cleaned


def test_in_page_watermark_removed():
    raw = "\n".join(["Prose line one.", "SAMPLE", "SAMPLE", "SAMPLE", "Prose line two."])
    cleaned = clean_page(raw, 1)
    assert "SAMPLE" not in cleaned
    assert "Prose line one." in cleaned


# --- F3: sentence and paragraph reconstruction ------------------------------


def test_join_hard_wrapped_lines():
    raw = "The process\nof photosynthesis\nhappens in leaves."
    assert reconstruct_sentences_paragraphs(raw) == "The process of photosynthesis happens in leaves."


def test_hyphenation_rejoined():
    raw = "The photo-\nsynthesis process."
    assert reconstruct_sentences_paragraphs(raw) == "The photosynthesis process."


def test_paragraphs_kept_separate():
    raw = "First paragraph line one.\nline two.\n\nSecond paragraph starts here."
    joined = reconstruct_sentences_paragraphs(raw)
    assert joined == "First paragraph line one. line two.\n\nSecond paragraph starts here."


# --- F4: structure preservation ---------------------------------------------


def test_protect_structures_marks_heading_and_table():
    raw = "Unit 3: Plants\n\nA | B\n1 | 2\n\nSome prose line."
    protected, body = protect_structures(raw)
    assert sorted(p.kind for p in protected) == ["heading", "table"]


def test_heading_never_merged_into_paragraph():
    raw = "Unit 3: Plants and their food\nPhotosynthesis happens\nin leaves."
    cleaned = clean_page(raw, 1)
    lines = [ln for ln in cleaned.splitlines() if ln.strip()]
    assert lines == ["Unit 3: Plants and their food", "Photosynthesis happens in leaves."]


def test_table_preserved_through_clean_page():
    raw = "Word   Meaning\ncat    an animal\nsun    a star\n\nThe cat sat on the mat."
    cleaned = clean_page(raw, 1)
    assert "cat    an animal" in cleaned
    assert "sun    a star" in cleaned
    assert "The cat sat on the mat." in cleaned


def test_pipe_table_preserved():
    raw = "A | B | C\n1 | 2 | 3\n2 | 4 | 6\n\nSome prose after the table."
    cleaned = clean_page(raw, 1)
    assert "A | B | C" in cleaned
    assert "1 | 2 | 3" in cleaned
    assert "Some prose after the table." in cleaned


def test_exercise_line_preserved():
    raw = "Exercise 2: Fill in the blanks\n\nPlants make their own food by ___."
    cleaned = clean_page(raw, 1)
    assert "Exercise 2: Fill in the blanks" in cleaned


# --- page numbers -----------------------------------------------------------


def test_find_page_number_bottom():
    assert find_page_number("Text here.\n\n42") == 42


def test_find_page_number_none():
    assert find_page_number("No numbers at the edges here.") is None


def test_find_page_number_ignores_middle_numbers():
    lines = ["Top", "Line two", "Line three", "42", "Line four", "Line five", "Bottom"]
    assert find_page_number("\n".join(lines)) is None


def test_page_number_removed_from_cleaned_text():
    raw = "Some content line.\n\n42"
    cleaned = clean_page(raw, 1, artifacts=["42"])
    assert "42" not in cleaned
    assert "Some content line." in cleaned


# --- book-level integration -------------------------------------------------


def test_clean_book_removes_footer_sets_page_numbers():
    records = [
        PageRecord(
            pdf_path="book.pdf", page_number=1,
            raw_text="New Oxford Modern English\n\nThe sun rises\nin the east.\n\n5",
        ),
        PageRecord(
            pdf_path="book.pdf", page_number=2,
            raw_text="New Oxford Modern English\n\nWater boils\nat hundred degrees.\n\n6",
        ),
    ]
    clean_book(records)
    for rec in records:
        assert "New Oxford Modern English" not in rec.clean_text
        assert rec.printed_page is not None
        assert rec.clean_text
    assert records[0].printed_page == 5
    assert records[1].printed_page == 6
    assert "The sun rises in the east." in records[0].clean_text
    assert "5" not in records[0].clean_text
    assert records[0].clean_text != records[0].raw_text


def test_clean_book_empty_ok():
    assert clean_book([]) == []


def test_clean_book_respects_existing_printed_page():
    records = [
        PageRecord(
            pdf_path="book.pdf", page_number=1, printed_page=10,
            raw_text="Some content here.\n\n99",
        )
    ]
    clean_book(records)
    assert records[0].printed_page == 10  # not overwritten


# --- F1b: position-agnostic cross-page watermark detection -----------------


def test_detect_repeated_lines_anywhere_catches_middle_of_page():
    """The exact scenario detect_repeated_headers_footers can't see (it only
    scans the top/bottom window) — a diagonal/centered watermark repeated in
    the middle of every page."""
    pages = [
        "Chapter 1: Plants\n\nPlants make food using sunlight.\nSAMPLE COPY\nThis is called photosynthesis.",
        "Water and carbon dioxide are needed.\nSAMPLE COPY\nLeaves capture sunlight.",
        "Chapter 2: Animals\n\nAnimals need food and water.\nSAMPLE COPY\nSome animals hunt for food.",
    ]
    artifacts = detect_repeated_lines_anywhere(pages, min_frequency=0.5)
    assert "sample copy" in artifacts
    assert "plants make food using sunlight." not in artifacts


def test_detect_repeated_lines_anywhere_requires_floor_of_two():
    """Regression test: at n=2 pages, ceil(2 * 0.5) rounds down to 1, which
    would previously flag content unique to a SINGLE page as "repeated" —
    silently gutting real body text on short documents. A line must appear
    on at least 2 pages, no matter how the percentage rounds."""
    pages = [
        "New Oxford Modern English\n\nThe sun rises in the east.",
        "New Oxford Modern English\n\nWater boils at a hundred degrees.",
    ]
    artifacts = detect_repeated_lines_anywhere(pages, min_frequency=0.5)
    assert "the sun rises in the east." not in artifacts
    assert "water boils at a hundred degrees." not in artifacts
    assert "new oxford modern english" in artifacts  # this one DOES repeat


def test_detect_repeated_headers_footers_also_respects_floor_of_two():
    """Same floor-of-2 regression guard, for the F1 (header/footer) detector."""
    pages = [
        "Unique Title A\n\nContent about plants.\n\n1",
        "Unique Title B\n\nContent about animals.\n\n2",
    ]
    artifacts = detect_repeated_headers_footers(pages, min_frequency=0.5)
    assert "unique title a" not in artifacts
    assert "unique title b" not in artifacts


def test_clean_book_removes_watermark_from_two_page_book_without_losing_content():
    """End-to-end: the exact bug caught during verification — clean_book on
    a short (2-page) book must not wipe out real content while still
    catching a genuinely repeated watermark line."""
    records = [
        PageRecord(pdf_path="y.pdf", page_number=1,
                   raw_text="New Oxford Modern English\n\nThe sun rises\nin the east.\n\n5"),
        PageRecord(pdf_path="y.pdf", page_number=2,
                   raw_text="New Oxford Modern English\n\nWater boils\nat hundred degrees.\n\n6"),
    ]
    clean_book(records, {"watermark": {"enabled": True, "cross_page_frequency": 0.5}})
    assert records[0].clean_text == "The sun rises in the east."
    assert records[1].clean_text == "Water boils at hundred degrees."


# --- F1c: keyword/regex watermark matching (dynamic watermarks) ------------


def test_flag_pattern_watermarks_matches_keyword_once_is_enough():
    text = "Plants make food.\nPROPERTY OF ABC SCHOOL\nSunlight helps them grow."
    flagged = flag_pattern_watermarks(text, keywords=["property of"])
    assert "property of abc school" in flagged
    assert "plants make food." not in flagged


def test_flag_pattern_watermarks_matches_dynamic_regex():
    """The class of watermark frequency-based detection can never catch:
    different text on every page (a baked-in username/timestamp)."""
    text = "Some real content.\nDownloaded by ali92 on 2026-01-02"
    flagged = flag_pattern_watermarks(
        text, patterns=compile_watermark_patterns([r"downloaded by .+ on \d{4}-\d{2}-\d{2}"])
    )
    assert "downloaded by ali92 on 2026-01-02" in flagged
    assert "some real content." not in flagged


def test_flag_pattern_watermarks_no_config_flags_nothing():
    text = "Plants make food.\nSAMPLE"
    assert flag_pattern_watermarks(text) == set()


def test_clean_book_removes_dynamic_watermark_via_regex_config():
    records = [
        PageRecord(pdf_path="x.pdf", page_number=1,
                   raw_text="Chapter 1: Plants\n\nPlants make food.\nDownloaded by ali92 on 2026-01-02"),
        PageRecord(pdf_path="x.pdf", page_number=2,
                   raw_text="Water is needed too.\nDownloaded by sara01 on 2026-01-03"),
    ]
    cfg = {"watermark": {"enabled": True, "regex_patterns": [r"downloaded by .+ on \d{4}-\d{2}-\d{2}"]}}
    clean_book(records, cfg)
    assert "downloaded by" not in records[0].clean_text.lower()
    assert "downloaded by" not in records[1].clean_text.lower()
    assert "plants make food." in records[0].clean_text.lower()


def test_clean_book_watermark_disabled_skips_all_new_detectors():
    records = [
        PageRecord(pdf_path="x.pdf", page_number=1, raw_text="SAMPLE\nContent one."),
        PageRecord(pdf_path="x.pdf", page_number=2, raw_text="SAMPLE\nContent two."),
    ]
    cfg = {"watermark": {"enabled": False, "keywords": ["sample"]}}
    clean_book(records, cfg)
    # keywords are configured but watermark detection is disabled -> "SAMPLE"
    # only gets removed if it independently trips F1 (it does, since it's a
    # top-line running head repeated on both pages) — but F1c/F1b must not add.
    assert "content one." in records[0].clean_text.lower()
    assert "content two." in records[1].clean_text.lower()


# --- style-flagged lines carried over from ocr.py (record.watermark_lines) --


def test_clean_book_removes_ocr_style_flagged_watermark_lines():
    """ocr.py's font/rotation/color detector (C1.5) flags lines on the
    PageRecord itself before cleaning even runs; clean_book must fold those
    into its artifact set."""
    rec = PageRecord(
        pdf_path="z.pdf", page_number=1,
        raw_text="Plants make food.\nCONFIDENTIAL DRAFT\nSunlight helps them grow.",
    )
    rec.watermark_lines = ["confidential draft"]
    clean_book([rec], {"watermark": {"enabled": True}})
    assert "confidential draft" not in rec.clean_text.lower()
    assert "plants make food." in rec.clean_text.lower()


# --- F4: table-protection must not bypass a known artifact ------------------


def test_watermark_rendered_as_wide_spaced_line_not_protected_as_table():
    """A watermark with wide letter-spacing looks like a 2-column 'table
    row' to _table_line's heuristic. If protect_structures grabbed it as a
    table before F2 could drop it, it would survive verbatim (F4 bypass)."""
    text = "S A M P L E   W A T E R M A R K\nReal content line one.\nReal content line two."
    artifacts = {"s a m p l e   w a t e r m a r k"}
    protected, body = protect_structures(text, artifacts=artifacts)
    # Must NOT have been captured as a protected table span.
    assert not any(p.kind == "table" and "watermark" in p.original.lower() for p in protected)
    # It stays in the body, where F2 (remove_watermarks_noise) will drop it.
    cleaned = remove_watermarks_noise(body, artifacts=artifacts)
    assert "watermark" not in cleaned.lower()
    assert "real content line one." in cleaned.lower()
