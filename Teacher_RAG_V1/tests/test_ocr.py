"""Unit + guarded integration tests for src/ocr.py (Phases 0-1)."""
from __future__ import annotations

import numpy as np
import pytest

from src.ocr import (
    RegionText,
    TextBlock,
    compare_quality,
    deskew,
    detect_text_blocks,
    detect_watermark_lines,
    estimate_skew_angle,
    extract_embedded_blocks,
    extract_embedded_text,
    hash_file,
    ingest_pdf,
    ocr_page_native,
    ocr_regions,
    preprocess,
    select_merge_best,
    tesseract_available,
    text_quality_score,
)

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import fitz

    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

needs_cv2 = pytest.mark.skipif(not HAS_CV2, reason="opencv not installed")
needs_fitz = pytest.mark.skipif(not HAS_FITZ, reason="PyMuPDF not installed")


# --- text_quality_score (pure) --------------------------------------------


def test_text_quality_empty_and_garbage():
    assert text_quality_score("") == 0.0
    assert text_quality_score("   ") == 0.0
    assert text_quality_score("####  !!  ...") == 0.0


def test_text_quality_sentence_scores_high():
    score = text_quality_score("Photosynthesis is the process by which plants make food.")
    assert score >= 0.5


def test_text_quality_garbage_scores_low():
    score = text_quality_score("qwerty12345 xyz98765 abc12345 !!!!")
    assert score < 0.5


# --- compare_quality (pure) -----------------------------------------------


def test_embedded_preferred_when_ocr_negligible():
    q = compare_quality("A solid embedded text paragraph from the text layer.", "", 0.0)
    assert q.method == "embedded"
    assert q.embedded_score > q.ocr_score


def test_ocr_preferred_when_embedded_missing():
    q = compare_quality(None, "OCR produced this text line.", 85.0)
    assert q.method == "ocr"
    assert q.ocr_score > 0.0


def test_merged_when_both_strong():
    emb = "The quick brown fox jumps over the lazy dog near the riverbank."
    ocr = "The quick brown fox jumps over the lazy dog near the riverbank."
    q = compare_quality(emb, ocr, 90.0)
    assert q.method in ("merged", "embedded")
    assert q.embedded_score > 0 and q.ocr_score > 0


def test_none_when_nothing_extracted():
    q = compare_quality(None, "", 0.0)
    assert q.method == "none"


# --- hash_file ------------------------------------------------------------


def test_hash_file_deterministic(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"hello world" * 1000)
    assert hash_file(f) == hash_file(f)
    f2 = tmp_path / "b.pdf"
    f2.write_bytes(b"hello world" * 1000 + b"!")
    assert hash_file(f) != hash_file(f2)


# --- preprocessing (cv2) --------------------------------------------------


@needs_cv2
def test_preprocess_returns_binary_image():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 80, size=(200, 200, 3), dtype=np.uint8)
    out = preprocess(img)
    assert out.ndim == 2
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 255}


@needs_cv2
def test_preprocess_rejects_even_kernel():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        preprocess(img, denoise_strength=4)


@needs_cv2
def test_deskew_straightens_rotated_text():
    # Synthetic "text": a thick horizontal bar, then rotated by -7 degrees.
    img = np.full((300, 400), 255, dtype=np.uint8)
    cv2.rectangle(img, (60, 130), (340, 170), 0, -1)
    center = (img.shape[1] / 2.0, img.shape[0] / 2.0)
    m = cv2.getRotationMatrix2D(center, -7.0, 1.0)
    rotated = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderValue=255)
    assert abs(estimate_skew_angle(rotated)) > 2.0
    corrected = deskew(rotated)
    assert abs(estimate_skew_angle(corrected)) < 1.5


@needs_cv2
def test_detect_text_blocks_finds_two_blocks():
    img = np.zeros((400, 300), dtype=np.uint8)
    cv2.rectangle(img, (50, 40), (250, 80), 255, -1)  # block 1
    cv2.rectangle(img, (50, 200), (250, 240), 255, -1)  # block 2
    boxes = detect_text_blocks(img)
    assert len(boxes) == 2
    ys = sorted(b[1] for b in boxes)
    assert ys[1] > ys[0]


# --- select_merge_best (pure) ---------------------------------------------


def _region(x, y, w, h, text, conf=0.9):
    return RegionText(bbox=(x, y, w, h), text=text, confidence=conf)


def test_merge_uses_embedded_text_for_overlapping_region():
    blocks = [TextBlock(0.1, 0.10, 0.5, 0.2, "Embedded paragraph text here")]
    regions = [_region(50, 48, 200, 30, "OCR noisy line")]  # normalized y 0.12-0.195
    merged = select_merge_best(blocks, regions, img_shape=(400, 400, 3))
    lines = [ln for ln in merged.splitlines() if ln.strip()]
    assert lines == ["Embedded paragraph text here"]


def test_merge_falls_back_to_ocr_when_no_overlap():
    blocks = [TextBlock(0.8, 0.8, 0.95, 0.9, "Far away embedded text")]
    regions = [_region(50, 48, 200, 30, "OCR only line here")]
    merged = select_merge_best(blocks, regions, img_shape=(400, 400, 3))
    lines = [ln for ln in merged.splitlines() if ln.strip()]
    # The OCR region keeps its own text; the uncovered embedded block is kept too.
    assert lines == ["OCR only line here", "Far away embedded text"]


def test_merge_includes_uncovered_blocks_and_orders_top_down():
    blocks = [
        TextBlock(0.1, 0.02, 0.5, 0.08, "Top embedded block"),
        TextBlock(0.1, 0.50, 0.5, 0.56, "Bottom embedded block"),
    ]
    regions = [_region(50, 160, 200, 30, "Middle OCR text")]  # normalized y 0.40-0.475
    merged = select_merge_best(blocks, regions, img_shape=(400, 400, 3))
    lines = [ln for ln in merged.splitlines() if ln.strip()]
    assert lines == ["Top embedded block", "Middle OCR text", "Bottom embedded block"]


def test_merge_empty_regions_returns_embedded_text():
    blocks = [TextBlock(0.1, 0.1, 0.5, 0.2, "Only embedded content here")]
    merged = select_merge_best(blocks, [], img_shape=(400, 400, 3))
    assert merged.strip() == "Only embedded content here"


def test_merge_ignores_tiny_embedded_blocks():
    blocks = [TextBlock(0.1, 0.1, 0.5, 0.2, "42")]
    regions = [_region(50, 48, 200, 30, "Real OCR paragraph here")]
    merged = select_merge_best(blocks, regions, img_shape=(400, 400, 3))
    assert merged.strip() == "Real OCR paragraph here"


def test_merge_does_not_duplicate_one_block_matched_by_multiple_regions():
    """Regression test for a real bug found via end-to-end testing: when a
    single embedded text block spans an area covered by SEVERAL OCR-detected
    line regions (the common case where the embedded layer has one coarse
    block per paragraph/page but OCR splits into per-line boxes), the block
    must be emitted only ONCE — not once per overlapping region. Before the
    fix, every matching region independently "won" the same block, so its
    full text was duplicated N times, and cleaning.py's in-page-repeat
    watermark filter would then strip the "repeated" prose entirely."""
    # One big embedded block spanning most of the page...
    blocks = [TextBlock(0.05, 0.05, 0.95, 0.95, "Plants make their own food using sunlight.")]
    # ...overlapped by three separate OCR line regions (as real per-line OCR
    # detection would produce against one coarse embedded block).
    regions = [
        _region(20, 20, 300, 20, "line one ocr text"),
        _region(20, 60, 300, 20, "line two ocr text"),
        _region(20, 100, 300, 20, "line three ocr text"),
    ]
    merged = select_merge_best(blocks, regions, img_shape=(400, 400, 3))
    occurrences = merged.count("Plants make their own food using sunlight.")
    assert occurrences == 1, f"embedded block duplicated {occurrences} times, expected 1"
    # The regions that lost the block still contribute their own OCR text,
    # rather than being silently dropped.
    assert "line two ocr text" in merged
    assert "line three ocr text" in merged


# --- ocr_regions with injected tesseract ----------------------------------


@needs_cv2
def test_ocr_regions_with_injected_tesseract():
    img = np.zeros((400, 300), dtype=np.uint8)
    cv2.rectangle(img, (50, 40), (250, 80), 255, -1)
    cv2.rectangle(img, (50, 200), (250, 240), 255, -1)

    calls: list[tuple[tuple, int]] = []

    def fake_tesseract(crop, langs, psm):
        calls.append((langs, psm))
        return f"block{len(calls)} text", 88.0

    regions = ocr_regions(img, ["eng"], psm=6, tesseract=fake_tesseract)
    assert len(regions) == 2
    assert regions[0].bbox[1] < regions[1].bbox[1]  # top-to-bottom order
    assert all(r.text.startswith("block") for r in regions)
    assert all(r.confidence == 88.0 for r in regions)
    assert all(langs == ("eng",) and psm == 6 for langs, psm in calls)


# --- ocr_page_native: Tesseract's own block/par/line grouping --------------
#
# Found via real-book testing: ocr_regions (above) badly fragments text on
# complex multi-column/callout-box textbook layouts. ocr_page_native groups
# Tesseract's own word-level output by (block_num, par_num, line_num)
# instead, in Tesseract's own reading order.


def _fake_image_to_data(rows: list[tuple[str, int, int, int, int, int, int, int, str]]):
    """Build a fake pytesseract.image_to_data()-shaped dict from
    ``(block, par, line, left, top, width, height, conf, text)`` tuples."""
    keys = ["block_num", "par_num", "line_num", "left", "top", "width", "height", "conf", "text"]
    data = {k: [] for k in keys}
    for row in rows:
        for k, v in zip(keys, row):
            data[k].append(v)
    return data


def test_ocr_page_native_groups_words_into_lines_by_tesseract_hierarchy():
    # Two words on the same (block, par, line) -> one joined line; a
    # different line key -> a separate region.
    rows = [
        (1, 1, 1, 10, 10, 30, 12, 95, "Reptiles"),
        (1, 1, 1, 45, 10, 20, 12, 96, "are"),
        (1, 1, 2, 10, 30, 40, 12, 94, "creeping"),
        (1, 1, 2, 55, 30, 40, 12, 93, "animals."),
    ]

    def fake_image_to_data(img, lang, config):
        return _fake_image_to_data(rows)

    regions = ocr_page_native(np.zeros((100, 100), dtype=np.uint8), image_to_data=fake_image_to_data)
    assert [r.text for r in regions] == ["Reptiles are", "creeping animals."]


def test_ocr_page_native_preserves_tesseract_reading_order():
    # Reading order follows FIRST appearance of each (block,par,line) key in
    # Tesseract's own output, not pixel position — this is the whole point:
    # trust Tesseract's own layout analysis instead of re-deriving order.
    rows = [
        (2, 1, 1, 0, 0, 10, 10, 90, "second"),
        (1, 1, 1, 0, 0, 10, 10, 90, "first"),
    ]

    def fake_image_to_data(img, lang, config):
        return _fake_image_to_data(rows)

    regions = ocr_page_native(np.zeros((10, 10), dtype=np.uint8), image_to_data=fake_image_to_data)
    assert [r.text for r in regions] == ["second", "first"]


def test_ocr_page_native_skips_blank_words():
    rows = [
        (1, 1, 1, 0, 0, 10, 10, 90, ""),
        (1, 1, 1, 10, 0, 10, 10, 90, "   "),
        (1, 1, 1, 20, 0, 10, 10, 90, "real"),
    ]

    def fake_image_to_data(img, lang, config):
        return _fake_image_to_data(rows)

    regions = ocr_page_native(np.zeros((10, 10), dtype=np.uint8), image_to_data=fake_image_to_data)
    assert [r.text for r in regions] == ["real"]


def test_ocr_page_native_bbox_spans_all_words_in_the_line():
    rows = [
        (1, 1, 1, 10, 10, 30, 12, 95, "one"),
        (1, 1, 1, 60, 8, 20, 14, 95, "two"),
    ]

    def fake_image_to_data(img, lang, config):
        return _fake_image_to_data(rows)

    regions = ocr_page_native(np.zeros((100, 100), dtype=np.uint8), image_to_data=fake_image_to_data)
    x, y, w, h = regions[0].bbox
    assert x == 10 and y == 8  # min left, min top across both words
    assert x + w == 80  # max(left+width) = 60+20
    assert y + h == 22  # max(top+height) = 8+14


def test_ocr_page_native_averages_confidence_ignoring_negative_conf():
    # Tesseract emits conf=-1 for non-text entries; those must not be
    # averaged into the line confidence.
    rows = [
        (1, 1, 1, 0, 0, 10, 10, 90, "word1"),
        (1, 1, 1, 10, 0, 10, 10, -1, "word2"),
        (1, 1, 1, 20, 0, 10, 10, 80, "word3"),
    ]

    def fake_image_to_data(img, lang, config):
        return _fake_image_to_data(rows)

    regions = ocr_page_native(np.zeros((10, 10), dtype=np.uint8), image_to_data=fake_image_to_data)
    assert regions[0].confidence == 85.0  # (90 + 80) / 2, -1 excluded


def test_ocr_page_native_default_uses_real_pytesseract_signature(monkeypatch):
    """The default (no injected image_to_data) path must call the real
    pytesseract with the documented signature — verified without requiring
    the Tesseract binary by monkeypatching pytesseract itself."""
    calls = []

    class _FakePytesseract:
        class Output:
            DICT = "dict"

        @staticmethod
        def image_to_data(img, lang=None, config=None, output_type=None):
            calls.append((lang, config, output_type))
            return _fake_image_to_data([(1, 1, 1, 0, 0, 10, 10, 90, "ok")])

    import sys
    monkeypatch.setitem(sys.modules, "pytesseract", _FakePytesseract())
    regions = ocr_page_native(np.zeros((10, 10), dtype=np.uint8), langs=("eng",), psm=3)
    assert [r.text for r in regions] == ["ok"]
    assert calls == [("eng", "--psm 3", "dict")]


# --- integration: ingest_pdf ----------------------------------------------


def _make_text_pdf(tmp_path, text: str) -> "fitz.Document":
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf = tmp_path / "sample.pdf"
    doc.save(str(pdf))
    doc.close()
    return pdf


@needs_fitz
def test_ingest_text_layer_pdf_embedded_only(tmp_path):
    pdf = _make_text_pdf(tmp_path, "Photosynthesis is the process by which plants make food.")
    records = ingest_pdf(pdf, {"dpi": 150, "ocr": {"enabled": False}})
    assert len(records) == 1
    rec = records[0]
    assert rec.page_number == 1
    assert rec.extraction_method == "embedded"
    assert "Photosynthesis" in rec.raw_text
    assert rec.ocr_confidence == 0.0
    assert rec.source_hash == hash_file(pdf)
    assert rec.pdf_path == str(pdf)


def test_ingest_raises_when_ocr_enabled_without_tesseract(tmp_path):
    if tesseract_available():
        pytest.skip("Tesseract present; this test covers the missing-binary case")
    if not HAS_FITZ:
        pytest.skip("PyMuPDF not installed")
    pdf = _make_text_pdf(tmp_path, "Hello text layer.")
    with pytest.raises(RuntimeError, match="Tesseract"):
        ingest_pdf(pdf, {"dpi": 150})


# --- C1.5: style-based watermark detection (font size / color / rotation) --
#
# Uses a fake PyMuPDF page (matching the get_text("dict") schema) so these
# run without needing PyMuPDF installed — no @needs_fitz guard required.


class _FakeRect:
    def __init__(self, width, height):
        self.width = width
        self.height = height


class _FakePage:
    """Minimal stand-in for a PyMuPDF page: only what detect_watermark_lines,
    extract_embedded_text, and extract_embedded_blocks actually touch."""

    def __init__(self, dict_data, plain_text="", width=600, height=800):
        self._dict_data = dict_data
        self._plain_text = plain_text
        self.rect = _FakeRect(width, height)

    def get_text(self, mode):
        if mode == "dict":
            return self._dict_data
        if mode == "text":
            return self._plain_text
        raise NotImplementedError(mode)


def _span(text, size=12.0, color=0x000000):
    return {"text": text, "size": size, "color": color}


def _line(text, size=12.0, color=0x000000, dir=(1.0, 0.0), bbox=(0, 0, 100, 12)):
    return {"dir": dir, "bbox": bbox, "spans": [_span(text, size, color)]}


def _text_block(lines, bbox=(0, 0, 100, 100)):
    return {"type": 0, "bbox": bbox, "lines": lines}


def _page_dict(lines):
    return {"blocks": [_text_block(lines)]}


def test_detect_watermark_flags_rotated_line():
    lines = [
        _line("Plants make their own food using sunlight.", size=12.0, dir=(1.0, 0.0)),
        _line("Sunlight, water and carbon dioxide are needed.", size=12.0, dir=(1.0, 0.0)),
        _line("SAMPLE WATERMARK", size=12.0, dir=(0.7, 0.7)),  # ~45 degrees
    ]
    page = _FakePage(_page_dict(lines))
    flagged, diag = detect_watermark_lines(page, {})
    assert "sample watermark" in flagged
    assert "plants make their own food using sunlight." not in flagged


def test_detect_watermark_flags_oversized_line():
    lines = [
        _line("Plants make their own food using sunlight.", size=12.0),
        _line("Sunlight, water and carbon dioxide are needed.", size=12.0),
        _line("CONFIDENTIAL", size=40.0),  # far above body median
    ]
    page = _FakePage(_page_dict(lines))
    flagged, diag = detect_watermark_lines(page, {})
    assert "confidential" in flagged
    assert "plants make their own food using sunlight." not in flagged


def test_detect_watermark_flags_light_colored_line():
    lines = [
        _line("Plants make their own food using sunlight.", size=12.0, color=0x000000),
        _line("Sunlight, water and carbon dioxide are needed.", size=12.0, color=0x000000),
        _line("NOT FOR SALE", size=12.0, color=0xEFEFEF),  # near-white
    ]
    page = _FakePage(_page_dict(lines))
    flagged, diag = detect_watermark_lines(page, {})
    assert "not for sale" in flagged
    assert "plants make their own food using sunlight." not in flagged


def test_detect_watermark_normal_body_text_not_flagged():
    lines = [
        _line("Plants make their own food using sunlight.", size=12.0),
        _line("Sunlight, water and carbon dioxide are needed.", size=12.0),
        _line("Leaves capture sunlight for energy.", size=12.0),
    ]
    page = _FakePage(_page_dict(lines))
    flagged, diag = detect_watermark_lines(page, {})
    assert flagged == set()


def test_detect_watermark_disabled_flags_nothing():
    lines = [_line("SAMPLE WATERMARK", size=12.0, dir=(0.7, 0.7))]
    page = _FakePage(_page_dict(lines))
    flagged, diag = detect_watermark_lines(page, {"watermark": {"enabled": False}})
    assert flagged == set()


def test_detect_watermark_respects_configured_thresholds():
    # A line only slightly rotated (below the configured threshold) should
    # NOT be flagged when the threshold is loosened.
    lines = [
        _line("Plants make their own food using sunlight.", size=12.0),
        _line("Sunlight, water and carbon dioxide are needed.", size=12.0),
        _line("Slightly tilted line of body text here.", size=12.0, dir=(0.999, 0.04)),
    ]
    page = _FakePage(_page_dict(lines))
    flagged, diag = detect_watermark_lines(page, {"watermark": {"rotation_sin_threshold": 0.5}})
    assert "slightly tilted line of body text here." not in flagged


# --- extract_embedded_text / extract_embedded_blocks with exclude_lines ----


def test_extract_embedded_text_no_exclusions_matches_plain_text():
    page = _FakePage({}, plain_text="Hello world.")
    assert extract_embedded_text(page) == "Hello world."


def test_extract_embedded_text_drops_excluded_lines():
    lines = [
        _line("Plants make food.", size=12.0),
        _line("SAMPLE COPY", size=12.0),
        _line("Sunlight helps them grow.", size=12.0),
    ]
    page = _FakePage(_page_dict(lines))
    text = extract_embedded_text(page, exclude_lines={"sample copy"})
    assert "SAMPLE COPY" not in text
    assert "Plants make food." in text
    assert "Sunlight helps them grow." in text


def test_extract_embedded_blocks_drops_excluded_lines_and_empty_blocks():
    watermark_only_block = _text_block([_line("SAMPLE COPY", size=12.0)], bbox=(0, 0, 100, 12))
    content_block = _text_block(
        [_line("Plants make food.", size=12.0), _line("Sunlight helps them grow.", size=12.0)],
        bbox=(0, 20, 100, 40),
    )
    page = _FakePage({"blocks": [watermark_only_block, content_block]})
    blocks = extract_embedded_blocks(page, exclude_lines={"sample copy"})
    # The watermark-only block disappears entirely (no surviving lines);
    # the content block survives with its lines intact.
    assert len(blocks) == 1
    assert "Plants make food." in blocks[0].text
    assert "SAMPLE COPY" not in blocks[0].text
