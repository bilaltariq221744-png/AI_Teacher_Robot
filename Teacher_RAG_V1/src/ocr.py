"""OCR and ingestion layer (SVG nodes A-E, PDF §3).

Responsibilities
----------------
* C1 - extract the embedded text layer from a PDF page.
* C2 - render a page to a high-DPI image (300 DPI default).
* C3 - preprocess the image: deskew, denoise, CLAHE, threshold.
* C4 - layout-aware OCR via Tesseract (per-block, reading order).
* D  - compare embedded-text vs OCR quality per page.
* E  - select or merge the best text per region.

The heavy dependencies (PyMuPDF, OpenCV, pytesseract) are imported lazily so
that the pure helpers (quality comparison, region merge) run in lightweight
environments (e.g. CI without the Tesseract binary). ``ocr_regions`` accepts
an injectable tesseract runner for the same reason.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RegionText:
    """A block of OCR text plus its location on the page image.

    ``bbox`` is ``(x, y, width, height)`` in image pixels.
    """

    bbox: tuple[int, int, int, int]
    text: str
    confidence: float = 0.0


@dataclass
class TextBlock:
    """A block from the embedded text layer.

    Coordinates are normalized to [0, 1] relative to the page, so they remain
    comparable to image-space regions regardless of DPI.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass
class PageQuality:
    """Result of the embedded-vs-OCR quality comparison (SVG D)."""

    embedded_score: float
    ocr_score: float
    method: str  # "embedded" | "ocr" | "merged" | "none"


@dataclass
class PageRecord:
    """Auditable per-page record (PDF §3: raw + cleaned text + quality notes)."""

    pdf_path: str
    page_number: int  # 1-based PDF page
    printed_page: Optional[int] = None  # filled later by the structure parser (G3)
    raw_text: str = ""
    clean_text: str = ""  # refined by cleaning.py (Phase 2); equals raw initially
    extraction_method: str = "embedded"
    ocr_confidence: float = 0.0
    source_hash: str = ""
    # Watermark audit trail (style-based defense, C1.5): normalized lines
    # dropped from the embedded-text layer because they looked like a stamp
    # (rotated / oversized / low-contrast), and how many were removed.
    watermark_lines: list[str] = field(default_factory=list)
    watermark_removed_count: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hash_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """sha256 hex digest of a file — the stable source hash stored on records."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _words(text: str) -> list[str]:
    return re.findall(r"\b[\w']+\b", text or "", flags=re.UNICODE)


def text_quality_score(text: str) -> float:
    """Heuristic 0..1 score for raw extracted text (used by D).

    Rewards content-rich text (word coverage, alphabetic ratio, average word
    length) and penalizes empty or garbage text. Compares the embedded text
    layer against OCR output.
    """
    t = (text or "").strip()
    if not t:
        return 0.0
    words = _words(t)
    if not words:
        return 0.0
    alpha_ratio = sum(1 for ch in t if ch.isalpha()) / max(1, len(t))
    avg_word_len = min(1.0, sum(len(w) for w in words) / len(words) / 5.0)
    coverage = min(1.0, len(words) / 40.0)
    score = 0.4 * coverage + 0.4 * alpha_ratio + 0.2 * avg_word_len
    return round(min(1.0, score), 3)


def _bbox_overlap_ratio(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection area of boxes a and b (x0,y0,x1,y1) divided by area of a."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    area_a = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
    return (ix * iy) / area_a


def tesseract_available() -> bool:
    """True when the Tesseract binary is on PATH (runtime OCR requirement)."""
    return shutil.which("tesseract") is not None


def _import_fitz():
    """PyMuPDF: prefer the modern ``pymupdf`` module, fall back to ``fitz``."""
    try:
        import pymupdf as fitz  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - older PyMuPDF releases
        import fitz  # type: ignore[no-redef]
    return fitz


def _norm_line(line: str) -> str:
    """Normalized form used for watermark-line matching (mirrors cleaning.py)."""
    return re.sub(r"\s+", " ", line or "").strip().lower()


# ---------------------------------------------------------------------------
# C1.5 - style-based watermark detection (font size / color / rotation)
# ---------------------------------------------------------------------------
#
# Born-digital watermarks ("SAMPLE", "Property of ...", diagonal publisher
# stamps) are real text in the PDF, so a plain get_text("text") call returns
# them mixed in with body content. They almost always differ from body text
# in at least one of: rotation (diagonal), font size (oversized stamp), or
# color (light gray / low-contrast overlay). This uses PyMuPDF's
# get_text("dict") output, which exposes per-line direction and per-span
# size/color, to flag those lines *before* they ever reach cleaning.py.
#
# This only helps the embedded-text path. Scanned/rasterized watermarks have
# no font metadata and fall through to cleaning.py's keyword/regex/frequency
# defenses instead (they see the OCR'd text like any other line).


def _line_text(line: dict) -> str:
    return "".join(span.get("text", "") for span in line.get("spans", [])).strip()


def _line_avg_size(line: dict) -> float:
    sizes = [s.get("size", 0.0) for s in line.get("spans", []) if (s.get("text") or "").strip()]
    return (sum(sizes) / len(sizes)) if sizes else 0.0


def _line_color(line: dict) -> int:
    for span in line.get("spans", []):
        if (span.get("text") or "").strip():
            return int(span.get("color", 0) or 0)
    return 0


def _is_light_color(color_int: int, threshold: int = 200) -> bool:
    """True when an sRGB packed color is near-white (low-contrast overlay text)."""
    r, g, b = (color_int >> 16) & 255, (color_int >> 8) & 255, color_int & 255
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance >= threshold


def _median(values: Sequence[float]) -> float:
    vals = sorted(v for v in values if v > 0)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2 == 0:
        return (vals[mid - 1] + vals[mid]) / 2.0
    return vals[mid]


def detect_watermark_lines(
    page, cfg: Optional[Mapping[str, object]] = None
) -> tuple[set[str], list[dict]]:
    """Flag watermark-styled lines on a page via font size/color/rotation (C1.5).

    Any one signal is enough to flag a line:
      * rotated text (line direction far from horizontal) — diagonal stamps
      * font size far above the page's median body size — oversized stamps
      * near-white / very light color — low-contrast overlay text

    Returns ``(normalized_flagged_lines, diagnostics)``. Controlled by the
    ``watermark:`` config section; returns ``(set(), [])`` when disabled.
    """
    wcfg = dict((cfg or {}).get("watermark") or {})
    if not wcfg.get("enabled", True):
        return set(), []

    rot_threshold = float(wcfg.get("rotation_sin_threshold", 0.05))
    size_ratio_min = float(wcfg.get("font_size_ratio_min_large", 1.6))
    light_threshold = int(wcfg.get("light_color_threshold", 200))

    data = page.get_text("dict")
    lines_info: list[dict] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block, 1 = image block
            continue
        for line in block.get("lines", []):
            text = _line_text(line)
            if not text:
                continue
            dirv = line.get("dir", (1.0, 0.0))
            rotated = abs(dirv[1]) > rot_threshold
            lines_info.append(
                {
                    "text": text,
                    "size": _line_avg_size(line),
                    "color": _line_color(line),
                    "rotated": rotated,
                }
            )

    # Body font size is the median of horizontal lines with enough text to be
    # representative prose (short horizontal lines skew the median low).
    body_size = _median([li["size"] for li in lines_info if not li["rotated"] and len(li["text"]) >= 8])

    flagged: set[str] = set()
    diagnostics: list[dict] = []
    for li in lines_info:
        size_ratio = (li["size"] / body_size) if body_size else 1.0
        too_large = body_size > 0 and size_ratio >= size_ratio_min
        too_light = _is_light_color(li["color"], light_threshold)
        if li["rotated"] or too_large or too_light:
            flagged.add(_norm_line(li["text"]))
            diagnostics.append(
                {
                    "text": li["text"],
                    "rotated": li["rotated"],
                    "size_ratio": round(size_ratio, 2),
                    "too_light": too_light,
                }
            )
    return flagged, diagnostics


# ---------------------------------------------------------------------------
# C1 - embedded text extraction
# ---------------------------------------------------------------------------


def extract_embedded_text(page, exclude_lines: Optional[set[str]] = None) -> Optional[str]:
    """Extract the embedded text layer of a PyMuPDF page (C1).

    ``exclude_lines`` (normalized, from ``detect_watermark_lines``) drops
    matching lines before reassembly. Default behavior (no exclusions) is
    unchanged from the original implementation.
    """
    if not exclude_lines:
        text = page.get_text("text")
        text = (text or "").strip()
        return text or None

    data = page.get_text("dict")
    kept: list[str] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = _line_text(line)
            if text and _norm_line(text) not in exclude_lines:
                kept.append(text)
    joined = "\n".join(kept).strip()
    return joined or None


def extract_embedded_blocks(page, exclude_lines: Optional[set[str]] = None) -> list[TextBlock]:
    """Text-layer blocks in reading order, normalized to [0,1] (input for E).

    ``exclude_lines`` drops watermark-flagged lines from each block before it
    is emitted (a block with no surviving lines is skipped entirely). Default
    behavior (no exclusions) is unchanged from the original implementation.
    """
    rect = page.rect
    if rect.width <= 0 or rect.height <= 0:
        return []

    if not exclude_lines:
        out = []
        for x0, y0, x1, y1, text, _block_no, _block_type in page.get_text("blocks"):
            if not (text or "").strip():
                continue
            out.append(
                TextBlock(
                    x0=x0 / rect.width,
                    y0=y0 / rect.height,
                    x1=x1 / rect.width,
                    y1=y1 / rect.height,
                    text=text.strip(),
                )
            )
        return out

    out = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        bx0, by0, bx1, by1 = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        lines_kept = [
            _line_text(line)
            for line in block.get("lines", [])
            if _line_text(line) and _norm_line(_line_text(line)) not in exclude_lines
        ]
        if not lines_kept:
            continue
        out.append(
            TextBlock(
                x0=bx0 / rect.width,
                y0=by0 / rect.height,
                x1=bx1 / rect.width,
                y1=by1 / rect.height,
                text="\n".join(lines_kept).strip(),
            )
        )
    return out


# ---------------------------------------------------------------------------
# C2 - page rendering
# ---------------------------------------------------------------------------


def render_page(page, dpi: int = 300) -> np.ndarray:
    """Render a PDF page to a BGR numpy image at ``dpi`` (C2)."""
    import cv2

    fitz = _import_fitz()
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img


# ---------------------------------------------------------------------------
# C3 - image preprocessing
# ---------------------------------------------------------------------------


def _to_gray(img: np.ndarray) -> np.ndarray:
    import cv2

    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate the text skew angle (degrees) of a grayscale image.

    Uses the minimum-area rectangle around all ink pixels. Returns ~0 for
    already-upright pages.
    """
    import cv2

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return 0.0
    _center, (w, h), angle = cv2.minAreaRect(coords)
    if w < h:  # first edge is the short side; re-anchor to the long side
        angle = 90.0 + angle
    if angle > 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0
    return float(angle)


def deskew(img: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """Rotate the image so text lines are horizontal (C3)."""
    import cv2

    gray = _to_gray(img)
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return img
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def clahe(gray: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization."""
    import cv2

    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)


def preprocess(
    img: np.ndarray, denoise_strength: int = 3, clahe_clip: float = 2.0, clahe_grid: int = 8
) -> np.ndarray:
    """C3 pipeline: deskew -> denoise -> CLAHE -> threshold.

    Returns a binary image (black text on white) ready for OCR.
    """
    import cv2

    if denoise_strength % 2 == 0:
        raise ValueError("denoise_strength must be an odd integer")
    gray = _to_gray(img)
    gray = deskew(gray)
    gray = cv2.medianBlur(gray, denoise_strength)
    gray = clahe(gray, clahe_clip, clahe_grid)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# C4 - layout-aware OCR
# ---------------------------------------------------------------------------


def detect_text_blocks(binary: np.ndarray, min_area: int = 100) -> list[tuple[int, int, int, int]]:
    """Group white-on-black text into blocks (layout detection).

    Returns ``(x, y, w, h)`` boxes in image pixels, top-to-bottom then
    left-to-right.
    """
    import cv2

    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
    block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, line_kernel)
    blocks = cv2.morphologyEx(joined, cv2.MORPH_CLOSE, block_kernel)
    _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(blocks)
    boxes = []
    for i in range(1, labels.max() + 1):
        x, y, w, h, area = stats[i]
        if w < 10 or h < 5 or area < min_area:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    boxes.sort(key=lambda b: (b[1] // 24, b[0]))
    return boxes


def _tesseract_block(crop: np.ndarray, langs: Sequence[str], psm: int) -> tuple[str, float]:
    """Run Tesseract on a single block crop; returns (text, mean confidence)."""
    import pytesseract

    lang = "+".join(langs)
    data = pytesseract.image_to_data(
        crop, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    parts: list[str] = []
    confs: list[float] = []
    for word, conf in zip(data["text"], data["conf"]):
        if not (word or "").strip():
            continue
        parts.append(word)
        try:
            c = float(conf)
        except (TypeError, ValueError):
            c = -1.0
        if c >= 0:
            confs.append(c)
    text = " ".join(parts).strip()
    confidence = sum(confs) / len(confs) if confs else 0.0
    return text, confidence


TesseractRunner = Callable[[np.ndarray, Sequence[str], int], tuple[str, float]]


def ocr_regions(
    img: np.ndarray,
    langs: Sequence[str] = ("eng",),
    psm: int = 6,
    tesseract: Optional[TesseractRunner] = None,
    min_block_text_words: int = 1,
) -> list[RegionText]:
    """Layout-aware OCR (C4, legacy path): detect blocks via connected
    components, OCR each crop independently in reading order.

    ``tesseract`` is injectable for testing without the Tesseract binary.

    This is the original block-detection approach: a hand-rolled
    morphological connected-component detector groups ink into boxes, then
    each box is OCR'd separately. It works reasonably for simple, single-
    column layouts, but real textbook pages with multi-column text and
    sidebar/callout boxes (two boxes side by side, dense mixed layouts)
    confuse the row-band sort and produce badly fragmented, out-of-order
    text — verified on a real illustrated textbook page during testing. For
    that case, prefer ``ocr_page_native`` (SVG C4b), which lets Tesseract's
    own page-segmentation engine do the layout analysis instead. Kept here,
    unmodified, as the ``ocr.layout_mode: blocks`` fallback and because
    several tests exercise it directly.
    """
    import cv2

    runner = tesseract or _tesseract_block
    gray = _to_gray(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Foreground (text) is the minority pixel class; keep it white.
    fg = cv2.bitwise_not(binary) if np.count_nonzero(binary) > binary.size // 2 else binary

    regions: list[RegionText] = []
    for x, y, w, h in detect_text_blocks(fg):
        pad = 4
        crop = fg[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad]
        text, conf = runner(crop, tuple(langs), psm)
        text = (text or "").strip()
        if len(_words(text)) < min_block_text_words:
            continue
        regions.append(RegionText(bbox=(x, y, w, h), text=text, confidence=round(float(conf), 3)))
    return regions


# ---------------------------------------------------------------------------
# C4b - native-layout OCR (Tesseract's own page segmentation)
# ---------------------------------------------------------------------------
#
# Found via real-book testing: the custom connected-component block detector
# above (detect_text_blocks + per-crop OCR) badly fragments text on complex
# textbook layouts — multi-column body text, side-by-side "Do you know?" /
# "Point to Ponder!" callout boxes, illustrated pages. Words end up as
# separate single-word "regions", scrambling reading order and duplicating
# content. Tesseract's own PSM 3 (fully automatic page segmentation) layout
# engine handles this far better: it groups words into its own block/
# paragraph/line hierarchy, which we can read directly from
# ``image_to_data`` instead of re-implementing layout detection ourselves.
# On a real sample page this turned "Their\nskin\nthick,\nand\ndry which\n
# protect their\ncreeping\nis\nare\ncoarse\nbody from" into the correct,
# coherent "Reptiles are called creeping animals. Their skin is thick,
# coarse and dry which protect their body from external effects."
#
# This is not a perfect layout analyzer either (e.g. it can still merge two
# side-by-side callout boxes onto one "line" if they sit in the same
# horizontal band), but it is a substantial, measured improvement over the
# connected-component approach for realistic textbook pages.

TesseractDataFn = Callable[[np.ndarray, str, str], dict]


def _tesseract_image_to_data(img: np.ndarray, lang: str, config: str) -> dict:
    import pytesseract

    return pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DICT)


def ocr_page_native(
    img: np.ndarray,
    langs: Sequence[str] = ("eng",),
    psm: int = 3,
    image_to_data: Optional[TesseractDataFn] = None,
    min_line_words: int = 1,
) -> list[RegionText]:
    """Native-layout OCR (C4b): one whole-page Tesseract call, grouped by
    Tesseract's own block/paragraph/line hierarchy instead of a custom
    connected-component detector.

    ``image_to_data`` is injectable for testing without the Tesseract
    binary; it must match ``pytesseract.image_to_data(img, lang=..,
    config=.., output_type=pytesseract.Output.DICT)``'s return shape (a
    dict of parallel lists: ``text, conf, block_num, par_num, line_num,
    left, top, width, height``).

    Words are grouped into lines by ``(block_num, par_num, line_num)``, in
    the order Tesseract first emits each line — this preserves Tesseract's
    own reading-order decision (its page segmentation already handles
    columns reasonably well), rather than re-deriving order from pixel
    coordinates the way ``ocr_regions`` does.
    """
    lang = "+".join(langs)
    getter = image_to_data or _tesseract_image_to_data
    data = getter(img, lang, f"--psm {psm}")

    texts = data.get("text", [])
    lines: dict[tuple, list[tuple[str, float, int, int, int, int]]] = {}
    order: list[tuple] = []
    for i in range(len(texts)):
        word = (texts[i] or "").strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = []
            order.append(key)
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        lines[key].append(
            (word, conf, data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        )

    regions: list[RegionText] = []
    for key in order:
        words = lines[key]
        if len(words) < min_line_words:
            continue
        text = " ".join(w[0] for w in words)
        xs = [w[2] for w in words]
        ys = [w[3] for w in words]
        x2s = [w[2] + w[4] for w in words]
        y2s = [w[3] + w[5] for w in words]
        x0, y0 = min(xs), min(ys)
        w_, h_ = max(x2s) - x0, max(y2s) - y0
        confs = [w[1] for w in words if w[1] >= 0]
        conf = sum(confs) / len(confs) if confs else 0.0
        regions.append(RegionText(bbox=(x0, y0, w_, h_), text=text, confidence=round(conf, 3)))
    return regions


# ---------------------------------------------------------------------------
# D - page-level quality comparison
# ---------------------------------------------------------------------------


def compare_quality(embedded: Optional[str], ocr_text: str, ocr_conf: float = 0.0) -> PageQuality:
    """Compare embedded text vs OCR quality for a page (D).

    ``method``: "embedded" (OCR negligible), "ocr" (embedded weak/missing),
    "merged" (both strong; region merge worthwhile), "none" (nothing usable).
    """
    emb = text_quality_score(embedded or "")
    ocr = text_quality_score(ocr_text)
    conf_factor = 0.6 + 0.4 * min(1.0, float(ocr_conf) / 90.0)
    ocr = round(min(1.0, ocr * conf_factor), 3)

    if emb < 0.05 and ocr < 0.05:
        method = "none"
    elif emb >= ocr:
        method = "merged" if ocr >= 0.05 else "embedded"
    else:
        method = "ocr"
    return PageQuality(embedded_score=emb, ocr_score=ocr, method=method)


# ---------------------------------------------------------------------------
# E - select or merge the best text by region
# ---------------------------------------------------------------------------


def select_merge_best(
    embedded_blocks: Sequence[TextBlock],
    ocr_regions_: Sequence[RegionText],
    img_shape: tuple[int, int, int],
    min_block_text_words: int = 3,
    overlap_threshold: float = 0.15,
) -> str:
    """Select or merge the best text per region (E).

    For each OCR region, prefer a substantive overlapping text-layer block;
    otherwise keep the OCR text. Text-layer blocks that cover no OCR region
    are also included, and everything is emitted in top-to-bottom order.
    """
    if not ocr_regions_:
        return "\n".join(
            b.text.strip() for b in embedded_blocks if len(_words(b.text)) >= min_block_text_words
        )

    h, w = int(img_shape[0]), int(img_shape[1])
    pieces: list[tuple[float, str]] = []
    used = [False] * len(embedded_blocks)

    for region in ocr_regions_:
        x, y, rw, rh = region.bbox
        region_box = (x / w, y / h, (x + rw) / w, (y + rh) / h)
        best_i, best_overlap = -1, overlap_threshold
        for i, block in enumerate(embedded_blocks):
            if used[i]:
                # Already matched to an earlier OCR region this pass — an
                # embedded block must never be emitted more than once. This
                # is the common case where OCR splits a page into many
                # per-line regions but the embedded text layer has one
                # coarser block (e.g. one block per paragraph); without this
                # guard, every OCR region that overlaps that single block
                # would independently "win" it, duplicating its full text
                # once per region (and cleaning.py's in-page-repeat
                # watermark filter would then strip the duplicated prose
                # entirely, since it looks exactly like a repeated stamp).
                continue
            if len(_words(block.text)) < min_block_text_words:
                continue
            overlap = _bbox_overlap_ratio(region_box, (block.x0, block.y0, block.x1, block.y1))
            if overlap > best_overlap:
                best_i, best_overlap = i, overlap
        if best_i >= 0:
            used[best_i] = True
            pieces.append((region_box[1], embedded_blocks[best_i].text.strip()))
        else:
            pieces.append((region_box[1], region.text.strip()))

    for i, block in enumerate(embedded_blocks):
        if not used[i] and len(_words(block.text)) >= min_block_text_words:
            pieces.append(((block.y0 + block.y1) / 2.0, block.text.strip()))

    pieces.sort(key=lambda p: p[0])
    return "\n".join(text for _top, text in pieces if text)


# ---------------------------------------------------------------------------
# Ingestion orchestrator (A -> E)
# ---------------------------------------------------------------------------


def ingest_pdf(
    pdf_path: str | Path, cfg: Optional[Mapping[str, object]] = None
) -> list[PageRecord]:
    """Run the ingestion pipeline over a PDF (A -> E) and return page records.

    ``cfg`` keys (all optional):
        dpi: int = 300
        ocr_langs: list[str] = ["eng"]
        ocr.enabled: bool = True   # False -> embedded-text-only mode
    """
    fitz = _import_fitz()

    cfg = dict(cfg or {})
    dpi = int(cfg.get("dpi", 300))
    langs = list(cfg.get("ocr_langs", ["eng"]))
    ocr_enabled = bool((cfg.get("ocr") or {}).get("enabled", True))
    layout_mode = str((cfg.get("ocr") or {}).get("layout_mode", "native")).lower()

    pdf_path = Path(pdf_path)
    source_hash = hash_file(pdf_path)

    if ocr_enabled and not tesseract_available():
        raise RuntimeError(
            "OCR is enabled but the Tesseract binary was not found on PATH. "
            "Install Tesseract, or set ocr.enabled=false for text-layer PDFs."
        )

    records: list[PageRecord] = []
    doc = fitz.open(str(pdf_path))
    try:
        for pno in range(len(doc)):
            page = doc[pno]
            watermark_lines, _wm_diag = detect_watermark_lines(page, cfg)
            embedded = extract_embedded_text(page, exclude_lines=watermark_lines)
            embedded_blocks = extract_embedded_blocks(page, exclude_lines=watermark_lines)

            if ocr_enabled:
                img = render_page(page, dpi)
                binary = preprocess(img)
                if layout_mode == "blocks":
                    regions = ocr_regions(binary, langs)
                else:
                    regions = ocr_page_native(binary, langs)
                ocr_text = "\n".join(r.text for r in regions)
                ocr_conf = sum(r.confidence for r in regions) / max(1, len(regions))
            else:
                regions = []
                ocr_text = ""
                ocr_conf = 0.0

            quality = compare_quality(embedded, ocr_text, ocr_conf)
            if ocr_enabled and regions:
                merged = select_merge_best(embedded_blocks, regions, img.shape)
            else:
                merged = embedded or ""

            records.append(
                PageRecord(
                    pdf_path=str(pdf_path),
                    page_number=pno + 1,
                    raw_text=merged.strip(),
                    clean_text=merged.strip(),
                    extraction_method=quality.method,
                    ocr_confidence=round(ocr_conf, 3),
                    source_hash=source_hash,
                    watermark_lines=sorted(watermark_lines),
                    watermark_removed_count=len(watermark_lines),
                )
            )
    finally:
        doc.close()
    return records
