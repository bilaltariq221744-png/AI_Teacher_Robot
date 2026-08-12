# NOME Teaching Guide 1 — FAISS Vector Database

A page-accurate, lightweight FAISS vector database built from
`New_Oxford_Modern_English_TG_1_SNC.pdf` (10 pages).

## How it was built (accuracy-first, not copy-paste)

1. **Rasterize** — every PDF page rendered to a 200 DPI JPEG
   (`data/page_images/page-XX.jpg`).
2. **Dual extraction per page** — for each page, text is pulled two
   ways: (a) the embedded PDF text layer, (b) Tesseract OCR run
   directly on the rasterized image. The cleaner of the two (by a
   character-quality score) is kept as that page's canonical text.
   This catches cases where a text layer is garbled or missing
   (common in scanned/complimentary-copy PDFs) instead of blindly
   trusting either source.
3. **Page-based output** — one clean `.txt` file per page
   (`data/pages/page_XX.txt`), plus `data/manifest.json` recording
   which extraction method won for each page and character counts,
   so you can audit the extraction.
4. **Chunking** — each page is split into paragraph-level chunks
   (merged if too short, split if too long), each chunk tagged with
   its page number and a link back to its source page image.
5. **Embeddings** — TF-IDF → Truncated SVD (LSA), L2-normalized,
   20-dim (auto-sized to the corpus). Fully local/offline — no model
   download required — which keeps this package small and fast while
   giving solid semantic retrieval for a document this size.
6. **Index** — `faiss.IndexFlatIP` (cosine similarity via inner
   product on normalized vectors), exact search, instant for a
   corpus this size.

## Structure

```
nome_vdb/
├── data/
│   ├── pages/            # page_01.txt ... page_10.txt (clean, page-based text)
│   ├── page_images/      # page-01.jpg ... page-10.jpg (source rasters, for audit)
│   └── manifest.json     # per-page extraction method + char counts
├── index/
│   ├── faiss.index       # the FAISS vector index
│   ├── metadata.json     # one record per chunk: page, text, source file/image
│   ├── vectorizer.pkl    # fitted TF-IDF vectorizer
│   └── svd.pkl           # fitted SVD (dense embedding) model
├── scripts/
│   ├── 01_extract_pages.py   # PDF -> page images -> OCR/text-layer -> data/pages/
│   ├── 02_build_index.py     # data/pages/ -> chunks -> embeddings -> FAISS index
│   └── search.py             # query the index from the command line
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
# tesseract-ocr binary must also be installed (only needed to rebuild, not to query)
```

## Query the database

```bash
python3 scripts/search.py "phonics vs look and say method" --k 5
```

Output includes, per result: page number, similarity score, matching
text snippet, and paths to the source `.txt` file and page image —
so every answer traces back to an exact page you can open and verify.

## Rebuild from scratch (if you replace the PDF)

```bash
python3 scripts/01_extract_pages.py   # re-extract text page by page
python3 scripts/02_build_index.py     # re-embed and rebuild the FAISS index
```

## Using it in your own code

```python
import sys
sys.path.insert(0, "scripts")
from search import search

results = search("pre-reading activities", k=3)
for r in results:
    print(r["page"], r["score"], r["text"][:150])
```

## Notes

- Page 2 is intentionally empty (a genuine blank page in the source
  PDF) — confirmed by checking its rasterized image, not just an
  extraction failure.

## Changelog — v2 pipeline (implements the review in
`OCR_and_Vector_Index_Analysis.docx`)

**`scripts/01_extract_pages.py`**
- Image preprocessing before OCR: deskew → denoise → CLAHE contrast →
  adaptive threshold (OpenCV; falls back to a PIL-only pipeline if
  `opencv-python` isn't installed).
- Layout-aware OCR via `pytesseract.image_to_data`: separates
  header-band / body / footer-band text by position, and reconstructs
  paragraphs from block/line grouping instead of one flat text dump —
  this is what fixes line-wrapped fragments ("The boy\nwent\nhome." →
  "The boy went home.").
- Corpus-wide boilerplate detection: header/footer lines that recur
  on ≥40% of pages (running headers, "Oxford University Press",
  page numbers) are stripped automatically.
- Conservative spell correction: fixes OCR noise like "Educatlon" →
  "Education", while leaving capitalized words alone unless the fix
  is a single-character edit — protects proper nouns (student/place
  names) that just aren't in the dictionary.
- Manifest now also records boilerplate-lines-removed and whether
  spell correction ran.
- **Known limitation:** header/footer detection is a positional
  heuristic (top/bottom 8% of the page), not a trained layout model.
  On very text-dense pages it can occasionally catch a line or two of
  real body text along with the header. A proper fix would mean
  adding a layout-detection model (e.g. LayoutParser/Detectron2),
  which is a much heavier dependency — flagged here rather than
  silently shipped as if it were fully solved.

**`scripts/02_build_index.py`**
- Real semantic embeddings (`BAAI/bge-small-en-v1.5`, falls back to
  `all-MiniLM-L6-v2`) replace TF-IDF+SVD — this is what makes
  retrieval understand paraphrases/synonyms instead of just keyword
  overlap. Generate once on a laptop; a Pi only needs to load the
  resulting FAISS index (embeddings aren't regenerated per-query).
- Chunking: sentence-split → sliding window sized 250–350 words, with
  ~20% sentence overlap between consecutive chunks, replacing the flat
  "cut every 600 characters" rule.
- Metadata expanded per chunk: `page`, `chapter`, `heading`,
  `chunk_id`, `chunk_position`, `previous_chunk_id`, `next_chunk_id`,
  `source_file`, `word_count` (up from just page/chunk/text).
  `chapter`/`heading` use a regex heuristic ("Unit N", "Chapter N",
  ALL-CAPS lines) — good enough for this teaching guide's structure,
  not a general-purpose heading classifier.
- No more `vectorizer.pkl` / `svd.pkl` — replaced by
  `index/embedding_model.json` recording which model built the index.

**`scripts/search.py`**
- Embeds queries with the same sentence-transformer model recorded at
  build time (loaded lazily/cached — cheap enough for single queries
  even on a Pi).
- Results now include chapter, heading, chunk position, and
  prev/next chunk ids for citation and context-expansion.

**New dependencies** (see `requirements.txt`): `opencv-python-headless`,
`pyspellchecker`, `sentence-transformers`. `scikit-learn` is no longer
required.


## extract_pages.py:

#!/usr/bin/env python3
"""
Page-accurate text extraction for the NOME Teaching Guide PDF.

Pipeline (v2 — see docs/OCR_and_Vector_Index_Analysis.docx for the
rationale behind each stage):

  PDF -> Convert Pages -> Image Enhancement -> Deskew -> Noise Removal ->
  Adaptive Threshold -> OCR -> Text Layer Extraction -> Compare ->
  Choose Best -> Header Removal -> Footer Removal -> Spell Correction ->
  Whitespace Cleanup -> Sentence Reconstruction -> Paragraph
  Reconstruction -> Save Clean Page

  1. Rasterize every page to an image (already done via pdftoppm, 200 DPI).
  2. Preprocess each page image before OCR: deskew, denoise, contrast
     enhancement (CLAHE), adaptive threshold. This is the single biggest
     lever for scanned-book OCR accuracy.
  3. Layout-aware OCR: instead of a single image_to_string() call, use
     Tesseract's word/line/paragraph/block boxes (image_to_data) to
     reconstruct paragraphs properly and to separate header-band /
     footer-band text from body text.
  4. Also pull the embedded PDF text layer per page via pdftotext.
  5. Pick the cleaner of the two per page (quality score), same as v1.
  6. Header/footer removal: lines that repeat near-verbatim across most
     pages (page numbers, running headers) are detected corpus-wide and
     stripped from the saved text.
  7. Spell correction: conservative, dictionary-based correction of
     tokens that look like OCR noise (e.g. "Educatlon" -> "Education"),
     skipping anything that looks intentional (numbers, short words,
     proper-noun-shaped words, words already in the dictionary).
  8. Sentence reconstruction: line-wrapped fragments ("The boy\\nwent\\n
     home.") are rejoined into flowing sentences using paragraph
     grouping from the OCR layout data, not naive newline stripping.
  9. Write one clean .txt file per page + a manifest recording exactly
     what happened at each stage, for auditing.
"""
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pytesseract import Output

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

try:
    from spellchecker import SpellChecker
    _SPELL = SpellChecker()
    HAVE_SPELLCHECK = True
except ImportError:
    _SPELL = None
    HAVE_SPELLCHECK = False
ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "source" / "New Oxford Modern English TG 5 SNC.pdf"
IMG_DIR = ROOT / "data" / "page_images"
OUT_DIR = ROOT / "data" / "pages"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# --- tunables -----------------------------------------------------------
HEADER_BAND = 0.08   # top 8% of page height treated as header band
FOOTER_BAND = 0.08   # bottom 8% of page height treated as footer band
BOILERPLATE_MIN_PAGE_FRACTION = 0.4  # a line repeated on >=40% of pages is boilerplate
SPELLCHECK_MIN_WORD_LEN = 4


# =========================================================================
# 1. Image preprocessing: deskew -> denoise -> contrast -> adaptive threshold
# =========================================================================
def preprocess_image(img_path: Path) -> Image.Image:
    """Return an OCR-ready PIL image: deskewed, denoised, contrast-enhanced,
    adaptively thresholded. Falls back to a PIL-only pipeline if OpenCV
    isn't available."""
    if HAVE_CV2:
        return _preprocess_cv2(img_path)
    return _preprocess_pil_fallback(img_path)


def _preprocess_cv2(img_path: Path) -> Image.Image:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return Image.open(img_path).convert("L")

    # --- deskew: find the dominant text angle and rotate to correct it
    img = _deskew_cv2(img)

    # --- denoise
    img = cv2.fastNlMeansDenoising(img, h=10)

    # --- contrast enhancement (CLAHE handles uneven scan lighting better
    #     than a flat histogram-equalize)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # --- adaptive threshold (per-region binarization; robust to shadows
    #     and uneven scanner lighting across a book page)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )

    return Image.fromarray(img)


def _deskew_cv2(img: "np.ndarray") -> "np.ndarray":
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect angle convention: normalize into [-45, 45]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.1:  # not worth rotating
        return img
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _preprocess_pil_fallback(img_path: Path) -> Image.Image:
    """Best-effort preprocessing without OpenCV: grayscale, autocontrast,
    a median filter for denoising, and a simple global threshold. Deskew
    is skipped in this path (needs OpenCV's minAreaRect)."""
    img = Image.open(img_path).convert("L")
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.point(lambda p: 255 if p > 180 else 0)
    return img


# =========================================================================
# 2. Layout-aware OCR: separate header band / body / footer band, and
#    reconstruct paragraphs from Tesseract's block/par/line grouping
#    instead of a flat image_to_string() dump.
# =========================================================================
def layout_ocr(img: Image.Image):
    """Returns (header_lines, body_paragraphs, footer_lines)."""
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    h = img.height

    words_by_block = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        conf = int(float(data["conf"][i])) if data["conf"][i] not in ("-1", -1) else -1
        if conf != -1 and conf < 20:
            continue  # drop very low-confidence noise tokens
        top = data["top"][i]
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        words_by_block.setdefault(key, {"top": top, "words": []})
        words_by_block[key]["words"].append(word)

    header_lines, footer_lines = [], []
    body_lines = []  # (block_num, par_num, line_num, top, text)
    for (block_num, par_num, line_num), info in sorted(
        words_by_block.items(), key=lambda kv: (kv[1]["top"])
    ):
        line_text = " ".join(info["words"])
        top_frac = info["top"] / h if h else 0
        if top_frac <= HEADER_BAND:
            header_lines.append(line_text)
        elif top_frac >= (1 - FOOTER_BAND):
            footer_lines.append(line_text)
        else:
            body_lines.append((block_num, par_num, line_num, info["top"], line_text))

    # --- paragraph reconstruction: group consecutive body lines that
    # belong to the same (block, par) into one flowing paragraph, joining
    # with spaces so line-wrapped sentences become single sentences
    # instead of "The boy\nwent\nhome."
    paragraphs = []
    cur_key, cur_lines = None, []
    for block_num, par_num, line_num, _top, text in body_lines:
        key = (block_num, par_num)
        if key != cur_key:
            if cur_lines:
                paragraphs.append(" ".join(cur_lines))
            cur_key, cur_lines = key, [text]
        else:
            cur_lines.append(text)
    if cur_lines:
        paragraphs.append(" ".join(cur_lines))

    return header_lines, paragraphs, footer_lines


# =========================================================================
# 3. PDF text-layer extraction (unchanged strategy from v1)
# =========================================================================
def get_layer_text(page_num: int) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", "-f", str(page_num), "-l", str(page_num),
         str(PDF_PATH), "-"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def quality_score(text: str) -> float:
    if not text:
        return 0.0
    ok = sum(1 for c in text if c.isalnum() or c.isspace() or c in ".,;:!?'\"()-—–")
    ratio = ok / len(text)
    return ratio * 1000 + len(text)


# =========================================================================
# 4. Corpus-wide boilerplate (header/footer) detection & removal
# =========================================================================
def normalize_line(line: str) -> str:
    """Normalize a line for boilerplate comparison: collapse whitespace,
    strip page-number-shaped tokens, lowercase."""
    line = re.sub(r"\d+", "#", line.strip().lower())
    line = re.sub(r"\s+", " ", line)
    return line


def find_boilerplate_lines(all_header_lines, all_footer_lines, n_pages):
    """A header/footer line that recurs (near-verbatim, after normalizing
    page numbers) across a large fraction of pages is running boilerplate
    (running headers, footers, page numbers) rather than page content."""
    counter = Counter()
    for lines in all_header_lines + all_footer_lines:
        seen_this_page = set()
        for line in lines:
            norm = normalize_line(line)
            if norm and norm not in seen_this_page:
                counter[norm] += 1
                seen_this_page.add(norm)
    threshold = max(2, int(n_pages * BOILERPLATE_MIN_PAGE_FRACTION))
    return {norm for norm, count in counter.items() if count >= threshold}


# =========================================================================
# 5. Spell correction — conservative, only fixes OCR-shaped noise
# =========================================================================
_WORD_RE = re.compile(r"[A-Za-z]+")


def _levenshtein_le1(a: str, b: str) -> bool:
    """True if edit distance between a and b is <= 1 (cheap check,
    avoids pulling in a dependency just for this)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = sum(1 for x, y in zip(a, b) if x != y)
        return diffs <= 1
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    i = j = skipped = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] != longer[j]:
            skipped += 1
            if skipped > 1:
                return False
            j += 1
        else:
            i += 1
            j += 1
    return True


def spell_correct(text: str) -> str:
    """Conservative OCR-noise correction. Lowercase unknown words are
    corrected normally (very unlikely to be a name deliberately
    lowercased). Capitalized unknown words are only corrected when the
    fix is a single-character slip (e.g. "Educatlon" -> "Education",
    an l/o OCR confusion) — this catches the OCR-noise case from the
    review doc while leaving genuine capitalized proper nouns (student
    names, place names) that just aren't in the dictionary untouched."""
    if not HAVE_SPELLCHECK or not text:
        return text

    def fix(match):
        word = match.group(0)
        if len(word) < SPELLCHECK_MIN_WORD_LEN:
            return word
        if word.isupper():
            return word  # skip acronyms / shouted headings
        lower = word.lower()
        if lower in _SPELL:
            return word  # already a real dictionary word
        candidate = _SPELL.correction(lower)
        if not candidate or candidate == lower:
            return word  # no confident correction
        is_capitalized = word[0].isupper()
        if is_capitalized and not _levenshtein_le1(lower, candidate):
            return word  # too big a change for a likely-proper-noun word
        if is_capitalized:
            candidate = candidate[0].upper() + candidate[1:]
        return candidate

    return _WORD_RE.sub(fix, text)


# =========================================================================
# 6. Whitespace cleanup / paragraph reconstruction
# =========================================================================
def clean_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def assemble_page_text(paragraphs, boilerplate, header_lines, footer_lines):
    """Drop boilerplate lines, spell-correct, and join paragraphs with
    blank lines (paragraph reconstruction)."""
    kept_paragraphs = []
    for para in paragraphs:
        if normalize_line(para) in boilerplate:
            continue
        kept_paragraphs.append(spell_correct(para))
    return clean_whitespace("\n\n".join(kept_paragraphs))


# =========================================================================
# main
# =========================================================================
def main():
    images = sorted(IMG_DIR.glob("page-*.jpg"))
    n_pages = len(images)

    per_page = []  # collected before boilerplate removal
    for img_path in images:
        page_num = int(img_path.stem.split("-")[1])

        layer_text = get_layer_text(page_num)

        preprocessed = preprocess_image(img_path)
        header_lines, body_paragraphs, footer_lines = layout_ocr(preprocessed)
        ocr_text = "\n\n".join(body_paragraphs)

        layer_score = quality_score(layer_text)
        ocr_score = quality_score(ocr_text)

        if layer_score >= ocr_score:
            chosen_paragraphs = [p.strip() for p in layer_text.split("\n\n") if p.strip()]
            source = "pdf_text_layer"
            # the text layer has no header/footer band split, so we fall
            # back to line-level boilerplate detection against the OCR
            # header/footer bands captured for this same page
        else:
            chosen_paragraphs = body_paragraphs
            source = "ocr_tesseract_preprocessed"

        per_page.append({
            "page_num": page_num,
            "source": source,
            "paragraphs": chosen_paragraphs,
            "header_lines": header_lines,
            "footer_lines": footer_lines,
            "layer_chars": len(layer_text),
            "ocr_chars": len(ocr_text),
        })

    # --- corpus-wide boilerplate detection (Weakness 4: duplicate removal)
    all_headers = [p["header_lines"] for p in per_page]
    all_footers = [p["footer_lines"] for p in per_page]
    boilerplate = find_boilerplate_lines(all_headers, all_footers, n_pages)

    manifest = []
    for p in per_page:
        final_text = assemble_page_text(
            p["paragraphs"], boilerplate, p["header_lines"], p["footer_lines"]
        )
        out_file = OUT_DIR / f"page_{p['page_num']:02d}.txt"
        out_file.write_text(final_text, encoding="utf-8")

        manifest.append({
            "page": p["page_num"],
            "source_used": p["source"],
            "chars": len(final_text),
            "layer_chars": p["layer_chars"],
            "ocr_chars": p["ocr_chars"],
            "boilerplate_lines_removed": sum(
                1 for h in p["header_lines"] + p["footer_lines"]
                if normalize_line(h) in boilerplate
            ),
            "spell_corrected": HAVE_SPELLCHECK,
            "file": str(out_file.relative_to(ROOT)),
        })
        print(f"page {p['page_num']:02d}: source={p['source']:26s} chars={len(final_text)}")

    (ROOT / "data" / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nDone. {len(manifest)} pages extracted -> {OUT_DIR}")
    print(f"Boilerplate lines identified corpus-wide: {len(boilerplate)}")
    if not HAVE_CV2:
        print("NOTE: opencv-python not installed — deskew/denoise/CLAHE/adaptive-"
              "threshold fell back to a PIL-only pipeline. `pip install opencv-python` "
              "for the full preprocessing chain.")
    if not HAVE_SPELLCHECK:
        print("NOTE: pyspellchecker not installed — spell correction was skipped. "
              "`pip install pyspellchecker` to enable it.")


if __name__ == "__main__":
    main()


## build_index.py

import json
import re
import uuid
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "data" / "pages"
INDEX_DIR = ROOT / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_MIN_WORDS = 250
CHUNK_MAX_WORDS = 350
CHUNK_OVERLAP_FRACTION = 0.20

# Preference order: try the stronger model first, fall back to the
# lighter one if it isn't available/downloadable in this environment.
EMBEDDING_MODEL_CANDIDATES = [
    "BAAI/bge-small-en-v1.5",
    "all-MiniLM-L6-v2",
]

# A heading in this teaching guide looks like "Unit 3", "Chapter 2",
# ALL CAPS lines, or a short Title Case line ending without punctuation.
HEADING_PATTERNS = [
    re.compile(r"^(unit|chapter|lesson)\s+\d+", re.IGNORECASE),
    re.compile(r"^[A-Z][A-Z \d:&'-]{3,60}$"),  # ALL CAPS line
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


# =========================================================================
# Sentence splitting (regex-based — no network-dependent tokenizer model
# needed, which matters since this may run on a Pi with no internet)
# =========================================================================
def split_sentences(text: str):
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def looks_like_heading(paragraph: str) -> bool:
    line = paragraph.strip()
    if not line or len(line) > 70:
        return False
    return any(p.search(line) for p in HEADING_PATTERNS)


# =========================================================================
# Sentence-aware sliding-window chunking with overlap
# =========================================================================
def chunk_sentences(sentences, min_words=CHUNK_MIN_WORDS, max_words=CHUNK_MAX_WORDS,
                     overlap_fraction=CHUNK_OVERLAP_FRACTION):
    """Greedily pack sentences into ~min-max word windows; each new
    window starts by re-including the trailing ~overlap_fraction of the
    previous window's sentences, so context carries across chunk
    boundaries instead of being cut mid-idea."""
    if not sentences:
        return []

    word_counts = [len(s.split()) for s in sentences]
    chunks = []
    i = 0
    n = len(sentences)
    while i < n:
        cur_sentences, cur_words = [], 0
        j = i
        while j < n and (cur_words + word_counts[j] <= max_words or not cur_sentences):
            cur_sentences.append(sentences[j])
            cur_words += word_counts[j]
            j += 1
            if cur_words >= min_words:
                break
        chunks.append(" ".join(cur_sentences))

        if j >= n:
            break

        # step forward but re-include the trailing overlap_fraction of
        # sentences for the next window
        overlap_words_target = cur_words * overlap_fraction
        back = 0
        acc = 0
        for k in range(len(cur_sentences) - 1, -1, -1):
            acc += word_counts[i + k] if (i + k) < n else 0
            back += 1
            if acc >= overlap_words_target:
                break
        i = max(i + 1, j - back)

    return chunks


# =========================================================================
# Chunk text -> records (with expanded metadata) for a single page
# =========================================================================
def build_page_records(text: str, page_num: int, source_file: str, running_chunk_id: int):
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Track the most recent heading-shaped paragraph so we can tag chunks
    # with the section they fall under.
    current_heading = None
    tagged_sentences = []  # (sentence, heading_at_time)
    for para in raw_paras:
        if looks_like_heading(para):
            current_heading = para.strip()
            continue  # headings aren't embedded as body content
        for sent in split_sentences(para):
            tagged_sentences.append((sent, current_heading))

    sentences = [s for s, _h in tagged_sentences]
    chunks = chunk_sentences(sentences)

    # crude chapter/topic-in-progress marker: chapter = the first
    # heading found before/within this page (NOME TG1 is short enough
    # that page-level granularity for "chapter" is reasonable)
    chapter = current_heading if current_heading else None

    records = []
    for i, chunk_text in enumerate(chunks):
        heading = None
        for sent, h in tagged_sentences:
            if sent in chunk_text:
                heading = h
                break
        records.append({
            "chunk_id": str(uuid.uuid4()),
            "page": page_num,
            "chapter": chapter,
            "heading": heading,
            "chunk_index_on_page": i,
            "chunk_position": running_chunk_id + i,  # global position
            "text": chunk_text,
            "word_count": len(chunk_text.split()),
            "source_file": source_file,
            "page_image": f"data/page_images/page-{page_num:02d}.jpg",
            "previous_chunk_id": None,  # linked in a second pass below
            "next_chunk_id": None,
        })
    return records


# =========================================================================
# Embedding model loading
# =========================================================================
def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers is required for semantic embeddings.\n"
            "  pip install sentence-transformers\n"
            "(This replaces the old TF-IDF+SVD pipeline — see Part 4 of "
            "the review doc for why.)"
        ) from e

    last_err = None
    for name in EMBEDDING_MODEL_CANDIDATES:
        try:
            model = SentenceTransformer(name)
            print(f"Loaded embedding model: {name}")
            return model, name
        except Exception as e:  # model download/network failure, etc.
            last_err = e
            print(f"Could not load '{name}' ({e}); trying next candidate...")
    raise RuntimeError(
        f"Could not load any embedding model from {EMBEDDING_MODEL_CANDIDATES}. "
        f"Generate embeddings on a machine with internet access first "
        f"(see README: 'generate embeddings once on your laptop'). "
        f"Last error: {last_err}"
    )


def main():
    page_files = sorted(PAGES_DIR.glob("page_*.txt"))

    records = []
    running_chunk_id = 0
    for pf in page_files:
        page_num = int(pf.stem.split("_")[1])
        text = pf.read_text(encoding="utf-8")
        page_records = build_page_records(
            text, page_num, str(pf.relative_to(ROOT)), running_chunk_id
        )
        records.extend(page_records)
        running_chunk_id += len(page_records)

    # link previous/next chunk ids (global reading order)
    for idx, rec in enumerate(records):
        rec["id"] = idx  # stable integer id == FAISS row index
        if idx > 0:
            rec["previous_chunk_id"] = records[idx - 1]["chunk_id"]
        if idx < len(records) - 1:
            rec["next_chunk_id"] = records[idx + 1]["chunk_id"]

    corpus = [r["text"] for r in records]
    print(f"Total chunks: {len(corpus)} across {len(page_files)} pages "
          f"(target {CHUNK_MIN_WORDS}-{CHUNK_MAX_WORDS} words/chunk, "
          f"{int(CHUNK_OVERLAP_FRACTION*100)}% overlap)")

    if not corpus:
        print("No chunks produced — nothing to index.")
        return

    # --- Embedding pipeline: real sentence embeddings, L2-normalized
    model, model_name = load_embedding_model()
    dense = model.encode(
        corpus, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")

    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dense = dense / norms

    # --- Build FAISS index (flat inner-product = cosine on normalized vecs)
    index = faiss.IndexFlatIP(dense.shape[1])
    index.add(dense)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    with open(INDEX_DIR / "embedding_model.json", "w", encoding="utf-8") as f:
        json.dump({"model_name": model_name, "dim": int(dense.shape[1])}, f, indent=2)

    print(f"FAISS index: {index.ntotal} vectors, dim={dense.shape[1]}, model={model_name}")
    print(f"Saved to: {INDEX_DIR}")


if __name__ == "__main__":
    main()
