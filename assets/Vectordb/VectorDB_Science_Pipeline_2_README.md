# Offline PDF Question-Answering Pipeline

A small, fully-offline pipeline that turns PDFs with a real text layer into a
searchable question-answering database. You ask a question, and the system
returns both the ranked source chunks (for citation) and a short direct answer
extracted from the best-matching sentences — no generative model required.

```
01_extract_text_pdf.py ──► page_XXX.txt files + manifest.json
        │
        ▼
chunking.py (imported by 02_build_index.py)
        │
        ▼
02_build_index.py ──► faiss.index, metadata.json, embedding_model.json,
                      bm25_corpus_tokens.json (+ vectorizer.pkl / svd.pkl)
        │
        ▼
search.py ──► answer + source chunks
```

## Scripts

| Script | Role |
|---|---|
| [`scripts/01_extract_text_pdf.py`](scripts/01_extract_text_pdf.py) | Extract page-based text from a PDF using its embedded text layer (no OCR). Writes one `page_XXX.txt` per page plus a `manifest.json` with per-page char counts. |
| [`scripts/chunking.py`](scripts/chunking.py) | Shared library: sentence splitting (English + Urdu), language detection, and small sentence-aware chunking. Imported by the index builder. |
| [`scripts/02_build_index.py`](scripts/02_build_index.py) | Load every `page_*.txt` from one or more page directories, chunk each page, embed the chunks, and build the FAISS vector index plus a BM25 lexical index. |
| [`scripts/search.py`](scripts/search.py) | Query the index. Returns ranked source chunks (hybrid semantic + lexical scoring) and a short direct answer extracted from the top chunks. |

## Requirements

- Python 3
- `PyMuPDF` (fitz) — used by `01_extract_text_pdf.py`
- `faiss`, `numpy` — used by `02_build_index.py` and `search.py`
- `sentence-transformers` — optional but recommended; enables real semantic embeddings (model downloaded once, needs internet for that first run)
- `rank-bm25` — optional; enables lexical hybrid scoring at search time
- `scikit-learn` — used only for the offline TF-IDF+SVD fallback embedding

## Usage

### 1. Extract text from a PDF

```bash
python3 scripts/01_extract_text_pdf.py input.pdf data/pages_en
# optional: --lang-tag ur  (stored in the manifest)
```

Produces `data/pages_en/page_001.txt` … `page_NNN.txt` and
`data/pages_en/manifest.json`. Empty pages (typically cover/blank pages) are
recorded in the manifest but left as empty files.

### 2. Build the index

```bash
python3 scripts/02_build_index.py --pages data/pages_en --out index/
```

`--pages` is repeatable, so you can index several languages or documents into
one combined index:

```bash
python3 scripts/02_build_index.py --pages data/pages_en --pages data/pages_ur --out index/
```

**Embedding backends (automatic selection):**

1. **Sentence-transformer** (best quality) — tries `BAAI/bge-m3`, then
   `intfloat/multilingual-e5-large`, then `multilingual-e5-small`. The first
   run downloads the model, so do it on a machine with internet access (e.g. a
   laptop, not a Pi).
2. **Offline TF-IDF + SVD** (fallback) — used when `sentence-transformers`
   isn't installed or no model can be downloaded. Works with zero internet
   access but gives weaker semantic matching. Saves `vectorizer.pkl` and
   `svd.pkl` alongside the index.

The chosen backend is recorded in `embedding_model.json` so search knows how to
embed queries.

**Outputs written to the `--out` directory:**

| File | Contents |
|---|---|
| `faiss.index` | Flat inner-product FAISS index (cosine similarity on normalized vectors) |
| `metadata.json` | One record per chunk: id, page, chunk text, sentences, word count, detected language, source file/dir, previous/next chunk ids |
| `embedding_model.json` | Which backend/model was used and its query/passage prefixes |
| `bm25_corpus_tokens.json` | Tokenized chunks for BM25 (only if `rank_bm25` is installed) |
| `vectorizer.pkl`, `svd.pkl` | TF-IDF vectorizer + SVD model (offline fallback only) |

### 3. Search

```bash
python3 scripts/search.py "what is a noun" --k 5
python3 scripts/search.py "سوال: اسم کیا ہے؟"
```

Options:

- `--k N` — number of top chunks to return (default `5`)
- `--hybrid-weight W` — weight of BM25 lexical scoring vs. embedding similarity
  in the final ranking (default `0.35`; `0` disables BM25 entirely)

Search prints the extracted answer with its source page, then the ranked source
chunks with their scores.

The index directory defaults to `index/` next to the `scripts/` folder; change
the `INDEX_DIR` constant in `search.py` if you build elsewhere.

## Design notes

- **Small chunks.** Chunks are 40–90 words (roughly 1–4 sentences), tuned for
  direct Q&A rather than summarization. A query matches a precise, topic-tight
  chunk, and the retrieved chunk is small enough to *be* the answer. Chunks
  never cross paragraph boundaries, and a 1-sentence overlap preserves facts
  that fall across a boundary.
- **Hybrid search.** Semantic (embedding cosine) + lexical (BM25) scores are
  combined at query time. This noticeably improves top-1 accuracy for queries
  with exact terms or proper nouns that a purely semantic embedding can
  under-weight. Search over-fetches 4× the requested `k` from FAISS so the
  hybrid re-ranking has room to reorder.
- **No LLM needed.** The direct answer is produced by a second, finer-grained
  pass of the same embedding model over sentences from the top chunks — fully
  offline.
- **Multilingual.** Chunking supports English and Urdu sentence boundaries
  (`.` `!` `?` plus Urdu `۔`/`؟`), and a cheap script-based detector tags each
  chunk `en`/`ur` for routing/metadata.

## Example output

```
Question: what is a noun
------------------------------------------------------------

Answer (page 12):
A noun is a word used to name a person, place, thing, or idea.

------------------------------------------------------------
Source chunks:

[1] page 12 | score 0.812 (semantic 0.823)
A noun is a word used to name a person, place, thing, or idea...
```

## extract_pages.py:

import argparse
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pdf")
    parser.add_argument("output_dir")
    parser.add_argument("--lang-tag", default="en", help="language tag stored in manifest")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(args.input_pdf)
    manifest = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        fname = out_dir / f"page_{i:03d}.txt"
        fname.write_text(text, encoding="utf-8")
        manifest.append({
            "page": i,
            "chars": len(text),
            "lang": args.lang_tag,
            "source_file": Path(args.input_pdf).name,
            "extraction_method": "text_layer",
        })
        print(f"page {i:3d}: {len(text):5d} chars")

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    empty = sum(1 for m in manifest if m["chars"] == 0)
    print(f"\n{len(manifest)} pages extracted, {empty} empty (likely cover/blank pages).")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()


## build_index.py:

import argparse
import json
import pickle
import sys
import uuid
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunking import chunk_text, split_sentences, detect_lang

EMBEDDING_MODEL_CANDIDATES = [
    {"name": "BAAI/bge-m3", "query_prefix": "", "passage_prefix": ""},
    {"name": "intfloat/multilingual-e5-large", "query_prefix": "query: ", "passage_prefix": "passage: "},
    {"name": "intfloat/multilingual-e5-small", "query_prefix": "query: ", "passage_prefix": "passage: "},
]


def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed - falling back to offline TF-IDF+SVD.")
        return None, None
    for cfg in EMBEDDING_MODEL_CANDIDATES:
        try:
            model = SentenceTransformer(cfg["name"])
            print(f"Loaded embedding model: {cfg['name']}")
            return model, cfg
        except Exception as e:
            print(f"Could not load '{cfg['name']}' ({e}); trying next...")
    print("No sentence-transformer model could be downloaded - falling back to offline TF-IDF+SVD.")
    print("(Run this script once on a machine with internet access to get real semantic embeddings.)")
    return None, None


def build_tfidf_svd_embeddings(corpus):
    """Fully offline fallback: no model download required. Weaker than a
    real sentence-transformer but works with zero internet access."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)
    tfidf = vectorizer.fit_transform(corpus)
    dim = min(256, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    dim = max(dim, 2)
    svd = TruncatedSVD(n_components=dim, random_state=42)
    dense = svd.fit_transform(tfidf).astype("float32")
    return dense, vectorizer, svd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", action="append", required=True,
                         help="a data/pages_* directory; repeatable for multiple sources")
    parser.add_argument("--out", default="index")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load all pages from all provided directories, chunk each page
    records = []
    for pages_dir in args.pages:
        pages_dir = Path(pages_dir)
        page_files = sorted(pages_dir.glob("page_*.txt"))
        for pf in page_files:
            text = pf.read_text(encoding="utf-8").strip()
            if not text:
                continue
            page_num = int(pf.stem.split("_")[1])
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                records.append({
                    "chunk_id": str(uuid.uuid4()),
                    "page": page_num,
                    "chunk_index_on_page": i,
                    "text": chunk,
                    "sentences": split_sentences(chunk),
                    "word_count": len(chunk.split()),
                    "lang": detect_lang(chunk),
                    "source_dir": str(pages_dir),
                    "source_file": pages_dir.name,
                })

    for idx, rec in enumerate(records):
        rec["id"] = idx
        rec["previous_chunk_id"] = records[idx - 1]["chunk_id"] if idx > 0 else None
        rec["next_chunk_id"] = records[idx + 1]["chunk_id"] if idx < len(records) - 1 else None

    corpus = [r["text"] for r in records]
    print(f"Total chunks: {len(corpus)} from {len(args.pages)} source dir(s) "
          f"(target {40}-{90} words/chunk)")

    if not corpus:
        print("No chunks produced - nothing to index.")
        return

    # ---- 2. Embeddings (semantic model, else offline fallback)
    model, model_cfg = load_embedding_model()
    if model is not None:
        prefixed = [model_cfg["passage_prefix"] + c for c in corpus]
        dense = model.encode(prefixed, batch_size=32, show_progress_bar=True,
                              convert_to_numpy=True).astype("float32")
        embedding_info = {"backend": "sentence_transformer", "model_name": model_cfg["name"],
                           "query_prefix": model_cfg["query_prefix"],
                           "passage_prefix": model_cfg["passage_prefix"]}
    else:
        dense, vectorizer, svd = build_tfidf_svd_embeddings(corpus)
        with open(out_dir / "vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        with open(out_dir / "svd.pkl", "wb") as f:
            pickle.dump(svd, f)
        embedding_info = {"backend": "tfidf_svd", "dim": int(dense.shape[1])}

    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dense = dense / norms

    # ---- 3. FAISS index (flat inner product = cosine on normalized vectors)
    index = faiss.IndexFlatIP(dense.shape[1])
    index.add(dense)
    faiss.write_index(index, str(out_dir / "faiss.index"))

    # ---- 4. BM25 lexical index (for hybrid scoring at search time)
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [c.split() for c in corpus]
        with open(out_dir / "bm25_corpus_tokens.json", "w", encoding="utf-8") as f:
            json.dump(tokenized, f, ensure_ascii=False)
        print("BM25 token corpus saved (hybrid scoring enabled at search time).")
    except ImportError:
        print("rank_bm25 not installed - hybrid scoring will be disabled at search time. "
              "pip install rank_bm25 to enable it.")

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open(out_dir / "embedding_model.json", "w", encoding="utf-8") as f:
        json.dump(embedding_info, f, indent=2)

    print(f"\nFAISS index: {index.ntotal} vectors, dim={dense.shape[1]}")
    print(f"Embedding backend: {embedding_info['backend']}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()

## chunking.py

import re

CHUNK_MIN_WORDS = 40
CHUNK_MAX_WORDS = 90
CHUNK_OVERLAP_SENTENCES = 1

# Sentence boundaries for English (. ! ?) and Urdu (\u06d4 Urdu full stop,
# \u061f Urdu question mark).
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?\u06d4\u061f])\s+(?=[A-Za-z0-9\u0600-\u06FF\"'])"
)

_URDU_RANGE_RE = re.compile(r"[\u0600-\u06FF]")


def detect_lang(text: str) -> str:
    """Cheap script-based language tag: 'ur' if Urdu/Arabic-script
    characters dominate, else 'en'. Good enough for routing/metadata;
    not a real language identifier."""
    urdu_chars = len(_URDU_RANGE_RE.findall(text))
    return "ur" if urdu_chars > len(text) * 0.15 else "en"


def split_sentences(text: str):
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def split_paragraphs(text: str):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_text(text: str, min_words=CHUNK_MIN_WORDS, max_words=CHUNK_MAX_WORDS,
                overlap_sentences=CHUNK_OVERLAP_SENTENCES):
    """Greedily pack sentences (within paragraph boundaries, so we never
    merge two unrelated paragraphs into one chunk) into small windows,
    with a 1-sentence overlap so a fact split across a boundary isn't
    lost entirely from either neighboring chunk."""
    chunks = []
    for para in split_paragraphs(text):
        sentences = split_sentences(para)
        if not sentences:
            continue
        word_counts = [len(s.split()) for s in sentences]
        i = 0
        n = len(sentences)
        while i < n:
            cur, cur_words = [], 0
            j = i
            while j < n and (cur_words + word_counts[j] <= max_words or not cur):
                cur.append(sentences[j])
                cur_words += word_counts[j]
                j += 1
                if cur_words >= min_words:
                    break
            chunks.append(cur)
            if j >= n:
                break
            i = max(i + 1, j - overlap_sentences)
    return [" ".join(c) for c in chunks]
