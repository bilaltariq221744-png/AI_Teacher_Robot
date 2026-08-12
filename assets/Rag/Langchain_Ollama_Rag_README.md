# NOME Vector DB — RAG with Ollama + LangChain

Same goal as before — ask a question, get a direct answer sourced from
your PDFs, works offline — rebuilt on **Ollama + LangChain** instead of
the custom FAISS pipeline. This version generates real, fluent answers
with a local LLM (not just extracted sentences), while still running
100% on your machine with no API keys and no internet needed after setup.

---

## 1. How it works

```
 PDFs → extract text → chunk → embed (Ollama) → Chroma vector store
   → hybrid retrieval (vector + BM25) → local LLM (Ollama) → direct answer
```

1. Text is pulled from your PDFs (or OCR'd via the browser tool for
   scanned books like the Urdu one).
2. Each page is split into small, focused chunks (~40–90 words).
3. Every chunk is embedded into a vector using a local embedding model
   served by **Ollama**, and stored in a **Chroma** vector database.
4. At query time, your question is matched against the database two
   ways at once — semantic similarity (meaning) and BM25 (exact keyword
   match) — and the two rankings are blended (**hybrid retrieval**), so
   both paraphrased questions and exact-term questions work well.
5. The retrieved chunks are handed to a local LLM (also served by
   **Ollama**) with instructions to answer directly, only from that
   context, in the same language the question was asked in.

Retrieval and generation both run locally via Ollama — nothing leaves
your machine, and once the models are downloaded, no internet is needed
at all, including on a Raspberry Pi.

---

## 2. Install Ollama

Download and install from **https://ollama.com** (Mac, Windows, Linux
all supported). Then pull the two models this project uses:

```bash
ollama pull bge-m3        # embedding model - multilingual (100+ languages incl. Urdu)
ollama pull qwen2.5       # generation model - strong multilingual instruction-following
```

Leave Ollama running in the background (the installer sets this up as a
service automatically on most platforms; if not, run `ollama serve`).

**On a low-power device (e.g. Raspberry Pi):** swap `qwen2.5` for a
smaller variant:
```bash
ollama pull qwen2.5:3b     # or even qwen2.5:1.5b for very constrained hardware
```
and pass `--llm-model qwen2.5:3b` when querying (see section 5).

---

## 3. Folder structure

```
pipeline_rag/
├── README.md
├── requirements.txt
├── source_pdfs/                ← your original PDFs
├── data/
│   ├── pages_en/                ← extracted English text, one .txt per page
│   └── pages_ur/                ← extracted Urdu text (from the OCR tool)
├── scripts/
│   ├── 01_extract_text_pdf.py   ← PDFs with a real text layer (no OCR needed)
│   ├── chunking.py              ← splits page text into small Q&A-sized chunks
│   ├── ingest.py                ← embeds chunks with Ollama, builds the Chroma index
│   └── rag.py                   ← hybrid retrieval + local LLM answer generation
└── index/                       ← created after you run ingest.py
    ├── chroma_db/                ← the vector store itself
    └── docs_for_bm25.json        ← chunk dump used to rebuild BM25 instantly at query time
```

---

## 4. Setup

```bash
pip install -r requirements.txt
```

### 4.1 Extract text

**PDFs with real text** (you can select/highlight text in a PDF viewer):
```bash
python3 scripts/01_extract_text_pdf.py source_pdfs/NOPS_TG_5.pdf data/pages_en --lang-tag en
```
Already done for `NOPS_TG_5.pdf` in this project.

**Scanned PDFs** (like the Urdu book — no selectable text):
1. Open the browser OCR tool shared earlier in the chat.
2. Upload the scanned PDF, test a small page range first, then process
   the rest in batches of ~20–30 pages.
3. Download each batch's `.zip` and unzip the `page_XXX.txt` files into
   `data/pages_ur/`.

This uses Claude's vision to transcribe pages directly in your browser —
much more accurate than Tesseract on Urdu's Nastaliq script.

### 4.2 Build the index
```bash
python3 scripts/ingest.py --pages data/pages_en --pages data/pages_ur --out index/chroma_db
```
This calls Ollama to embed every chunk and stores the result in
`index/chroma_db`. Re-run this any time you add or change source pages —
it clears and rebuilds the collection so nothing gets duplicated.

---

## 5. Asking questions

**Single question:**
```bash
python3 scripts/rag.py "what is a noun"
python3 scripts/rag.py "اسم کیا ہے"
```

**Interactive chat:**
```bash
python3 scripts/rag.py
```

**See which chunks the answer came from:**
```bash
python3 scripts/rag.py "what is a noun" --show-sources
```

**Use a smaller model (e.g. on a Pi):**
```bash
python3 scripts/rag.py "what is a noun" --llm-model qwen2.5:3b
```

Example output:
```
Question: what is a noun
------------------------------------------------------------
A noun is a word that names a person, place, thing, animal, or idea.
```

---

## 6. Adding more PDFs later

1. Extract the new PDF into a new folder, e.g. `data/pages_grade7/`
   (text-layer PDF → `01_extract_text_pdf.py`; scanned → the OCR tool).
2. Re-run ingest with every folder you want included:
   ```bash
   python3 scripts/ingest.py \
     --pages data/pages_en \
     --pages data/pages_ur \
     --pages data/pages_grade7 \
     --out index/chroma_db
   ```
3. Query as usual — everything lives in one combined index.

---

## 7. Tuning

| Want to change | Where |
|---|---|
| Answer length / chunk granularity | `CHUNK_MIN_WORDS` / `CHUNK_MAX_WORDS` in `scripts/chunking.py`, then re-run `ingest.py` |
| Exact terms/names not surfacing | raise `--vector-weight` toward 1.0 lowers keyword weight; lower it (e.g. `0.4`) to favor BM25/exact matches |
| Paraphrased questions not matching | raise `--vector-weight` (e.g. `0.8`) to favor semantic search |
| Answer tone/strictness/length rules | edit `SYSTEM_PROMPT` in `scripts/rag.py` |
| How many chunks feed the answer | `--k` (default 5) |
| More deterministic vs. more natural phrasing | `temperature` in `build_chain()` inside `scripts/rag.py` (default 0.1 — low, for factual consistency) |

---

## 8. Which models are used, and why

| Role | Model | Why |
|---|---|---|
| Embeddings | `bge-m3` (via Ollama) | Multilingual across 100+ languages including English and Urdu; same model family used in the previous version of this project, proven to work well here. |
| Generation | `qwen2.5` (via Ollama) | Strong multilingual instruction-following, including Urdu — noticeably better at non-English generation than most similarly-sized open models. Swap for `qwen2.5:3b`/`qwen2.5:1.5b` on constrained hardware, or `llama3.1`/`gemma2` if you prefer. |
| Vector store | Chroma | Simple, local, file-based — no server to manage, persists to disk automatically. |
| Keyword retriever | BM25 (`rank_bm25` via LangChain) | Catches exact terms/names/numbers that a purely semantic model can under-weight; blended with vector search via LangChain's `EnsembleRetriever`. |

---

## 9. A note on package versions

`requirements.txt` pins LangChain to the **0.3.x** line on purpose.
LangChain released a `1.0` restructure that moved and renamed several
things (including the retriever classes this project uses) and is still
actively shifting; `0.3.x` is the stable, well-documented line most
current tutorials and this project were built against. If you want to
move to LangChain 1.x later, the main things likely to need updating are
the `EnsembleRetriever`/`BM25Retriever` import paths in `scripts/rag.py`.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ConnectionError` / refused connection | Ollama isn't running | Start the Ollama app, or run `ollama serve` |
| `model not found` | Model wasn't pulled | `ollama pull bge-m3` / `ollama pull qwen2.5` |
| Ingest is slow | Normal for large books on CPU-only machines | Expected — this only happens once per book, not per query |
| Answers ignore the books / sound generic | Retrieval found nothing relevant, or `index/` wasn't rebuilt after adding pages | Run `--show-sources` to check what was retrieved; re-run `ingest.py` |
| Urdu answers look garbled | Book was OCR'd with Tesseract instead of the browser tool | Re-run that book through the browser OCR tool, re-ingest |
| `ModuleNotFoundError` | Dependencies not installed | `pip install -r requirements.txt` |

---

## 11. Quick reference

```bash
# Install
pip install -r requirements.txt
ollama pull bge-m3
ollama pull qwen2.5

# Extract
python3 scripts/01_extract_text_pdf.py source_pdfs/NOPS_TG_5.pdf data/pages_en --lang-tag en
# (Urdu book: use the browser OCR tool, unzip into data/pages_ur/)

# Build the index
python3 scripts/ingest.py --pages data/pages_en --pages data/pages_ur --out index/chroma_db

# Ask
python3 scripts/rag.py "what is a noun"
python3 scripts/rag.py "اسم کیا ہے"
```


## extract_text.py

#!/usr/bin/env python3
"""
Extract page-based text from a PDF that already has a real embedded text
layer (no OCR needed) — e.g. NOPS_TG_5.pdf.

Usage:
    python3 01_extract_text_pdf.py <input.pdf> <output_dir> [--lang-tag en]

Writes one page_XXX.txt file per page to <output_dir>, plus a
manifest.json recording char counts per page (empty pages, usually
cover/blank pages, are recorded but left as empty files).
"""
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


## chunking.py:

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

## ingest.py:

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunking import chunk_text, split_sentences, detect_lang

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def load_documents(pages_dirs):
    """Read every page_*.txt in the given directories, chunk each page,
    and wrap each chunk as a LangChain Document with useful metadata."""
    docs = []
    for pages_dir in pages_dirs:
        pages_dir = Path(pages_dir)
        page_files = sorted(pages_dir.glob("page_*.txt"))
        for pf in page_files:
            text = pf.read_text(encoding="utf-8").strip()
            if not text:
                continue
            page_num = int(pf.stem.split("_")[1])
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "page": page_num,
                        "chunk_index_on_page": i,
                        "lang": detect_lang(chunk),
                        "source_dir": str(pages_dir),
                        "source_book": pages_dir.name,
                        "sentence_count": len(split_sentences(chunk)),
                    },
                ))
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", action="append", required=True,
                         help="a data/pages_* directory; repeatable for multiple books")
    parser.add_argument("--out", default="index/chroma_db")
    parser.add_argument("--embedding-model", default="bge-m3",
                         help="Ollama embedding model name (must be pulled already: `ollama pull <name>`)")
    parser.add_argument("--collection", default="nome_vdb")
    args = parser.parse_args()

    docs = load_documents(args.pages)
    print(f"Loaded {len(docs)} chunks from {len(args.pages)} source folder(s).")
    if not docs:
        print("No chunks found - check that your --pages folders contain page_*.txt files.")
        return

    print(f"Embedding with Ollama model '{args.embedding_model}' "
          f"(make sure `ollama pull {args.embedding_model}` has been run, "
          f"and `ollama serve` / the Ollama app is running)...")

    embeddings = OllamaEmbeddings(model=args.embedding_model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=str(out_dir),
    )
    # Wipe any previous run of this collection so re-ingesting doesn't duplicate chunks
    existing = vectorstore.get()
    if existing and existing.get("ids"):
        vectorstore.delete(ids=existing["ids"])

    batch_size = 64
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  embedded {min(i + batch_size, len(docs))}/{len(docs)}")

    # Also dump the raw docs to disk so the BM25 (keyword) retriever can be
    # rebuilt instantly at query time without re-reading every page file.
    dump = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    with open(out_dir.parent / "docs_for_bm25.json", "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(docs)} chunks embedded and stored in: {out_dir}")
    print(f"Embedding model used: {args.embedding_model}")


if __name__ == "__main__":
    main()

## rag.py:

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = """You are a study assistant answering questions from a specific set of textbooks.

Rules:
- Answer ONLY using the provided context. If the context doesn't contain the answer, say so plainly - do not make anything up.
- Answer in the SAME language the question was asked in (English question -> English answer, Urdu question -> Urdu answer).
- Give the answer itself directly, in 1-2 sentences, the way a textbook definition reads. No preamble like "Based on the context...".
- Do NOT add "Key features", numbered lists, bullet points, extra examples, or elaboration UNLESS the question explicitly asks for a list, examples, or steps (e.g. "list the types of...", "give examples of...", "what are the steps to...").
- Never restate the question before answering. Never add a closing summary sentence.
- Hard limit: {max_words} words. Stop as soon as the question is answered - do not keep adding detail to fill space.

Context:
{context}"""


def format_docs(docs):
    parts = []
    for d in docs:
        parts.append(f"[page {d.metadata.get('page', '?')}] {d.page_content}")
    return "\n\n".join(parts)


def build_retriever(chroma_dir, docs_json_path, embedding_model, k=5, vector_weight=0.6):
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        collection_name="nome_vdb",
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # BM25 (keyword) retriever rebuilt from the same chunks dumped at ingest time.
    with open(docs_json_path, encoding="utf-8") as f:
        raw_docs = json.load(f)
    bm25_docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in raw_docs]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, 1 - vector_weight],
    )


def build_chain(retriever, llm_model, temperature=0.1, max_words=40):
    # num_predict caps how many tokens Ollama can generate, as a hard backstop
    # in case the model ignores the word-limit instruction in the prompt.
    # ~1.3 tokens/word is a safe rough estimate; +20 gives a little headroom
    # so answers don't get cut off mid-sentence.
    num_predict = int(max_words * 1.3) + 20
    llm = ChatOllama(model=llm_model, temperature=temperature, num_predict=num_predict)
    system_prompt = SYSTEM_PROMPT.replace("{max_words}", str(max_words))
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None,
                         help="question to ask; omit for an interactive chat loop")
    parser.add_argument("--index", default="index/chroma_db")
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--llm-model", default="qwen2.5",
                         help="Ollama chat model, e.g. qwen2.5, qwen2.5:3b, llama3.1")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.6,
                         help="0-1, how much semantic search counts vs keyword (BM25) search")
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument("--max-words", type=int, default=40,
                         help="hard cap on answer length in words (default 40, ~1-2 sentences). "
                              "Raise this for questions that genuinely need lists/steps.")
    args = parser.parse_args()

    chroma_dir = ROOT / args.index
    docs_json_path = chroma_dir.parent / "docs_for_bm25.json"
    if not chroma_dir.exists() or not docs_json_path.exists():
        print(f"No index found at {chroma_dir}. Run scripts/ingest.py first.")
        return

    retriever = build_retriever(chroma_dir, docs_json_path, args.embedding_model,
                                 k=args.k, vector_weight=args.vector_weight)
    chain = build_chain(retriever, args.llm_model, max_words=args.max_words)

    def ask(question):
        docs = retriever.invoke(question)
        answer = chain.invoke(question)
        print(f"\nQuestion: {question}")
        print("-" * 60)
        print(answer.strip())
        if args.show_sources:
            print("-" * 60)
            print("Sources:")
            for d in docs:
                snippet = d.page_content[:150].replace("\n", " ")
                print(f"  [page {d.metadata.get('page', '?')}] {snippet}...")
        print()

    if args.query:
        ask(args.query)
    else:
        print("Interactive mode. Type a question, or 'quit' to exit.\n")
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("quit", "exit"):
                break
            ask(q)


if __name__ == "__main__":
    main()
