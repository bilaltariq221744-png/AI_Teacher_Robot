#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    PAGES_DIR,
    CHROMA_DIR,
    BM25_DOCS_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from src.chunking import chunk_text, detect_lang, split_sentences


def load_page_documents(pages_dir: Path) -> List[Document]:
    docs = []
    page_files = sorted(pages_dir.glob("*.txt"))

    for page_file in page_files:
        text = page_file.read_text(encoding="utf-8").strip()

        if not text:
            continue

        chunks = chunk_text(text)

        for chunk_index, chunk in enumerate(chunks):
            metadata = {
                "source_file": page_file.name,
                "chunk_index": chunk_index,
                "lang": detect_lang(chunk),
                "sentence_count": len(split_sentences(chunk)),
            }

            docs.append(
                Document(
                    page_content=chunk,
                    metadata=metadata,
                )
            )

    return docs


def build_vector_db(reset: bool = False, batch_size: int = 64) -> None:
    if not PAGES_DIR.exists():
        raise FileNotFoundError(f"Pages folder not found: {PAGES_DIR}")

    docs = load_page_documents(PAGES_DIR)

    if not docs:
        raise RuntimeError(
            f"No text chunks found in {PAGES_DIR}. "
            "First extract PDF text into data/pages/."
        )

    CHROMA_DIR.parent.mkdir(parents=True, exist_ok=True)

    if reset and CHROMA_DIR.exists():
        print(f"Removing old Chroma DB: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    print(f"Loaded {len(docs)} chunks.")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("Make sure Ollama is running before continuing.")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    if reset:
        existing = vectorstore.get()
        if existing and existing.get("ids"):
            vectorstore.delete(ids=existing["ids"])

    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        vectorstore.add_documents(batch)
        print(f"Embedded {min(start + batch_size, len(docs))}/{len(docs)}")

    # Save the same chunks for BM25 keyword retrieval.
    bm25_dump = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]

    BM25_DOCS_PATH.write_text(
        json.dumps(bm25_dump, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nVector DB created successfully.")
    print(f"Chroma DB: {CHROMA_DIR}")
    print(f"BM25 docs: {BM25_DOCS_PATH}")
    print("\nNow copy the whole index/ folder to Raspberry Pi.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete old vector DB before building. Use this when rebuilding all books.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    args = parser.parse_args()

    build_vector_db(reset=args.reset, batch_size=args.batch_size)


if __name__ == "__main__":
    main()