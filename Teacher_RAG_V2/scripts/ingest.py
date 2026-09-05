#!/usr/bin/env python3
"""
Build the RAG knowledge base: load page text -> chunk -> embed with a
local Ollama embedding model -> persist into a Chroma vector store.

Requires Ollama running locally (https://ollama.com) with the embedding
model already pulled:
    ollama pull bge-m3

Usage:
    python3 scripts/ingest.py --pages data/pages_en --pages data/pages_ur --out index/chroma_db
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.chunking import chunk_text, split_sentences, detect_lang

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
