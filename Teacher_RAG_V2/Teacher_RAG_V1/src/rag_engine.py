import json
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

from config import (
    CHROMA_DIR,
    BM25_DOCS_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    RETRIEVAL_K,
    VECTOR_WEIGHT,
    MAX_WORDS,
    TEMPERATURE,
)


SYSTEM_PROMPT = """
You are an AI Teacher for school students.

Your job is to answer the student's question accurately using ONLY the provided
textbook context.

Rules:

- Answer ONLY from the provided textbook context.
- If the context does not contain the answer, say:
  "This answer is not available in the selected textbook."
- Answer in the same language as the student's question.
- If the student asks in Roman Urdu, answer in simple Roman Urdu.
- Understand the student's question and decide the appropriate answer length yourself.
- Give only as much information as necessary to properly answer the question.
- Simple questions should receive concise answers.
- Questions requiring explanation should receive enough explanation to make the
  concept clear.
- If the question has multiple parts, answer every part.
- Do not make answers unnecessarily long.
- Do not give unnecessary background information.
- Do not repeat the same information.
- Use simple, clear language suitable for a school student.
- Use bullet points when they make the answer easier to understand.
- Do not add facts from your own knowledge.
- Do not make up examples unless the textbook context supports them.
- Prioritize accuracy, relevance, and clarity.

Context:
{context}
"""


def format_docs(docs: List[Document]) -> str:
    parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "?")
        text = doc.page_content.strip()

        parts.append(f"[source {i}: {source}, chunk {chunk_index}]\n{text}")

    return "\n\n".join(parts)


def load_bm25_docs(path: Path) -> List[Document]:
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 file not found: {path}. "
            "Copy index/docs_for_bm25.json from laptop to Raspberry Pi."
        )

    raw_docs = json.loads(path.read_text(encoding="utf-8"))

    return [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )
        for item in raw_docs
    ]


def build_retriever():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma DB not found: {CHROMA_DIR}. "
            "Copy index/chroma_db from laptop to Raspberry Pi."
        )

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )

    bm25_docs = load_bm25_docs(BM25_DOCS_PATH)
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = RETRIEVAL_K

    retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[VECTOR_WEIGHT, 1 - VECTOR_WEIGHT],
    )

    return retriever


def build_chain(retriever):
    num_predict = int(MAX_WORDS * 2.0) + 50

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        num_predict=num_predict,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT.replace("{max_words}", str(MAX_WORDS))),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


class TeacherRAG:
    def __init__(self):
        self.retriever = build_retriever()
        self.chain = build_chain(self.retriever)

    def ask(self, question: str, show_sources: bool = False) -> Tuple[str, List[Document]]:
        question = question.strip()

        if not question:
            return "Please ask a question.", []

        docs = self.retriever.invoke(question)
        answer = self.chain.invoke(question).strip()

        if show_sources:
            return answer, docs

        return answer, []