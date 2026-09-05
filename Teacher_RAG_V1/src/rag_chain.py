"""RAG chain (SVG W, X; PDF §8).

W - LangChain prompt: a system message with the teacher-robot rules (grounding,
    page citations, deterministic decline) plus the retrieved context blocks,
    and a human message with the student question.
X - ``call_llm``: Ollama + Qwen2.5 1.5B Q4 via langchain-ollama.
    ``answer`` orchestrates retrieve -> decline-or-prompt -> LLM and returns
    an ``Answer`` with sources (page citations for the UI/audit).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .retriever import DECLINE_MESSAGE, ContextBlock, retrieve

SYSTEM_TEMPLATE = """You are an offline teacher robot helping a school student. Answer using ONLY the textbook passages provided below.

Rules:
- Answer in the same language the student used.
- Base every answer strictly on the passages. If the passages do not cover the question, reply exactly: "{decline_message}"
- When you use a passage, cite its page number in parentheses, e.g. (page 30-31).
- Keep the answer short, clear and age-appropriate.
- The passages may still contain stray publisher marks, "sample"/"not for sale" stamps, or other non-lesson text left over from the source book. Ignore anything that is clearly not part of the lesson content itself and never repeat it back to the student.

Passages:
{context}
"""

LLM_ERROR_MESSAGE = (
    "I'm having trouble reaching the answering engine right now. "
    "Please try again in a moment."
)


class LLMConnectionError(RuntimeError):
    """Raised when the LLM backend (Ollama) can't be reached or fails."""


def decline() -> str:
    """Deterministic not-covered answer (PDF §8, rule 10)."""
    return DECLINE_MESSAGE


def format_context(blocks: Sequence[ContextBlock]) -> str:
    """Render context blocks with page labels for the prompt (W)."""
    parts = []
    for i, block in enumerate(blocks, 1):
        meta = block.metadata
        start, end = meta.get("page_start"), meta.get("page_end")
        if start is not None and end is not None and end != start:
            pages = f" (pages {start}-{end})"
        elif start is not None:
            pages = f" (page {start})"
        else:
            pages = ""
        parts.append(f"Passage {i}{pages}:\n{block.text}")
    return "\n\n".join(parts)


def _template():
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            ("human", "{question}"),
        ]
    )


def build_messages(blocks: Sequence[ContextBlock], question: str) -> list:
    """LangChain messages: system rules + context, human question (W)."""
    return _template().format_messages(
        context=format_context(blocks),
        question=question,
        decline_message=DECLINE_MESSAGE,
    )


def build_prompt(blocks: Sequence[ContextBlock], question: str) -> str:
    """Rendered prompt string (for logging/tests)."""
    return _template().format(
        context=format_context(blocks),
        question=question,
        decline_message=DECLINE_MESSAGE,
    )


def call_llm(messages: list, cfg: Optional[dict] = None) -> str:
    """Send the messages to Ollama/Qwen via langchain-ollama (X).

    Raises ``LLMConnectionError`` on any connectivity/backend failure so
    ``answer()`` can return a friendly message instead of a raw traceback —
    Ollama not being up yet on a freshly booted Pi is an expected condition,
    not a bug.
    """
    from langchain_ollama import ChatOllama

    llm_cfg = (cfg or {}).get("llm", {})
    try:
        llm = ChatOllama(
            model=llm_cfg.get("model", "qwen2.5:1.5b"),
            num_ctx=int(llm_cfg.get("num_ctx", 2048)),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        )
        response = llm.invoke(messages)
        return str(response.content).strip()
    except Exception as exc:  # noqa: BLE001 - any backend failure -> one clean error type
        raise LLMConnectionError(f"Ollama call failed: {exc}") from exc


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)
    declined: bool = False
    error: bool = False  # True when the LLM backend failed (not a content decline)


def _sources(blocks: Sequence[ContextBlock]) -> list[dict]:
    out = []
    for block in blocks:
        meta = block.metadata
        out.append(
            {
                "chunk_id": block.chunk_id,
                "parent_id": block.parent_id,
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "unit": meta.get("unit", ""),
                "chapter": meta.get("chapter", ""),
                "topic": meta.get("topic", ""),
            }
        )
    return out


def answer(
    question: str,
    store,
    backend,
    cfg: Optional[dict] = None,
    llm: Optional[Callable[[list, Optional[dict]], str]] = None,
) -> Answer:
    """Full student answer: retrieve -> decline or prompt -> LLM (W, X).

    ``llm`` is injectable for tests; default is ``call_llm``.
    """
    blocks = retrieve(question, store, backend, cfg)
    if not blocks:
        return Answer(text=decline(), declined=True)
    messages = build_messages(blocks, question)
    responder = llm or call_llm
    try:
        text = responder(messages, cfg)
    except LLMConnectionError:
        return Answer(text=LLM_ERROR_MESSAGE, sources=_sources(blocks), declined=False, error=True)
    return Answer(text=str(text).strip(), sources=_sources(blocks), declined=False)
