import re
from typing import List

CHUNK_MIN_WORDS = 40
CHUNK_MAX_WORDS = 90
CHUNK_OVERLAP_SENTENCES = 1

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?\u06d4\u061f])\s+(?=[A-Za-z0-9\u0600-\u06FF\"'])"
)

_URDU_RANGE_RE = re.compile(r"[\u0600-\u06FF]")


def detect_lang(text: str) -> str:
    """
    Simple language detection.
    Returns 'ur' if Urdu/Arabic script is common, otherwise 'en'.
    """
    if not text:
        return "en"

    urdu_chars = len(_URDU_RANGE_RE.findall(text))
    return "ur" if urdu_chars > len(text) * 0.15 else "en"


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    parts = _SENTENCE_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part.strip()]


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_text(
    text: str,
    min_words: int = CHUNK_MIN_WORDS,
    max_words: int = CHUNK_MAX_WORDS,
    overlap_sentences: int = CHUNK_OVERLAP_SENTENCES,
) -> List[str]:
    """
    Create small topic-tight chunks for direct question answering.
    """
    chunks = []

    for paragraph in split_paragraphs(text):
        sentences = split_sentences(paragraph)

        if not sentences:
            continue

        word_counts = [len(sentence.split()) for sentence in sentences]
        i = 0
        n = len(sentences)

        while i < n:
            current = []
            current_words = 0
            j = i

            while j < n and (current_words + word_counts[j] <= max_words or not current):
                current.append(sentences[j])
                current_words += word_counts[j]
                j += 1

                if current_words >= min_words:
                    break

            if current:
                chunks.append(" ".join(current).strip())

            if j >= n:
                break

            i = max(i + 1, j - overlap_sentences)

    return chunks